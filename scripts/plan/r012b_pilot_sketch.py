"""R012b: per-sample score-vector sketches on theta_w from the R012a pilot dump (also R014, B0b formula path).

One process per GPU, each with a bf16 replica of the policy (gradient checkpointing); questions are
dealt to ranks by position, so `torchrun --nproc_per_node 4` (no collectives) or 4 separate
launches with RANK/WORLD_SIZE both work. For every question of the dump (G trajectories) it
backpropagates, one scalar at a time,
  ('turn', i, k)  sum of log pi over the tokens of decision turn k of trajectory i
  ('kl', i)       sum over policy tokens of low_var_kl(pi, ref)            (ref log-probs from the dump)
  ('sib', i, j)   sum of log pi over the tokens of sibling j at k*_i
and stores the count-sketch (D = 2^18) of each gradient plus its exact squared norm:
  OUT/q_<batch>_<n>.pt = {qid, uid, split, index: [keys], sketch: (n_vec, D) fp32, sqnorm: (n_vec,)}
Re-running skips finished questions.

--exact_prefix P [--exact_questions N] (R014): for the first N questions (in dump order) also
accumulate exact fp32 aggregates, restricted to parameters whose name starts with one of P, of the
vectors g5 and Y for p2_all / p2_sel / p3_m4 (main reader), and sketches of the same restricted
vectors, so r014_check.py can compare exact and sketched traces.

--b0b_method p0|p2_all|p3 --b0b_beta B --b0b_M M (B0b formula path): instead of the R014
aggregates, accumulate the exact restricted training gradient of the whole dump (formula.training_coef),
partial per rank (OUT/b0b_exact_rank<r>.pt); the full-parameter sketch is combined offline from the
per-vector sketches by r004_compare.py.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from search_r1.plan.sketch import sketch_add, clean_name, D_DEFAULT
from search_r1.plan import formula


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump_dir', required=True)
    ap.add_argument('--model', required=True, help='HF checkpoint of theta_w (actor/global_step_*)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--D', type=int, default=D_DEFAULT)
    ap.add_argument('--sketch_seed', type=int, default=0)
    ap.add_argument('--kl_coef', type=float, default=0.001)
    ap.add_argument('--no_grad_ckpt', action='store_true')
    ap.add_argument('--exact_prefix', nargs='*', default=[])
    ap.add_argument('--exact_questions', type=int, default=20)
    ap.add_argument('--b0b_method', default=None)
    ap.add_argument('--b0b_beta', type=float, default=0.0)
    ap.add_argument('--b0b_M', type=int, default=0)
    ap.add_argument('--loss_len', type=float, default=500.0)
    ap.add_argument('--device', default='cuda', help='cuda (one GPU per rank) or cpu (tests)')
    ap.add_argument('--dtype', default='bf16', choices=['bf16', 'fp32'], help='fp32 only for exactness tests')
    ap.add_argument('--split_files', nargs='*', default=['prereg/splits/pilot_calib.txt', 'prereg/splits/pilot_eval.txt'])
    return ap.parse_args()


def load_split(paths):
    out = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        with open(os.path.join(ROOT, p) if not os.path.isabs(p) else p) as f:
            for line in f:
                if line.strip():
                    out[line.strip()] = name
    return out


def low_var_kl(logp, ref):
    kl = ref - logp
    return torch.clamp(torch.exp(kl) - kl - 1, min=-10, max=10)


class Replica:

    def __init__(self, path, device, grad_ckpt=True, dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM
        attn = 'flash_attention_2' if torch.device(device).type == 'cuda' else 'sdpa'
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation=attn)
        self.model.to(device)
        if grad_ckpt:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.train()  # HF only checkpoints in train mode; Qwen2 dropout is 0
        self.params = [(clean_name(n), p) for n, p in self.model.named_parameters()]
        self.device = device

    def row_forward(self, t, r, need_kl, ref=None):
        """Log-probs (fp32) of the policy tokens of row r, their turn ids, and per-token KL."""
        P = t['prompts'].shape[1]
        am = t['attention_mask'][r].bool()
        ids = t['input_ids'][r][am].to(self.device)
        p_valid = int(t['attention_mask'][r, :P].sum())
        r_valid = int(t['attention_mask'][r, P:].sum())
        tid = t['turn_ids'][r, :r_valid]
        idx = torch.nonzero(tid > 0).view(-1)
        pos = torch.arange(ids.numel(), device=self.device).unsqueeze(0)
        hidden = self.model.model(input_ids=ids.unsqueeze(0), position_ids=pos, use_cache=False).last_hidden_state[0]
        h = hidden[(p_valid + idx - 1).to(self.device)]
        labels = ids[(p_valid + idx).to(self.device)]
        logp = torch.log_softmax(self.model.lm_head(h).float(), -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        kld = None
        if need_kl:
            kld = low_var_kl(logp, ref[r, :r_valid][idx].to(self.device).float())
        return logp, tid[idx].to(self.device), kld

    def grads(self, scalars, on_grad):
        """Backpropagate each (key, scalar) separately through one graph; on_grad(key) reads .grad."""
        for n, (key, s) in enumerate(scalars):
            self.model.zero_grad(set_to_none=True)
            s.backward(retain_graph=n < len(scalars) - 1)
            on_grad(key)
        self.model.zero_grad(set_to_none=True)


def question_groups(meta):
    groups = {}
    for rec in meta['trajectories']:
        groups.setdefault(rec['uid'], []).append(rec)
    return [sorted(v, key=lambda r: r['row']) for v in groups.values()]


def main():
    a = parse()
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', rank))
    if a.device == 'cuda':
        torch.cuda.set_device(local)
        dev = torch.device('cuda', local)
    else:
        dev = torch.device('cpu')
    os.makedirs(a.out_dir, exist_ok=True)
    split_of = load_split(a.split_files)

    batches = sorted(glob.glob(os.path.join(a.dump_dir, 'pilot_*.json')))
    batches = [b for b in batches if os.path.exists(b[:-5] + '.done')]
    work = []  # (batch path, question index, records)
    for b in batches:
        meta = json.load(open(b))
        for n, recs in enumerate(question_groups(meta)):
            work.append((b, n, recs))
    print(f'[rank {rank}] {len(work)} questions in {len(batches)} dump batches')

    rep = Replica(a.model, dev, grad_ckpt=not a.no_grad_ckpt,
                  dtype=torch.float32 if a.dtype == 'fp32' else torch.bfloat16)
    exact_names = None
    if a.exact_prefix:
        exact_names = [n for n, _ in rep.params if any(n.startswith(p) for p in a.exact_prefix)]
        assert exact_names, f'no parameter matches {a.exact_prefix}'
        n_ex = sum(p.numel() for n, p in rep.params if n in exact_names)
        print(f'[rank {rank}] exact subset: {len(exact_names)} tensors, {n_ex / 1e6:.1f}M params')
    b0b = a.b0b_method is not None
    b0b_coef, b0b_acc = None, None
    if b0b:
        assert exact_names, 'B0b needs --exact_prefix (e.g. model.layers.0. model.layers.1. model.layers.2.)'
        all_q = [recs for _, _, recs in work]
        n_orig = sum(len(r) for r in all_q)
        b0b_coef = formula.training_coef(all_q, a.b0b_method, a.b0b_beta, a.b0b_M, a.kl_coef, n_orig, a.loss_len)
        b0b_acc = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in rep.params if n in exact_names}

    cache_t = {}
    for w, (bpath, qn, recs) in enumerate(work):
        if w % world != rank:
            continue
        out_path = os.path.join(a.out_dir, f'q_{os.path.basename(bpath)[6:10]}_{qn:03d}.pt')
        if os.path.exists(out_path) and not b0b:  # B0b re-accumulates the exact gradient, never skips
            continue
        if bpath not in cache_t:
            cache_t = {bpath: torch.load(bpath[:-5] + '.pt', map_location='cpu')}
        tens = cache_t[bpath]
        t0 = time.time()
        index, sk, sq, sk_sub = [], [], [], []
        exact_aggs = None
        if exact_names and not b0b and w < a.exact_questions:
            coefs = {'g5': formula.g5_coef(recs, a.kl_coef), 'p2_all': formula.p2_all_coef(recs),
                     'p2_sel': formula.p2_sel_coef(recs), 'p3_m4': formula.p3_coef(recs, 4)}
            exact_aggs = {m: {n: torch.zeros_like(p, dtype=torch.float32) for n, p in rep.params if n in exact_names}
                          for m in coefs}

        def on_grad(key):
            s = torch.zeros(a.D, dtype=torch.float32, device=dev)
            tot = 0.0
            sub = torch.zeros(a.D, dtype=torch.float32, device=dev) if exact_aggs is not None else None
            for n, p in rep.params:
                if p.grad is None:
                    continue
                sketch_add(s, n, p.grad, 0, a.sketch_seed)
                tot += float(p.grad.float().pow(2).sum())
                if exact_names and n in exact_names:
                    if sub is not None:
                        sketch_add(sub, n, p.grad, 0, a.sketch_seed)
                    if exact_aggs is not None:
                        for m, agg in exact_aggs.items():
                            c = coefs[m].get(key, 0.0)
                            if c:
                                agg[n].add_(p.grad.float(), alpha=c)
                    if b0b:
                        c = b0b_coef.get((w,) + key, 0.0)
                        if c:
                            b0b_acc[n].add_(p.grad.float(), alpha=c)
            index.append(key)
            sk.append(s.cpu())
            sq.append(tot)
            if sub is not None:
                sk_sub.append(sub.cpu())

        for i, rec in enumerate(recs):
            row = rec['row']
            logp, tid, kld = rep.row_forward(tens['orig'], row, True, tens['orig']['ref_log_prob'])
            scalars = [(('turn', i, k), logp[tid == k].sum()) for k in range(1, rec['n_turns'] + 1)
                       if bool((tid == k).any())]
            scalars.append((('kl', i), kld.sum()))
            rep.grads(scalars, on_grad)
            if rec['fork'] is not None:
                k = rec['k_star']
                for j, srow in enumerate(rec['fork']['sib_rows'], 1):
                    logp_s, tid_s, _ = rep.row_forward(tens['sib'], srow, False)
                    rep.grads([(('sib', i, j), logp_s[tid_s == k].sum())], on_grad)
        out = {'qid': recs[0]['qid'], 'uid': recs[0]['uid'], 'split': split_of.get(recs[0]['qid'], 'unknown'),
               'batch': os.path.basename(bpath), 'records': recs, 'index': index,
               'sketch': torch.stack(sk), 'sqnorm': torch.tensor(sq, dtype=torch.float64),
               'seconds': time.time() - t0, 'D': a.D, 'sketch_seed': a.sketch_seed, 'kl_coef': a.kl_coef}
        if exact_aggs is not None:
            out['exact_names'] = exact_names
            out['sketch_sub'] = torch.stack(sk_sub)
            ex_path = os.path.join(a.out_dir, f'exact_{os.path.basename(out_path)}')
            torch.save({m: {n: v.cpu() for n, v in agg.items()} for m, agg in exact_aggs.items()}, ex_path)
        torch.save(out, out_path + '.tmp')
        os.replace(out_path + '.tmp', out_path)
        print(f'[rank {rank}] {out["qid"]} ({out["split"]}): {len(index)} vectors, {out["seconds"]:.1f}s')

    if b0b:
        torch.save({n: v.cpu() for n, v in b0b_acc.items()}, os.path.join(a.out_dir, f'b0b_exact_rank{rank}.pt'))
        print(f'[rank {rank}] saved partial exact training gradient on {len(b0b_acc)} tensors')


if __name__ == '__main__':
    main()
