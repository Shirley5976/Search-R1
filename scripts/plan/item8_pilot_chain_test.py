"""CPU test of the pilot chain (checklist item 8; R012a dump -> R012b sketches -> R014 check -> R015).

A tiny random Qwen2 model stands in for theta_w; fake vLLM / retriever / reader / ref as in item7.
  1. fit_pilot writes one dump per batch with M=4 siblings and the phi variants;
  2. r012b (CPU) stores one sketch per score vector;
  3. B0b logic on CPU: for P0, P2-all and P3, the gradient of the actual training loss built by
     advantages.py + step.py (dp_actor loss formula) equals formula.training_coef applied to the
     per-vector sketches. Sketching is linear, so with an fp32 replica the two agree to float
     round-off (relative L2 < 1e-4) and beta != 0 must move the gradient by far more than that;
  4. r014_check passes on the exact-subset questions;
  5. r015 statistics: trace_cov == numpy, promotion rule on synthetic vectors (P3 wins when it
     cuts variance cheaply, P0 wins when the cost ratio eats the gain), full r015 run on the dump.

Run: HF_HUB_OFFLINE=1 python -I scripts/plan/item8_pilot_chain_test.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from transformers import Qwen2Config, Qwen2ForCausalLM

from verl import DataProto
from search_r1.plan import formula
from search_r1.plan.phi import PhiEngine
from search_r1.plan.sketch import grad_sketch, rel_l2
from search_r1.plan.step import build_update_batch, compute_plan_advantages
from p3_fork_test import TOK, make_batch
from item7_plan_step_test import FakeReaderWG, em_reward, make_trainer, rollout
import r015_pilot_analysis as R015
from r012b_pilot_sketch import Replica, low_var_kl

FAILS = []
PY = sys.executable


def check(c, m):
    print(f"[{'PASS' if c else 'FAIL'}] {m}")
    if not c:
        FAILS.append(m)


class RefWG:
    def compute_ref_log_prob(self, data):
        g = torch.Generator().manual_seed(len(data))
        L = data.batch['responses'].shape[1]
        return DataProto.from_dict(tensors={'ref_log_prob': -12.0 + 0.3 * torch.randn(len(data), L, generator=g)})


def batch_dict(qs, G=5):
    gb = make_batch(len(qs))
    golds = [[f'ans{100 + i}'] for i in qs]
    return {'input_ids': gb.batch['input_ids'], 'attention_mask': gb.batch['attention_mask'],
            'position_ids': gb.batch['position_ids'],
            'question': np.array([f'Question {i}: who?' for i in qs], dtype=object),
            'id': np.array([f'train_{i}' for i in qs], dtype=object),
            'data_source': np.array(['nq' if i % 2 else 'hotpotqa' for i in qs], dtype=object),
            'index': np.array(qs, dtype=object),
            'reward_model': np.array([{'ground_truth': {'target': g}, 'style': 'rule'} for g in golds], dtype=object)}


def make_dump(tmp, n_batches=3, q_per_batch=4):
    qids = [i for i in range(n_batches * q_per_batch)]
    neg = {f"{'nq' if i % 2 else 'hotpotqa'}/train_{i}": {'qid': 'x', 'aliases': [f'alias {i + 7}']} for i in qids}
    with open(os.path.join(tmp, 'neg.json'), 'w') as f:
        json.dump(neg, f)
    tr = make_trainer('p3')
    tr.config.plan.mode = 'pilot'
    tr.config.plan.M = 4
    tr.config.plan.negctrl_map = os.path.join(tmp, 'neg.json')
    tr.config.plan.pilot_dump_dir = os.path.join(tmp, 'dump')
    tr.config.trainer.n_gpus_per_node, tr.config.trainer.nnodes = 4, 1
    tr._plan_init()
    tr.ref_policy_wg = RefWG()
    tr.reader_wg = FakeReaderWG()
    tr.phi_engine = PhiEngine(TOK, tr.reader_wg.compute_phi, 4)
    tr.reward_fn = em_reward
    tr.train_dataloader = [batch_dict(qids[b * q_per_batch:(b + 1) * q_per_batch]) for b in range(n_batches)]
    mgr, _ = rollout(M=4)
    tr.fit_pilot(mgr)
    calib = [f"{'nq' if i % 2 else 'hotpotqa'}/train_{i}" for i in qids[:len(qids) // 2]]
    evals = [f"{'nq' if i % 2 else 'hotpotqa'}/train_{i}" for i in qids[len(qids) // 2:]]
    for name, ids in (('pilot_calib', calib), ('pilot_eval', evals)):
        with open(os.path.join(tmp, f'{name}.txt'), 'w') as f:
            f.write('\n'.join(ids) + '\n')
    return os.path.join(tmp, 'dump')


def tiny_model(tmp):
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(TOK), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                      tie_word_embeddings=True)
    m = Qwen2ForCausalLM(cfg)
    path = os.path.join(tmp, 'model')
    m.save_pretrained(path)
    TOK.save_pretrained(path)
    return path


# ------------------------------------------------------------------ 3: B0b logic on CPU
def records_to_trajs(recs):
    trajs = []
    for r in recs:
        trajs.append(dict(n_turns=r['n_turns'], k_star=r['k_star'],
                          phi_prev={k: r['phi_prev']['main'][k - 1] for k in range(1, r['n_turns'] + 1)},
                          search_turns={}, fork_phis=None if r['fork'] is None else list(r['fork']['main'])))
    return trajs


def impl_gradient(rep, dump_dir, method, beta, M, kl_coef, L_const):
    """Gradient of the single-step loss (dp_actor formula) on the batch step.py builds."""
    rep.model.zero_grad(set_to_none=True)
    files = sorted(f for f in os.listdir(dump_dir) if f.endswith('.json'))
    n_orig = 0
    batches = []
    for f in files:
        meta = json.load(open(os.path.join(dump_dir, f)))
        tens = torch.load(os.path.join(dump_dir, f[:-5] + '.pt'))
        recs = meta['trajectories']
        n_orig += len(recs)
        batches.append((recs, tens))
    for recs, tens in batches:
        o = {k: v for k, v in tens['orig'].items()}
        uids = [r['uid'] for r in recs]
        orig = DataProto.from_dict(tensors=o, non_tensors={'uid': np.array(uids, dtype=object)})
        sib = DataProto.from_dict(tensors=dict(tens['sib']))
        sib_recs = [None] * len(sib)
        for i, r in enumerate(recs):
            if r['fork'] is not None:
                for srow in r['fork']['sib_rows'][:M]:
                    sib_recs[srow] = {'parent': i, 'k': r['k_star']}
        # P3 with M < dumped siblings uses the first M of each parent (same rule as formula.p3_coef)
        keep = [s for s, x in enumerate(sib_recs) if x is not None]
        sib = DataProto.from_dict(tensors={k: v[keep] for k, v in sib.batch.items()}) if keep else None
        sib_recs = [sib_recs[s] for s in keep]
        trajs = records_to_trajs(recs)
        for t in trajs:
            if t['fork_phis'] is not None:
                t['fork_phis'] = t['fork_phis'][:M + 1]
        z, adj, sib_adv = compute_plan_advantages(method, [r['score'] for r in recs], uids, trajs, beta, 0.0)
        upd, _, _ = build_update_batch(orig, sib, sib_recs, z, adj, sib_adv, M, world_size=1)
        ub = upd.batch
        for r in range(len(upd)):
            t = {k: ub[k] for k in ('prompts', 'input_ids', 'attention_mask', 'turn_ids')}
            logp, tid, _ = rep.row_forward(t, r, False)
            P = ub['prompts'].shape[1]
            r_valid = int(ub['attention_mask'][r, P:].sum())
            idx = torch.nonzero(ub['turn_ids'][r, :r_valid] > 0).view(-1)
            A = ub['advantages'][r, :r_valid][idx]
            mask = ub['loss_mask'][r, :r_valid][idx].float()
            w, iso = float(ub['loss_weight'][r]), float(ub['is_original'][r])
            ratio = torch.exp(logp - logp.detach())
            loss = (-A * ratio * mask * w).sum() / (n_orig * L_const)
            if iso:
                kld = low_var_kl(logp, ub['ref_log_prob'][r, :r_valid][idx].float())
                loss = loss + kl_coef * (kld * mask).sum() / (n_orig * L_const)
            (-loss).backward()  # accumulate g = -grad(loss)
    g = grad_sketch(rep.model, D=2 ** 14)
    rep.model.zero_grad(set_to_none=True)
    return g


def formula_gradient(sketch_dir, method, beta, M, kl_coef, L_const):
    files = sorted(f for f in os.listdir(sketch_dir) if f.startswith('q_'))
    qs = [torch.load(os.path.join(sketch_dir, f)) for f in files]
    n_orig = sum(len(q['records']) for q in qs)
    coef = formula.training_coef([q['records'] for q in qs], method, beta, M, kl_coef, n_orig, L_const)
    out = 0
    for w, q in enumerate(qs):
        c = {k[1:]: v for k, v in coef.items() if k[0] == w}
        out = out + formula.combine(c, q['index'], q['sketch'])
    return out


# ------------------------------------------------------------------ 5: statistics
def test_stats():
    rng = np.random.RandomState(0)
    X, Y = rng.randn(30, 7), rng.randn(30, 7)
    G = X @ Y.T
    ref = np.trace(np.cov(X.T, Y.T)[:7, 7:])
    check(abs(R015.trace_cov(G, np.ones(30)) - ref) < 1e-9, 'trace_cov == numpy trace of the cross-covariance')
    w = np.zeros(30)
    w[:15] = 2
    ref2 = np.trace(np.cov(np.repeat(X[:15], 2, 0).T, np.repeat(Y[:15], 2, 0).T)[:7, 7:])
    check(abs(R015.trace_cov(G, w) - ref2) < 1e-9, 'trace_cov with multiplicities == duplicated-sample covariance')


def synthetic(rho_p3, rho_p2, n=200, D=256, seed=1):
    rng = np.random.RandomState(seed)
    meta, vecs = [], {}
    u = rng.randn(n, D)
    e = rng.randn(n, D)
    g = u + e
    vecs['g'] = torch.tensor(g, dtype=torch.float32)
    for t in R015.y_types():
        rho = rho_p3 if t[0].startswith('p3') else rho_p2
        noise = rng.randn(n, D)
        # Y = a*u_part + b*noise with corr(g, Y) set so that rho2 ~ rho
        y = np.sqrt(rho) * g / np.sqrt(2) + np.sqrt(1 - rho) * noise
        vecs[t] = torch.tensor(y, dtype=torch.float32)
    for i in range(n):
        meta.append({'split': 'pilot_calib' if i < n // 2 else 'pilot_eval', 'source': 'nq' if i % 2 else 'hotpotqa'})
    return meta, vecs


def test_promotion():
    meta, vecs = synthetic(0.6, 0.0)
    costs = {'p0': 1.0, 'p2_all': 1.1, 'p3_m1': 1.2, 'p3_m2': 1.3, 'p3_m4': 1.5}
    res = R015.analyse(meta, vecs, costs, n_boot=300)
    r = res['rows'][('p3_m1', 'main')]
    check(res['winner'] == 'p3' and res['p3_M_star'] == 1 and abs(r['rho2_eval'] - 0.6) < 0.1,
          f"P3 wins with rho2~0.6 and cheap cost (winner {res['winner']}, M* {res['p3_M_star']}, rho2 {r['rho2_eval']:.2f})")
    check(abs(r['W_ratio_eval'] - 1.2 * (1 - r['rho2_eval']) - r['fit_error_term_eval'] / r['V0_eval'] * 1.2) < 1e-6,
          'W/W0 = (c_m/c0) (1 - rho2) + fit-error term / V0  (V_beta_hat decomposition)')
    check(res['predicted_training_ranking'][0] == 'p3', 'predicted ranking puts P3 first')
    costs2 = {'p0': 1.0, 'p2_all': 1.1, 'p3_m1': 3.0, 'p3_m2': 3.0, 'p3_m4': 3.0}
    res2 = R015.analyse(meta, vecs, costs2, n_boot=300)
    check(res2['winner'] == 'p0', f"P0 wins when c3/c0 = 3 > 1/(1-rho2) (winner {res2['winner']})")


def main():
    test_stats()
    test_promotion()
    tmp = tempfile.mkdtemp(prefix='plan_item8_')
    dump = make_dump(tmp)
    files = sorted(os.listdir(dump))
    check(sum(f.endswith('.done') for f in files) == 3, f'fit_pilot wrote 3 dump batches ({files[:4]}...)')
    meta0 = json.load(open(os.path.join(dump, 'pilot_0000.json')))
    forked = [r for r in meta0['trajectories'] if r['fork'] is not None]
    check(forked and all(len(r['fork'][v]) == 5 for r in forked for v in ('main', 'neg', 'lex')),
          'dump: forked trajectories carry 1+4 candidate phis for main / neg / lex')
    model = tiny_model(tmp)
    out = os.path.join(tmp, 'sketch')
    cmd = [PY, '-I', os.path.join(HERE, 'r012b_pilot_sketch.py'), '--dump_dir', dump, '--model', model,
           '--out_dir', out, '--device', 'cpu', '--D', str(2 ** 14), '--exact_prefix', 'model.layers.1.',
           '--exact_questions', '6', '--dtype', 'fp32', '--split_files', os.path.join(tmp, 'pilot_calib.txt'),
           os.path.join(tmp, 'pilot_eval.txt')]
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, 'RANK': '0', 'WORLD_SIZE': '1'})
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:])
    qfiles = sorted(f for f in os.listdir(out) if f.startswith('q_'))
    check(p.returncode == 0 and len(qfiles) == 12, f'r012b wrote {len(qfiles)} question files')

    rep = Replica(model, 'cpu', grad_ckpt=True, dtype=torch.float32)
    for method, beta, M in (('p0', 0.0, 4), ('p2_all', 0.7, 4), ('p3', 0.7, 4), ('p3', 0.7, 2)):
        gi = impl_gradient(rep, dump, method, beta, M, 0.001, 500.0)
        gf = formula_gradient(out, method, beta, M, 0.001, 500.0)
        g0 = formula_gradient(out, 'p0', 0.0, M, 0.001, 500.0)
        err = rel_l2(gf, gi)
        sep = rel_l2(gf, g0) if method != 'p0' else float('nan')
        check(err < 1e-4 and (method == 'p0' or sep > 100 * err),
              f'B0b logic [{method}, beta={beta}, M={M}]: |formula - impl| / |impl| = {err:.2e}; '
              f'|formula(beta) - formula(0)| / |formula(0)| = {sep:.2e}')

    p = subprocess.run([PY, '-I', os.path.join(HERE, 'r014_check.py'), '--sketch_dir', out], capture_output=True,
                       text=True)
    print(p.stdout[-1500:])
    check(p.returncode == 0, 'r014_check passes on the exact-subset questions')

    costs = os.path.join(tmp, 'costs.json')
    json.dump({'p0': 10.0, 'p2_all': 11.0, 'p3_m1': 12.0, 'p3_m2': 13.0, 'p3_m4': 15.0}, open(costs, 'w'))
    p = subprocess.run([PY, '-I', os.path.join(HERE, 'r015_pilot_analysis.py'), '--sketch_dir', out, '--costs',
                        costs, '--out_prefix', os.path.join(tmp, 'pilot_result'), '--n_boot', '200'],
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-3000:])
    ok = p.returncode == 0
    if ok:
        res = json.load(open(os.path.join(tmp, 'pilot_result.json')))
        ok = res['winner'] in ('p0', 'p2_all', 'p3') and 'p3_m4|neg' in res['rows'] and 'reader_vs_lex' in res['diagnostics']
    check(ok, 'r015 runs end to end on the dump and writes the W table, winner and diagnostics')
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}   (artifacts in {tmp})")
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
