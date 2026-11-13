"""CPU test of the driver-side plan step (checklist items 3, 4, 7): phi assembly, expanded update
batch, and RayPPOTrainer._plan_step with fake worker groups (fake vLLM, retriever, reader, ref, actor).

Checks:
  1. collect_phis: Phi(S_{k-1}) equals the reader score of question + docs of executed searches
     before turn k; executed search -> Phi(S_{k-1}+docs); answer -> EM vs gold; final-turn search
     and invalid -> Phi(S_{k-1}); fork_phis = [true action, siblings in order];
  2. reader requests are deduplicated and rows are returned in the caller's order after the
     length-sorted round-robin dispatch;
  3. build_update_batch: originals keep z+adj on their turns and info_mask as loss mask; siblings
     have loss only on their own turn, advantage beta*(phi_j-C_j), weight 1/M, is_original 0,
     ref 0; padding rows contribute nothing and make len % world_size == 0;
  4. _plan_step for every method: one update_actor call, n_orig == bs, ref log-probs requested
     for originals only, old log-probs for every row, P0 advantages == verl GRPO outcome advantage.

Run: HF_HUB_OFFLINE=1 python -I scripts/plan/item7_plan_step_test.py
"""

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from search_r1.llm_agent.generation import GenerationConfig
from search_r1.plan.phi import PhiEngine, collect_phis, state_docs
from search_r1.plan.reader_prompt import ReaderPromptBuilder
from search_r1.plan.advantages import p3_deltas, grpo_z
from search_r1.plan.step import build_update_batch, phi_needs, fork_m_for
from verl.utils.reward_score.qa_em import em_check
from p3_fork_test import TOK, FakeRollout, Manager, make_batch

FAILS = []


def check(cond, msg):
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)


def fake_mean_logp(ids, tmask):
    """Deterministic stand-in for the reader: depends on context length and target tokens."""
    n_ctx = int((tmask == 0).sum())
    t = ids[tmask.bool()].double()
    return float((-(t % 13) / 10).mean() - 1e-3 * n_ctx)


class FakeReaderWG:
    world_size = 4

    def __init__(self):
        self.calls = []

    def compute_phi(self, data):
        b = data.batch
        n = b['input_ids'].shape[0]
        assert n % self.world_size == 0
        self.calls.append(n)
        out = []
        for r in range(n):
            L = int(b['attention_mask'][r].sum())
            out.append(fake_mean_logp(b['input_ids'][r, :L], b['target_mask'][r, :L]))
        return DataProto.from_dict(tensors={'mean_logp': torch.tensor(out)})


def expected_phi(builder, q, docs, aliases):
    rows = builder.build(q, docs, aliases)
    best = max(fake_mean_logp(torch.tensor(ids), torch.tensor([0] * (len(ids) - tl) + [1] * tl))
               for ids, tl in rows)
    return float(np.exp(best))


def rollout(bs=10, G=5, M=3, seed=0):
    cfg = GenerationConfig(max_turns=3, max_start_length=256, max_prompt_length=4096, max_response_length=128,
                           max_obs_length=256, num_gpus=1, search_url='fake', topk=3, plan_fork_m=M,
                           plan_kstar_max=4, plan_seed=seed)
    roll = FakeRollout(seed + 7)
    mgr = Manager(tokenizer=TOK, actor_rollout_wg=roll, config=cfg)
    return mgr, cfg


def make_full_batch(n_q=4, G=5):
    gb = make_batch(n_q)
    golds = [[f'ans{100 + i}', f'alias {i}'] for i in range(n_q)]
    nt = {'question': np.array([f'Question {i}: who?' for i in range(n_q)], dtype=object),
          'golden_answers': np.array(golds + [], dtype=object),
          'id': np.array([f'train_{i}' for i in range(n_q)], dtype=object),
          'data_source': np.array(['nq'] * n_q, dtype=object),
          'index': np.array(list(range(n_q)), dtype=object),
          'reward_model': np.array([{'ground_truth': {'target': g}, 'style': 'rule'} for g in golds], dtype=object)}
    batch = DataProto.from_dict(tensors=dict(gb.batch), non_tensors=nt)
    return batch.repeat(repeat_times=G, interleave=True)


# ------------------------------------------------------------------ 1-2: collect_phis
def test_collect():
    mgr, cfg = rollout()
    batch = make_full_batch()
    gen = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
    out = mgr.run_llm_loop(gen, gen.batch['input_ids'].clone())
    trace, recs, k_star = mgr.last_trace, mgr.last_sib_records, mgr.last_k_star
    # make some answers correct so EM is exercised
    for i, tr in enumerate(trace):
        for r in tr:
            if r['action'] == 'answer' and i % 2 == 0:
                r['content'] = f'ans{100 + i // 5}'
    for r in recs:
        if r['action'] == 'answer' and r['parent'] % 3 == 0:
            r['content'] = f'alias {r["parent"] // 5}'
    qs = list(batch.non_tensor_batch['question'])
    golds = [list(rm['ground_truth']['target']) for rm in batch.non_tensor_batch['reward_model']]
    wg = FakeReaderWG()
    eng = PhiEngine(TOK, wg.compute_phi, wg.world_size)
    trajs, sib_phis = collect_phis(eng, trace, recs, k_star, qs, golds, golds, {'prev_all', 'fork', 'search_turns'})
    b = ReaderPromptBuilder(TOK)
    ok_prev, ok_st, ok_fork = True, True, True
    for i, tr in enumerate(trace):
        for k in range(1, len(tr) + 1):
            e = expected_phi(b, qs[i], state_docs(tr, k - 1), golds[i])
            ok_prev &= abs(trajs[i]['phi_prev'][k] - e) < 1e-6
        for k, (a, c) in trajs[i]['search_turns'].items():
            ok_st &= tr[k - 1]['executed'] and abs(a - trajs[i]['phi_prev'][k]) < 1e-9
            ok_st &= abs(c - expected_phi(b, qs[i], state_docs(tr, k), golds[i])) < 1e-6

    def exp_action(i, r):
        if r['action'] == 'answer':
            return float(em_check(r['content'], golds[i]))
        docs = state_docs(trace[i], r['k'] - 1)
        if r['action'] == 'search' and r['executed']:
            docs = docs + list(r['docs'])
        return expected_phi(b, qs[i], docs, golds[i])

    n_forked, n_em, n_final_search = 0, 0, 0
    for i, t in enumerate(trajs):
        if t['fork_phis'] is None:
            ok_fork &= k_star[i] > len(trace[i])
            continue
        n_forked += 1
        sibs = [r for r in recs if r['parent'] == i]
        cands = [trace[i][k_star[i] - 1]] + sibs
        exp = [exp_action(i, c) for c in cands]
        ok_fork &= len(t['fork_phis']) == 1 + 3 and np.allclose(t['fork_phis'], exp, atol=1e-6)
        n_em += sum(1 for c, v in zip(cands, exp) if c['action'] == 'answer' and v == 1.0)
        n_final_search += sum(1 for c in cands if c['action'] == 'search' and not c['executed'])
    check(ok_prev, 'Phi(S_{k-1}) = reader(question + docs of executed searches before k), every turn')
    check(ok_st, 'P1 search turns: (Phi(S_{k-1}), Phi(S_k)) on executed searches only')
    check(ok_fork and n_forked > 0, f'fork phis = [true, siblings]: search/answer(EM)/invalid rules '
                                    f'({n_forked} forks, {n_em} EM hits, {n_final_search} unexecuted searches)')
    st = eng.stats
    check(st['unique'] < st['requests'] and len(wg.calls) == 1,
          f"one reader call per step, requests deduplicated ({st['requests']} -> {st['unique']})")
    return mgr, out, trajs


# ------------------------------------------------------------------ 3: build_update_batch
def test_build(mgr, out, trajs):
    recs, sib = mgr.last_sib_records, mgr.last_siblings
    n = len(out)
    L = out.batch['responses'].shape[1]
    uids = [f'g{i // 5}' for i in range(n)]
    scores = np.array([float(i % 3 == 0) for i in range(n)])
    z = grpo_z(scores, uids)
    beta, M = 0.7, 3
    adj = [dict() for _ in range(n)]
    sib_adv = [[] for _ in range(n)]
    for i, t in enumerate(trajs):
        if t['fork_phis'] is not None:
            d = p3_deltas(t['fork_phis'])
            adj[i][t['k_star']] = -beta * d[0]
            sib_adv[i] = [beta * x for x in d[1:]]
    orig = DataProto.from_dict(tensors=dict(out.batch), non_tensors={'uid': np.array(uids, dtype=object)})
    orig.batch['ref_log_prob'] = torch.randn(n, L)
    upd, n_sib, n_pad = build_update_batch(orig, sib, recs, z, adj, sib_adv, M, world_size=4)
    ub = upd.batch
    check(len(upd) % 4 == 0 and len(upd) == n + n_sib + n_pad and n_sib == len(recs),
          f'expanded batch {n} + {n_sib} siblings + {n_pad} pad rows, divisible by 4')
    tid = ub['turn_ids'][:n]
    ok = torch.equal(ub['loss_mask'][:n], out.batch['info_mask'][:, -L:].long())
    for i in range(n):
        for k in range(1, int(tid[i].max()) + 1):
            v = ub['advantages'][i][tid[i] == k]
            ok &= bool(torch.allclose(v, torch.full_like(v, z[i] + adj[i].get(k, 0.0)), atol=1e-5))
        ok &= bool((ub['advantages'][i][tid[i] == 0] == 0).all())
    check(ok, 'originals: advantage z + adj on turn tokens, 0 elsewhere; loss mask = info_mask')
    ok = True
    seen = {}
    for s, r in enumerate(recs):
        row = n + s
        j = seen.get(r['parent'], 0)
        seen[r['parent']] = j + 1
        own = ub['turn_ids'][row] == r['k']
        ok &= torch.equal(ub['loss_mask'][row].bool(), own) and bool(own.any())
        ok &= bool(torch.allclose(ub['advantages'][row][own], torch.tensor(sib_adv[r['parent']][j]).float(), atol=1e-6))
        ok &= bool((ub['advantages'][row][~own] == 0).all())
        ok &= abs(float(ub['loss_weight'][row]) - 1 / M) < 1e-7 and float(ub['is_original'][row]) == 0
        ok &= bool((ub['ref_log_prob'][row] == 0).all())
        ok &= upd.non_tensor_batch['uid'][row] == uids[r['parent']]
    check(ok, 'siblings: loss on own turn only, beta*(phi_j - C_j), weight 1/M, non-original, ref 0, parent uid')
    pad = slice(n + n_sib, len(upd))
    check(all(float(ub[k][pad].abs().sum()) == 0 for k in ('advantages', 'loss_mask', 'loss_weight', 'is_original')),
          'padding rows have zero advantage, loss mask, weight and is_original')
    check(float(ub['is_original'].sum()) == n, 'N_orig = number of original trajectories')


# ------------------------------------------------------------------ 4: _plan_step
class FakeActorWG:
    world_size = 4

    def __init__(self):
        self.updates, self.logprob_rows = [], []

    def compute_log_prob(self, data):
        self.logprob_rows.append(len(data))
        return DataProto.from_dict(tensors={'old_log_probs': torch.zeros(len(data), data.batch['responses'].shape[1])})

    def update_actor(self, data):
        self.updates.append(data)
        return DataProto(meta_info={'metrics': {'actor/pg_loss': [0.0]}})


class FakeRefWG:
    def __init__(self):
        self.rows = []

    def compute_ref_log_prob(self, data):
        self.rows.append(len(data))
        return DataProto.from_dict(tensors={'ref_log_prob': torch.zeros(len(data), data.batch['responses'].shape[1])})


def em_reward(data):
    out = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
    for i in range(len(data)):
        L = int(data.batch['attention_mask'][i, data.batch['prompts'].shape[1]:].sum())
        out[i, max(L - 1, 0)] = float(i % 4 == 0)
    return out


def make_trainer(method):
    cfg = OmegaConf.create({
        'plan': {'enable': True, 'method': method, 'mode': 'train', 'M': 3 if method == 'p3' else 0, 'beta': 0.5,
                 'alpha': 0.3, 'reader_target': 'main', 'negctrl_map': None, 'pilot_dump_dir': None,
                 'reader': {'path': 'unused'}},
        'actor_rollout_ref': {'actor': {'plan_single_step': True, 'state_masking': True, 'use_kl_loss': True,
                                        'plan_loss_len': 128},
                              'rollout': {'temperature': 1.0, 'top_p': 1.0, 'top_k': -1, 'n_agent': 5}},
        'trainer': {'seed': 3}, 'max_turns': 3})
    tr = object.__new__(RayPPOTrainer)
    tr.config = cfg
    tr._plan_init()
    tr.actor_rollout_wg, tr.ref_policy_wg = FakeActorWG(), FakeRefWG()
    tr.reward_fn = em_reward
    tr.global_steps = 1
    if tr.plan_use_reader:
        tr.reader_wg = FakeReaderWG()
        tr.phi_engine = PhiEngine(TOK, tr.reader_wg.compute_phi, 4)
    return tr


def test_plan_step():
    for method in ('p0', 'p1', 'p2_all', 'p2_sel', 'p3'):
        tr = make_trainer(method)
        mgr, _ = rollout(M=tr.plan_fork_m)
        batch = make_full_batch()
        gen = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
        metrics, timing = {}, {}
        view = tr._plan_step(batch, gen, mgr, metrics, timing)
        n = len(view)
        upd = tr.actor_rollout_wg.updates
        ok = len(upd) == 1 and float(upd[0].batch['is_original'].sum()) == n
        ok &= tr.ref_policy_wg.rows == [n] and tr.actor_rollout_wg.logprob_rows == [len(upd[0])]
        ok &= len(upd[0]) % 4 == 0 and upd[0].meta_info['loss_len_const'] == 128.0
        n_sib = int(metrics['plan/n_sib'])
        ok &= (n_sib > 0) == (method == 'p3')
        ok &= (method in ('p0',)) == (not tr.plan_use_reader)
        if method == 'p0':
            score = em_reward(view)
            resp_mask = view.batch['info_mask'][:, -view.batch['responses'].shape[1]:]
            ref_adv, _ = core_algos.compute_grpo_outcome_advantage(score, resp_mask, np.array(
                [f'nq/{i // 5}' for i in range(n)], dtype=object))
            ok &= bool(torch.allclose(view.batch['advantages'], ref_adv * resp_mask, atol=1e-5))
        check(ok, f'_plan_step[{method}]: one update, n_orig={n}, siblings={n_sib}, ref on originals only')


def main():
    torch.manual_seed(0)
    mgr, out, trajs = test_collect()
    test_build(mgr, out, trajs)
    test_plan_step()
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
