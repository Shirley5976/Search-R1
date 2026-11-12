"""CPU test of the plan additions to LLMGenerationManager.run_llm_loop (checklist item 2).

A scripted fake policy replaces vLLM and a fake retriever replaces the HTTP server. Checks:
  1. turn_ids mark exactly the response tokens (== info_mask on the response part) and the tokens
     of turn k decode to the turn-k response;
  2. the trace has one record per decision turn and matches the generated actions;
  3. at turn k* every reaching trajectory gets M siblings sampled from the *same* rolling prefix,
     in the same generate call as the true action; trajectories ending before k* get none;
  4. a sibling row = parent's right side before turn k* + its own response (turn id k*), with the
     same prompt as the parent; siblings are never continued;
  5. sibling searches go to the same retrieval request as the originals (one request per turn);
  6. validation mode never forks.

Run: python -I scripts/plan/p3_fork_test.py   (needs the Qwen2.5-3B tokenizer in the HF cache)
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from transformers import AutoTokenizer

from verl import DataProto
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

TOK = AutoTokenizer.from_pretrained(os.environ.get('PLAN_TEST_TOKENIZER', 'Qwen/Qwen2.5-3B'))
FAILS = []


def check(cond, msg):
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)


class FakeRollout:
    """Each call samples per row: search (p=.5), answer (p=.35) or invalid text."""

    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
        self.calls = []

    def generate_sequences(self, data):
        ids = data.batch['input_ids']
        self.calls.append(ids.clone())
        outs = []
        for r in range(ids.shape[0]):
            u = self.rng.rand()
            tag = self.rng.randint(1000)
            if u < 0.5:
                outs.append(f'<think>t{tag}</think><search> query {tag} </search> trailing')
            elif u < 0.85:
                outs.append(f'<think>t{tag}</think><answer> ans{tag} </answer>')
            else:
                outs.append(f'just text {tag}')
        resp = TOK(outs, add_special_tokens=False, return_tensors='pt', padding='longest')['input_ids']
        return DataProto.from_dict({'responses': resp})


class Manager(LLMGenerationManager):

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.search_calls = []

    def _batch_search(self, queries):
        self.search_calls.append(list(queries))
        return {'result': [[{'document': {'contents': f'T{q}\nbody of {q} doc{d}'}} for d in range(3)]
                           for q in queries]}


def make_batch(bs):
    prompts = [f'Question {i}: who?' for i in range(bs)]
    enc = TOK(prompts, add_special_tokens=False, return_tensors='pt', padding='longest', padding_side='left')
    attn = enc['attention_mask']
    pos = (attn.cumsum(-1) - 1).clamp_min(0)
    return DataProto.from_dict({'input_ids': enc['input_ids'], 'attention_mask': attn, 'position_ids': pos})


def run(bs=12, M=3, max_turns=3, seed=0, validation=False):
    cfg = GenerationConfig(max_turns=max_turns, max_start_length=256, max_prompt_length=4096,
                           max_response_length=128, max_obs_length=256, num_gpus=1,
                           search_url='fake', topk=3, plan_fork_m=M, plan_kstar_max=max_turns + 1, plan_seed=seed)
    roll = FakeRollout(seed + 100)
    mgr = Manager(tokenizer=TOK, actor_rollout_wg=roll, config=cfg, is_validation=validation)
    gb = make_batch(bs)
    out = mgr.run_llm_loop(gb, gb.batch['input_ids'].clone())
    return mgr, roll, out


def main():
    pad = TOK.pad_token_id
    M = 3
    mgr, roll, out = run(M=M)
    resp = out.batch['responses']
    tid = out.batch['turn_ids']
    L = resp.shape[1]
    info = out.batch['info_mask'][:, -L:]
    check(torch.equal((tid > 0).long(), info.long()), 'turn_ids > 0 coincides with the response-part info_mask')

    trace, k_star = mgr.last_trace, mgr.last_k_star
    ok_trace = True
    for i, tr in enumerate(trace):
        ks = [r['k'] for r in tr]
        ok_trace &= ks == list(range(1, len(ks) + 1))
        for r in tr:
            toks = resp[i][tid[i] == r['k']]
            text = TOK.decode(toks, skip_special_tokens=True)
            if r['action'] in ('search', 'answer'):
                ok_trace &= f"<{r['action']}>" in text and r['content'] in text
            if r['action'] == 'search' and r['k'] <= 3:
                ok_trace &= r['executed'] and len(r['docs']) == 3
        # turn ids never exceed the number of recorded turns
        ok_trace &= int(tid[i].max()) <= len(tr)
    check(ok_trace, 'trace records one entry per decision turn and matches the turn-k tokens')

    recs = mgr.last_sib_records
    reach = [i for i in range(len(trace)) if k_star[i] <= len(trace[i])]
    expected = sorted(p for p in reach for _ in range(M))
    check(sorted(r['parent'] for r in recs) == expected,
          f'siblings exactly for trajectories reaching k* ({len(reach)} x M={M}); none for early stoppers')
    check(all(r['k'] == k_star[r['parent']] for r in recs), 'every sibling sits at its parent\'s k*')

    # same prefix, same generate call: find the call for each k and compare prompt rows
    ok_prefix = True
    for r in recs:
        call = roll.calls[r['k'] - 1]
        n_active = sum(1 for tr in trace if len(tr) >= r['k'])
        parent_pos = sum(1 for i in range(r['parent']) if len(trace[i]) >= r['k'])
        sib_rows = call[n_active:]
        ok_prefix &= any(torch.equal(call[parent_pos], row) for row in sib_rows)
        ok_prefix &= call.shape[0] == n_active + M * sum(1 for i in reach if k_star[i] == r['k'])
    check(ok_prefix, 'siblings are generated in the same call and from the identical rolling prefix')

    sib = mgr.last_siblings
    ok_rows = True
    s_resp, s_tid = sib.batch['responses'], sib.batch['turn_ids']
    for s, r in enumerate(recs):
        p, k = r['parent'], r['k']
        ok_rows &= torch.equal(sib.batch['prompts'][s], out.batch['prompts'][p])
        before = resp[p][(tid[p] < k) | (tid[p] == 0)]
        # parent's tokens up to (excluding) turn k: everything before the first turn-k token
        first_k = (tid[p] == k).nonzero()
        prefix_len = int(first_k[0]) if len(first_k) else int((resp[p] != pad).sum())
        ok_rows &= torch.equal(s_resp[s][:prefix_len], resp[p][:prefix_len])
        ok_rows &= bool((s_tid[s][:prefix_len] < k).all())
        own = s_resp[s][s_tid[s] == k]
        ok_rows &= len(own) > 0 and int(s_tid[s].max()) == k
        ok_rows &= r['content'] in TOK.decode(own, skip_special_tokens=True)
        # nothing after the sibling's own response (no continuation)
        ok_rows &= int((s_resp[s] != pad).sum()) == prefix_len + len(own)
    check(ok_rows, 'sibling row = parent prefix before k* + own response only, same prompt')
    check(s_resp.shape[1] == resp.shape[1], 'siblings and originals share the response width')

    n_loop_turns = 3
    check(len(mgr.search_calls) <= n_loop_turns, f'one retrieval request per turn ({len(mgr.search_calls)} calls)')
    sib_search = [r for r in recs if r['action'] == 'search' and r['k'] <= n_loop_turns]
    all_q = [q for c in mgr.search_calls for q in c]
    check(all(f"query {r['content'].split()[-1]}" in all_q for r in sib_search) and all(r['executed'] for r in sib_search),
          f'{len(sib_search)} sibling searches executed inside the per-turn batch')
    final_sib = [r for r in recs if r['k'] == n_loop_turns + 1]
    check(all(not r['executed'] for r in final_sib), 'final-turn sibling searches are not executed (as originals)')

    mgr_v, _, out_v = run(M=M, validation=True)
    check(mgr_v.last_siblings is None and not mgr_v.last_sib_records, 'validation never forks')

    mgr0, _, out0 = run(M=0)
    check(mgr0.last_siblings is None, 'M=0 never forks')

    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
