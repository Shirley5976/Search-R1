"""Driver-side phi assembly (checklist items 6, 7): rollout trace -> reader requests -> phis.

State S_k = question + docs of every executed search in turns 1..k (S_0: no docs, closed-book).
Action phi (prereg/decisions.md):
  executed search -> Phi(S_{k-1} + docs)   (reader)
  answer          -> EM(answer, true gold) (also under the negative control)
  otherwise       -> Phi(S_{k-1})          (invalid action, or a final-turn search that is not executed)
Reader requests are deduplicated per call (identical question/docs/target), scored in one batch on
the reader worker group, and phi = exp(max over aliases of the mean target log-prob).
"""

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score.qa_em import em_check
from search_r1.plan.reader_prompt import ReaderPromptBuilder, collate, dedup_docs, clean_aliases
from search_r1.plan.lexical_phi import lexical_phi


def state_docs(trace_i, upto_k):
    """Docs of S_upto_k: executed searches of turns 1..upto_k, in retrieval order."""
    docs = []
    for r in trace_i:
        if r['k'] <= upto_k and r['action'] == 'search' and r['executed']:
            docs += list(r['docs'])
    return docs


class PhiEngine:
    """Collects reader requests, scores them in one batched call, caches within the call."""

    def __init__(self, tokenizer, score_fn, world_size, pad_id=None):
        self.builder = ReaderPromptBuilder(tokenizer)
        self.score_fn = score_fn  # DataProto(input_ids, attention_mask, position_ids, target_mask) -> mean_logp
        self.world_size = int(world_size)
        self.pad_id = tokenizer.pad_token_id if pad_id is None else pad_id
        self.reset()

    def reset(self):
        self._pending = {}
        self._values = {}
        self.stats = {'requests': 0, 'unique': 0, 'rows': 0, 'truncated': 0, 'empty_target': 0}

    def ask(self, question, docs, aliases):
        key = (question, tuple(dedup_docs(docs)), tuple(clean_aliases(aliases)))
        self.stats['requests'] += 1
        if key not in self._pending and key not in self._values:
            self._pending[key] = None
        return key

    def run(self):
        keys = list(self._pending)
        self._pending = {}
        if not keys:
            return
        rows, owner = [], []
        for n, (q, docs, aliases) in enumerate(keys):
            info = {}
            r = self.builder.build(q, list(docs), list(aliases), info=info)
            self.stats['truncated'] += int(info['n_kept'] < info['n_docs'])
            if not r:
                self.stats['empty_target'] += 1
            rows += r
            owner += [n] * len(r)
        self.stats['unique'] += len(keys)
        self.stats['rows'] += len(rows)
        best = np.full(len(keys), -np.inf)
        if rows:
            mean_logp = self._score_rows(rows)
            for o, v in zip(owner, mean_logp):
                best[o] = max(best[o], v)
        for k, b in zip(keys, best):
            self._values[k] = float(np.exp(b)) if np.isfinite(b) else 0.0

    def _score_rows(self, rows):
        """Length-sorted, dealt round-robin to ranks so each rank's contiguous chunk is balanced
        and its micro-batches have similar lengths. Padded with copies of the shortest row."""
        n = len(rows)
        W = self.world_size
        n_pad = (-n) % W
        order = sorted(range(n), key=lambda i: -len(rows[i][0]))
        order += [order[-1]] * n_pad
        per_rank = [order[r::W] for r in range(W)]
        perm = [i for chunk in per_rank for i in chunk]
        ids, attn, pos, tm = collate([rows[i] for i in perm], self.pad_id)
        out = self.score_fn(DataProto.from_dict(tensors={'input_ids': ids, 'attention_mask': attn,
                                                         'position_ids': pos, 'target_mask': tm}))
        vals = out.batch['mean_logp'].float().numpy()
        res = np.empty(n)
        for v, i in zip(vals, perm):
            res[i] = v
        return res

    def get(self, key):
        return self._values[key]


def collect_phis(engine, trace, sib_records, k_star, questions, golds, reader_targets, need):
    """Build the per-trajectory inputs of advantages.turn_adjustments.

    need: subset of {'prev_all', 'prev_kstar', 'fork', 'search_turns'}.
    golds: true gold aliases (EM of answer actions); reader_targets: aliases the reader scores
    (= golds for the main reader, the mapped aliases for the negative control).
    Returns (trajs_partial, sib_phis): trajs_partial[i] has n_turns, k_star, phi_prev, search_turns,
    fork_phis, fork_actions; sib_phis[s] is the phi of sibling record s.
    """
    bs = len(trace)
    plan = []  # deferred lookups: (kind, i, extra, key_or_value)

    def s_key(i, upto_k, extra_docs=()):
        return engine.ask(questions[i], state_docs(trace[i], upto_k) + list(extra_docs), reader_targets[i])

    def action(i, rec):
        if rec['action'] == 'answer':
            return ('value', float(em_check(rec['content'], list(golds[i]))))
        if rec['action'] == 'search' and rec['executed']:
            return ('key', s_key(i, rec['k'] - 1, rec['docs']))
        return ('key', s_key(i, rec['k'] - 1))

    sib_by_parent = {}
    for s, r in enumerate(sib_records):
        sib_by_parent.setdefault(r['parent'], []).append(s)

    for i in range(bs):
        K = len(trace[i])
        if 'prev_all' in need:
            for k in range(1, K + 1):
                plan.append(('prev', i, k, ('key', s_key(i, k - 1))))
        if 'prev_kstar' in need and k_star is not None and k_star[i] <= K:
            plan.append(('prev', i, int(k_star[i]), ('key', s_key(i, k_star[i] - 1))))
        if 'search_turns' in need:
            for rec in trace[i]:
                if rec['action'] == 'search' and rec['executed']:
                    plan.append(('st0', i, rec['k'], ('key', s_key(i, rec['k'] - 1))))
                    plan.append(('st1', i, rec['k'], ('key', s_key(i, rec['k']))))
        if 'fork' in need and i in sib_by_parent:
            k = sib_records[sib_by_parent[i][0]]['k']
            plan.append(('fork0', i, k, action(i, trace[i][k - 1])))
    sib_plan = [action(r['parent'], r) for r in sib_records] if 'fork' in need else []

    engine.run()

    def val(x):
        return x[1] if x[0] == 'value' else engine.get(x[1])

    trajs = [dict(n_turns=len(trace[i]), k_star=None if k_star is None else int(k_star[i]), phi_prev={},
                  search_turns={}, fork_phis=None, fork_actions=None) for i in range(bs)]
    st = {}
    for kind, i, k, x in plan:
        if kind == 'prev':
            trajs[i]['phi_prev'][k] = val(x)
        elif kind in ('st0', 'st1'):
            st.setdefault((i, k), [None, None])[0 if kind == 'st0' else 1] = val(x)
        elif kind == 'fork0':
            trajs[i]['fork_phis'] = [val(x)]
            trajs[i]['fork_actions'] = [trace[i][k - 1]['action']]
    for (i, k), (a, b) in st.items():
        trajs[i]['search_turns'][k] = (a, b)
    sib_phis = [val(x) for x in sib_plan]
    for i, idx in sib_by_parent.items():
        if trajs[i]['fork_phis'] is not None:
            trajs[i]['fork_phis'] += [sib_phis[s] for s in idx]
            trajs[i]['fork_actions'] += [sib_records[s]['action'] for s in idx]
    return trajs, sib_phis


def lexical_action_phi(trace_i, rec, golds_i):
    """Lexical variant of the action phi (Block 3): searches use lexical_phi over docs(S)."""
    if rec['action'] == 'answer':
        return float(em_check(rec['content'], list(golds_i)))
    docs = state_docs(trace_i, rec['k'] - 1)
    if rec['action'] == 'search' and rec['executed']:
        docs += list(rec['docs'])
    return float(lexical_phi(docs, list(golds_i)))


def lexical_state_phi(trace_i, upto_k, golds_i):
    return float(lexical_phi(state_docs(trace_i, upto_k), list(golds_i)))
