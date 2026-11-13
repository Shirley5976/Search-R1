"""Driver-side assembly of one plan training step (checklist items 3, 4, 7).

Given the rollout (originals + P3 siblings), the outcome scores and the phis, build the expanded
batch the single-step actor update consumes:
  originals: advantages z(R) + adj[turn], loss_mask = response info_mask, loss_weight 1, is_original 1
  siblings:  advantages beta*(phi_j - C_j) on their own turn only, loss_mask = that turn's tokens,
             loss_weight 1/M, is_original 0, ref_log_prob 0 (KL is masked by is_original anyway)
  padding:   copies of row 0 with loss_mask 0, loss_weight 0, advantages 0, is_original 0, so the
             row count divides the world size without touching the loss or N_orig.
"""

import numpy as np
import torch

from verl import DataProto
from search_r1.plan.advantages import grpo_z, turn_adjustments, token_advantages

METHODS = ('p0', 'p1', 'p2_all', 'p2_sel', 'p3')
TENSOR_KEYS = ('prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids', 'info_mask', 'turn_ids')


def phi_needs(method, mode='train'):
    if mode == 'pilot':
        return {'prev_all', 'fork'}
    return {'p0': set(), 'p1': {'search_turns'}, 'p2_all': {'prev_all'}, 'p2_sel': {'prev_kstar'},
            'p3': {'fork'}}[method]


def fork_m_for(method, M, mode='train'):
    if mode == 'pilot':
        return int(M)
    return int(M) if method == 'p3' else 0


def compute_plan_advantages(method, scores, uids, trajs, beta, alpha):
    """scores: (bs,) outcome scores. trajs: from phi.collect_phis (uid added here).
    Returns (z, adj, sib_adv)."""
    z = grpo_z(scores, uids)
    for t, u in zip(trajs, uids):
        t['uid'] = u
    adj, sib_adv = turn_adjustments(method, trajs, beta=beta, alpha=alpha)
    return z, adj, sib_adv


def build_update_batch(orig, sib, sib_records, z, adj, sib_adv, M, world_size):
    """orig: DataProto of originals (TENSOR_KEYS, optional ref_log_prob, non-tensors).
    sib: DataProto of siblings (TENSOR_KEYS) in sib_records order, or None.
    sib_adv[i]: list of M sibling advantages for parent i (P3), else [].
    Returns (batch, n_sib, n_pad)."""
    L = orig.batch['responses'].shape[1]
    has_ref = 'ref_log_prob' in orig.batch.keys()

    o = {k: orig.batch[k] for k in TENSOR_KEYS}
    o['advantages'] = token_advantages(o['turn_ids'], z, adj)
    o['loss_mask'] = orig.batch['info_mask'][:, -L:].long()
    n = o['responses'].shape[0]
    o['loss_weight'] = torch.ones(n, dtype=torch.float32)
    o['is_original'] = torch.ones(n, dtype=torch.float32)
    if has_ref:
        o['ref_log_prob'] = orig.batch['ref_log_prob']
    parts = [DataProto.from_dict(tensors=o, non_tensors=dict(orig.non_tensor_batch))]

    # siblings whose parent got sibling advantages (P3); in pilot mode no sibling enters the update
    keep, adv_rows = [], []
    pos_in_parent = {}
    for s, r in enumerate(sib_records):
        p = r['parent']
        j = pos_in_parent.get(p, 0)
        pos_in_parent[p] = j + 1
        if sib_adv[p]:
            assert len(sib_adv[p]) == M
            keep.append(s)
            adv_rows.append({r['k']: sib_adv[p][j]})
    n_sib = len(keep)
    if n_sib:
        idx = torch.tensor(keep, dtype=torch.long)
        sb = {k: sib.batch[k][idx] for k in TENSOR_KEYS}
        assert sb['responses'].shape[1] == L and sb['input_ids'].shape[1] == o['input_ids'].shape[1]
        sb['advantages'] = token_advantages(sb['turn_ids'], np.zeros(n_sib), adv_rows)
        own = torch.tensor([list(a)[0] for a in adv_rows]).view(-1, 1)
        sb['loss_mask'] = (sb['turn_ids'] == own).long()
        sb['loss_weight'] = torch.full((n_sib,), 1.0 / M, dtype=torch.float32)
        sb['is_original'] = torch.zeros(n_sib, dtype=torch.float32)
        if has_ref:
            sb['ref_log_prob'] = torch.zeros(n_sib, L, dtype=o['ref_log_prob'].dtype)
        parents = np.array([sib_records[s]['parent'] for s in keep])
        nt = {k: v[parents] for k, v in orig.non_tensor_batch.items()}
        parts.append(DataProto.from_dict(tensors=sb, non_tensors=nt))

    batch = DataProto.concat(parts)
    n_pad = (-len(batch)) % world_size
    if n_pad:
        pad = batch[:1]
        pb = {k: v.repeat(n_pad, *([1] * (v.dim() - 1))) for k, v in pad.batch.items()}
        for k in ('advantages', 'loss_mask', 'loss_weight', 'is_original'):
            pb[k] = torch.zeros_like(pb[k])
        nt = {k: np.repeat(v, n_pad, axis=0) for k, v in pad.non_tensor_batch.items()}
        batch = DataProto.concat([batch, DataProto.from_dict(tensors=pb, non_tensors=nt)])
    batch.meta_info.update(orig.meta_info)
    return batch, n_sib, n_pad


def fork_stats(trajs, sib_records):
    """Fraction of trajectories forked and the action mix at k* (for logging)."""
    n = len(trajs)
    forked = [t for t in trajs if t.get('fork_phis') is not None]
    acts = [r['action'] or 'invalid' for r in sib_records]
    out = {'plan/fork_frac': len(forked) / max(n, 1)}
    for a in ('search', 'answer', 'invalid'):
        out[f'plan/sib_{a}_frac'] = acts.count(a) / max(len(acts), 1)
    return out
