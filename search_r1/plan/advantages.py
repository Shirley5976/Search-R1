"""Per-turn advantages for P0 / P1 / P2-all / P2-sel / P3 (implementation checklist items 3, 4, 7).

Everything here is plain numpy/torch on the driver and detached from the policy. A trajectory's
response tokens carry a turn id (1-based decision turn; 0 for prompt, observation and padding), so
an advantage is "z(R) + adj[turn]" per token. Siblings get "beta*(phi_j - C_j)" on their single
response turn and loss weight 1/M.

Definitions (refine-logs/FINAL_PROPOSAL.md):
  z(R)    GRPO group z-score, identical to verl core_algos.compute_grpo_outcome_advantage
          (unbiased std, eps=1e-6, singleton group -> mean 0, std 1). Zero-variance group -> 0.
  P2-all  adj[k] = -beta * Phi(S_{k-1}) for every turn k.
  P2-sel  adj[k*] = -beta * Phi(S_{k*-1}) if the trajectory reaches k*.
  P3      adj[k*] = -beta * (phi_0 - C_0), sibling j advantage = +beta * (phi_j - C_j),
          C_0 = mean of the M sibling phis, C_j = mean of the other M candidates (true action included).
  P1      adj[k] = alpha * z_turn(Phi(S_k) - Phi(S_{k-1})) on executed-search turns, where z_turn
          standardises the per-turn rewards over every executed-search turn of every trajectory in
          the GRPO group (prereg/decisions.md: turns without a retrieval carry no process reward).
"""

from collections import defaultdict

import numpy as np
import torch

GRPO_EPS = 1e-6


def grpo_z(scores, uids, eps=GRPO_EPS):
    scores = np.asarray(scores, dtype=np.float64)
    groups = defaultdict(list)
    for i, u in enumerate(uids):
        groups[u].append(i)
    z = np.zeros_like(scores)
    for idx in groups.values():
        s = scores[idx]
        if len(idx) == 1:
            mean, std = 0.0, 1.0
        else:
            mean, std = s.mean(), s.std(ddof=1)
        z[idx] = (s - mean) / (std + eps)
    return z


def p3_baselines(phis):
    """phis[0] = true action, phis[1:] = M siblings. Returns C with C[0] = mean(siblings),
    C[j] = mean over the other M candidates (leave-one-out over all 1+M)."""
    phis = np.asarray(phis, dtype=np.float64)
    m = len(phis) - 1
    assert m >= 1, 'P3 needs at least one sibling'
    total = phis.sum()
    c = (total - phis) / m
    return c


def p3_deltas(phis):
    phis = np.asarray(phis, dtype=np.float64)
    return phis - p3_baselines(phis)


def p1_turn_z(turn_rewards_by_group, eps=GRPO_EPS):
    """turn_rewards_by_group: {uid: [(traj_index, k, r), ...]} -> {(traj_index, k): z}.
    Same standardisation convention as grpo_z over the pooled turn rewards of a group."""
    out = {}
    for items in turn_rewards_by_group.values():
        r = np.array([x[2] for x in items], dtype=np.float64)
        if len(r) == 0:
            continue
        if len(r) == 1:
            mean, std = 0.0, 1.0
        else:
            mean, std = r.mean(), r.std(ddof=1)
        zr = (r - mean) / (std + eps)
        for (ti, k, _), v in zip(items, zr):
            out[(ti, k)] = float(v)
    return out


def turn_adjustments(method, trajs, beta=0.0, alpha=0.0):
    """Compute per-trajectory turn adjustments and sibling advantages.

    trajs: list of dicts with keys
        uid, n_turns (K_i), k_star, phi_prev {k: Phi(S_{k-1})} for k=1..K_i,
        search_turns {k: (Phi(S_{k-1}), Phi(S_k))} for executed searches (P1),
        fork_phis [phi_0, phi_1..phi_M] or None when no fork happened.
    Returns (adj, sib_adv): adj[i] = {k: value}; sib_adv[i] = [adv_j for j=1..M] (P3 only, else []).
    """
    adj = [dict() for _ in trajs]
    sib_adv = [[] for _ in trajs]
    if method == 'p0':
        return adj, sib_adv
    if method == 'p2_all':
        for i, t in enumerate(trajs):
            for k in range(1, t['n_turns'] + 1):
                adj[i][k] = -beta * t['phi_prev'][k]
        return adj, sib_adv
    if method == 'p2_sel':
        for i, t in enumerate(trajs):
            k = t['k_star']
            if k <= t['n_turns']:
                adj[i][k] = -beta * t['phi_prev'][k]
        return adj, sib_adv
    if method == 'p3':
        for i, t in enumerate(trajs):
            if t.get('fork_phis') is None:
                continue
            assert t['k_star'] <= t['n_turns'], 'fork recorded on a turn the trajectory never reached'
            d = p3_deltas(t['fork_phis'])
            adj[i][t['k_star']] = -beta * d[0]
            sib_adv[i] = [beta * x for x in d[1:]]
        return adj, sib_adv
    if method == 'p1':
        by_group = defaultdict(list)
        for i, t in enumerate(trajs):
            for k, (before, after) in sorted(t['search_turns'].items()):
                by_group[t['uid']].append((i, k, after - before))
        zt = p1_turn_z(by_group)
        for (i, k), v in zt.items():
            adj[i][k] = alpha * v
        return adj, sib_adv
    raise ValueError(f'unknown plan method {method!r}')


def token_advantages(turn_ids, base, adj):
    """turn_ids: (bs, T) long, 0 outside response tokens. base: (bs,) z(R) (0 for sibling rows).
    adj: list of {k: value}. Returns (bs, T) float32 advantages, zero where turn_ids == 0."""
    turn_ids = turn_ids.long()
    bs, T = turn_ids.shape
    max_k = int(turn_ids.max().item()) if turn_ids.numel() else 0
    table = torch.zeros(bs, max_k + 1, dtype=torch.float64)
    table[:, 1:] = torch.as_tensor(np.asarray(base, dtype=np.float64)).view(-1, 1)
    for i, a in enumerate(adj):
        for k, v in a.items():
            if k <= max_k:
                table[i, k] += v
    out = table.gather(1, turn_ids)
    return out.float()
