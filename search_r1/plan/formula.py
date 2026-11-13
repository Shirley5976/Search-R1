"""Pilot estimators as linear combinations of per-sample score vectors (R012b, R014, R015, B0b).

Written from the FINAL_PROPOSAL definitions, independently of advantages.py / step.py (B0b
compares the two). Per question q with G trajectories i (pilot dump records), the stored vectors are
  ('turn', i, k): u_{i,k} = grad sum_{tokens of turn k} log pi       (k = 1..K_i)
  ('kl', i):      kappa_i = grad sum_{policy tokens} KL_tok(pi, ref)  (low_var_kl)
  ('sib', i, j):  u_{i,j} = grad sum_{tokens of sibling j} log pi     (j = 1..M_dump)
and every estimator is  sum_v coef[v] * v. Conventions (FINAL_PROPOSAL "Core Mechanism"):
  g5(q)    = (1/G) sum_i [ z_i sum_k u_{i,k} - kl_coef * kappa_i ]
  P2-all   G = g5 - beta * Y,  Y = (1/G) sum_i sum_k u_{i,k} Phi(S_{i,k-1})
  P2-sel   Y = (1/G) sum_i 1{reach k*} u_{i,k*} Phi(S_{i,k*-1})
  P3(M)    Y = (1/G) sum_i 1{reach} [u_{i,k*}(phi_0 - C_0) - (1/M) sum_{j<=M} u_{i,j}(phi_j - C_j)]
           using the first M of the dumped siblings, leave-one-out C recomputed on 1+M candidates.
phi variants: main (reader), neg (negative-control reader), lex (lexical), off (reader-off: every
non-answer candidate gets the constant phi_bar; answer candidates keep EM).
"""

import numpy as np
import torch

GRPO_EPS = 1e-6


def group_z(scores, eps=GRPO_EPS):
    """verl GRPO z within one group (singleton: mean 0, std 1)."""
    s = np.asarray(scores, dtype=np.float64)
    if len(s) == 1:
        return s / (1.0 + eps)
    return (s - s.mean()) / (s.std(ddof=1) + eps)


def loo(phis):
    """C_c = mean of the other candidates."""
    phis = np.asarray(phis, dtype=np.float64)
    return (phis.sum() - phis) / (len(phis) - 1)


def fork_phis(rec, variant, phi_bar=None, M=None):
    f = rec['fork']
    n = len(f['actions']) if M is None else 1 + M
    if variant == 'off' and 'off' not in f:
        assert phi_bar is not None
        main = f['main'][:n]
        return [main[c] if f['actions'][c] == 'answer' else phi_bar for c in range(n)]
    return list(f[variant][:n])


def _add(coef, key, val):
    if val != 0.0:
        coef[key] = coef.get(key, 0.0) + float(val)


def g5_coef(recs, kl_coef):
    G = len(recs)
    z = group_z([r['score'] for r in recs])
    coef = {}
    for i, r in enumerate(recs):
        for k in range(1, r['n_turns'] + 1):
            _add(coef, ('turn', i, k), z[i] / G)
        _add(coef, ('kl', i), -kl_coef / G)
    return coef


def p2_all_coef(recs, variant='main'):
    G = len(recs)
    coef = {}
    for i, r in enumerate(recs):
        for k in range(1, r['n_turns'] + 1):
            _add(coef, ('turn', i, k), r['phi_prev'][variant][k - 1] / G)
    return coef


def p2_sel_coef(recs, variant='main'):
    G = len(recs)
    coef = {}
    for i, r in enumerate(recs):
        k = r['k_star']
        if k <= r['n_turns']:
            _add(coef, ('turn', i, k), r['phi_prev'][variant][k - 1] / G)
    return coef


def p3_coef(recs, M, variant='main', phi_bar=None, action_filter=None):
    """action_filter: if set ('search' / 'answer' / 'invalid'), keep only the terms of candidates
    with that action (per-candidate action-type decomposition of Y; the parts sum to Y)."""
    G = len(recs)
    coef = {}
    for i, r in enumerate(recs):
        if r['fork'] is None:
            continue
        phis = fork_phis(r, variant, phi_bar, M)
        d = np.asarray(phis) - loo(phis)
        acts = r['fork']['actions']
        if action_filter is None or acts[0] == action_filter:
            _add(coef, ('turn', i, r['k_star']), d[0] / G)
        for j in range(1, M + 1):
            if action_filter is None or acts[j] == action_filter:
                _add(coef, ('sib', i, j), -d[j] / (G * M))
    return coef


def method_y(recs, method, variant='main', phi_bar=None):
    """method: 'p2_all' | 'p2_sel' | 'p3_m{1,2,4}'."""
    if method == 'p2_all':
        return p2_all_coef(recs, variant)
    if method == 'p2_sel':
        return p2_sel_coef(recs, variant)
    if method.startswith('p3_m'):
        return p3_coef(recs, int(method[4:]), variant, phi_bar)
    raise ValueError(method)


def combine(coef, index, sketches):
    """coef: {vec key: c}; index: list of vec keys (row order of sketches (n, D)). Returns c @ sketches."""
    pos = {tuple(k): n for n, k in enumerate(index)}
    w = np.zeros(len(index), dtype=np.float64)
    for key, c in coef.items():
        if key in pos:  # a turn with no policy tokens has a zero gradient and is not stored
            w[pos[key]] += c
    return (torch.from_numpy(w).to(sketches.device) @ sketches.double()).float()


def training_coef(recs_all, method, beta, M, kl_coef, n_orig, loss_len):
    """Gradient of the single-step training loss of one step (B0b formula path), as coefficients on
    per-sample vectors: (1/(N_orig L)) * sum_q G * [g5(q) - beta * Y(q)], where recs_all is a list of
    per-question record lists. Keys get the question index prepended: (q, kind, i, ...)."""
    out = {}
    for q, recs in enumerate(recs_all):
        G = len(recs)
        parts = [(g5_coef(recs, kl_coef), 1.0)]
        if method != 'p0':
            name = {'p2_all': 'p2_all', 'p2_sel': 'p2_sel', 'p3': f'p3_m{M}'}[method]
            parts.append((method_y(recs, name), -beta))
        for coef, s in parts:
            for key, c in coef.items():
                _add(out, (q,) + key, s * c * G / (n_orig * loss_len))
    return out
