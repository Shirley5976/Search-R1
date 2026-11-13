"""R015 (+ R013 negctrl, Block 2/3 diagnostics): W table, beta_hat, promotion rule, prereg output.

Reads the R012b per-question sketches (q_*.pt) and the stage-0.5 costs (R011, r011_costs.py) and writes
prereg/pilot_result.json and prereg/pilot_result.md. Definitions follow FINAL_PROPOSAL.md:
  per question q: g5(q) and Y_m(q) (search_r1/plan/formula.py), G_beta(q) = g5(q) - beta * Y_m(q)
  V0 = tr Var_q g5, VZ = tr Var_q Y, C = tr Cov_q(g5, Y)   (unbiased, n - 1)
  beta_hat = C / VZ on calibration (0 if VZ = 0), V_beta = V0 + beta^2 VZ - 2 beta C
  W(m) / W(P0) = (c_m / c_0) * V_beta_hat(m) / V0 on evaluation with the frozen beta_hat
  rho2 = C^2 / (V0 VZ); fit-error term VZ (beta_hat - beta*)^2 with beta* fitted on evaluation.
Bootstrap: question resampling stratified by data source, B draws, seed 0; the promotion upper
bound is the one-sided 97.5% quantile (Bonferroni over the two candidates).
Promotion: candidates P2-all and the P3 M in {1,2,4} with the smallest calibration W; checked in
calibration-W order, the first with evaluation upper bound < 1 wins, otherwise P0.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from search_r1.plan import formula

P3_MS = (1, 2, 4)
VARIANTS = ('main', 'off', 'lex', 'neg')


def y_types():
    out = [('p2_all', v) for v in VARIANTS] + [('p2_sel', v) for v in VARIANTS]
    out += [(f'p3_m{m}', v) for m in P3_MS for v in VARIANTS]
    return out


# ------------------------------------------------------------------ statistics on Gram matrices
def trace_cov(Gab, w):
    """Unbiased tr Cov over a (re)sample with multiplicities w: (sum w_i Gab_ii - w'Gab w / n) / (n-1)."""
    n = w.sum()
    return (float(w @ np.diag(Gab)) - float(w @ Gab @ w) / n) / (n - 1)


def stats(G, w):
    """G: dict with 'gg', 'gy', 'yy' (n x n). Returns V0, VZ, C."""
    return trace_cov(G['gg'], w), trace_cov(G['yy'], w), trace_cov(G['gy'], w)


def v_beta(V0, VZ, C, beta):
    return V0 + beta * beta * VZ - 2 * beta * C


def fit_beta(VZ, C):
    return C / VZ if VZ > 0 else 0.0


def rho2(V0, VZ, C):
    return C * C / (V0 * VZ) if V0 > 0 and VZ > 0 else 0.0


def strata_weights(groups, rng):
    """Multiplicity vector of a stratified bootstrap draw (groups: list of index arrays)."""
    n = sum(len(g) for g in groups)
    w = np.zeros(n)
    for g in groups:
        np.add.at(w, rng.choice(g, size=len(g), replace=True), 1)
    return w


# ------------------------------------------------------------------ loading
def phi_bars(questions):
    """reader-off constants from calibration: mean main phi of non-answer fork candidates, and mean Phi(S_{k-1})."""
    a, b = [], []
    for q in questions:
        if q['split'] != 'pilot_calib':
            continue
        for r in q['records']:
            b += list(r['phi_prev']['main'])
            if r['fork'] is not None:
                a += [p for p, act in zip(r['fork']['main'], r['fork']['actions']) if act != 'answer']
    return float(np.mean(a)) if a else 0.0, float(np.mean(b)) if b else 0.0


def inject_off(records, bar_fork, bar_prev):
    for r in records:
        r['phi_prev']['off'] = [bar_prev] * len(r['phi_prev']['main'])
        if r['fork'] is not None:
            f = r['fork']
            f['off'] = [m if act == 'answer' else bar_fork for m, act in zip(f['main'], f['actions'])]


def load(sketch_dir, kl_coef=None, device='cpu'):
    files = sorted(glob.glob(os.path.join(sketch_dir, 'q_*.pt')))
    assert files, f'no q_*.pt under {sketch_dir}'
    heads = [torch.load(f, map_location='cpu') for f in files]
    bar_fork, bar_prev = phi_bars(heads)
    vecs = {'g': []}
    for t in y_types():
        vecs[t] = []
    meta = []
    for q in heads:
        kc = q['kl_coef'] if kl_coef is None else kl_coef
        recs = q['records']
        inject_off(recs, bar_fork, bar_prev)
        sk = q['sketch'].to(device)
        vecs['g'].append(formula.combine(formula.g5_coef(recs, kc), q['index'], sk))
        for m, v in y_types():
            vecs[(m, v)].append(formula.combine(formula.method_y(recs, m, v), q['index'], sk))
        meta.append({'qid': q['qid'], 'split': q['split'], 'source': q['qid'].split('/')[0], 'records': recs,
                     'index': q['index'], 'seconds': q.get('seconds', 0.0)})
        q['sketch'] = None
    return meta, {k: torch.stack(v) for k, v in vecs.items()}, (bar_fork, bar_prev), heads


def grams(vecs, idx, y_keys):
    g = vecs['g'][idx].double()
    out = {}
    gg = (g @ g.T).cpu().numpy()
    for t in y_keys:
        y = vecs[t][idx].double()
        out[t] = {'gg': gg, 'gy': (g @ y.T).cpu().numpy(), 'yy': (y @ y.T).cpu().numpy()}
    return out


# ------------------------------------------------------------------ main analysis
def cost_of(m, costs):
    """GPU s per question. P2-sel is never trained; it is charged like P2-all (both score prefixes).
    Action-type parts of a P3 Y only enter covariance decompositions; they get the P0 cost."""
    if m in costs:
        return costs[m]
    if m == 'p2_sel':
        return costs['p2_all']
    return costs['p0']


def analyse(meta, vecs, costs, n_boot=2000, seed=0, decomp_vecs=None):
    calib = np.array([i for i, m in enumerate(meta) if m['split'] == 'pilot_calib'])
    evals = np.array([i for i, m in enumerate(meta) if m['split'] == 'pilot_eval'])
    assert len(calib) > 2 and len(evals) > 2, (len(calib), len(evals))
    keys = list(y_types()) + list((decomp_vecs or {}).keys())
    allv = dict(vecs)
    allv.update(decomp_vecs or {})
    Gc, Ge = grams(allv, torch.as_tensor(calib), keys), grams(allv, torch.as_tensor(evals), keys)
    wc, we = np.ones(len(calib)), np.ones(len(evals))
    src = np.array([meta[i]['source'] for i in evals])
    strata = [np.nonzero(src == s)[0] for s in sorted(set(src))]
    rng = np.random.RandomState(seed)
    draws = [strata_weights(strata, rng) for _ in range(n_boot)]

    c0 = costs['p0']
    rows = {}
    for t in keys:
        m, v = t
        V0c, VZc, Cc = stats(Gc[t], wc)
        V0e, VZe, Ce = stats(Ge[t], we)
        beta_hat = fit_beta(VZc, Cc)
        cratio = cost_of(m, costs) / c0
        w_calib = cratio * v_beta(V0c, VZc, Cc, beta_hat) / V0c
        w_eval = cratio * v_beta(V0e, VZe, Ce, beta_hat) / V0e
        boot = np.array([cratio * v_beta(*stats(Ge[t], w), beta_hat) / stats(Ge[t], w)[0] for w in draws])
        beta_star = fit_beta(VZe, Ce)
        rows[t] = {'method': m, 'variant': v, 'beta_hat': beta_hat, 'cost_ratio': cratio,
                   'V0_calib': V0c, 'VZ_calib': VZc, 'C_calib': Cc, 'rho2_calib': rho2(V0c, VZc, Cc),
                   'V0_eval': V0e, 'VZ_eval': VZe, 'C_eval': Ce, 'rho2_eval': rho2(V0e, VZe, Ce),
                   'W_ratio_calib': w_calib, 'W_ratio_eval': w_eval,
                   'W_ratio_eval_ub975': float(np.quantile(boot, 0.975)),
                   'W_ratio_eval_ci95': [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                   'var_ratio_eval': v_beta(V0e, VZe, Ce, beta_hat) / V0e,
                   'break_even_rho2': 1 - 1 / cratio, 'beta_star_eval': beta_star,
                   'fit_error_term_eval': VZe * (beta_hat - beta_star) ** 2}
    # negative control keeps the main reader's beta_hat (Block 4 protocol)
    for m in ['p2_all', 'p2_sel'] + [f'p3_m{k}' for k in P3_MS]:
        t = (m, 'neg')
        b = rows[(m, 'main')]['beta_hat']
        V0e, VZe, Ce = stats(Ge[t], we)
        rows[t]['beta_main'] = b
        rows[t]['W_ratio_eval_main_beta'] = rows[t]['cost_ratio'] * v_beta(V0e, VZe, Ce, b) / V0e

    def paired(t1, t2):
        """Ratio of evaluation variances V_beta_hat(t1) / V_beta_hat(t2), percentile 95% CI."""
        b1, b2 = rows[t1]['beta_hat'], rows[t2]['beta_hat']
        f = lambda w: v_beta(*stats(Ge[t1], w), b1) / v_beta(*stats(Ge[t2], w), b2)
        bs = np.array([f(w) for w in draws])
        return {'ratio': f(we), 'ci95': [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}

    # promotion
    p3_best = min(P3_MS, key=lambda k: rows[(f'p3_m{k}', 'main')]['W_ratio_calib'])
    cands = sorted([('p2_all', 'main'), (f'p3_m{p3_best}', 'main')], key=lambda t: rows[t]['W_ratio_calib'])
    winner = 'p0'
    for t in cands:
        if rows[t]['W_ratio_eval_ub975'] < 1.0:
            winner = t[0]
            break
    winner_method = 'p3' if winner.startswith('p3') else winner
    pred = sorted([('p0', 1.0), ('p2_all', rows[('p2_all', 'main')]['W_ratio_eval']),
                   ('p3', rows[(f'p3_m{p3_best}', 'main')]['W_ratio_eval'])], key=lambda x: x[1])
    p3t = (f'p3_m{p3_best}', 'main')
    diag = {'p3_vs_p2_sel': paired(p3t, ('p2_sel', 'main')),
            'reader_vs_off': paired(p3t, (p3t[0], 'off')),
            'reader_vs_lex': paired(p3t, (p3t[0], 'lex')),
            'one_sibling_vs_four': paired(('p3_m1', 'main'), ('p3_m4', 'main'))}
    return {'rows': rows, 'p3_M_star': p3_best, 'calib_order': [t[0] for t in cands], 'winner': winner_method,
            'winner_config': winner, 'predicted_training_ranking': [p for p, _ in pred],
            'diagnostics': diag, 'n_calib': int(len(calib)), 'n_eval': int(len(evals)), 'n_boot': n_boot}


def decomposition_vecs(meta, heads_sketch_dir, M, device='cpu', kl_coef=None):
    """Per-candidate action-type parts of Y for P3(M), main reader (Block 5; parts sum to Y)."""
    files = sorted(glob.glob(os.path.join(heads_sketch_dir, 'q_*.pt')))
    out = {(f'p3_m{M}_{a}', 'main'): [] for a in ('search', 'answer', 'invalid')}
    for f, m in zip(files, meta):
        q = torch.load(f, map_location='cpu')
        sk = q['sketch'].to(device)
        for a in ('search', 'answer', 'invalid'):
            out[(f'p3_m{M}_{a}', 'main')].append(
                formula.combine(formula.p3_coef(m['records'], M, 'main', action_filter=a), q['index'], sk))
    return {k: torch.stack(v) for k, v in out.items()}


def to_markdown(res, costs, bars):
    L = ['# Pilot result (R015)', '',
         f"- calibration questions: {res['n_calib']}, evaluation questions: {res['n_eval']}, bootstrap draws: {res['n_boot']}",
         f"- costs (GPU s / question): " + ', '.join(f'{k} {v:.2f}' for k, v in costs.items()),
         f"- reader-off constants: fork phi_bar = {bars[0]:.4f}, Phi(S_k-1) bar = {bars[1]:.4f}",
         f"- **pilot winner: {res['winner']}** (config {res['winner_config']}); P3 M* = {res['p3_M_star']}; "
         f"calibration order: {res['calib_order']}",
         f"- predicted training ranking (best first, by evaluation W): {res['predicted_training_ranking']}", '',
         '| method | phi | beta_hat | c_m/c_0 | rho2 calib | rho2 eval | break-even | W/W0 calib | W/W0 eval | UB 97.5% | fit-error term |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for (m, v), r in res['rows'].items():
        L.append(f"| {m} | {v} | {r['beta_hat']:.4g} | {r['cost_ratio']:.3f} | {r['rho2_calib']:.3f} | "
                 f"{r['rho2_eval']:.3f} | {r['break_even_rho2']:.3f} | {r['W_ratio_calib']:.3f} | "
                 f"{r['W_ratio_eval']:.3f} | {r['W_ratio_eval_ub975']:.3f} | {r['fit_error_term_eval']:.4g} |")
    L += ['', '## Diagnostics (variance ratios on evaluation, 95% CI)', '']
    for k, d in res['diagnostics'].items():
        L.append(f"- {k}: {d['ratio']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]")
    L += ['', '## Negative control with the main reader beta_hat', '']
    for (m, v), r in res['rows'].items():
        if v == 'neg':
            L.append(f"- {m}: W/W0 eval = {r['W_ratio_eval_main_beta']:.3f} (own beta_hat {r['beta_hat']:.4g})")
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sketch_dir', required=True)
    ap.add_argument('--costs', required=True, help='json from r011_costs.py: GPU seconds per question per method')
    ap.add_argument('--out_prefix', default=os.path.join(ROOT, 'prereg', 'pilot_result'))
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    costs = json.load(open(a.costs))
    meta, vecs, bars, _ = load(a.sketch_dir, device=a.device)
    res0 = analyse(meta, vecs, costs, n_boot=200, seed=0)
    dec = decomposition_vecs(meta, a.sketch_dir, res0['p3_M_star'], device=a.device)
    res = analyse(meta, vecs, costs, n_boot=a.n_boot, seed=0, decomp_vecs=dec)
    res['phi_bar'] = {'fork': bars[0], 'prev': bars[1]}
    res['costs'] = costs
    ser = dict(res)
    ser['rows'] = {f'{m}|{v}': r for (m, v), r in res['rows'].items()}
    with open(a.out_prefix + '.json', 'w') as f:
        json.dump(ser, f, indent=1)
    with open(a.out_prefix + '.md', 'w') as f:
        f.write(to_markdown(res, costs, bars))
    print(to_markdown(res, costs, bars))


if __name__ == '__main__':
    main()
