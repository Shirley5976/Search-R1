"""R014: exact vs sketched traces on the R014 subset of questions (from r012b with --exact_prefix).

Exact: fp32 aggregates of g5 and Y (p2_all, p2_sel, p3_m4; main reader) restricted to the
--exact_prefix parameters, saved per question as exact_q_*.pt. Sketched: the same restricted vectors
rebuilt from the per-vector restricted sketches (sketch_sub) with formula.combine. Compares
V0, VZ, C, rho2 and beta* = C / VZ, plus, for every stored vector of every question, the
full-parameter squared norm ||S(v)||^2 against the exact ||v||^2.
Pass (prereg/decisions.md): relative error <= 5% on V0, VZ and on C relative to sqrt(V0 VZ);
|rho2 error| <= 0.05; median full-norm relative error <= 1%.
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

METHODS = ('p2_all', 'p2_sel', 'p3_m4')


def tr_stats(A, B):
    """Unbiased trace covariance between rows of A and B (n x d)."""
    A = A - A.mean(0, keepdim=True)
    B = B - B.mean(0, keepdim=True)
    return float((A * B).sum() / (A.shape[0] - 1))


def summarize(g, ys):
    out = {}
    V0 = tr_stats(g, g)
    for m, y in ys.items():
        VZ, C = tr_stats(y, y), tr_stats(g, y)
        out[m] = {'V0': V0, 'VZ': VZ, 'C': C, 'rho2': C * C / (V0 * VZ) if VZ > 0 else 0.0,
                  'beta_star': C / VZ if VZ > 0 else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sketch_dir', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ex_files = sorted(glob.glob(os.path.join(a.sketch_dir, 'exact_q_*.pt')))
    assert len(ex_files) >= 3, 'need >= 3 questions with exact aggregates'
    g_ex, y_ex, g_sk, y_sk = [], {m: [] for m in METHODS}, [], {m: [] for m in METHODS}
    for ef in ex_files:
        q = torch.load(os.path.join(a.sketch_dir, os.path.basename(ef)[6:]), map_location='cpu')
        ex = torch.load(ef, map_location='cpu')
        names = q['exact_names']
        flat = lambda d: torch.cat([d[n].reshape(-1).double() for n in names])
        g_ex.append(flat(ex['g5']))
        recs, kc = q['records'], q['kl_coef']
        g_sk.append(formula.combine(formula.g5_coef(recs, kc), q['index'], q['sketch_sub']).double())
        for m in METHODS:
            y_ex[m].append(flat(ex[m]))
            y_sk[m].append(formula.combine(formula.method_y(recs, m, 'main'), q['index'], q['sketch_sub']).double())
    exact = summarize(torch.stack(g_ex), {m: torch.stack(v) for m, v in y_ex.items()})
    sk = summarize(torch.stack(g_sk), {m: torch.stack(v) for m, v in y_sk.items()})

    norm_err = []
    for f in sorted(glob.glob(os.path.join(a.sketch_dir, 'q_*.pt'))):
        q = torch.load(f, map_location='cpu')
        est = q['sketch'].double().pow(2).sum(1)
        norm_err += ((est - q['sqnorm']).abs() / q['sqnorm'].clamp_min(1e-30)).tolist()
    norm_err = np.array(norm_err)

    report, ok = {'n_questions': len(ex_files), 'methods': {}}, True
    for m in METHODS:
        e, s = exact[m], sk[m]
        err = {'V0': abs(s['V0'] - e['V0']) / e['V0'], 'VZ': abs(s['VZ'] - e['VZ']) / max(e['VZ'], 1e-30),
               'C_rel_sqrt': abs(s['C'] - e['C']) / max((e['V0'] * e['VZ']) ** 0.5, 1e-30),
               'rho2_abs': abs(s['rho2'] - e['rho2'])}
        passed = err['V0'] <= .05 and err['VZ'] <= .05 and err['C_rel_sqrt'] <= .05 and err['rho2_abs'] <= .05
        ok &= passed
        report['methods'][m] = {'exact': e, 'sketch': s, 'error': err, 'pass': passed}
        print(f"{m:7s} V0 {e['V0']:.4g}/{s['V0']:.4g}  VZ {e['VZ']:.4g}/{s['VZ']:.4g}  C {e['C']:.4g}/{s['C']:.4g}  "
              f"rho2 {e['rho2']:.4f}/{s['rho2']:.4f}  -> {'PASS' if passed else 'FAIL'}")
    report['full_norm_rel_err'] = {'median': float(np.median(norm_err)), 'p95': float(np.quantile(norm_err, .95)),
                                   'max': float(norm_err.max()), 'n': int(len(norm_err))}
    ok &= report['full_norm_rel_err']['median'] <= 0.01
    report['pass'] = bool(ok)
    print(f"full-parameter ||S(v)||^2 vs ||v||^2: {report['full_norm_rel_err']}")
    print('R014', 'PASS' if ok else 'FAIL')
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(report, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
