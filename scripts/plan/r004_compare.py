"""R004 (B0b) comparison: implementation gradient (r004_b0b_impl.py) vs the formula path (r012b).

For every spec tag found as impl_<tag>_ws<w>.pt under --impl_dir:
  full sketch:  impl sketch  vs  sum_q formula.training_coef . per-vector sketches (r012b q_*.pt)
  exact subset: impl full grads of the --exact_prefix tensors  vs  sum_r b0b_exact_<tag>_rank<r>.pt
                (r012b --b0b_method/--b0b_beta/--b0b_M, if present)
The implementation computes grad(loss); the formula path stores g = -grad(loss), so the impl side is
negated. Pass: relative L2 <= --tol (1e-2) on both, and for every beta != 0 spec the formula moves
away from P0 by more than 5x the error (beta = 0 vs beta != 0 separation). Also checks ws1 vs ws4
(<= tol); with both runs present, an error <= 3x that 1-vs-4-GPU noise floor also passes (bf16).
"""

import argparse
import glob
import json
import os
import re
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from search_r1.plan import formula
from search_r1.plan.sketch import rel_l2


def formula_sketch(sketch_dir, method, beta, M, kl_coef, loss_len):
    files = sorted(glob.glob(os.path.join(sketch_dir, 'q_*.pt')))
    qs = [torch.load(f, map_location='cpu') for f in files]
    n_orig = sum(len(q['records']) for q in qs)
    coef = formula.training_coef([q['records'] for q in qs], method, beta, M, kl_coef, n_orig, loss_len)
    out = 0
    for w, q in enumerate(qs):
        c = {k[1:]: v for k, v in coef.items() if k[0] == w}
        out = out + formula.combine(c, q['index'], q['sketch'])
    return out, n_orig


def flat(d, names):
    return torch.cat([d[n].reshape(-1).double() for n in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl_dir', required=True)
    ap.add_argument('--sketch_dir', required=True, help='r012b output on the same dump (q_*.pt, b0b_exact_*)')
    ap.add_argument('--tol', type=float, default=1e-2)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    impl = {}
    for f in sorted(glob.glob(os.path.join(a.impl_dir, 'impl_*_ws*.pt'))):
        m = re.match(r'impl_(.+)_ws(\d+)\.pt$', os.path.basename(f))
        impl.setdefault(m.group(1), {})[int(m.group(2))] = torch.load(f, map_location='cpu')
    assert impl, f'no impl_*.pt under {a.impl_dir}'
    report, ok = {}, True
    p0_formula = None
    for tag in sorted(impl, key=lambda t: not t.startswith('p0')):
        method, beta, M = re.match(r'(\w+?)_b([-\d.e]+)_m(\d+)$', tag).groups()
        beta, M = float(beta), int(M)
        any_run = next(iter(impl[tag].values()))
        fs, n_orig = formula_sketch(a.sketch_dir, method, beta, M, any_run['kl_coef'], any_run['loss_len'])
        if method == 'p0':
            p0_formula = fs
        r = {'n_orig_formula': n_orig}
        for ws, run in sorted(impl[tag].items()):
            assert run['D'] == fs.numel(), 'impl and r012b sketches use different D'
            e = rel_l2(fs.double(), -run['sketch'].double())
            r[f'ws{ws}'] = {'sketch_rel_l2': e, 'n_orig_impl': run['n_orig']}
            passed = e <= a.tol and abs(run['n_orig'] - n_orig) < 0.5
            parts = sorted(glob.glob(os.path.join(a.sketch_dir, f'b0b_exact_{tag}_rank*.pt')))
            if parts and run['exact']:
                ex = [torch.load(p, map_location='cpu') for p in parts]
                names = sorted(set(ex[0]) & set(run['exact']))
                fe = sum(flat(x, names) for x in ex)
                ee = rel_l2(fe, -flat(run['exact'], names))
                r[f'ws{ws}']['exact_rel_l2'] = ee
                r[f'ws{ws}']['exact_tensors'] = len(names)
                passed &= ee <= a.tol
            r[f'ws{ws}']['pass'] = bool(passed)
        if len(impl[tag]) > 1:
            runs = [impl[tag][w] for w in sorted(impl[tag])]
            noise = rel_l2(runs[-1]['sketch'].double(), runs[0]['sketch'].double())
            r['ws_consistency_rel_l2'] = noise
            # bf16 fallback (prereg/decisions.md): an error within 3x the implementation's own
            # 1-GPU vs 4-GPU noise also passes, provided the beta separation check below holds
            for k, v in r.items():
                if k.startswith('ws') and isinstance(v, dict) and not v['pass']:
                    errs = [v['sketch_rel_l2']] + ([v['exact_rel_l2']] if 'exact_rel_l2' in v else [])
                    v['pass_noise_floor'] = bool(max(errs) <= 3 * noise and abs(v['n_orig_impl'] - n_orig) < 0.5)
            ok &= noise <= a.tol
        if beta != 0 and p0_formula is not None:
            sep = rel_l2(fs.double(), p0_formula.double())
            worst = max(v['sketch_rel_l2'] for k, v in r.items() if k.startswith('ws') and isinstance(v, dict))
            r['separation_from_p0'] = sep
            r['separation_ok'] = bool(sep > 5 * worst)
            ok &= r['separation_ok']
        for k, v in r.items():
            if k.startswith('ws') and isinstance(v, dict):
                ok &= v['pass'] or v.get('pass_noise_floor', False)
        report[tag] = r
        print(tag, json.dumps(r))
    report['pass'] = bool(ok)
    print('R004', 'PASS' if ok else 'FAIL')
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(report, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
