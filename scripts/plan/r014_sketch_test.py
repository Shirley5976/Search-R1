"""CPU checks of search_r1/plan/sketch.py (R012b/R014/B0b support).
  1. linear: S(x+y) = S(x)+S(y);
  2. shard invariance: sketching a tensor in slices with offsets == sketching it whole
     (what fsdp_grad_sketch relies on);
  3. <S(x),S(y)> is unbiased for <x,y> over sketch seeds, with relative sd ~ sqrt(2/D)-scale;
  4. name-dependence: different tensor names use different hashes.
Run: python -I scripts/plan/r014_sketch_test.py
"""
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from search_r1.plan.sketch import sketch_add, sketch_named

FAILS = []
def check(c, m):
    print(f"[{'PASS' if c else 'FAIL'}] {m}")
    if not c: FAILS.append(m)

torch.manual_seed(0)
D = 2 ** 12
x = {'a.weight': torch.randn(300, 70), 'b.bias': torch.randn(5000)}
y = {k: 0.6 * v + 0.8 * torch.randn_like(v) for k, v in x.items()}
Sx = sketch_named(x.items(), D=D); Sy = sketch_named(y.items(), D=D)
Sxy = sketch_named({k: x[k] + y[k] for k in x}.items(), D=D)
check(torch.allclose(Sx + Sy, Sxy, atol=1e-4), 'linearity')

out = torch.zeros(D)
for name, t in x.items():
    flat = t.reshape(-1); cuts = [0, 777, 4000, flat.numel()]
    for s, e in zip(cuts[:-1], cuts[1:]):
        sketch_add(out, name, flat[s:e], offset=s)
check(torch.allclose(out, Sx, atol=1e-4), 'slices with offsets == whole tensor (FSDP shard invariance)')

true = sum(float((x[k] * y[k]).sum()) for k in x)
nx = sum(float((v * v).sum()) for v in x.values()); ny = sum(float((v * v).sum()) for v in y.values())
est = torch.tensor([float(sketch_named(x.items(), D=D, seed=s) @ sketch_named(y.items(), D=D, seed=s))
                    for s in range(200)])
se = float(est.std() / 200 ** 0.5)
check(abs(float(est.mean()) - true) < 4 * se, f'unbiased <x,y>: mean {est.mean():.1f} vs {true:.1f} (se {se:.1f})')
bound = ((nx * ny + true ** 2) / D) ** 0.5
check(float(est.std()) < 1.5 * bound, f'sd {est.std():.1f} within 1.5x theory {bound:.1f}')
n2 = torch.tensor([float(sketch_named(x.items(), D=D, seed=s).pow(2).sum()) for s in range(50)])
check(abs(float(n2.mean()) / nx - 1) < 0.02, 'squared norm unbiased')

from search_r1.plan.sketch import ref_hash, _hash, tensor_coeffs
js = [0, 1, 2, 12345, 2 ** 31 + 7, 2 ** 33 - 1, 3 * 10 ** 9]
h, s = _hash(torch.tensor(js, dtype=torch.int64), tensor_coeffs('w', 3), D)
ok = all(ref_hash('w', j, D, 3) == (int(h[n]), int(s[n])) for n, j in enumerate(js))
check(ok, 'int64 wrap-around hash == exact 64-bit integer reference (incl. j > 2^32)')
hb, _ = _hash(torch.arange(200000, dtype=torch.int64), tensor_coeffs('w', 0), D)
cnt = torch.bincount(hb, minlength=D).float()
check(float(cnt.std() / cnt.mean()) < 1.2 * (D / 200000) ** 0.5 + 0.02, 'buckets are near-uniform')

z = torch.randn(1000)
s1 = torch.zeros(D); sketch_add(s1, 'p', z); s2 = torch.zeros(D); sketch_add(s2, 'q', z)
check(not torch.allclose(s1, s2), 'hash depends on tensor name')
print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}"); sys.exit(1 if FAILS else 0)
