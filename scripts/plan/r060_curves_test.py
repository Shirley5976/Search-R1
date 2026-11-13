"""CPU test of scripts/plan/curves.py (R060) on synthetic eval dumps with known answers.

Run: python -I scripts/plan/r060_curves_test.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curves as C

FAILS = []


def check(c, m):
    print(f"[{'PASS' if c else 'FAIL'}] {m}")
    if not c:
        FAILS.append(m)


def unit():
    t, y = np.array([0., 10, 20]), np.array([0., 1, 1])
    check(abs(C.auc(t, y, 20) - 15) < 1e-12, 'AUC trapezoid on [0, 20]')
    check(abs(C.auc(t, y, 40) - 35) < 1e-12, 'AUC holds the last value flat to T')
    check(abs(C.auc(t, y, 10) - 5) < 1e-12, 'AUC truncated at T')
    check(C.crossing(t, y, 0.5, 20) == 5.0, 'crossing interpolates linearly')
    check(C.crossing(t, y, 0.5, 4) is None, 'crossing beyond t_limit is not reached')
    check(C.crossing(t, y, 2.0, 20) is None, 'target never reached')
    check(C.delta_tau(50, 10, 30) == 10 and C.delta_tau(50, None, 30) == -np.inf
          and C.delta_tau(None, 10, 30) == np.inf and np.isnan(C.delta_tau(None, None, 30)), 'delta_tau censoring codes')
    S = np.array([[1, 0, 1, 1.]])
    src = np.array([0, 0, 1, 1])
    check(abs(C.em7(S, np.ones(4), src)[0] - 0.75) < 1e-12, 'em7 is the unweighted mean over sources')
    check(C.verdict(0.1, 0.2) == 'supports' and C.verdict(-0.2, -0.1) == 'contradicts'
          and C.verdict(-0.1, 0.1) == 'inconclusive', 'verdicts')


def write_run(root, name, ts, p_of_t, n_per_src, rng, full=False):
    d = os.path.join(root, name, 'eval')
    os.makedirs(d)
    for k, t in enumerate(ts):
        tag = f'gpuh{t:.2f}' if k < len(ts) - 1 else f'final_gpuh{t:.2f}'
        with open(os.path.join(d, f'eval_{tag}_step{k * 10}.jsonl'), 'w') as f:
            for s in C.SOURCES:
                for i in range(n_per_src):
                    f.write(json.dumps({'qid': f'{s}/q{i}', 'score': float(rng.rand() < p_of_t(t))}) + '\n')


def end_to_end():
    rng = np.random.RandomState(0)
    tmp = tempfile.mkdtemp(prefix='plan_r060_')
    ck = os.path.join(tmp, 'ck')
    ts = list(np.arange(0, 161, 10.0))
    slow = lambda t: 0.2 + 0.2 * t / 160
    fast = lambda t: 0.2 + 0.3 * min(t / 60, 1)
    for s in (1, 2, 3):
        write_run(ck, f'R02{s - 1}-p0-main-s{s}', ts, slow, 200, rng)
        write_run(ck, f'R02{s + 2}-p3-main-s{s}', ts[:13], fast, 200, rng)  # stops at 120 = T - H
    write_run(ck, 'R026-p2_all-main-s1', ts[:13], slow, 200, rng)
    for al, f in ((0.1, slow), (0.3, slow)):
        write_run(ck, f'R03{int(al * 10)}-p1_{al}-main-s1', ts, f, 200, rng)
    write_run(ck, 'R040-p1_0.3-negctrl-s1', ts, lambda t: 0.1, 200, rng)
    json.dump({'winner': 'p3', 'predicted_training_ranking': ['p3', 'p2_all', 'p0'],
               'diagnostics': {'reader_vs_off': {'ci95': [0.5, 0.8]}}}, open(os.path.join(tmp, 'pilot.json'), 'w'))
    json.dump({'_ledger': {'H_unbiased_gpu_h': 40.0}}, open(os.path.join(tmp, 'led.json'), 'w'))
    json.dump({'alpha': 0.3}, open(os.path.join(tmp, 'p1.json'), 'w'))
    p = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'curves.py'), '--ckpt', ck, '--pilot_result',
                        os.path.join(tmp, 'pilot.json'), '--ledger', os.path.join(tmp, 'led.json'), '--p1_alpha',
                        os.path.join(tmp, 'p1.json'), '--out_prefix', os.path.join(tmp, 'res'), '--n_boot', '200'],
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-3000:])
    check(p.returncode == 0, 'curves.py runs end to end')
    r = json.load(open(os.path.join(tmp, 'res.json')))
    check(r['pairs']['p3>p0']['verdict'] == 'supports', f"fast P3 beats P0 on AUC ({r['pairs']['p3>p0']['ci95']})")
    check(r['pairs']['p3>p2_all']['verdict'] == 'supports' and 'note' in r['pairs']['p3>p2_all'],
          'P3 > P2-all supported, flagged as a 1-seed comparison')
    check(r['pairs']['p2_all>p0']['verdict'] != 'supports', 'P2-all (same curve as P0, shorter) not better than P0')
    d = r['delta_tau']['p3']
    # P0 reaches its own final EM ~0.4 only at ~160; P3 reaches 0.4 at t = 40 -> delta ~ 160 - (40 + 40) = 80
    check(d['saving_established'] and 40 < d['point'] < 120, f"delta_tau(P3) ~ 80 and > 0 ({d['point']:.1f})")
    check(r['delta_tau']['p2_all']['frac_method_not_reached'] > 0.5, 'P2-all mostly censored as not reached')
    check(r['p1_alpha'] == 0.3 and r['n_seeds'].get('p1') == 1, 'only the selected P1 alpha enters the analysis')
    check('p1@neg' in r['negctrl'] and r['negctrl']['p1@neg']['ci95'][1] < 0, 'negative control hurts P1 (CI < 0)')
    it = r['interpretation']
    # B2 needs every adjacent ranking pair supported; p2_all > p0 is inconclusive here (same curve)
    check(it['B1'].startswith('supports') and it['B2'].startswith('inconclusive') and '1-seed' in it['B2']
          and it['C'].startswith('holds') and it['A'].startswith('confirmed'), f"interpretation table: {it}")
    p = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'p1_select.py'), '--ckpt', ck, '--runs', 'x', '--alphas', '1'],
                       capture_output=True, text=True)
    check(p.returncode != 0, 'p1_select refuses runs without a validation eval')


if __name__ == '__main__':
    unit()
    end_to_end()
    print('\nALL PASS' if not FAILS else f'\n{len(FAILS)} FAILED')
    sys.exit(1 if FAILS else 0)
