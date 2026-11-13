"""R011 stage 0.5 costs c(m) and the cost ledger H_m (FINAL_PROPOSAL "成本账本").

c(m): GPU seconds per question of one training step of method m, from the stage-0.5 runs
(<ckpt>/r011-<m>/eval/metrics.jsonl): first step dropped, mean of timing_s/step * n_gpus / B.
Output keys p0, p2_all, p3_m1, p3_m2, p3_m4 (input of r015_pilot_analysis.py --costs).

H_m (P2-all and P3; P0 = 0): main-reader pilot GPU hours + stage-0.5 GPU hours. The pilot part is
the R012a dump step time with the negative-control half of the reader time removed (both readers
run inside timing 'reader', with identical request sets) plus the R012b sketch time (q_*.pt seconds,
one GPU each). Stage 0.5 counts every step of all five runs. Written next to the costs as
'_ledger' (ignored by r015).
"""

import argparse
import glob
import json
import os

import numpy as np

RUNS = {'p0': 'r011-p0', 'p2_all': 'r011-p2_all', 'p3_m1': 'r011-p3-M1', 'p3_m2': 'r011-p3-M2',
        'p3_m4': 'r011-p3-M4'}


def steps(path):
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    return [r['timing_s/step'] for r in rows if 'timing_s/step' in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='verl_checkpoints')
    ap.add_argument('--B', type=int, default=128)
    ap.add_argument('--n_gpus', type=int, default=4)
    ap.add_argument('--pilot_dump', default=None, help='R012a dump dir (for H_m)')
    ap.add_argument('--sketch_dir', default=None, help='R012b output dir (for H_m)')
    ap.add_argument('--out', default='prereg/stage05_costs.json')
    a = ap.parse_args()
    costs, stage05_s = {}, 0.0
    for m, run in RUNS.items():
        s = steps(os.path.join(a.ckpt, run, 'eval', 'metrics.jsonl'))
        assert len(s) >= 2, f'{run}: need >= 2 logged steps, got {len(s)}'
        costs[m] = float(np.mean(s[1:])) * a.n_gpus / a.B
        stage05_s += sum(s) * a.n_gpus
        print(f'{m:7s} {len(s) - 1} steps kept, {np.mean(s[1:]):.1f} s/step -> c = {costs[m]:.2f} GPU s / question')
    ledger = {'stage05_gpu_h': stage05_s / 3600}
    if a.pilot_dump:
        tot = 0.0
        for f in glob.glob(os.path.join(a.pilot_dump, 'pilot_*.json')):
            meta = json.load(open(f))
            t = meta['timing_s']
            tot += (t['step'] - 0.5 * t.get('reader', 0.0)) * meta['n_gpus']
        ledger['pilot_dump_main_gpu_h'] = tot / 3600
    if a.sketch_dir:
        import torch
        ledger['pilot_sketch_gpu_h'] = sum(torch.load(f, map_location='cpu')['seconds']
                                           for f in glob.glob(os.path.join(a.sketch_dir, 'q_*.pt'))) / 3600
    ledger['H_unbiased_gpu_h'] = sum(ledger.values())
    costs['_ledger'] = ledger
    print(json.dumps(ledger, indent=1))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(costs, f, indent=1)


if __name__ == '__main__':
    main()
