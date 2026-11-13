"""M1 summary (R007-R009): per-step wall clock and its sections, peak GPU memory, retrieval share.

Reads <ckpt>/<run>/eval/metrics.jsonl (timing_s/*) and logs/<run>.gpumem.csv. The first step is
dropped (vLLM / CUDA warm-up). Projects the steps and questions a 160 GPU-hour run would get.
"""

import argparse
import csv
import json
import os

import numpy as np

SECTIONS = ('gen', 'search', 'reader', 'reward', 'ref', 'old_log_prob', 'update_actor')


def load_metrics(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if 'timing_s/step' in r:
                rows.append(r)
    return rows


def peak_mem(path):
    if not os.path.exists(path):
        return None
    peak = {}
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('timestamp'):
                continue
            gpu, mem = row[1].strip(), float(row[2].strip().split()[0])
            peak[gpu] = max(peak.get(gpu, 0.0), mem)
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='verl_checkpoints')
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--n_gpus', type=int, default=4)
    ap.add_argument('--budget_gpu_h', type=float, default=160.0)
    a = ap.parse_args()
    out = {}
    for run in a.runs:
        rows = load_metrics(os.path.join(a.ckpt, run, 'eval', 'metrics.jsonl'))[1:]
        if not rows:
            print(f'{run}: no steps logged')
            continue
        step = np.array([r['timing_s/step'] for r in rows])
        sec = {s: float(np.mean([r.get(f'timing_s/{s}', 0.0) for r in rows])) for s in SECTIONS}
        steps_in_budget = a.budget_gpu_h * 3600 / (step.mean() * a.n_gpus)
        B = rows[0].get('plan/n_orig', None)
        out[run] = {'n_steps': len(rows), 'step_s_mean': float(step.mean()), 'step_s_sd': float(step.std()),
                    'sections_s': sec, 'retrieval_share': sec['search'] / step.mean(),
                    'steps_in_budget': steps_in_budget, 'peak_mem_mib': peak_mem(os.path.join('logs', f'{run}.gpumem.csv'))}
        print(f"{run}: {len(rows)} steps, {step.mean():.0f} +- {step.std():.0f} s/step, "
              f"{steps_in_budget:.0f} steps per {a.budget_gpu_h:g} GPU h, retrieval {100 * out[run]['retrieval_share']:.0f}%")
        print('   ' + ', '.join(f'{k} {v:.0f}s' for k, v in sec.items()))
        print(f"   peak memory (MiB): {out[run]['peak_mem_mib']}")
    with open(os.path.join('logs', 'm1_summary.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
