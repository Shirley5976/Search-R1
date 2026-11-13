"""R030-R032 selection: alpha* = argmax of the validation mean EM (unweighted over data sources) of the
screening checkpoints (eval tag p1_screen_val). Ties go to the smaller alpha. Writes {alpha, scores}."""

import argparse
import glob
import json
import os

import numpy as np


def val_em(run_dir):
    fs = sorted(glob.glob(os.path.join(run_dir, 'eval', 'eval_p1_screen_val_step*.jsonl')))
    assert fs, f'no p1_screen_val eval under {run_dir}'
    by = {}
    for r in map(json.loads, open(fs[-1])):
        by.setdefault(r['qid'].split('/')[0], []).append(r['score'])
    return float(np.mean([np.mean(v) for v in by.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='verl_checkpoints')
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--alphas', nargs='+', type=float, required=True)
    ap.add_argument('--out', default='prereg/p1_alpha.json')
    a = ap.parse_args()
    scores = {al: val_em(os.path.join(a.ckpt, r)) for al, r in zip(a.alphas, a.runs)}
    best = max(sorted(scores), key=lambda al: scores[al])  # sorted + strict max keeps the smaller alpha on ties
    print(f'validation EM: {scores} -> alpha* = {best:g}')
    with open(a.out, 'w') as f:
        json.dump({'alpha': float(f'{best:g}'), 'val_em': {f'{k:g}': v for k, v in scores.items()}}, f, indent=1)


if __name__ == '__main__':
    main()
