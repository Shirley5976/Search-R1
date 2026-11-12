"""Build the frozen negative-control target map (checklist item 6, Block 3 negctrl).

Every training-side question (train_pool, pilot_calib, pilot_eval) is mapped, within its answer
class, to another question whose gold aliases the reader scores instead. Writes
data/prereg/negctrl_map.json ({"data_source/id": {"qid": ..., "aliases": [...]}}) and prints the
sha256 and class counts that go into prereg/decisions.md.

Run: python -I scripts/plan/make_negctrl_map.py
"""

import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from search_r1.plan.reader_prompt import build_negctrl_map, mapping_sha256, clean_aliases

SPLITS = ('train_pool', 'pilot_calib', 'pilot_eval')


def main():
    aliases = {}
    for s in SPLITS:
        df = pd.read_parquet(os.path.join(ROOT, 'data/prereg', f'{s}.parquet'), columns=['id', 'data_source', 'reward_model'])
        for qid, ds, rm in zip(df['id'], df['data_source'], df['reward_model']):
            key = f'{ds}/{qid}'
            assert key not in aliases, f'duplicate qid {key}'
            aliases[key] = clean_aliases(rm['ground_truth']['target'])
    mapping, counts = build_negctrl_map(aliases, seed=0)
    out = {q: {'qid': t, 'aliases': aliases[t]} for q, t in sorted(mapping.items())}
    path = os.path.join(ROOT, 'data/prereg/negctrl_map.json')
    with open(path, 'w') as f:
        json.dump(out, f, ensure_ascii=False)
    fixed = sum(q == t for q, t in mapping.items())
    print(json.dumps({'path': os.path.relpath(path, ROOT), 'n': len(mapping), 'sha256': mapping_sha256(mapping),
                      'class_counts': counts, 'fixed_points': fixed}, indent=1))


if __name__ == '__main__':
    main()
