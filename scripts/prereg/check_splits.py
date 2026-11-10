"""R001 checks: disjointness, counts, no question-text leakage, reproducibility."""

import filecmp
import os
import subprocess
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from make_splits import normalize  # noqa: E402

SPLIT_DIR = 'prereg/splits'
EXPECTED = {'pilot_calib': 200, 'pilot_eval': 300, 'val': 1000, 'curve_eval': 3125}


def read_ids(name, split_dir=SPLIT_DIR):
    with open(os.path.join(split_dir, f'{name}.txt')) as f:
        return f.read().split()


def main():
    names = list(EXPECTED) + ['train_pool']
    ids = {n: read_ids(n) for n in names}

    for n, want in EXPECTED.items():
        assert len(ids[n]) == want, (n, len(ids[n]), want)
    for n in names:
        assert len(set(ids[n])) == len(ids[n]), f'duplicate ids in {n}'
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(ids[a]) & set(ids[b]), f'{a} and {b} overlap'

    for n in ['pilot_calib', 'pilot_eval', 'val']:
        srcs = pd.Series([u.split('/')[0] for u in ids[n]]).value_counts().to_dict()
        assert srcs == {'nq': EXPECTED[n] // 2, 'hotpotqa': EXPECTED[n] // 2}, (n, srcs)

    train = pd.read_parquet('data/nq_hotpotqa_train/train.parquet')
    assert len(ids['train_pool']) + 1500 == len(train)
    pool = pd.read_parquet('data/prereg/train_pool.parquet')
    held = pd.concat([pd.read_parquet(f'data/prereg/{n}.parquet') for n in ['pilot_calib', 'pilot_eval', 'val']])
    assert not set(pool['question'].map(normalize)) & set(held['question'].map(normalize)), 'question text leak'
    assert list(pool['data_source'] + '/' + pool['id']) == ids['train_pool']

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'make_splits.py'),
                        '--split_dir', f'{tmp}/splits', '--data_dir', f'{tmp}/data'],
                       check=True, capture_output=True)
        for n in names:
            assert filecmp.cmp(f'{SPLIT_DIR}/{n}.txt', f'{tmp}/splits/{n}.txt', shallow=False), f'{n} not reproducible'

    print('all R001 checks passed')


if __name__ == '__main__':
    main()
