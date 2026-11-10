"""R001: generate and freeze the data splits defined in EXPERIMENT_PLAN.md.

Question IDs are `data_source/id` because `id` alone repeats across sources.
ID lists go to prereg/splits/ (tracked in git); parquet files go to
data/prereg/ (ignored by git, reproducible from this script).
"""

import argparse
import hashlib
import json
import os
import re
import string

import numpy as np
import pandas as pd

SEED = 0
HELD_OUT = [('pilot_calib', 100), ('pilot_eval', 150), ('val', 500)]  # per source, NQ:HotpotQA = 1:1
TRAIN_SOURCES = ['hotpotqa', 'nq']
CURVE_PER_SOURCE = 500


def uid(df):
    return df['data_source'] + '/' + df['id'].astype(str)


def normalize(q):
    q = q.lower()
    q = ''.join(ch for ch in q if ch not in set(string.punctuation))
    return re.sub(r'\s+', ' ', q).strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_ids(path, ids):
    with open(path, 'w') as f:
        f.write('\n'.join(ids) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/nq_hotpotqa_train/train.parquet')
    parser.add_argument('--test', default='data/nq_hotpotqa_train/test.parquet')
    parser.add_argument('--split_dir', default='prereg/splits')
    parser.add_argument('--data_dir', default='data/prereg')
    args = parser.parse_args()
    os.makedirs(args.split_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    train = pd.read_parquet(args.train)
    test = pd.read_parquet(args.test)
    train['uid'] = uid(train)
    test['uid'] = uid(test)
    assert train['uid'].is_unique and test['uid'].is_unique
    train['qnorm'] = train['question'].map(normalize)

    # Held-out questions must not share text with any other training row.
    qcount = train['qnorm'].value_counts()
    eligible = train[train['qnorm'].map(qcount) == 1]

    rng = np.random.default_rng(SEED)
    splits = {name: [] for name, _ in HELD_OUT}
    for src in TRAIN_SOURCES:
        pool = eligible[eligible['data_source'] == src]['uid'].to_numpy()
        pool = pool[rng.permutation(len(pool))]
        start = 0
        for name, n in HELD_OUT:
            splits[name].extend(pool[start:start + n].tolist())
            start += n

    held_out = set().union(*splits.values())
    held_out_q = set(train.loc[train['uid'].isin(held_out), 'qnorm'])
    train_pool = train[~train['uid'].isin(held_out) & ~train['qnorm'].isin(held_out_q)]
    assert len(train_pool) == len(train) - len(held_out)

    curve = []
    for src in sorted(test['data_source'].unique()):
        ids = test[test['data_source'] == src]['uid'].to_numpy()
        n = min(CURVE_PER_SOURCE, len(ids))
        curve.extend(ids[np.sort(rng.choice(len(ids), size=n, replace=False))].tolist())
    splits['curve_eval'] = curve
    splits['train_pool'] = train_pool['uid'].tolist()

    for name, ids in splits.items():
        write_ids(os.path.join(args.split_dir, f'{name}.txt'), ids)

    by_uid_train = train.set_index('uid', drop=False)
    by_uid_test = test.set_index('uid', drop=False)
    outputs = {}
    for name, ids in splits.items():
        src_df = by_uid_test if name == 'curve_eval' else by_uid_train
        out = src_df.loc[ids].drop(columns=['uid', 'qnorm'], errors='ignore').reset_index(drop=True)
        path = os.path.join(args.data_dir, f'{name}.parquet')
        out.to_parquet(path)
        outputs[name] = {'parquet': path, 'sha256': sha256(path)}

    manifest = {
        'seed': SEED,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'sources': {p: sha256(p) for p in [args.train, args.test]},
        'id_format': 'data_source/id',
        'counts': {name: len(ids) for name, ids in splits.items()},
        'counts_by_source': {
            name: pd.Series([i.split('/')[0] for i in ids]).value_counts().sort_index().to_dict()
            for name, ids in splits.items()
        },
        'parquet': outputs,
    }
    with open(os.path.join(args.split_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest['counts'], indent=2))


if __name__ == '__main__':
    main()
