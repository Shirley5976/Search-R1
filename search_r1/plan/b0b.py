"""B0b (R004): rebuild the training update batch of a method from a pilot-format dump.

The same B=16 rollout (plan.mode=pilot dump, M=4 siblings) is turned into the expanded batch that
advantages.py + step.py feed to dp_actor.update_policy_single_step, for any (method, beta, M <= 4).
P3 with M < 4 keeps the first M dumped siblings of each parent and their phis (the same rule as
formula.p3_coef), so the implementation path and the formula path see identical samples.
"""

import glob
import json
import os

import numpy as np
import torch

from verl import DataProto
from search_r1.plan.step import build_update_batch, compute_plan_advantages


def parse_spec(spec):
    """'p3:0.5:4' -> ('p3', 0.5, 4); 'p0' -> ('p0', 0.0, 1)."""
    parts = spec.split(':')
    beta = float(parts[1]) if len(parts) > 1 else 0.0
    M = int(parts[2]) if len(parts) > 2 else 1
    return parts[0], beta, M


def spec_tag(method, beta, M):
    """File tag shared with r012b --b0b_method / --b0b_beta / --b0b_M."""
    return f'{method}_b{beta:g}_m{M}'


def load_dump(dump_dir):
    """[(records, tensors)] for every finished batch of the dump, in file order."""
    out = []
    for js in sorted(glob.glob(os.path.join(dump_dir, 'pilot_*.json'))):
        if not os.path.exists(js[:-5] + '.done'):
            continue
        meta = json.load(open(js))
        out.append((meta['trajectories'], torch.load(js[:-5] + '.pt', map_location='cpu')))
    assert out, f'no finished pilot batches under {dump_dir}'
    return out


def records_to_trajs(recs, M):
    trajs = []
    for r in recs:
        fork = None if r['fork'] is None else list(r['fork']['main'])[:M + 1]
        trajs.append(dict(n_turns=r['n_turns'], k_star=r['k_star'],
                          phi_prev={k: r['phi_prev']['main'][k - 1] for k in range(1, r['n_turns'] + 1)},
                          search_turns={}, fork_phis=fork))
    return trajs


def _concat_padded(parts):
    """Concat batches whose prompt / response widths differ (each dump batch pads independently):
    the prompt side is left-padded and the response side right-padded with zeros (masks, turn ids
    and advantages are 0 there, so padding is inert)."""
    if len(parts) == 1:
        return parts[0]
    P = max(p.batch['prompts'].shape[1] for p in parts)
    L = max(p.batch['responses'].shape[1] for p in parts)
    F = torch.nn.functional
    out = []
    for p in parts:
        p0, l0 = p.batch['prompts'].shape[1], p.batch['responses'].shape[1]
        t = {}
        for k, v in p.batch.items():
            if v.dim() == 2 and v.shape[1] == p0 + l0 and k != 'responses':
                v = torch.cat([F.pad(v[:, :p0], (P - p0, 0)), F.pad(v[:, p0:], (0, L - l0))], 1)
            elif v.dim() == 2 and v.shape[1] == l0:
                v = F.pad(v, (0, L - l0))
            elif k == 'prompts':
                v = F.pad(v, (P - p0, 0))
            t[k] = v
        out.append(DataProto.from_dict(tensors=t))
    return DataProto.concat(out)


def update_batch(batches, method, beta, M, world_size):
    """Expanded update batch over all dump batches (originals then siblings per dump batch),
    padded with zero-weight rows to a multiple of world_size."""
    parts = []
    for recs, tens in batches:
        uids = [r['uid'] for r in recs]
        orig = DataProto.from_dict(tensors=dict(tens['orig']), non_tensors={'uid': np.array(uids, dtype=object)})
        sib, sib_recs = None, []
        if tens['sib'] is not None and method == 'p3':
            owner = {}
            for i, r in enumerate(recs):
                if r['fork'] is not None:
                    for srow in r['fork']['sib_rows'][:M]:
                        owner[srow] = {'parent': i, 'k': r['k_star']}
            keep = sorted(owner)
            if keep:
                sib = DataProto.from_dict(tensors={k: v[keep] for k, v in tens['sib'].items()})
                sib_recs = [owner[s] for s in keep]
        z, adj, sib_adv = compute_plan_advantages(method, [r['score'] for r in recs], uids,
                                                  records_to_trajs(recs, M), beta, 0.0)
        upd, _, _ = build_update_batch(orig, sib, sib_recs, z, adj, sib_adv, M, world_size=1)
        parts.append(DataProto.from_dict(tensors={k: upd.batch[k] for k in upd.batch.keys()}))
    batch = _concat_padded(parts)
    pad = (-len(batch)) % world_size
    if pad:
        t = {k: v[:1].repeat(pad, *([1] * (v.dim() - 1))) for k, v in batch.batch.items()}
        for k in ('advantages', 'loss_mask', 'loss_weight', 'is_original'):
            t[k] = torch.zeros_like(t[k])
        batch = DataProto.concat([batch, DataProto.from_dict(tensors=t)])
    return batch
