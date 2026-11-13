"""CPU test of the R004 chain without FSDP: search_r1.plan.b0b.update_batch + r004_compare.py.

Reuses the item-8 fake pilot dump and tiny fp32 Qwen2. The "implementation" gradient is the
dp_actor single-step loss evaluated row by row on the batch b0b.update_batch builds (padded to
world 4); it is written in the impl_<tag>_ws<w>.pt format (as grad(loss), like r004_b0b_impl.py),
r012b provides the formula path (with --b0b_method for the exact subset), and r004_compare must pass.
A deliberately wrong beta must fail.

Run: HF_HUB_OFFLINE=1 python -I scripts/plan/r004_cpu_test.py
"""

import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
sys.path.insert(0, HERE)

from search_r1.plan import b0b
from search_r1.plan.sketch import grad_sketch, clean_name
from item8_pilot_chain_test import make_dump, tiny_model
from r012b_pilot_sketch import Replica, low_var_kl

PY = sys.executable
D = 2 ** 14
PREFIX = 'model.layers.1.'
FAILS = []


def check(c, m):
    print(f"[{'PASS' if c else 'FAIL'}] {m}")
    if not c:
        FAILS.append(m)


def impl_grad(rep, batch, kl_coef, L):
    rep.model.zero_grad(set_to_none=True)
    ub = batch.batch
    n_orig = float(ub['is_original'].sum())
    P = ub['prompts'].shape[1]
    for r in range(len(batch)):
        if float(ub['loss_weight'][r]) == 0.0:
            continue
        t = {k: ub[k] for k in ('prompts', 'input_ids', 'attention_mask', 'turn_ids')}
        logp, _, _ = rep.row_forward(t, r, False)
        r_valid = int(ub['attention_mask'][r, P:].sum())
        idx = torch.nonzero(ub['turn_ids'][r, :r_valid] > 0).view(-1)
        A = ub['advantages'][r, :r_valid][idx]
        mask = ub['loss_mask'][r, :r_valid][idx].float()
        loss = (-A * torch.exp(logp - logp.detach()) * mask * float(ub['loss_weight'][r])).sum() / (n_orig * L)
        if float(ub['is_original'][r]):
            kld = low_var_kl(logp, ub['ref_log_prob'][r, :r_valid][idx].float())
            loss = loss + kl_coef * (kld * mask).sum() / (n_orig * L)
        loss.backward()
    sk = grad_sketch(rep.model, D=D)
    exact = {clean_name(n): p.grad.detach().float().clone() for n, p in rep.model.named_parameters()
             if p.grad is not None and clean_name(n).startswith(PREFIX)}
    rep.model.zero_grad(set_to_none=True)
    return sk, exact, n_orig


def main():
    tmp = tempfile.mkdtemp(prefix='plan_r004_')
    dump = make_dump(tmp)
    model = tiny_model(tmp)
    sk_dir, impl_dir = os.path.join(tmp, 'sketch'), os.path.join(tmp, 'impl')
    os.makedirs(impl_dir)
    specs = ['p0', 'p2_all:0.5:1', 'p3:0.5:4', 'p3:0.5:2']
    base = [PY, '-I', os.path.join(HERE, 'r012b_pilot_sketch.py'), '--dump_dir', dump, '--model', model,
            '--out_dir', sk_dir, '--device', 'cpu', '--dtype', 'fp32', '--D', str(D), '--exact_prefix', PREFIX,
            '--split_files', os.path.join(tmp, 'pilot_calib.txt'), os.path.join(tmp, 'pilot_eval.txt')]
    env = {**os.environ, 'RANK': '0', 'WORLD_SIZE': '1'}
    p = subprocess.run(base + ['--exact_questions', '0'], capture_output=True, text=True, env=env)
    check(p.returncode == 0, 'r012b per-vector sketches')
    for spec in specs:
        m, b, M = b0b.parse_spec(spec)
        p = subprocess.run(base + ['--b0b_method', m, '--b0b_beta', str(b), '--b0b_M', str(M)],
                           capture_output=True, text=True, env=env)
        if p.returncode:
            print(p.stderr[-2000:])
        check(p.returncode == 0, f'r012b B0b exact subset for {spec}')

    rep = Replica(model, 'cpu', grad_ckpt=True, dtype=torch.float32)
    batches = b0b.load_dump(dump)
    for spec in specs:
        m, b, M = b0b.parse_spec(spec)
        batch = b0b.update_batch(batches, m, b, M, world_size=4)
        check(len(batch) % 4 == 0, f'{spec}: update batch padded to world 4 ({len(batch)} rows)')
        sk, exact, n_orig = impl_grad(rep, batch, 0.001, 500.0)
        torch.save({'spec': spec, 'sketch': sk, 'exact': exact, 'n_orig': n_orig, 'D': D, 'kl_coef': 0.001,
                    'loss_len': 500.0}, os.path.join(impl_dir, f'impl_{b0b.spec_tag(m, b, M)}_ws4.pt'))
    p = subprocess.run([PY, '-I', os.path.join(HERE, 'r004_compare.py'), '--impl_dir', impl_dir, '--sketch_dir',
                        sk_dir, '--tol', '1e-4'], capture_output=True, text=True)
    print(p.stdout[-2500:], p.stderr[-1500:])
    check(p.returncode == 0, 'r004_compare passes (sketch, exact subset, separation from P0)')

    # negative check: implementation run with a different beta must fail the comparison
    bad = os.path.join(tmp, 'impl_bad')
    os.makedirs(bad)
    batch = b0b.update_batch(batches, 'p3', 0.9, 4, world_size=4)
    sk, exact, n_orig = impl_grad(rep, batch, 0.001, 500.0)
    torch.save({'spec': 'p3:0.5:4', 'sketch': sk, 'exact': exact, 'n_orig': n_orig, 'D': D, 'kl_coef': 0.001,
                'loss_len': 500.0}, os.path.join(bad, 'impl_p3_b0.5_m4_ws4.pt'))
    p = subprocess.run([PY, '-I', os.path.join(HERE, 'r004_compare.py'), '--impl_dir', bad, '--sketch_dir', sk_dir,
                        '--tol', '1e-2'], capture_output=True, text=True)
    check(p.returncode != 0, 'r004_compare fails when the implementation used beta 0.9 instead of 0.5')
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}   (artifacts in {tmp})")
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
