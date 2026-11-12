"""Unit test for search_r1.plan.budget.GpuBudget (implementation checklist item 9).

Run: python scripts/plan/item9_budget_test.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from search_r1.plan.budget import GpuBudget


def main():
    fails = 0

    def check(name, ok):
        nonlocal fails
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    # 4 GPUs, eval every 10 GPU-h = every 9000 wall seconds of step time
    b = GpuBudget(n_gpus=4, eval_every_gpu_h=10, budget_gpu_h=25)
    fired = []
    for step in range(1, 31):
        b.add_step(1000)  # 1000 s wall -> 4000 GPU-s per step
        if b.eval_due():
            fired.append(step)
    # thresholds 36000, 72000, 108000 GPU-s -> steps 9, 18, 27
    check(f'eval fires once per 10 GPU-h threshold (got steps {fired})', fired == [9, 18, 27])
    check('GPU hours = wall * n_gpus', abs(b.used_gpu_h - 30 * 4000 / 3600) < 1e-9)

    b = GpuBudget(n_gpus=4, budget_gpu_h=25)
    n = 0
    while not b.exhausted():
        b.add_step(1000)
        n += 1
    check(f'budget stop at first step reaching 25 GPU-h (got {n}, expect 23)', n == 23)

    b = GpuBudget(n_gpus=4, eval_every_gpu_h=10)
    b.add_step(25 * 900)  # one huge step crosses two thresholds
    check('skipped thresholds collapse into one eval', b.eval_due() and not b.eval_due())
    b.add_step(5 * 900)  # now at 30 GPU-h
    check('next threshold after collapse is 30 GPU-h', b.eval_due())

    off = GpuBudget(n_gpus=4)
    off.add_step(1e9)
    check('defaults: no eval, no budget stop', not off.eval_due() and not off.exhausted())

    # resume round-trip
    a = GpuBudget(n_gpus=4, eval_every_gpu_h=10)
    for _ in range(5):
        a.add_step(1000)
        a.eval_due()
    r = GpuBudget(n_gpus=4, eval_every_gpu_h=10)
    r.load_state_dict(a.state_dict())
    seq_a, seq_r = [], []
    for _ in range(20):
        a.add_step(1000); r.add_step(1000)
        seq_a.append(a.eval_due()); seq_r.append(r.eval_due())
    check('resume restores used time and next eval threshold', seq_a == seq_r and a.used_s == r.used_s)

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
