"""CPU test of search_r1/plan/advantages.py (checklist items 3, 4, 7).

  1. grpo_z equals verl core_algos.compute_grpo_outcome_advantage (incl. singleton / constant groups);
  2. P3 baselines are the leave-one-out means; C_0 = mean of the siblings;
  3. exact enumeration on a softmax bandit: the expected P3 builder correction
     -beta(phi_0 - C_0) grad log pi(a_0) + (1/M) sum_j beta(phi_j - C_j) grad log pi(a_j) is zero;
  4. P2-all / P2-sel / P1 formulas on hand-made trajectories;
  5. token_advantages puts z + adj[k] on turn-k tokens and 0 where turn id = 0.

Run: python -I scripts/plan/item34_advantages_test.py
"""

import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from search_r1.plan.advantages import (grpo_z, p3_baselines, p3_deltas, turn_adjustments, token_advantages)

FAILS = []


def check(cond, msg):
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)


def test_grpo_z():
    rng = np.random.RandomState(0)
    uids = ['a'] * 5 + ['b'] * 5 + ['c'] + ['d'] * 5
    scores = np.concatenate([rng.randint(0, 2, 5), [1, 1, 1, 1, 1], [1], rng.rand(5)]).astype(np.float64)
    L = 7
    tlr = torch.zeros(len(scores), L)
    tlr[:, 3] = torch.tensor(scores, dtype=torch.float32)
    mask = torch.ones(len(scores), L)
    ref, _ = compute_grpo_outcome_advantage(tlr, mask, np.array(uids, dtype=object))
    z = grpo_z(scores, uids)
    check(np.allclose(ref[:, 0].numpy(), z, atol=1e-5), 'grpo_z == verl compute_grpo_outcome_advantage')
    check(np.allclose(z[5:10], 0) and np.isclose(z[10], 1.0), 'constant group -> 0, singleton -> score (mean 0, std 1)')


def test_p3_baselines():
    phis = np.array([0.9, 0.1, 0.4, 0.7, 0.3])
    c = p3_baselines(phis)
    loo = np.array([np.delete(phis, j).mean() for j in range(5)])
    check(np.allclose(c, loo) and np.isclose(c[0], phis[1:].mean()), 'C_j = leave-one-out mean, C_0 = sibling mean')
    check(np.allclose(p3_deltas(phis), phis - loo), 'p3_deltas = phi - C')


def test_p3_unbiased_bandit():
    """Exact expectation over (a_0, a_1..a_M) iid ~ softmax(theta) on A actions."""
    rng = np.random.RandomState(1)
    ok = True
    for A, M in [(3, 1), (3, 2), (4, 3), (2, 4)]:
        theta = rng.randn(A)
        pi = np.exp(theta) / np.exp(theta).sum()
        phi = rng.rand(A)
        beta = 0.7
        grad_logpi = np.eye(A) - pi[None, :]  # row a: d log pi(a) / d theta
        total = np.zeros(A)
        for acts in itertools.product(range(A), repeat=M + 1):
            p = np.prod(pi[list(acts)])
            d = p3_deltas(phi[list(acts)])
            g = -beta * d[0] * grad_logpi[acts[0]]
            g += sum(beta * d[j] * grad_logpi[acts[j]] for j in range(1, M + 1)) / M
            total += p * g
        ok &= np.allclose(total, 0, atol=1e-12)
        # and the sibling part alone is a nonzero (beta * grad E phi) term, so the test is not vacuous
        g_phi = (pi[:, None] * phi[:, None] * grad_logpi).sum(0)
        ok &= np.linalg.norm(g_phi) > 1e-3
    check(ok, 'P3 builder correction has exact zero expectation (softmax bandit, M in {1,2,3,4})')


def make_trajs():
    return [
        dict(uid='q', n_turns=3, k_star=2, phi_prev={1: .1, 2: .3, 3: .6},
             search_turns={1: (.1, .3), 2: (.3, .6)}, fork_phis=[.6, .2, .5]),
        dict(uid='q', n_turns=1, k_star=3, phi_prev={1: .1}, search_turns={}, fork_phis=None),
        dict(uid='q', n_turns=2, k_star=1, phi_prev={1: .1, 2: .2},
             search_turns={1: (.1, .2)}, fork_phis=[.2, .2, .8]),
        dict(uid='r', n_turns=2, k_star=2, phi_prev={1: .5, 2: .5}, search_turns={1: (.5, .5)}, fork_phis=None),
    ]


def test_methods():
    T = make_trajs()
    beta, alpha = 0.5, 0.2
    adj, sib = turn_adjustments('p0', T, beta, alpha)
    check(all(not a for a in adj) and all(not s for s in sib), 'P0: no adjustment')

    adj, _ = turn_adjustments('p2_all', T, beta)
    check(adj[0] == {1: -.05, 2: -.15, 3: -.3} and adj[1] == {1: -.05}, 'P2-all: -beta*Phi(S_{k-1}) on every turn')

    adj, _ = turn_adjustments('p2_sel', T, beta)
    check(adj[0] == {2: -.15} and adj[1] == {} and adj[2] == {1: -.05} and adj[3] == {2: -.25},
          'P2-sel: only k*, nothing when the trajectory ends before k*')

    adj, sib = turn_adjustments('p3', T, beta)
    d0 = p3_deltas([.6, .2, .5])
    ok = np.isclose(adj[0][2], -beta * d0[0]) and np.allclose(sib[0], beta * d0[1:])
    ok &= adj[1] == {} and sib[1] == [] and adj[3] == {} and sib[3] == []
    check(ok, 'P3: -beta(phi0-C0) at k*, siblings +beta(phi_j-C_j), none without a fork')

    adj, _ = turn_adjustments('p1', T, alpha=alpha)
    r = np.array([.2, .3, .1])  # group q pooled turn rewards: traj0 k1, traj0 k2, traj2 k1
    zr = (r - r.mean()) / (r.std(ddof=1) + 1e-6)
    ok = np.isclose(adj[0][1], alpha * zr[0]) and np.isclose(adj[0][2], alpha * zr[1])
    ok &= np.isclose(adj[2][1], alpha * zr[2]) and adj[1] == {} and 3 not in adj[0]
    ok &= np.isclose(adj[3][1], 0.0)  # singleton group r: (0 - 0) / 1
    check(ok, 'P1: alpha * z over the group\'s executed-search turns; non-search turns get nothing')


def test_token_advantages():
    tid = torch.tensor([[1, 1, 0, 0, 2, 2, 0, 3, 0],
                        [1, 0, 0, 2, 2, 0, 0, 0, 0],
                        [1, 1, 0, 0, 2, 0, 0, 0, 0]])
    base = [0.5, -1.0, 0.0]
    adj = [{2: 0.1, 3: -0.2}, {}, {2: 0.3}]
    a = token_advantages(tid, base, adj)
    exp = torch.tensor([[.5, .5, 0, 0, .6, .6, 0, .3, 0],
                        [-1, 0, 0, -1, -1, 0, 0, 0, 0],
                        [0, 0, 0, 0, .3, 0, 0, 0, 0]])
    check(torch.allclose(a, exp), 'token_advantages: z + adj[k] on turn-k tokens, 0 on turn id 0')
    check(a.dtype == torch.float32 and a.shape == tid.shape, 'token_advantages dtype/shape')


def main():
    test_grpo_z()
    test_p3_baselines()
    test_p3_unbiased_bandit()
    test_methods()
    test_token_advantages()
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
