"""R003 (Block 0a): exhaustive exact-rational checks of the P0/P2/P3 estimators on a toy MDP.

Toy: one question, K=2 decision turns. Turn 1 at s0: search_a -> sa, search_b -> sb, ans_x
(correct) or ans_y. Turn 2 at sa/sb: ans_x or ans_y. Tabular softmax policy, so the score of
action a at state s is e_{s,a} - pi(.|s). R = EM of the final answer.

GRPO advantage z_i = (R_i - k/G) * s_k, where k is the number of correct answers in the group
and s_k = 1/(std_k + eps) is kept as a free symbol: every identity is checked for any per-k
scale, including verl's std + 1e-6. Groups with k in {0, G} get z = 0. KL and entropy are
omitted (the plan applies KL only to original trajectories and sets entropy_coeff=0).

Definitions follow refine-logs/FINAL_PROPOSAL.md (Core Mechanism):
  Z_i = 1{reach k*_i} [u_0(phi_0 - C_0) - (1/M) sum_j u_j(phi_j - C_j)]
  V_beta = V0 + beta^2 V_Z - 2 beta C,  beta* = C / V_Z,  rho^2 = C^2 / (V0 V_Z)
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from fractions import Fraction as F
from itertools import product

# ----------------------------------------------------------------------------- exact algebra


def mono_mul(a, b):
    if not a:
        return b
    if not b:
        return a
    d = dict(a)
    for v, e in b:
        d[v] = d.get(v, 0) + e
    return tuple(sorted(d.items()))


class Poly:
    """Polynomial with Fraction coefficients; a monomial is a sorted ((var, exp), ...) tuple."""

    def __init__(self, terms=None):
        self.t = {m: c for m, c in (terms or {}).items() if c != 0}

    @staticmethod
    def var(name):
        return Poly({((name, 1),): F(1)})

    @staticmethod
    def lift(x):
        return x if isinstance(x, Poly) else Poly({(): F(x)})

    def __add__(self, o):
        t = dict(self.t)
        for m, c in Poly.lift(o).t.items():
            t[m] = t.get(m, 0) + c
        return Poly(t)

    __radd__ = __add__

    def __neg__(self):
        return Poly({m: -c for m, c in self.t.items()})

    def __sub__(self, o):
        return self + (-Poly.lift(o))

    def __rsub__(self, o):
        return Poly.lift(o) - self

    def __mul__(self, o):
        o = Poly.lift(o)
        t = defaultdict(F)
        for m1, c1 in self.t.items():
            for m2, c2 in o.t.items():
                t[mono_mul(m1, m2)] += c1 * c2
        return Poly(t)

    __rmul__ = __mul__

    def is_zero(self):
        return not self.t

    def eval(self, values):
        return sum(float(c) * math.prod(values[v] ** e for v, e in m) for m, c in self.t.items())


class VecPoly:
    """Vector-valued polynomial: monomial -> list of Fractions."""

    def __init__(self):
        self.t = {}

    def add(self, vec, coef=F(1), mono=()):
        if coef == 0:
            return self
        cur = self.t.get(mono)
        if cur is None:
            self.t[mono] = [coef * x for x in vec]
        else:
            for i, x in enumerate(vec):
                if x:
                    cur[i] += coef * x
        return self

    def add_vp(self, other, coef=F(1), mono=()):
        for m, v in other.t.items():
            self.add(v, coef, mono_mul(m, mono))
        return self

    def dot(self, other):
        t = defaultdict(F)
        for m1, v1 in self.t.items():
            for m2, v2 in other.t.items():
                d = sum(a * b for a, b in zip(v1, v2))
                if d:
                    t[mono_mul(m1, m2)] += d
        return Poly(t)

    def is_zero(self):
        return all(not any(v) for v in self.t.values())


def vp(vec):
    return VecPoly().add(vec)


def zeros():
    return [F(0)] * D


def vadd(a, b):
    return [x + y for x, y in zip(a, b)]


def vscale(a, c):
    return [x * c for x in a]


def vsum(vs):
    out = zeros()
    for v in vs:
        out = vadd(out, v)
    return out


def vdot(a, b):
    return sum(x * y for x, y in zip(a, b))


def fprod(xs):
    return math.prod(xs, start=F(1))


# ----------------------------------------------------------------------------- toy MDP

ACTIONS = {'s0': ['search_a', 'search_b', 'ans_x', 'ans_y'], 'sa': ['ans_x', 'ans_y'], 'sb': ['ans_x', 'ans_y']}
NEXT = {'search_a': 'sa', 'search_b': 'sb'}
OFFSET = {'s0': 0, 'sa': 4, 'sb': 6}
D = 8
K = 2


def truncate_top_p(probs, top_p):
    """Nucleus sampling: keep the smallest top set with cumulative mass >= top_p, renormalize."""
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    keep, total = set(), F(0)
    for i in order:
        keep.add(i)
        total += probs[i]
        if total >= top_p:
            break
    return [probs[i] / total if i in keep else F(0) for i in range(len(probs))]


class Toy:

    def __init__(self, pi, phi_state, top_p=F(1)):
        self.pi = pi  # the policy that is differentiated
        self.q = {s: truncate_top_p(p, top_p) for s, p in pi.items()}  # the sampling distribution
        self.phi_state = phi_state  # reader score Phi of each state (s0 = no documents yet)
        self.trajs = self._trajectories()

    def score(self, s, a):
        v = zeros()
        for j, p in enumerate(self.pi[s]):
            v[OFFSET[s] + j] = (1 if j == a else 0) - p
        return v

    def phi(self, s, a):
        name = ACTIONS[s][a]
        if name in NEXT:
            return self.phi_state[NEXT[name]]
        return F(int(name == 'ans_x'))

    def _trajectories(self):
        out = []
        for a1, p1 in enumerate(self.q['s0']):
            if p1 == 0:
                continue
            name = ACTIONS['s0'][a1]
            if name in NEXT:
                s1 = NEXT[name]
                for a2, p2 in enumerate(self.q[s1]):
                    if p2:
                        out.append(dict(prob=p1 * p2, steps=[('s0', a1), (s1, a2)],
                                        R=int(ACTIONS[s1][a2] == 'ans_x')))
            else:
                out.append(dict(prob=p1, steps=[('s0', a1)], R=int(name == 'ans_x')))
        for t in out:
            t['S'] = vsum(self.score(s, a) for s, a in t['steps'])
        return out


# ----------------------------------------------------------------------------- estimators


def z_p3(model, s, a0, sibs):
    cands = [a0, *sibs]
    M = len(sibs)
    ph = [model.phi(s, a) for a in cands]
    tot = sum(ph)
    C = [(tot - ph[c]) / M for c in range(M + 1)]  # leave-one-out mean over the other M candidates
    z = vscale(model.score(s, a0), ph[0] - C[0])
    for j in range(1, M + 1):
        z = vadd(z, vscale(model.score(s, cands[j]), -(ph[j] - C[j]) / M))
    return z


def ext_p3(toy, t, M):
    """(prob, Z) outcomes of one trajectory's correction, over k* and its M siblings."""
    out = []
    for kstar in range(1, K + 1):
        pk = F(1, K)
        if kstar > len(t['steps']):
            out.append((pk, zeros()))
            continue
        s, a0 = t['steps'][kstar - 1]
        q = toy.q[s]
        for sibs in product(range(len(q)), repeat=M):
            p = fprod(q[b] for b in sibs)
            if p:
                out.append((pk * p, z_p3(toy, s, a0, sibs)))
    return out


def ext_p2_all(toy, t):
    y = vsum(vscale(toy.score(s, a), toy.phi_state[s]) for s, a in t['steps'])
    return [(F(1), y)]


def ext_p2_sel(toy, t):
    out = []
    for kstar in range(1, K + 1):
        if kstar > len(t['steps']):
            out.append((F(1, K), zeros()))
        else:
            s, a = t['steps'][kstar - 1]
            out.append((F(1, K), vscale(toy.score(s, a), toy.phi_state[s])))
    return out


def corrections_for(toy):
    T = range(len(toy.trajs))
    corr = {'P2-all': [ext_p2_all(toy, toy.trajs[t]) for t in T],
            'P2-sel': [ext_p2_sel(toy, toy.trajs[t]) for t in T]}
    for M in (1, 2, 4):
        corr[f'P3-M{M}'] = [ext_p3(toy, toy.trajs[t], M) for t in T]
    return corr


def g_group(toy, combo, G):
    """Base GRPO gradient g_G of one group (trajectory indices), as a VecPoly in s_k."""
    g = VecPoly()
    R = [toy.trajs[t]['R'] for t in combo]
    k = sum(R)
    if 0 < k < G:
        mono = ((f's{k}', 1),)
        for t, r in zip(combo, R):
            g.add(toy.trajs[t]['S'], (r - F(k, G)) / G, mono)
    return g


def factorized(toy, G, corrections):
    """Exact V0 and, per correction X, E[X], V_X = tr Var(mean_i X_i), C_X = tr Cov(g_G, mean_i X_i).

    Uses only that X_i depends on trajectory i (plus its own k* and siblings), which holds by construction.
    """
    T = len(toy.trajs)
    xbar = {n: [vsum(vscale(x, p) for p, x in ext[t]) for t in range(T)] for n, ext in corrections.items()}
    E_g, E_gg = VecPoly(), Poly()
    E_gx = {n: Poly() for n in corrections}
    for combo in product(range(T), repeat=G):
        p = fprod(toy.trajs[t]['prob'] for t in combo)
        g = g_group(toy, combo, G)
        E_g.add_vp(g, p)
        E_gg = E_gg + p * g.dot(g)
        for n in corrections:
            E_gx[n] = E_gx[n] + p * g.dot(vp(vscale(vsum(xbar[n][t] for t in combo), F(1, G))))
    V0 = E_gg - E_g.dot(E_g)
    stats = {}
    for n, ext in corrections.items():
        EX = vsum(vscale(xbar[n][t], toy.trajs[t]['prob']) for t in range(T))
        EXX = sum(toy.trajs[t]['prob'] * p * vdot(x, x) for t in range(T) for p, x in ext[t])
        stats[n] = dict(EX=EX, VX=(EXX - vdot(EX, EX)) / G, CX=E_gx[n] - E_g.dot(vp(EX)))
    return V0, E_g, stats


def direct(toy, G, ext):
    """Brute-force joint enumeration of a whole group, with beta kept as a free symbol b."""
    outcomes = [(t, toy.trajs[t]['prob'] * p, x) for t in range(len(toy.trajs)) for p, x in ext[t]]
    E_G, E_GG = VecPoly(), Poly()
    bmono = (('b', 1),)
    for combo in product(outcomes, repeat=G):
        p = fprod(c[1] for c in combo)
        Gb = g_group(toy, [c[0] for c in combo], G)
        Gb.add(vsum(c[2] for c in combo), F(-1, G), bmono)
        E_G.add_vp(Gb, p)
        E_GG = E_GG + p * Gb.dot(Gb)
    return E_G, E_GG - E_G.dot(E_G)


# ----------------------------------------------------------------------------- checks


def random_toy(rng, top_p=F(1)):
    def dist(n):
        w = [rng.randint(1, 9) for _ in range(n)]
        return [F(x, sum(w)) for x in w]
    pi = {s: dist(len(a)) for s, a in ACTIONS.items()}
    if top_p < 1:  # make sure nucleus truncation drops at least one action at s0
        pi['s0'] = [F(40, 100), F(30, 100), F(27, 100), F(3, 100)]
    phi = {s: F(rng.randint(1, 10), 10) for s in ACTIONS}
    return Toy(pi, phi, top_p)


def exact_eval(poly, values):
    return sum(c * fprod(F(values[v]) ** e for v, e in m) for m, c in poly.t.items())


def grpo_scale(k, G, eps=1e-6):
    """verl GRPO: (R - mean) / (torch.std + eps), torch.std is unbiased."""
    return 1.0 / (math.sqrt(k * (G - k) / (G * (G - 1))) + eps)


class Report:

    def __init__(self):
        self.rows, self.failed = [], 0

    def check(self, name, ok, detail=''):
        self.rows.append(dict(check=name, ok=bool(ok), detail=detail))
        self.failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def conditional_zero(toy, M):
    """E[Z | state s, true action a0] summed over a0 ~ q(.|s) must be 0 at every state."""
    for s, q in toy.q.items():
        tot = zeros()
        for a0, p0 in enumerate(q):
            for sibs in product(range(len(q)), repeat=M):
                p = p0 * fprod(q[b] for b in sibs)
                if p:
                    tot = vadd(tot, vscale(z_p3(toy, s, a0, sibs), p))
        if any(tot):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--G', type=int, default=5)
    ap.add_argument('--out', default='logs/r003_toy_exact.json')
    args = ap.parse_args()
    rep = Report()
    numeric = {}

    for seed in range(args.seeds):
        toy = random_toy(random.Random(seed))
        tag = f'seed{seed}'
        corr = corrections_for(toy)
        for M in (1, 2, 4):
            rep.check(f'{tag} P3-M{M}: E[Z | prefix] = 0 at every state', conditional_zero(toy, M))
        V0, E_g, stats = factorized(toy, args.G, corr)
        for n, st in stats.items():
            rep.check(f'{tag} {n}: E[mean_i X_i] = 0 (top_p=1, G={args.G})', not any(st['EX']))
        # biased control: dropping the sibling term must break E[Z]=0, otherwise the check is blind
        bad = []
        for t in toy.trajs:
            outs = []
            for kstar in range(1, K + 1):
                if kstar > len(t['steps']):
                    outs.append((F(1, K), zeros()))
                    continue
                s, a0 = t['steps'][kstar - 1]
                for b, pb in enumerate(toy.q[s]):
                    outs.append((F(1, K) * pb, vscale(toy.score(s, a0), toy.phi(s, a0) - toy.phi(s, b))))
            bad.append(outs)
        _, _, bst = factorized(toy, args.G, {'ctrl': bad})
        rep.check(f'{tag} control (no sibling score term) is detected as biased', any(bst['ctrl']['EX']))

        sval = {f's{k}': grpo_scale(k, args.G) for k in range(1, args.G)}
        num = {'V0': V0.eval(sval)}
        for n, st in stats.items():
            VX, CX = float(st['VX']), st['CX'].eval(sval)
            num[n] = dict(VX=VX, C=CX, beta_star=CX / VX if VX else 0.0,
                          rho2=CX * CX / (num['V0'] * VX) if VX else 0.0)
        numeric[tag] = num

    # direct joint enumeration: E[G_beta] = E[g] for all beta, Var = V0 + b^2 V_X - 2 b C_X exactly
    toy = random_toy(random.Random(0))
    corr = corrections_for(toy)
    for n, Gd in (('P2-all', 3), ('P2-sel', 3), ('P3-M1', 3), ('P3-M2', 2)):
        V0, E_g, stats = factorized(toy, Gd, {n: corr[n]})
        E_G, V = direct(toy, Gd, corr[n])
        st = stats[n]
        diff = E_G.add_vp(E_g, F(-1))
        rep.check(f'direct {n} G={Gd}: E[G_beta] = E[g_G] for every beta', diff.is_zero())
        bb = Poly.var('b')
        rep.check(f'direct {n} G={Gd}: Var(G_beta) = V0 + b^2 V_Z - 2 b C (symbolic in b, s_k)',
                  (V - (V0 + bb * bb * st['VX'] - 2 * bb * st['CX'])).is_zero())

    # V_beta-hat decomposition, beta*, break-even: exact at rational s_k = 1 (any fixed scale works)
    toy = random_toy(random.Random(3))
    V0, _, stats = factorized(toy, args.G, corrections_for(toy))
    one = {f's{k}': 1 for k in range(1, args.G)}
    v0 = exact_eval(V0, one)
    bb = Poly.var('b')
    for n, st in stats.items():
        VX, CX = st['VX'], exact_eval(st['CX'], one)
        bstar, rho2 = CX / VX, CX * CX / (v0 * VX)
        Vb = v0 + bb * bb * VX - 2 * bb * CX
        rep.check(f'{n}: V_b = V0(1-rho2) + V_Z(b-beta*)^2 (symbolic in b)',
                  (Vb - (v0 * (1 - rho2) + VX * (bb - bstar) * (bb - bstar))).is_zero())
        rep.check(f'{n}: beta* = C/V_Z minimizes V_b', exact_eval(Vb, {'b': bstar}) == min(
            exact_eval(Vb, {'b': bstar + d}) for d in (F(0), F(1, 100), F(-1, 100))))
        c0, c3 = F(1), F(7, 5)
        W0, W3 = c0 * v0, c3 * exact_eval(Vb, {'b': bstar})
        rep.check(f'{n}: W(P3)/W(P0) = (c3/c0)(1-rho2) at beta*', W3 / W0 == (c3 / c0) * (1 - rho2))

    # first round: using the other G-1 group members' first actions as siblings cancels exactly
    toy = random_toy(random.Random(1))
    q0 = toy.q['s0']
    ok = True
    for acts in product(range(len(q0)), repeat=args.G):
        tot = vsum(z_p3(toy, 's0', acts[i], [acts[j] for j in range(args.G) if j != i]) for i in range(args.G))
        ok &= not any(tot)
    rep.check(f'first round: sum_i Z_i == 0 when siblings are the other {args.G - 1} group members', ok)

    # top_p = 0.95: sampling q != pi. P2 loses its zero mean, P3 keeps it (exchangeable candidates).
    toy = random_toy(random.Random(0), top_p=F(95, 100))
    _, _, stats = factorized(toy, args.G, corrections_for(toy))
    rep.check('top_p=0.95: P2-all mean != 0', any(stats['P2-all']['EX']))
    rep.check('top_p=0.95: P2-sel mean != 0', any(stats['P2-sel']['EX']))
    for M in (1, 2, 4):
        rep.check(f'top_p=0.95: P3-M{M} mean = 0', not any(stats[f'P3-M{M}']['EX']))

    # EM-only, Bernoulli-logit p=1/2, phi(a)=a, M=1: Z is identically 0
    toy = Toy({'s0': [F(1, 4)] * 4, 'sa': [F(1, 2)] * 2, 'sb': [F(1, 2)] * 2}, {s: F(1, 2) for s in ACTIONS})
    rep.check('EM-only p=1/2, M=1: Z == 0 for every (a0, a1)',
              all(not any(z_p3(toy, s, a0, [a1])) for s in ('sa', 'sb') for a0 in range(2) for a1 in range(2)))

    print('\nnumeric (s_k = 1/(std_k + 1e-6), G=%d):' % args.G)
    for tag, num in numeric.items():
        print(f"  {tag}: V0={num['V0']:.4g}  " + '  '.join(
            f"{n}: rho2={v['rho2']:.3f} beta*={v['beta_star']:.3g}" for n, v in num.items() if n != 'V0'))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(dict(checks=rep.rows, failed=rep.failed, numeric=numeric, G=args.G, seeds=args.seeds), f, indent=1)
    print(f'\n{len(rep.rows) - rep.failed}/{len(rep.rows)} checks passed -> {args.out}')
    sys.exit(1 if rep.failed else 0)


if __name__ == '__main__':
    main()
