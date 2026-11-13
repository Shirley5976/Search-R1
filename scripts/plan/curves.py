"""R060: preregistered endpoints, verdicts and the interpretation table (FINAL_PROPOSAL "Claim-Driven Validation").

Input: <ckpt>/<run>/eval/eval_<tag>_step<N>.jsonl ({qid, score} per question), run names
<RUN_ID>-<method>-<target>-s<seed> from scripts/plan/runs/train_run.sh. Curve points are the tags
gpuh<t> / final_gpuh<t> (curve subsample); final_full (7 full test sets) is reported separately.

- em7(t): unweighted mean over the 7 data sources of the per-source mean EM.
- Main endpoint (H-pred): AUC of the seed-mean curve over training GPU hours t in [0, T] (trapezoid;
  a curve that stops early is held flat to T).
- Net endpoint: target = mean over P0 seeds of the final em7; tau = first crossing (linear between
  observed checkpoints) of the seed-mean curve, P0 within [0, T], method m within t <= T - H_m;
  delta_tau = tau_P0 - (H_m + tau_m). Not reached is kept: m not reached -> -inf, only P0 not
  reached -> +inf, neither -> nan (counts as not > 0). Savings are claimed only if the 2.5%
  quantile (nan as -inf) is > 0.
- Bootstrap (seed 0): each draw resamples seeds per method, then one stratified question draw
  (within data source) shared by all methods. 1-seed methods: questions only, flagged.
- Pair verdicts: supports / contradicts / inconclusive from the 95% interval of
  AUC(better-predicted) - AUC(worse-predicted).
"""

import argparse
import glob
import json
import os
import re

import numpy as np

SOURCES = ('nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle')
T_MAX = 160.0
H_DEFAULT = 45.0  # preregistered estimate; the r011_costs.py ledger replaces it when present
NAME_RE = re.compile(r'(R\d{3})-(p0|p2_all|p3|p1_[0-9.]+)-(main|negctrl)-s(\d+)$')
TAG_RE = re.compile(r'(?:final_)?gpuh(\d+(?:\.\d+)?)')


# ------------------------------------------------------------------ loading
def load_run(run_dir):
    """{'t': (n_t,), 'qids': [...], 'S': (n_t, n_q) scores, 'full': {qid: score} or None}."""
    tags, step_of = {}, {}
    for f in glob.glob(os.path.join(run_dir, 'eval', 'eval_*_step*.jsonl')):
        m = re.match(r'eval_(.+)_step(\d+)\.jsonl$', os.path.basename(f))
        tag, step = m.group(1), int(m.group(2))
        if step >= step_of.get(tag, -1):
            step_of[tag] = step
            tags[tag] = {r['qid']: r['score'] for r in map(json.loads, open(f))}
    pts = {}
    for tag, sc in tags.items():
        m = TAG_RE.fullmatch(tag)
        if m:
            pts[float(m.group(1))] = sc
    if not pts:
        return None
    ts = sorted(pts)
    qids = sorted(set.intersection(*[set(pts[t]) for t in ts]))
    S = np.array([[pts[t][q] for q in qids] for t in ts], dtype=np.float64)
    return {'t': np.array(ts), 'qids': qids, 'S': S, 'full': tags.get('final_full')}


def align(runs):
    """Restrict every run to the common question set, same column order; returns qids, src index."""
    common = sorted(set.intersection(*[set(r['qids']) for r in runs.values()]))
    for r in runs.values():
        pos = {q: i for i, q in enumerate(r['qids'])}
        r['S'] = r['S'][:, [pos[q] for q in common]]
        r['qids'] = common
    src = np.array([SOURCES.index(q.split('/')[0]) for q in common])
    return common, src


# ------------------------------------------------------------------ curves and endpoints
def em7(S, w, src):
    """S: (n_t, n_q); w: question multiplicities. Unweighted mean over sources present."""
    out = []
    for s in range(len(SOURCES)):
        ws = w * (src == s)
        if ws.sum() > 0:
            out.append(S @ ws / ws.sum())
    return np.mean(out, axis=0)


def mean_curve(curves):
    """curves: list of (t, y). Seed-mean on the union grid (each seed linearly interpolated)."""
    grid = np.unique(np.concatenate([t for t, _ in curves]))
    return grid, np.mean([np.interp(grid, t, y) for t, y in curves], axis=0)


def auc(t, y, t_max=T_MAX):
    keep = t <= t_max
    t, y = t[keep], y[keep]
    if t[-1] < t_max:
        t, y = np.append(t, t_max), np.append(y, y[-1])
    return float(np.sum((t[1:] - t[:-1]) * (y[1:] + y[:-1]) / 2))


def crossing(t, y, target, t_limit):
    """First t <= t_limit where the piecewise-linear curve reaches target, else None."""
    if y[0] >= target:
        return float(t[0])
    for i in range(1, len(t)):
        if t[i] > t_limit + 1e-9:
            if y[i] >= target and y[i] != y[i - 1]:
                tc = t[i - 1] + (target - y[i - 1]) / (y[i] - y[i - 1]) * (t[i] - t[i - 1])
                return float(tc) if tc <= t_limit else None
            return None
        if y[i] >= target:
            return float(t[i - 1] + (target - y[i - 1]) / (y[i] - y[i - 1]) * (t[i] - t[i - 1]))
    return None


def delta_tau(tau_p0, tau_m, H):
    if tau_m is None:
        return -np.inf if tau_p0 is not None else np.nan
    if tau_p0 is None:
        return np.inf
    return tau_p0 - (H + tau_m)


def verdict(lo, hi):
    """Interval of (predicted better - predicted worse)."""
    return 'supports' if lo > 0 else ('contradicts' if hi < 0 else 'inconclusive')


# ------------------------------------------------------------------ bootstrap
class Analysis:

    def __init__(self, groups, src, H, n_boot=2000, seed=0, t_max=T_MAX):
        """groups: {method: [run dict]} (aligned); H: {method: H_m}."""
        self.g, self.src, self.H, self.t_max = groups, src, H, t_max
        self.n_boot, self.seed = n_boot, seed
        self.n_q = len(src)

    def curves(self, m, w, seeds):
        return [(r['t'], em7(r['S'], w, self.src)) for r in (self.g[m][i] for i in seeds)]

    def endpoints(self, w, pick):
        """AUC per method, P0 target, tau per method, on one (re)sample."""
        mc = {m: mean_curve(self.curves(m, w, pick[m])) for m in self.g}
        out = {'auc': {m: auc(t, y, self.t_max) for m, (t, y) in mc.items()}}
        if 'p0' in self.g:
            target = float(np.mean([y[-1] for _, y in self.curves('p0', w, pick['p0'])]))
            tau = {m: crossing(t, y, target, self.t_max - self.H.get(m, 0.0)) for m, (t, y) in mc.items()}
            out['target'] = target
            out['dtau'] = {m: delta_tau(tau['p0'], tau[m], self.H.get(m, 0.0)) for m in self.g if m != 'p0'}
            out['tau'] = tau
        return out

    def run(self):
        ones = np.ones(self.n_q)
        point = self.endpoints(ones, {m: list(range(len(r))) for m, r in self.g.items()})
        rng = np.random.RandomState(self.seed)
        strata = [np.nonzero(self.src == s)[0] for s in range(len(SOURCES)) if (self.src == s).any()]
        draws = []
        for _ in range(self.n_boot):
            pick = {m: list(rng.choice(len(r), len(r), replace=True)) for m, r in self.g.items()}
            w = np.zeros(self.n_q)
            for g in strata:
                np.add.at(w, rng.choice(g, len(g), replace=True), 1)
            draws.append(self.endpoints(w, pick))
        return point, draws


def pair(point, draws, better, worse, n_seeds):
    d = np.array([x['auc'][better] - x['auc'][worse] for x in draws])
    lo, hi = np.quantile(d, [0.025, 0.975])
    out = {'better': better, 'worse': worse, 'auc_diff': point['auc'][better] - point['auc'][worse],
           'ci95': [float(lo), float(hi)], 'verdict': verdict(lo, hi),
           'seeds': {better: n_seeds[better], worse: n_seeds[worse]}}
    if min(out['seeds'].values()) < 2:
        out['note'] = 'involves a 1-seed method: observed order only, cross-seed stability not established'
    return out


def dtau_summary(point, draws, m):
    d = np.array([x['dtau'][m] for x in draws], dtype=np.float64)
    d_ = np.where(np.isnan(d), -np.inf, d)
    lo = float(np.quantile(d_, 0.025))
    return {'point': point['dtau'][m], 'tau': {k: point['tau'][k] for k in ('p0', m)},
            'ci95_lo': lo, 'ci95_hi': float(np.quantile(d_, 0.975)),
            'frac_method_not_reached': float(np.mean(np.isneginf(d))),
            'frac_p0_not_reached': float(np.mean(np.isposinf(d))), 'frac_neither': float(np.mean(np.isnan(d))),
            'saving_established': bool(lo > 0)}


# ------------------------------------------------------------------ interpretation table
def interpret(res, pilot):
    winner = pilot['winner']
    T = {}
    T['A'] = ('confirmed: the local estimation-efficiency gain of %s is confirmed on the evaluation split' % winner
              if winner != 'p0' else
              'not confirmed: neither candidate gain was confirmed (do not write "worse than P0" or "anchor unreachable")')
    if winner != 'p0' and f'{winner}>p0' in res['pairs']:
        v = res['pairs'][f'{winner}>p0']['verdict']
        T['B1'] = {'supports': "supports: the predicted winner's advantage over the baseline is supported",
                   'contradicts': 'contradicts: report as is; diagnose with later checkpoints',
                   'inconclusive': 'inconclusive (not equivalence)'}[v]
    else:
        T['B1'] = 'not applicable (pilot winner is P0)' if winner == 'p0' else 'missing runs'
    need = res['ranking_pairs']
    vs = [res['pairs'][k]['verdict'] for k in need if k in res['pairs']]
    if len(vs) < len(need):
        T['B2'] = 'inconclusive (missing runs, e.g. R026 cut)'
    elif 'contradicts' in vs:
        T['B2'] = 'contradicts: the ranking part of H-pred is refuted'
    elif all(v == 'supports' for v in vs):
        T['B2'] = 'supports: the full predicted ranking is supported'
    else:
        T['B2'] = 'inconclusive'
    if any('note' in res['pairs'][k] for k in need if k in res['pairs']):
        T['B2'] += ' (1-seed comparison: cross-seed stability not established)'
    c = res['delta_tau'].get(winner)
    T['C'] = ('holds: GPU-hour saving can be claimed' if c and c['saving_established'] else
              'does not hold (includes "not reached within budget"); this does not refute H-pred')
    d = pilot['diagnostics']['reader_vs_off']['ci95']
    T['mechanism'] = {'p3': 'P3 wins: the paper centres on the sibling estimator',
                      'p2_all': 'P2-all wins: retitle as a prospective injection-location study; report the P3 UB separately',
                      'p0': 'no reader injection improvement confirmed in this setting'}[winner]
    if d[0] <= 1 <= d[1]:
        T['mechanism'] += '; reader-off vs P3 W interval contains 1: no incremental local gain from the reader score shown'
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='verl_checkpoints')
    ap.add_argument('--pilot_result', default='prereg/pilot_result.json')
    ap.add_argument('--ledger', default='prereg/stage05_costs.json')
    ap.add_argument('--p1_alpha', default='prereg/p1_alpha.json')
    ap.add_argument('--out_prefix', default='prereg/results')
    ap.add_argument('--n_boot', type=int, default=2000)
    a = ap.parse_args()
    pilot = json.load(open(a.pilot_result))
    H = H_DEFAULT
    if os.path.exists(a.ledger):
        H = json.load(open(a.ledger)).get('_ledger', {}).get('H_unbiased_gpu_h', H)
    alpha = json.load(open(a.p1_alpha))['alpha'] if os.path.exists(a.p1_alpha) else None
    groups = {}
    for d in sorted(glob.glob(os.path.join(a.ckpt, 'R0*'))):
        m = NAME_RE.match(os.path.basename(d))
        if not m:
            continue
        meth, target = m.group(2), m.group(3)
        if meth.startswith('p1_'):
            if alpha is None or float(meth[3:]) != float(alpha):
                continue  # screening runs of the other alphas
            meth = 'p1'
        r = load_run(d)
        if r is not None:
            groups.setdefault(meth if target == 'main' else meth + '@neg', []).append(r)
    assert 'p0' in groups, 'no P0 runs found'
    qids, src = align({f'{m}{i}': r for m, rs in groups.items() for i, r in enumerate(rs)})
    Hm = {m: (H if m.split('@')[0] in ('p2_all', 'p3') else 0.0) for m in groups}
    main_groups = {m: v for m, v in groups.items() if '@' not in m}
    point, draws = Analysis(main_groups, src, Hm, a.n_boot).run()
    n_seeds = {m: len(v) for m, v in groups.items()}
    ranking = pilot['predicted_training_ranking']
    res = {'H_gpu_h': H, 'n_questions': len(qids), 'n_boot': a.n_boot, 'n_seeds': n_seeds, 'ranking': ranking,
           'winner': pilot['winner'], 'p1_alpha': alpha, 'auc': point['auc'], 'p0_target_em7': point.get('target'),
           'pairs': {}, 'delta_tau': {}}
    ranking_pairs = []
    for i in range(len(ranking) - 1):
        b, w = ranking[i], ranking[i + 1]
        ranking_pairs.append(f'{b}>{w}')
    res['ranking_pairs'] = ranking_pairs
    for k in set(ranking_pairs) | {f'{m}>p0' for m in main_groups if m != 'p0'}:
        b, w = k.split('>')
        if b in main_groups and w in main_groups:
            res['pairs'][k] = pair(point, draws, b, w, n_seeds)
    for m in main_groups:
        if m != 'p0':
            res['delta_tau'][m] = dtau_summary(point, draws, m)
    # negative control (Claim 2): final em7 change, question bootstrap, 1 seed each
    res['negctrl'] = {}
    for m in groups:
        if m.endswith('@neg') and m[:-4] in groups:
            base = m[:-4]
            sub = {base: groups[base][:1], m: groups[m][:1]}
            pt, dr = Analysis(sub, src, {}, a.n_boot).run()
            fin = lambda x, k: x['auc'][k]
            d = np.array([fin(x, m) - fin(x, base) for x in dr]) / T_MAX
            res['negctrl'][m] = {'auc_mean_em_change': (fin(pt, m) - fin(pt, base)) / T_MAX,
                                 'ci95': [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                                 'final_em7': {k: float(em7(sub[k][0]['S'], np.ones(len(src)), src)[-1]) for k in sub}}
    res['interpretation'] = interpret(res, pilot)
    with open(a.out_prefix + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    L = ['# Results (R060)', '', f"- H = {H:.1f} GPU h, {len(qids)} curve questions, {a.n_boot} bootstrap draws",
         f"- seeds: {n_seeds}", f"- predicted ranking: {ranking}; pilot winner: {pilot['winner']}", '',
         '| pair | AUC diff | 95% CI | verdict |', '|---|---|---|---|']
    for k, p in res['pairs'].items():
        L.append(f"| {k} | {p['auc_diff']:.3f} | [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] | {p['verdict']} |")
    L += ['', '| method | delta_tau | 2.5% | method not reached | P0 not reached | saving |', '|---|---|---|---|---|---|']
    for m, d in res['delta_tau'].items():
        L.append(f"| {m} | {d['point']} | {d['ci95_lo']:.2f} | {d['frac_method_not_reached']:.2f} | "
                 f"{d['frac_p0_not_reached']:.2f} | {d['saving_established']} |")
    L += ['', '## Negative control', ''] + [f"- {m}: mean-EM change {v['auc_mean_em_change']:+.4f} "
                                            f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for m, v in res['negctrl'].items()]
    L += ['', '## Interpretation (preregistered table)', ''] + [f'- **{k}**: {v}' for k, v in res['interpretation'].items()]
    with open(a.out_prefix + '.md', 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
