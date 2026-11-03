"""Numerical checks of the math behind history merging (kmaze.py).

1. Merge test: for two Dirichlet-categorical rows with counts n1, n2 and m outcomes,
       dF = ln p(one row) - ln p(two rows) ~ -(n1 + n2) JS + (m - 1)/2 ln(n1 n2 / (n1 + n2)) + c
   with JS the weighted Jensen-Shannon divergence in nats. Equal rows merge at a log rate;
   different rows split at a linear rate.
2. Coverage law: when "which state comes next" is evidence, a distinction passes back one
   edge at a time, so the episodes needed should follow N* = c / d_min, where d_min is the
   visits per episode of the rarest (predictive state, action) pair. The test is whether
   N* d_min stays constant across environments, while the depth of the first observable
   difference varies. c is measured, not derived.

Run from the repo root:  python src/merge_theory.py [--merge | --coverage]
"""
import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import kmaze as M  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / 'results' / 'merge_theory.json'
NS = [20, 30, 50, 80, 120, 200, 300, 500, 800, 1200, 2000, 3000]
SEEDS = 8
RECOVERED = 0.75                     # share of seeds that must be exact
ALPHA = 0.5


def log_dm(n, a):
    return gammaln(a * len(n)) - gammaln(a * len(n) + n.sum()) + (gammaln(a + n) - gammaln(a)).sum()


def dF(n1, n2, a=ALPHA):
    return log_dm(n1 + n2, a) - log_dm(n1, a) - log_dm(n2, a)


def js(p, q, w):
    m = w * p + (1 - w) * q
    kl = lambda x: (x[x > 0] * np.log(x[x > 0] / m[x > 0])).sum()
    return w * kl(p) + (1 - w) * kl(q)


def check_merge():
    print('merge test: exact dF against -n JS + (m-1)/2 ln(n1 n2/n), in nats')
    print('  case             n     exact   approx   exact-approx')
    p = np.array([0.5, 0.3, 0.2])
    for name, q in (('equal', p), ('different', np.array([0.3, 0.3, 0.4]))):
        for n in (100, 1000, 10000, 100000):
            n1 = n2 = n // 2
            c1, c2 = n1 * p, n2 * q
            e = dF(c1, c2)
            a = -n * js(p, q, 0.5) + (len(p) - 1) / 2 * np.log(n1 * n2 / n)
            print(f'  {name:12s} {n:7d} {e:9.2f} {a:9.2f} {e - a:9.2f}')


def key(env, loc, bits, seen):
    return loc, tuple(b if s else None for b, s in zip(bits, seen))


def observe(env, loc, seen):
    """Which bits the observation at loc shows."""
    if isinstance(env, M.Corridor):
        return (True,)
    k = env.k
    if loc == 0:
        return seen
    i = (loc - 1) % k
    return tuple(s or j == i for j, s in enumerate(seen))


def coverage(env, T):
    """Exact visits per episode of each (predictive state, action), random policy."""
    dist, d = {}, {}
    for bits in itertools.product((0, 1), repeat=env.k):
        s0 = observe(env, 0, (False,) * env.k)
        dist[(0, bits, s0)] = dist.get((0, bits, s0), 0) + 0.5 ** env.k
    for _ in range(T):
        nxt = {}
        for (loc, bits, seen), p in dist.items():
            for a in range(env.n_act):
                q = p / env.n_act
                e = (key(env, loc, bits, seen), a)
                d[e] = d.get(e, 0) + q
                l2 = env.move(loc, a)
                z = (l2, bits, observe(env, l2, seen))
                nxt[z] = nxt.get(z, 0) + q
        dist = nxt
    keys = {e[0] for e in d}
    return min(d.values()), len(keys), keys


def exact(env, trajs, S):
    """Each visited predictive state has one learned state and back, to 99%."""
    pairs = {}
    for (obs, acts, true), s in zip(trajs, S):
        seen = observe(env, 0, (False,) * env.k)
        for t in range(len(acts) + 1):
            loc, bits = true[t]
            seen = observe(env, loc, seen)
            if s[t] >= 0:
                pairs.setdefault(key(env, loc, bits, seen), {}).setdefault(int(s[t]), 0)
                pairs[key(env, loc, bits, seen)][int(s[t])] += 1
    owner, good, total = {}, 0, 0
    for k, v in pairs.items():
        best = max(v, key=v.get)
        if best in owner:
            return False
        owner[best] = k
        good += v[best]
        total += sum(v.values())
    return good >= 0.99 * total


def check_coverage():
    envs = [('hub k=1', M.KMaze(1), 12), ('star k=1', M.KMaze(1, layout='star'), 12)]
    envs += [(f'corridor D={D}', M.Corridor(D), 10 + 3 * D) for D in (1, 2, 3, 4)]
    out = []
    for name, env, T in envs:
        d_min, truth, _ = coverage(env, T)
        rate = []
        t0 = time.time()
        for n in NS:
            hits = 0
            for s in range(SEEDS):
                trajs = env.sample(n, T, 1000 * s + n)
                S, _ = M.learn_states(trajs, env.n_obs)
                hits += M.n_states(S) == truth and exact(env, trajs, S)
            rate.append(hits / SEEDS)
            print(f'{name} N={n} exact {hits}/{SEEDS} ({time.time() - t0:.0f}s)', flush=True)
        ok = [n for n, r in zip(NS, rate) if r >= RECOVERED]
        first = ok[0] if ok else None
        out.append(dict(name=name, T=T, truth=truth, d_min=d_min, rate=rate, N_star=first,
                        product=first * d_min if first else None))
        OUT.write_text(json.dumps(dict(ns=NS, seeds=SEEDS, runs=out), indent=1))
    print('\n  env              states   d_min     N*    N* d_min')
    for r in out:
        print(f"  {r['name']:16s} {r['truth']:4d} {r['d_min']:8.4f} {r['N_star']} {r['product']}")
    for name, env, T in (('hub k=2', M.KMaze(2), 20), ('hub k=3', M.KMaze(3), 30)):
        d_min, truth, _ = coverage(env, T)
        print(f'  {name}: {truth} predictive states, d_min {d_min:.2e}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--coverage', action='store_true')
    a = ap.parse_args()
    if a.merge or not a.coverage:
        check_merge()
    if a.coverage or not a.merge:
        check_coverage()
