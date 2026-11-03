"""Test of rdp_language.adact_language, and the variants behind ENV_CHECK.md. No truth reaches a
learner: n_layered, belief_logloss and mu_0 are computed from the environment, for scoring only.
FAIL lines are real failures, not skipped. Exit code 1 on any FAIL in sections 1 and 2.

Sections
  1  internals and processes with a known answer
  2  the six environments at n = 2000, five seeds: held-out log-loss through Machine, states, time
  3  exact X_{3,1}-distinguishability under the uniform policy: which minimal states the metric
     can separate at all, and from which n
  4  states against n, one seed, to tell too little data from a metric that cannot see
  5  rdp_envs vs the ICLR text: T-maze(c) as drawn in Figures 4 and 6, Cookie without replacement
  6  Figure 2: the Example 1 T-maze, K = 100

Run:  python src/rdp_language_test.py [sections, e.g. 1 2]
"""
import sys
import time
from fractions import Fraction as F
from itertools import combinations

import numpy as np

from rdp_baselines import adact_h, symbols
from rdp_envs import ENVS, PAPER_H, Cookie, Corridor, TabEnv, TMaze
from rdp_language import (adact_language, language_metric, language_table, n_languages, patterns,
                          threshold)

TABLE1 = {'corridor': 11, 'tmaze': 18, 'cookie': 91, 'cheese': 1178, 'minihall': 6098}  # language U
fails = []


def check(name, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {name}  {detail}', flush=True)
    if not ok:
        fails.append(name)


def fit(env, trajs, key='ao', **kw):
    return adact_language(trajs, env.n_act, env.n_obs, len(env.rewards), key=key, **kw)


class Iid(TabEnv):
    """A fair coin, whatever the action. One state per layer: H + 2."""
    rewards = (0,)

    def __init__(self):
        self.n_act, self.n_codes, self.n_obs = 2, 2, 2
        self._tab, self._n_pred = {}, None

    def start(self):
        return [(F(1, 2), 0, c, 0) for c in (0, 1)]

    def step(self, s, a):
        return self.start()


class Sticky(Iid):
    """A shown bit kept w.p. 0.9: 2H + 2."""

    def start(self):
        return [(F(1, 2), c, c, 0) for c in (0, 1)]

    def step(self, s, a):
        return [(F(9, 10), s, s, 0), (F(1, 10), 1 - s, 1 - s, 0)]


class TMazeFig(TMaze):
    """T-maze(c) as Figures 4 and 6 and Example 1 give it: 011 means goal north, 101 all along the
    corridor including the step that arrives at the junction, 010 on the step taken at the
    junction ('at the T-junction and beyond'). No fifth code."""
    CUES = (0, 1)

    def code(self, x, g):
        return self.CORR

    def step(self, s, a):
        if s == 'end':
            return [(1, 'end', self.JUNC, 0)]
        if s[0] == self.N:
            return [(1, 'end', self.JUNC, 4 if a == s[1] else -1)]
        return super().step(s, a)


class TMazeFigNoW(TMazeFig):
    """Same, W not available at S, as q1 and q12 of Figure 6 show."""

    def legal(self, s):
        return (2,) if s != 'end' and s[0] == 0 else super().legal(s)


class CookieKeep(Cookie):
    """Pressing with a cookie out does nothing (rdp_envs follows Icarte 2019: it replaces it)."""

    def step(self, s, a):
        return [(1, s, 1, 0)] if a == 4 and s[0] == 1 and s[1] else super().step(s, a)


class TMazeEx1(TMaze):
    """Example 1 and 3: corridor obs uniform on 101, 111 (the arrival at the junction included,
    tau reads theta_o(u_i) of the cell left), 010 at the junction and beyond, 1 for the correct
    N or S and 0 otherwise, H = N + 2. Behaviour of Example 4: East, then N or S."""
    rewards = (0, 1)
    CUES, NOISE = (0, 1), 4

    def __init__(self, N):
        super().__init__(N=N, actions='east')
        self.n_obs = self.n_codes * len(self.rewards)

    def step(self, s, a):
        if s == 'end':
            return [(1, 'end', self.JUNC, 0)]
        if s[0] == self.N:
            return [(1, 'end', self.JUNC, 1 if a == s[1] else 0)]
        return [(F(1, 2), (s[0] + 1, s[1]), c, 0) for c in (self.CORR, self.NOISE)]


def layered_classes(env, T):
    """As TabEnv.n_layered, kept per k so the classes can be read. Returns edges, reach, per_k."""
    edges, reach, first = env.belief_graph(depth=T)
    per_k = [{i: 0 for i in range(len(edges))}]
    for k in range(1, T + 1):
        ids, cur = {}, {}
        for i in range(len(edges)):
            if first[i] <= T - k:
                v = tuple(sorted((a, o, p, per_k[-1][j]) for (a, o), (p, j) in edges[i].items()))
                cur[i] = ids.setdefault(v, len(ids))
        per_k.append(cur)
    return edges, reach, per_k


def distinguishability(env, T, delta=0.05):
    """Exact, uniform policy, X_{3,1}: for each pair of minimal states of a layer, L = max |p(X) -
    p'(X)|. Returns (mu_0, blind pairs (L = 0), n below which the worst seeing pair can't pass the
    test on expected counts: max 2 log(2|X| / delta) / (L^2 min occupancy)), layer of that pair."""
    R, Oc = len(env.rewards), env.n_obs // len(env.rewards)
    n_x, A = n_languages(env.n_act, Oc, R), env.n_act
    edges, reach, per_k = layered_classes(env, T)
    memo = {}

    def miss(i, k):
        """P(no symbol of X in the next k steps | belief i), one entry per X."""
        if k == 0 or not edges[i]:
            return np.ones(n_x)
        if (i, k) not in memo:
            tot = np.zeros(n_x)
            for (a, o), (p, j) in edges[i].items():
                keep = np.ones(n_x)
                keep[patterns(a, o // R, o % R, A, Oc, R)] = 0
                tot += float(p) / A * keep * miss(j, k - 1)
            memo[i, k] = tot
        return memo[i, k]

    occ, mu, blind, need, where = {0: 1.0}, np.inf, 0, 0.0, None
    for t in range(T + 1):
        nxt = {}                                          # belief occupancy after o_t
        for i, v in occ.items():
            for (a, _), (p, j) in edges[i].items():
                nxt[j] = nxt.get(j, 0.0) + v * float(p) / (1 if a is None else A)
        occ = nxt
        cls = {}
        for i in reach[t]:
            c = cls.setdefault(per_k[T - t][i], [i, 0.0])
            c[1] += occ.get(i, 0.0)
        for (i1, d1), (i2, d2) in combinations(cls.values(), 2):
            L = np.abs(miss(i1, T - t) - miss(i2, T - t)).max()
            if L < 1e-12:
                blind += 1
                continue
            mu = min(mu, L)
            n = 2 * np.log(2 * n_x / delta) / (L ** 2 * min(d1, d2))
            if n > need:
                need, where = n, (t, round(L, 4), round(min(d1, d2), 5))
    return mu, blind, need, where


def section1():
    print('1. internals and known answers')
    check('|X_{3,1}| formula vs summing the groups', n_languages(4, 5, 3) ==
          4 + 5 + 3 + 20 + 12 + 15 + 60)
    check('threshold', abs(threshold(100, 119) - np.sqrt(2 * np.log(2 * 119 / .05) / 100)) < 1e-12)
    ids = patterns(*(np.array(v) for v in (2, 3, 1)), 4, 5, 3)
    check('a symbol is in exactly 7 distinct patterns, all in range',
          len(set(ids.tolist())) == 7 and ids.max() < n_languages(4, 5, 3))
    # by hand, A = 2, O = 2, R = 1: packed a * 2 + o. All with a = 0 vs all with a = 1
    check('L_X by hand: a pattern in every trace of one side only',
          language_metric([(0, 1)], [(2, 3)], 2, 2, 1) == 1.0)
    check('L_X by hand: same symbols, other order, is not seen',
          language_metric([(0, 3)], [(3, 0)], 2, 2, 1) == 0.0)
    z1, z2 = [(0, 0), (1, 1)], [(0, 0), (0, 0)]
    check('L_X by hand: half the traces', language_metric(z1, z2, 2, 2, 1) == 0.5)
    env = TMaze()
    tr = env.sample(120, 5, seed=1)
    Es = symbols(tr, 4, env.n_obs)
    last, rng, bad = language_table(Es, 4, env.n_obs, 3), np.random.default_rng(0), 0
    for t in (0, 2, 4):
        for _ in range(4):
            i = rng.permutation(120)
            a, b = i[:30], i[30:90]
            fast = np.abs((last[a] > t).mean(0) - (last[b] > t).mean(0)).max()
            bad += abs(fast - language_metric([tuple(x) for x in Es[a, t + 1:].tolist()],
                                              [tuple(x) for x in Es[b, t + 1:].tolist()], 4, env.n_obs, 3)) > 1e-12
    check('fast L_X agrees with the definition, 12 random pairs', bad == 0, f'{bad} wrong')
    for name, e, H, n, want in (('iid coin', Iid(), 6, 20000, 8), ('sticky bit', Sticky(), 6, 40000, 14),
                                ('Corridor', Corridor(), 5, 5000, 11),
                                ('T-maze east', TMaze(actions='east'), 5, 5000, 12)):
        m = fit(e, e.sample(n, H, seed=2))
        check(name, m.n_states == want, f'n={n} states={m.n_states} want={want} layers={m.info["layers"]}')
    check('too little data merges all', fit(Sticky(), Sticky().sample(10, 6, seed=2)).n_states == 8)


def section2():
    print('2. six environments, n = 2000, held-out 2000, five seeds. ex = log-loss - Bayes floor')
    for k, E in ENVS.items():
        env, T = E(), PAPER_H[k]
        truth, rows = env.n_layered(T), {}
        for seed in range(5):
            train, test = env.sample(2000, T, seed=100 + seed), env.sample(2000, T, seed=200 + seed)
            floor = env.belief_logloss(test)
            for name, f in (('language ao', lambda: fit(env, train)),
                            ('language aor', lambda: fit(env, train, 'aor')),
                            ('L-inf adact_h', lambda: adact_h(train, env.n_act, env.n_obs))):
                t0 = time.time()
                m = f()
                dt = time.time() - t0
                ll, cov = m.logloss(test)
                rows.setdefault(name, []).append((m.n_states, ll, ll - floor, cov, dt))
        print(f'  {k} H={T} A={env.n_act} O={env.n_obs // len(env.rewards)} R={len(env.rewards)} '
              f'|X|={n_languages(env.n_act, env.n_obs // len(env.rewards), len(env.rewards))} '
              f'minimal layered={truth} Table 1={TABLE1.get(k, "-")}')
        for name, r in rows.items():
            r = np.array(r)
            print(f'    {name:14s} states {r[:, 0].mean():7.1f} [{r[:, 0].min():.0f}, {r[:, 0].max():.0f}]  '
                  f'log-loss {r[:, 1].mean():.4f} ex {r[:, 2].mean():.4f} +- {r[:, 2].std():.4f}  '
                  f'routed {r[:, 3].mean():.3f}  {r[:, 4].mean():.3f}s')
        s = np.array(rows['language ao'])[:, 0]
        check(f'{k}: language finds the minimal layered size at n=2000', (s == truth).all(),
              f'states {s.astype(int).tolist()} vs {truth}')
        if k in TABLE1:
            check(f'{k}: language size equals Table 1', (s == TABLE1[k]).any(),
                  f'{s.astype(int).tolist()} vs {TABLE1[k]} (their n is unknown)')


def section3():
    print('3. exact X_{3,1}-distinguishability, uniform policy (n needed: see distinguishability)')
    for k, E in ENVS.items():
        env, T = E(), PAPER_H[k]
        t0 = time.time()
        mu, blind, need, where = distinguishability(env, T)
        print(f'  {k}: mu_0={mu:.4g} blind pairs={blind} n needed={need:.3g} '
              f'at (t, L, occupancy)={where}  {time.time() - t0:.1f}s', flush=True)


def section4():
    print('4. states and excess log-loss against n, one seed')
    for k, E in ENVS.items():
        env, T = E(), PAPER_H[k]
        test = env.sample(2000, T, seed=299)
        floor, out = env.belief_logloss(test), []
        for n in (300, 2000, 10000, 30000, 100000, 300000):
            train = env.sample(n, T, seed=199)
            t0 = time.time()
            m = fit(env, train)
            out.append(f'{n}: {m.n_states} ({m.logloss(test)[0] - floor:.4f}, {time.time() - t0:.1f}s)')
        print(f'  {k} minimal {env.n_layered(T)} | ' + ' | '.join(out), flush=True)


def section5():
    print('5. variants of the ICLR text, exact minimal layered sizes, H = 5 (Cookie H = 9)')
    for name, e in (('rdp_envs full', TMaze()), ('Fig obs, full', TMazeFig()),
                    ('rdp_envs ew', TMaze(actions='ew')), ('Fig obs, ew', TMazeFig(actions='ew')),
                    ('Fig obs, ew, no W at S', TMazeFigNoW(actions='ew')),
                    ('Fig obs, full, no W at S', TMazeFigNoW()),
                    ('rdp_envs east', TMaze(actions='east')), ('Fig obs, east', TMazeFig(actions='east'))):
        print(f'  T-maze {name}: {e.n_layered(5)}')
    print(f'  Cookie rdp_envs: {Cookie().n_layered(9)}, without replacement: {CookieKeep().n_layered(9)}')
    print('  same learner on rdp_envs T-maze vs the Figure version, seeds 0..4')
    for n in (2000, 30000, 300000):
        for name, E in (('rdp_envs', TMaze), ('Figure', TMazeFig)):
            env, s = E(), []
            for seed in range(5):
                train, test = env.sample(n, 5, seed=100 + seed), env.sample(2000, 5, seed=200 + seed)
                m = fit(env, train)
                s.append((m.n_states, m.logloss(test)[0] - env.belief_logloss(test)))
            s = np.array(s)
            print(f'    n={n} {name}: states {s[:, 0].astype(int).tolist()} ex {s[:, 1].mean():.4f}', flush=True)


def section6():
    print('6. Figure 2: Example 1 T-maze, East-only behaviour. 5 seeds, mean states')
    for N in (2, 6, 10, 14, 18):
        env, H = TMazeEx1(N), N + 2
        truth = env.n_layered(H)
        for K in (100, 300, 1000, 3000):
            lang, linf, ok = [], [], 0
            for seed in range(5):
                tr = env.sample(K, H, seed=seed)
                lang.append(fit(env, tr).n_states)
                ok += lang[-1] == truth
                linf.append(adact_h(tr, env.n_act, env.n_obs).n_states)
            print(f'  N={N} H={H} K={K}: minimal {truth} language {np.mean(lang):.1f} '
                  f'(exact {ok}/5) L-inf {np.mean(linf):.1f}', flush=True)


if __name__ == '__main__':
    sel = sys.argv[1:] or list('123456')
    for s in sel:
        globals()[f'section{s}']()
    print('\nFAILS:', fails if fails else 'none')
    sys.exit(1 if fails else 0)
