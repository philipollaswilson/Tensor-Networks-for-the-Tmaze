"""Smoke test of rdp_envs and rdp_baselines: small data, no tuning, no full comparison.
Prints PASS / FAIL per check, exit code 1 on any FAIL.
Run:  python src/rdp_smoke_test.py
"""
import sys
import time
from fractions import Fraction as F

import numpy as np

from rdp_baselines import adact_h, flexfringe, linf_prefix, threshold, symbols, Prefixes
from rdp_envs import ENVS, PAPER_H, Corridor, TabEnv, TMaze

fails = []


def check(name, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {name}  {detail}')
    if not ok:
        fails.append(name)


class Iid(TabEnv):
    """A fair coin, whatever the action. Known answer: one state per layer, H + 2."""
    rewards = (0,)

    def __init__(self):
        self.n_act, self.n_codes, self.n_obs = 2, 2, 2
        self._tab, self._n_pred = {}, None

    def start(self):
        return [(F(1, 2), 0, c, 0) for c in (0, 1)]

    def step(self, s, a):
        return self.start()


class Sticky(Iid):
    """A shown bit kept w.p. 0.9. By hand: q_0, two states in layers 1 .. H, one final: 2H + 2."""

    def start(self):
        return [(F(1, 2), c, c, 0) for c in (0, 1)]

    def step(self, s, a):
        return [(F(9, 10), s, s, 0), (F(1, 10), 1 - s, 1 - s, 0)]


print('1. environments: sampling, format, truth')
for k, E in ENVS.items():
    env, T = E(), PAPER_H[k]
    tr = env.sample(50, T, seed=0)
    ok = all(len(o) == T + 1 and len(a) == T and len(s) == T + 1 for o, a, s in tr)
    ok &= all(0 <= x < env.n_obs for o, _, _ in tr for x in o)
    ok &= all(0 <= x < env.n_act for _, a, _ in tr for x in a)
    same = str(tr[:3]) == str(E().sample(50, T, seed=0)[:3])
    r = float(np.mean([sum(map(env.reward, o)) for o, _, _ in tr]))
    check(k, ok and same, f'A={env.n_act} O={env.n_obs} H={T} n_pred={env.n_pred} '
          f'n_layered(H)={env.n_layered(T)} mean return={r:.2f} floor={env.belief_logloss(tr):.3f}')
check('cookie: eating with no cookie pays nothing', all(r[3] == 0 for s in ((0, 0), (2, 3), (2, 0))
                                                        for r in ENVS['cookie']().step(s, 5)))
check('tiger reports unbounded', ENVS['tiger']().n_pred is None, ENVS['tiger']().pred_note[:40] + '...')
check('truth vs paper: Corridor layered 11 (Table 1, Fig 2)', Corridor().n_layered(5) == 11)
check('truth vs Ronca: Corridor cyclic 2m', [Corridor(m=m).n_pred for m in (3, 6, 9)] == [6, 12, 18])
check('truth vs paper: T-maze east layered 12 (Fig 3)', TMaze(actions='east').n_layered(5) == 12)
ew = TMaze(actions='ew').n_layered(5)
check('truth vs paper: T-maze ew layered 19 (Fig 5)', ew == 19, f'mine {ew}')

print('2. AdaCT-H internals')
check('threshold formula', abs(threshold(2, 5, 100, 4, 3, 5, .1)
                               - np.sqrt(2 * np.log(8 * 60 ** 3 / .1) / 100)) < 1e-12)
z1, z2 = [(1, 2), (1, 3), (2, 2), (2, 2)], [(1, 2), (1, 2), (1, 2), (3, 3)]
check('L-inf prefix, by hand', abs(linf_prefix(z1, z2) - 0.5) < 1e-12, f'{linf_prefix(z1, z2)}')
tr = TMaze().sample(400, 5, seed=1)
Es = symbols(tr, 4, 15)
pre, bad = Prefixes(Es, 1), 0
rng = np.random.default_rng(0)
for _ in range(100):
    i = rng.permutation(400)
    a, b = i[:30], i[30:200]
    d = linf_prefix([tuple(x) for x in Es[a, 1:].tolist()], [tuple(x) for x in Es[b, 1:].tolist()])
    bad += pre.at_least(a, b, d + 1e-9) or not pre.at_least(a, b, d - 1e-9)
check('fast prefix test agrees with the definition, 100 random pairs', bad == 0, f'{bad} wrong')

print('3. AdaCT-H against processes with a known answer')
for name, env, H, n, want in (('iid coin', Iid(), 6, 20000, 8),
                              ('sticky bit', Sticky(), 6, 40000, 14),
                              ('Corridor', Corridor(), 5, 5000, 11),
                              ('Corridor p = .8, .2', Corridor(p0=.8, p1=.2), 5, 20000, 11),
                              ('T-maze east', TMaze(actions='east'), 5, 5000, 12)):
    t0 = time.time()
    m = adact_h(env.sample(n, H, seed=2), env.n_act, env.n_obs)
    check(name, m.n_states == want, f'n={n} H={H} states={m.n_states} want={want} '
          f'layers={m.info["layers"]} {time.time() - t0:.2f}s')
m = adact_h(Sticky().sample(30, 6, seed=2), 2, 2)
check('too little data merges all (threshold > 1)', m.n_states == 8, f'n=30 states={m.n_states}')

print('4. both baselines end to end, each env, n=2000, held-out n=500')
for k, E in ENVS.items():
    env, T = E(), PAPER_H[k]
    train, test = env.sample(2000, T, seed=3), env.sample(500, T, seed=4)
    floor = env.belief_logloss(test)
    for name, fit in (('adact_h', adact_h), ('flexfringe', flexfringe)):
        t0 = time.time()
        try:
            m = fit(train, env.n_act, env.n_obs)
            ll, cov = m.logloss(test)
            p = m.predict(test[0][0][:3], test[0][1][:3])
            ok = abs(p.sum() - 1) < 1e-9 and ll >= floor - 0.02 and np.isfinite(ll)
            check(f'{k} {name}', ok, f'states={m.n_states} truth_layered={env.n_layered(T)} log-loss={ll:.3f} '
                  f'floor={floor:.3f} routed={cov:.2f} {time.time() - t0:.1f}s {m.info if name == "flexfringe" else ""}')
        except Exception as ex:
            check(f'{k} {name}', False, repr(ex)[:200])

print('\nFAILS:', fails if fails else 'none')
sys.exit(1 if fails else 0)
