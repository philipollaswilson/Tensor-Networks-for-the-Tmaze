"""Numerical checks of the statements in MERGE_PAC.md, on small synthetic cases.

Each section tests one numbered statement, prints the bound beside the simulated value, and
records PASS (bound holds), or FAIL. Tightness = bound / simulated. A vacuous bound (>= 1) passes
but is listed as vacuous. Read-only on kmaze.py: it uses `Rows`, `bmr_rows`, `future_windows`
and `bisimulate` as they stand.

Run from the repo root:  python src/merge_pac_check.py  (about 1 to 2 minutes)
The measured environments of question 4 are kept apart, as they take longer:
    python src/merge_pac_check.py --repo                exact machines, fixed-point test
    python src/merge_pac_check.py --variant minveto     whole learner under one rule
"""
import itertools
import math
import pathlib
import sys
import time

import numpy as np
from scipy.special import comb, gammaln as lg
from scipy.stats import binom, chi2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import kmaze as M  # noqa: E402

RNG = np.random.default_rng(20261101)
LOG = []


def report(name, ok, note=''):
    LOG.append((name, bool(ok), note))
    print(f'  [{"PASS" if ok else "FAIL"}] {name} {note}')


# ------------------------------------------------------------------ the merge statistic
def dF(x, y, mp):
    """x, y: (..., m) counts; mp: outcomes in the modality (alpha = 1/mp). As in bmr_rows."""
    a = 1.0 / mp
    n1, n2 = x.sum(-1), y.sum(-1)
    return (lg(a + x + y) + lg(a) - lg(a + x) - lg(a + y)).sum(-1) \
        - (lg(1 + n1 + n2) - lg(1 + n1) - lg(1 + n2))


def xlx(x):
    return np.where(x > 0, x * np.log(np.where(x > 0, x, 1)), 0.0)


def nJ(x, y):
    """n JS_w of the empirical rows, in nats."""
    n1, n2 = x.sum(-1), y.sum(-1)
    n = n1 + n2
    return xlx(x).sum(-1) + xlx(y).sum(-1) - xlx(x + y).sum(-1) - xlx(n1) - xlx(n2) + xlx(n)


def J(p, q, w=0.5):
    p, q = np.asarray(p, float), np.asarray(q, float)
    r = w * p + (1 - w) * q
    kl = lambda a: (a[a > 0] * np.log(a[a > 0] / r[a > 0])).sum()
    return w * kl(p) + (1 - w) * kl(q)


def tail(t, M_):
    """Lemma 4: P(n JS >= t) <= exp(M - t) (t/M)^M for t > M; here M = sum of (support - 1)."""
    t = np.asarray(t, float)
    if M_ == 0:
        return np.where(t > 0, 0.0, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        b = np.exp(M_ - t + M_ * np.log(np.maximum(t, 1e-300) / M_))
    return np.where(t > M_, np.minimum(b, 1.0), 1.0)


def K_(a):
    return lg(a) - 0.5 * np.log(2 * np.pi)


def e_range(a):
    """e(x) = r(x) - x - (1/2 - a) ln x - K(a), x >= 1: inf and sup over x, evaluated."""
    X = np.unique(np.round(np.logspace(0, 8, 3000)))
    e = X * np.log(X) - lg(X + a) + lg(a) - X - (0.5 - a) * np.log(X) - K_(a)
    return min(e.min(), 0.0), max(e.max(), 0.0)


def floor_G(n1, n2, a, C):
    """Lemma 2(a): G >= this, for any rows with C common outcomes."""
    h = n1 * n2 / (n1 + n2)
    cell = np.log((1 + a) / (4 * a)) if a <= 1 / 3 + 1e-12 else -0.5 * np.log(2)
    return 0.5 * np.log(2 * np.pi * h) + C * cell


def upper_G(n1, n2, a, Cmax):
    """Lemma 2(c): G <= this, for rows with at most Cmax common outcomes."""
    h = n1 * n2 / (n1 + n2)
    lo, hi = e_range(a)
    best = 0.0
    for C in range(1, Cmax + 1):
        best = max(best, C * ((0.5 - a) * np.log(max(h / C, 1.0)) + K_(a) + 2 * hi - lo))
    return 0.5 * np.log(2 * np.pi * h) + 1 / (12 * n1) + 1 / (12 * n2) + best


def typical_G(n1, n2, p, a):
    """Lemma 2(b): lower bound on G when every cell has x_i >= n1 p_i / 2 and same for y."""
    p = p[p > 0]
    h = n1 * n2 / (n1 + n2)
    lo, hi = e_range(a)
    hi_cell = (n1 * p / 2) * (n2 * p / 2) / (n1 * p / 2 + n2 * p / 2)
    return 0.5 * np.log(2 * np.pi * h) + ((0.5 - a) * np.log(hi_cell) + K_(a) + 2 * lo - hi).sum()


def p_atypical(n1, n2, p):
    p = p[p > 0]
    return (np.exp(-n1 * p / 8) + np.exp(-n2 * p / 8)).sum()


def merge_tail(n1, n2, p, q, t):
    """Lemma 5: P(n JS <= t) <= inf_{l, u} exp(l (t + u) - n psi(l)) + tail(u, Mq)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    n, w = n1 + n2, n1 / (n1 + n2)
    r = w * p + (1 - w) * q
    Mq = int((p > 0).sum() - 1 + (q > 0).sum() - 1)
    best = 1.0
    for l in np.linspace(0.02, 0.98, 49):
        sp = (p[p > 0] ** (1 - l) * r[p > 0] ** l).sum()
        sq = (q[q > 0] ** (1 - l) * r[q > 0] ** l).sum()
        psi = -w * np.log(sp) - (1 - w) * np.log(sq)
        us = np.linspace(max(Mq + 1e-9, 0.1), Mq + 60, 240)
        v = np.exp(np.minimum(l * (t + us) - n * psi, 0.0)) + tail(us, Mq)
        best = min(best, float(v.min()))
    return min(best, 1.0)


def rows(n, p, R):
    return RNG.multinomial(n, p, size=R).astype(float)


# ------------------------------------------------------------------ Lemmas 1-3 on tables
def check_lemmas():
    print('\nLemmas 1-3: exact statements on 200,000 random tables')
    bad = [0, 0, 0, 0]
    margin = np.inf
    for _ in range(4000):
        s = RNG.integers(2, 8)
        mp = s + RNG.integers(0, 5)
        a = 1 / mp
        n1, n2 = RNG.integers(1, 60, 2)
        x = RNG.multinomial(n1, RNG.dirichlet(np.ones(s) * 0.5), size=50).astype(float)
        y = RNG.multinomial(n2, RNG.dirichlet(np.ones(s) * 0.5), size=50).astype(float)
        d = dF(x, y, mp)
        C = ((x > 0) & (y > 0)).sum(1)
        G = d + nJ(x, y)
        # G from its cell decomposition, Lemma 1
        r = lambda v: xlx(v) - (lg(v + a) - lg(a))
        t = lambda v: lg(1 + v) - xlx(v)
        G2 = t(np.float64(n1)) + t(np.float64(n2)) - t(np.float64(n1 + n2)) \
            + (((x > 0) & (y > 0)) * (r(x) + r(y) - r(x + y))).sum(1)
        bad[0] += (np.abs(G - G2) > 1e-7).sum()
        bad[1] += (d < -np.log(comb(n1 + n2, n1)) - 1e-9).sum()
        bad[2] += (G < floor_G(n1, n2, a, C) - 1e-9).sum()
        bad[3] += (G > upper_G(n1, n2, a, min(s, mp)) + 1e-9).sum()
        margin = min(margin, (G - floor_G(n1, n2, a, C)).min())
    report('Lemma 1 identity dF = -n JS + G', bad[0] == 0, f'violations {bad[0]}')
    report('Lemma 3 floor dF >= -ln C(n, n1)', bad[1] == 0, f'violations {bad[1]}')
    report('Lemma 2a Occam floor', bad[2] == 0, f'violations {bad[2]}, smallest margin {margin:.4f}')
    report('Lemma 2c Occam ceiling', bad[3] == 0, f'violations {bad[3]}')
    n1, n2 = 3, 3
    x, y = np.array([3.0, 0, 0]), np.array([0.0, 3, 0])
    print(f'  disjoint rows (3,3): dF {dF(x, y, 3):.4f} = -ln C(6,3) = {-np.log(20):.4f}; veto '
          'needs C > e^3 = 20.09, so (3,3) does not veto and (3,4) does')


# ------------------------------------------------------------------ Theorem 1a: wrong split
def check_split():
    print('\nTheorem 1a: P(wrong split | equal rows), single modality, 400k pairs')
    print('  s  n1=n2  P(dF<0)   floor-bd  typ-bd    tight  chi2 proxy  P(dF<-3)  bd')
    ok, tight = True, []
    for s, p in ((2, [.5, .5]), (2, [.9, .1]), (3, [.5, .3, .2]), (5, [.2] * 5)):
        p = np.array(p)
        for n in (30, 300, 3000, 30000):
            x, y = rows(n, p, 400000), rows(n, p, 400000)
            d = dF(x, y, s)
            emp, emp3 = (d < 0).mean(), (d < -3).mean()
            M_ = 2 * (s - 1)
            a = 1 / s
            g = floor_G(n, n, a, 0 if a <= 1 / 3 + 1e-12 else s)
            b1 = float(tail(g, M_))
            gt = typical_G(n, n, p, a)
            b2 = min(1.0, p_atypical(n, n, p) + float(tail(gt, M_)))
            b = min(b1, b2)
            b3 = min(1.0, min(float(tail(g + 3, M_)), p_atypical(n, n, p) + float(tail(gt + 3, M_))))
            proxy = chi2.sf(2 * d.mean() + (s - 1), s - 1)     # mean dF ~ G - (s-1)/2
            ok &= emp <= b + 3e-3 and emp3 <= b3 + 3e-3
            if b < 1 and emp > 0:
                tight.append(b / emp)
            print(f'  {s}  {n:6d} {emp:9.5f} {b1:8.4f} {b2:8.4f} {b / max(emp, 1e-9):8.1f} '
                  f'{proxy:9.5f} {emp3:9.6f} {b3:8.4f}')
    report('Theorem 1a split bound', ok, f'tightness when non-vacuous: {min(tight):.0f}x to {max(tight):.0f}x'
           if tight else 'always vacuous on this grid')
    x, y = rows(100, np.array([1.0, 0, 0]), 1000), rows(100, np.array([1.0, 0, 0]), 1000)
    report('Theorem 1a point masses never split', (dF(x, y, 3) >= 0).all())


# ------------------------------------------------------------------ Theorem 1b: wrong merge
def check_merge():
    print('\nTheorem 1b: P(wrong merge), single modality, 200k pairs (tau = 0 and tau = 3)')
    print('  case               J    n1=n2  P(dF>=0)    bound   P(dF>=-3)    bound')
    ok = True
    cases = (('near', [.5, .3, .2], [.3, .3, .4]), ('far', [.8, .1, .1], [.1, .1, .8]),
             ('noisy det', [.95, .05], [.05, .95]), ('rare diff', [.98, .02], [.9, .1]))
    worst_ratio = []
    for name, p, q in cases:
        p, q = np.array(p), np.array(q)
        s = len(p)
        Jv = J(p, q)
        for k in (2, 8, 24, 60):
            n = max(4, int(k / Jv))
            x, y = rows(n, p, 200000), rows(n, q, 200000)
            d = dF(x, y, s)
            e0, e3 = (d >= 0).mean(), (d >= -3).mean()
            U = upper_G(n, n, 1 / s, s)
            b0, b3 = merge_tail(n, n, p, q, U), merge_tail(n, n, p, q, U + 3)
            ok &= e0 <= b0 + 3e-3 and e3 <= b3 + 3e-3
            if b3 < 1 and e3 > 0:
                worst_ratio.append(b3 / e3)
            print(f'  {name:16s} {Jv:.3f} {n:7d} {e0:9.5f} {b0:8.4f} {e3:10.5f} {b3:8.4f}')
    # required n at delta = 0.05, bound vs simulation
    p, q = np.array([.5, .3, .2]), np.array([.3, .3, .4])
    grid = np.unique(np.round(np.logspace(1, 4.5, 40)).astype(int))
    nb = next((n for n in grid if merge_tail(n, n, p, q, upper_G(n, n, 1 / 3, 3)) <= .05), None)
    ns = next((n for n in grid if (dF(rows(n, p, 40000), rows(n, q, 40000), 3) >= 0).mean() <= .05), None)
    report('Theorem 1b merge bound', ok, f'n for 5% wrong merge, near case: bound {nb}, simulated {ns} '
           f'(factor {nb / ns:.1f})')
    for n1, n2 in ((3, 3), (3, 4), (1, 19), (1, 20), (2, 4), (2, 5)):
        print(f'  disjoint rows ({n1},{n2}): -dF = ln C = {np.log(comb(n1 + n2, n1)):.3f}, '
              f'veto {"fires" if np.log(comb(n1 + n2, n1)) > 3 else "silent"}')


# ------------------------------------------------------------------ Theorem 1c, 1d: veto
def check_veto_merge():
    print('\nTheorem 1c: wrong merge with one informative modality among many equal ones')
    print('  equal modalities   P(merge) sum rule   P(merge) veto   bound (one modality)')
    n, R = 300, 20000
    p, q = np.array([.7, .3]), np.array([.3, .7])
    r = np.array([.5, .5])
    d0 = dF(rows(n, p, R), rows(n, q, R), 2)
    b = merge_tail(n, n, p, q, upper_G(n, n, .5, 2) + 3)
    ok = b < 1
    for E in (0, 5, 20, 40, 80):
        eq = np.stack([dF(rows(n, r, R), rows(n, r, R), 2) for _ in range(E)], 1) if E else np.zeros((R, 0))
        tot = d0 + eq.sum(1)
        neg = np.minimum(d0, 0) + np.minimum(eq, 0).sum(1)
        psum, pveto = (tot >= 0).mean(), ((tot >= 0) & (neg >= -3)).mean()
        ok &= pveto <= b + 3e-3
        print(f'  {E:8d} {psum:18.4f} {pveto:15.4f} {b:15.4f}')
    report('Theorem 1c veto bound is independent of the other modalities', ok)


def stream(n, s, A=2):
    return list(RNG.integers(s, size=n + 1)), list(RNG.integers(A, size=n))


def pair_table(n, s, L, fresh):
    """Two items of equal law in one iid stream, windows as `future_windows` builds them.
    fresh: each depth uses its own stream, so modalities do not share samples."""
    if not fresh:
        obs, acts = stream(2 * n, s)
        items = list(RNG.integers(2, size=2 * n))
        return M.future_windows(items, obs, acts, L)
    w = []
    for d in range(1, L + 1):
        obs, acts = stream(2 * n, s)
        items = list(RNG.integers(2, size=2 * n))
        w += [e for e in M.future_windows(items, obs, acts, d)
              if len(e[1][0]) == d and isinstance(e[1][0][0], (int, np.integer))]
    ev_obs, ev_acts = stream(2 * n, s)
    items = list(RNG.integers(2, size=2 * n))
    w += [e for e in M.future_windows(items, ev_obs, ev_acts, 1) if e[1][0][0] == 'event']
    return w


def per_modality(R):
    x, y = R.X[0], R.X[1]
    a = R.alpha
    per = np.zeros(R.M.shape[1])
    np.add.at(per, R.mod, lg(a + x + y) + lg(a) - lg(a + x) - lg(a + y))
    t1, t2 = x @ R.M, y @ R.M
    return per - (lg(1 + t1 + t2) - lg(1 + t1) - lg(1 + t2))


def analytic_neg(R, s):
    """Theorem 1d: sum over modalities of the bound on E[(dF_j)-], with tail integrals."""
    t1, t2 = R.X[0] @ R.M, R.X[1] @ R.M
    total = 0.0
    for j in range(R.M.shape[1]):
        n1, n2 = t1[j], t2[j]
        if n1 < 1 or n2 < 1:
            continue
        cap = lg(n1 + n2 + 1) - lg(n1 + 1) - lg(n2 + 1)
        mp = R.M[:, j].sum()
        if mp < 2:
            continue
        a = 1 / mp
        g = floor_G(n1, n2, a, 0 if a <= 1 / 3 + 1e-12 else 2)
        us = np.linspace(0, cap, 13)
        M_ = 2 * (mp - 1)
        total += np.minimum(1.0, tail(g + us, int(M_))).mean() * cap
    return total


def check_veto_split():
    print('\nTheorem 1d: wrong split of two equal items through the real bmr_rows')
    print('  iid noise on s symbols, 2 actions, future_windows of depth 1..L plus event windows')
    print('  L  s    n  modal  shared: P(split) P(veto) E[-F-]/3 analytic/3 | no sharing: P(split) P(veto)')
    ok, rowsout = True, []
    for L, s, n in ((1, 2, 200), (3, 2, 200), (3, 3, 200), (3, 3, 2000)):
        out = {}
        R_ = 40 if n <= 200 else 8
        for fresh in (False, True):
            sp, ve, neg, mods, ana = 0, 0, 0.0, 0, 0.0
            for _ in range(R_):
                w = pair_table(n, s, L, fresh)
                T = M.Rows([0, 1], w, s ** L, L)
                per = per_modality(T)
                F = np.minimum(per, 0).sum()
                sp += M.bmr_rows(T).max() > 0
                ve += F < -3
                neg += -F
                mods += T.M.shape[1]
                ana += analytic_neg(T, s)
            out[fresh] = (sp / R_, ve / R_, neg / R_ / 3, mods / R_, ana / R_ / 3)
        a, b = out[False], out[True]
        ok &= a[1] <= a[2] + 3 * np.sqrt(max(a[1], 1 / R_) / R_) + 1e-9 and a[2] <= a[4]
        rowsout.append((L, s, n, a, b))
        print(f'  {L}  {s} {n:5d} {a[3]:5.0f}   {a[0]:12.3f} {a[1]:7.3f} {a[2]:8.3f} {a[4]:9.2f} | {b[0]:12.3f} {b[1]:7.3f}')
    report('Theorem 1d Markov bound P(veto) <= E[-F-]/tau', ok, '(dependence-free, plug-in expectation)')
    print('  accumulation on M independent noisy modalities of equal rows, n = 30, m = 2, 40k pairs')
    print('    M   P(sum of negatives < -3)   P(any single dF < -3)   P(F < 0)')
    p = np.array([.5, .5])
    vals = []
    for Mm in (1, 5, 20, 80, 160):
        d = np.stack([dF(rows(30, p, 40000), rows(30, p, 40000), 2) for _ in range(Mm)], 1)
        v, one, neg = (np.minimum(d, 0).sum(1) < -3).mean(), (d.min(1) < -3).mean(), (d.sum(1) < 0).mean()
        vals.append((v, one))
        print(f'    {Mm:3d} {v:18.4f} {one:22.4f} {neg:14.4f}')
    report('Theorem 1d sum veto accumulates faster than the single-modality union', vals[3][0] > 2 * vals[3][1],
           f'at M = 80: {vals[3][0]:.3f} against {vals[3][1]:.3f}; the single-modality rate also grows')
    print('  worst case, c identical copies of one modality: the veto threshold per copy is 3/c')
    n = 200
    p = np.array([.5, .5])
    d = dF(rows(n, p, 200000), rows(n, p, 200000), 2)
    for c in (1, 2, 4, 7):
        print(f'    c = {c}: P(c min(dF, 0) < -3) = {(c * np.minimum(d, 0) < -3).mean():.4f}')


# ------------------------------------------------------------------ Theorem 2: BISIM given U
class Ring:
    """Unifilar machine with Q true states, each copied c times. Action 0 moves round the
    ring, action 1 stays; the emission has bias eps, and state Q - 1 has the opposite bias,
    so the difference must walk back Q - 1 steps. The copy flips with the observation."""

    def __init__(self, Q, c, eps):
        self.Q, self.c, self.eps = Q, c, eps
        self.K, self.A = Q * c, 2

    def theta(self, v, a):
        i = v // self.c
        e = self.eps
        return np.array([e, 1 - e] if i == self.Q - 1 else [1 - e, e])

    def step(self, v, a, o):
        i, k = v // self.c, v % self.c
        i2 = (i + 1) % self.Q if a == 0 else i
        return i2 * self.c + (k + o) % self.c


def moore(mc):
    """Exact Moore rounds. Returns the partitions and, for each separated pair, its JS at the
    round it separates (best action)."""
    K, A = mc.K, mc.A
    blocks = np.zeros(K, int)
    gaps, parts = {}, [blocks.copy()]

    def law(v, a, b):
        out = {}
        th = mc.theta(v, a)
        for o, p in enumerate(th):
            if p > 0:
                k = (o, b[mc.step(v, a, o)])
                out[k] = out.get(k, 0) + p
        return out
    for _ in range(K):
        sig = [tuple(tuple(sorted((k, round(p, 12)) for k, p in law(v, a, blocks).items()))
                     for a in range(A)) for v in range(K)]
        ids = {}
        new = np.array([ids.setdefault((blocks[v], sig[v]), len(ids)) for v in range(K)])
        for u, v in itertools.combinations(range(K), 2):
            if blocks[u] == blocks[v] and new[u] != new[v]:
                best = 0.0
                for a in range(A):
                    lu, lv = law(u, a, blocks), law(v, a, blocks)
                    keys = sorted(set(lu) | set(lv))
                    best = max(best, J([lu.get(k, 0) for k in keys], [lv.get(k, 0) for k in keys]))
                gaps[(u, v)] = best
        if len(ids) == len(set(blocks)):
            break
        blocks = new
        parts.append(blocks.copy())
    return parts, gaps


def rule(xs, ys, mps, tau=3.0):
    """The repo's rule on one pair, modalities given as count pairs."""
    per = np.array([dF(x, y, mp) for x, y, mp in zip(xs, ys, mps)])
    return per.sum() >= -1e-9 and np.minimum(per, 0).sum() >= -tau


def stack(mc, n0):
    return {(v, a): RNG.choice(2, size=n0, p=mc.theta(v, a)) for v in range(mc.K) for a in range(mc.A)}


def bisim(mc, st, n0, greedy=False):
    """Moore refinement on fixed-size rows. greedy=False: pairwise tests inside each block,
    then connected components (Theorem 2). greedy=True: the real bmr_rows on all items."""
    K, A = mc.K, mc.A
    b = np.zeros(K, int)
    for _ in range(K + 1):
        if greedy:
            w = []
            for (v, a), o in st.items():
                nxt = [b[mc.step(v, a, int(x))] for x in o]
                w += [(v, (('next', a), (int(x), int(nb)))) for x, nb in zip(o, nxt)]
            new = M.bmr_rows(M.Rows(list(range(K)), w, 2, 1))
        else:
            cols = {}
            X = np.zeros((K, A, 2 * K))
            for (v, a), o in st.items():
                c = (o * K + b[[mc.step(v, a, x) for x in (0, 1)]][o])
                X[v, a] += np.bincount(c, minlength=2 * K)
            mp = np.maximum((X.sum(0) > 0).sum(1), 1)
            par = list(range(K))

            def find(i):
                while par[i] != i:
                    par[i] = par[par[i]]
                    i = par[i]
                return i
            for u, v in itertools.combinations(range(K), 2):
                if b[u] == b[v] and find(u) != find(v):
                    if rule(X[u], X[v], mp):
                        par[find(u)] = find(v)
            roots = [(b[v], find(v)) for v in range(K)]
            ids = {}
            new = np.array([ids.setdefault(r, len(ids)) for r in roots])
        if len(set(zip(b, new))) == len(set(b)) == len(set(new)):
            return b
        b = new
    return b


def same_partition(a, b):
    return len(set(zip(a, b))) == len(set(a)) == len(set(b))


def bound_T2(mc, gaps, n0, tau=3.0):
    """Failure bound of Theorem 2 given coverage. Wrong-split candidates: pairs with equal
    emissions at every action (they stay together for at least one round, and their row over
    (o, block) is their row over o, so one event per pair covers all rounds). Wrong-merge
    candidates: every pair, at the round it separates."""
    s = max((mc.theta(v, a) > 0).sum() for v in range(mc.K) for a in range(mc.A))
    split_pairs = sum(all(np.allclose(mc.theta(u, a), mc.theta(v, a)) for a in range(mc.A))
                      for u, v in itertools.combinations(range(mc.K), 2))
    A = mc.A
    pS = 0.0
    if s > 1:
        M1 = 2 * (s - 1)
        g = 0.5 * np.log(np.pi * n0 / 4)                # floor, worst alpha = 1/2
        pS = A * float(tail(g + tau / A, M1)) + float(tail(A * g, A * M1))
    pM = 0.0
    U = max(upper_G(n0, n0, 1 / mp, 2) for mp in (2, 3, 4)) + tau
    for (u, v), gam in gaps.items():
        if gam >= np.log(2) - 1e-9:                     # disjoint rows: Lemma 3
            pM += 0.0 if np.log(comb(2 * n0, n0)) > tau else 1.0
        else:
            e = mc.eps
            pM += merge_tail(n0, n0, np.array([1 - e, e]), np.array([e, 1 - e]), U)
    return min(1.0, split_pairs * pS + pM), split_pairs, split_pairs * pS, pM


def pair_rate(eps, n0, tau=3.0, R_=100000):
    """Simulated P(wrong split) of one equal pair under the rule, 2 actions, worst alpha."""
    p = np.array([1 - eps, eps])
    d1 = dF(rows(n0, p, R_), rows(n0, p, R_), 2)
    d2 = dF(rows(n0, p, R_), rows(n0, p, R_), 2)
    return ((d1 + d2 < -1e-9) | (np.minimum(d1, 0) + np.minimum(d2, 0) < -tau)).mean()


def check_T2():
    print('\nTheorem 2: Moore refinement on a given unifilar U, fixed-size rows, 2 actions')
    print('  success rates; "union" = N_E x simulated pair rate + merge bound; "bound" = Theorem 2')
    print('  Q c eps    n0  pairwise+CC  greedy bmr_rows  1-union  N_E  split term  merge term')
    ok = True
    for Q, c, eps in ((4, 2, 0.1), (4, 2, 0.02), (4, 3, 0.1)):
        mc = Ring(Q, c, eps)
        parts, gaps = moore(mc)
        final = parts[-1]
        for n0 in (10, 30, 100, 1000, 10000):
            R_ = 30
            ok1, ok2 = 0, 0
            for _ in range(R_):
                st = stack(mc, n0)
                ok1 += same_partition(bisim(mc, st, n0), final)
                if n0 <= 300:
                    ok2 += same_partition(bisim(mc, st, n0, greedy=True), final)
            b, Ne, ps, pm = bound_T2(mc, gaps, n0)
            un = min(1.0, Ne * pair_rate(eps, n0) + pm)
            ok &= 1 - ok1 / R_ <= b + 2 * np.sqrt(.25 / R_)
            g2 = f'{ok2 / R_:.2f}' if n0 <= 300 else '  - '
            print(f'  {Q} {c} {eps:.2f} {n0:5d} {ok1 / R_:10.2f} {g2:>13s} {1 - un:10.2f} {Ne:4d} {ps:10.3g} {pm:10.3g}')
    report('Theorem 2 failure <= bound (given coverage)', ok, '(the bound is vacuous below n0 ~ 1e9)')
    mc = Ring(4, 2, .1)
    gaps = moore(mc)[1]
    nb = next((n for n in np.logspace(1, 14, 131) if bound_T2(mc, gaps, float(n))[0] <= .05), None)
    print(f'  bound reaches 95% on Ring(4,2,0.1) at n0 = {nb:.3g}')


def check_rate_delta():
    print('\nTheorem 2 remark: the false-split rate of the pairwise rule is polynomial, not exponential')
    print('  n     P(dF<0) sim   sim*sqrt(n)   (Bernoulli 0.5, 2M pairs, m=2)')
    vals = []
    for n in (100, 1000, 10000, 100000, 1000000):
        p = np.array([.5, .5])
        d = dF(rows(n, p, 2000000), rows(n, p, 2000000), 2)
        e = (d < 0).mean()
        vals.append(e * np.sqrt(n))
        print(f'  {n:8d} {e:10.5f} {e * np.sqrt(n):12.3f}')
    report('polynomial rate: P(split) ~ n^(-1/2) up to a log', max(vals) / min(vals) < 3,
           f'P sqrt(n) spans {min(vals):.2f}-{max(vals):.2f}')


# ------------------------------------------------------------------ Theorem 3
class CueCorridor:
    """Fair bit b; the start shows it; D blank cells; a last cell shows b again, absorbing.
    go moves on, stay stays. Given (b, position) every row is a point mass."""
    A = 2

    def __init__(self, D):
        self.D = D
        self.P = D + 2
        self.K = 2 * self.P

    def obs(self, b, pos):
        return b if pos == 0 else 2 if pos <= self.D else 3 + b

    def nxt(self, pos, a):
        return min(pos + 1, self.D + 1) if a == 0 else pos

    def episode(self, T, r):
        b = int(r.integers(2))
        pos, ob, ac, st = 0, [self.obs(b, 0)], [], [(b, 0)]
        for a in r.integers(2, size=T):
            pos = self.nxt(pos, int(a))
            ac.append(int(a)); ob.append(self.obs(b, pos)); st.append((b, pos))
        return ob, ac, st

    def occupancy(self, T):
        """Exact visits per episode and P(visited at all) of each (state, action), for one bit;
        layered occupancy as well."""
        P = self.P
        vis = np.zeros((P, 2))
        dist = np.zeros(P); dist[0] = 1
        layered = np.full((T, P, 2), 0.0)
        for t in range(T):
            layered[t] = dist[:, None] * 0.25          # bit 1/2 x action 1/2
            vis += dist[:, None] * 0.5
            new = np.zeros(P)
            for p in range(P):
                for a in (0, 1):
                    new[self.nxt(p, a)] += dist[p] * .5
            dist = new
        first = np.zeros((P, 2))
        for tp in range(P):
            for ta in (0, 1):
                dist = np.zeros(P); dist[0] = 1
                hit = 0.0
                for t in range(T):
                    hit += dist[tp] * .5
                    new = np.zeros(P)
                    for p in range(P):
                        for a in (0, 1):
                            if p == tp and a == ta:
                                continue
                            new[self.nxt(p, a)] += dist[p] * .5
                    dist = new
                first[tp, ta] = hit
        return vis / 2, first / 2, layered


def check_T3():
    print('\nTheorem 3: deterministic rows. n0 = 4 suffices for any delta, given U and coverage')
    print('  Exact on CueCorridor: separation needs ln C(2n0, n0) > 3 only')
    for n0 in (1, 2, 3, 4):
        print(f'    n0 = {n0}: ln C(2n0, n0) = {np.log(comb(2 * n0, n0)):.3f}')
    print('  D  T  K_U  d_U     N(T3)   coverage rate  bisimulate on U exact (real code, all visits)')
    ok = True
    delta = 0.05
    for D in (1, 2, 3):
        cc = CueCorridor(D)
        T = 4 * (D + 2)
        # U: two copies of each state (parity of stays). Exact occupancy on U
        dist = np.zeros((cc.P, 2)); dist[0, 0] = 1
        du = np.zeros((cc.P, 2, 2))
        for _ in range(T):
            du += dist[:, :, None] * .25                 # bit, action
            new = np.zeros((cc.P, 2))
            for p_ in range(cc.P):
                new[cc.nxt(p_, 0)] += dist[p_] * .5
                new[p_, ::-1] += dist[p_] * .5
            dist = new
        d = du[du > 0].min()
        KA = 2 * cc.K * cc.A
        pv = cc.occupancy(T)[1].min() / 2              # one copy holds about half the entries
        N = int(math.ceil(min(max(2 * 4 / d, 8 * T * np.log(KA / delta) / d),
                              max(2 * 4 / pv, 8 * np.log(KA / delta) / pv))))
        R_ = 12
        cov, real = 0, 0
        for r_ in range(R_):
            r = np.random.default_rng(1000 + r_ + 100 * D)
            eps = [cc.episode(T, r) for _ in range(N)]
            ids, sts, acts = {}, [], []
            for ob, ac, st in eps:
                c, lab = 0, []
                for t, s in enumerate(st):
                    lab.append(ids.setdefault((s, c), len(ids)))
                    if t < len(ac):
                        c ^= int(ac[t] == 1)        # the copy is the parity of stays
                sts.append(np.array(lab)); acts.append(ac)
            cnt = {}
            for s, a in zip(sts, acts):
                for t, x in enumerate(a):
                    cnt[(int(s[t]), x)] = cnt.get((int(s[t]), x), 0) + 1
            covered = len(cnt) == cc.K * 2 * 2 and min(cnt.values()) >= 4
            cov += covered
            trajs = [(ob, ac, st) for ob, ac, st in eps]
            S = M.bisimulate(trajs, sts, 5, 1)
            lab = {}
            good = True
            for st, s in zip([e[2] for e in eps], S):
                for a_, b_ in zip(st, s):
                    good &= lab.setdefault(a_, int(b_)) == int(b_)
            good &= len(set(lab.values())) == len(lab)
            real += good
        ok &= cov / R_ >= 1 - delta
        print(f'  {D}  {T:2d} {2 * cc.K:2d} {d:7.4f} {N:6d} {cov / R_:10.2f} {real / R_:14.2f}')
    report('Theorem 3 coverage count gives n0 = 4 everywhere', ok)
    # determinism also needs no statistics: simulate the idealised rule at n0 = 4
    mc = Ring(5, 2, 0.0)
    parts, _ = moore(mc)
    res = [same_partition(bisim(mc, stack(mc, n0), n0), parts[-1]) for n0 in (1, 2, 3, 4, 5)]
    report('Theorem 3 deterministic Ring: exact recovery at n0 >= 4', res[3] and res[4],
           f'n0 = 1..5 -> {[int(x) for x in res]}')


def c0(n0):
    """Least total of minority counts (a + b) with dF < 0, m = 2 (alpha = 1/2)."""
    for c in range(1, 2 * n0):
        for a_ in range(0, c + 1):
            b_ = c - a_
            if a_ <= n0 and b_ <= n0:
                x, y = np.array([n0 - a_, a_], float), np.array([n0 - b_, b_], float)
                if dF(x, y, 2) < 0:
                    return c
    return 2 * n0


def check_near_det():
    print('\nTheorem 3b: near-determinism. P(wrong split) <= P(Bin(2n0, eps) >= c0(n0))')
    print('  eps     n0  c0  P(split) sim    bound   tight')
    ok = True
    for eps in (.001, .01, .05):
        for n0 in (10, 30, 100, 300):
            p = np.array([1 - eps, eps])
            R_ = 2000000
            e = (dF(rows(n0, p, R_), rows(n0, p, R_), 2) < 0).mean()
            c = c0(n0)
            b = binom.sf(c - 1, 2 * n0, eps)
            ok &= e <= b + 1e-5
            print(f'  {eps:.3f} {n0:4d} {c:3d} {e:12.2e} {b:9.2e} {b / max(e, 5e-7):7.1f}')
    report('Theorem 3b near-deterministic split bound', ok)


# ------------------------------------------------------------------ Theorem 4: GROW
def rows_of_stat(cc, T, N, r, stat, H):
    """Episodes -> {(state or candidate): outcome list}, hold-out odd / even. First-visit rows
    with a fixed horizon H after the visit. stat 'event': (go steps to the first change,
    symbol); stat 'window': the observation sequence of the all-go action sequence of length
    H, used only if the actions match."""
    odd, even = {}, {}
    for e in range(N):
        ob, ac, st = cc.episode(T, r)
        pool = odd if e % 2 else even
        first = {}
        for t in range(len(ac) - H):
            if st[t] in first:
                continue
            first[st[t]] = t
            if stat == 'event':
                g, val = 0, (ob[t], 'none')
                for u in range(t, t + H):
                    if ob[u + 1] != ob[u]:
                        val = (ob[t], g + (ac[u] == 0), ob[u + 1])
                        break
                    g += ac[u] == 0
            else:
                if any(a != 0 for a in ac[t:t + H]):
                    continue
                val = tuple(ob[t:t + H + 1])
            pool.setdefault(st[t], []).append(val)
    return odd, even


def grow_ok(cc, odd, even, n0):
    """Pairs of true states are tested with odd rows against even rows; success = equal
    states merge and different states do not, on n0 samples each. All K states are candidates."""
    S = sorted(set(odd) | set(even))
    for s in S:
        if len(odd.get(s, [])) < n0 or len(even.get(s, [])) < n0:
            return False
    for u, v in itertools.product(S, S):
        col = {}
        for val in odd[u][:n0] + even[v][:n0]:
            col.setdefault(val, len(col))
        x, y = np.zeros(len(col)), np.zeros(len(col))
        for val in odd[u][:n0]:
            x[col[val]] += 1
        for val in even[v][:n0]:
            y[col[val]] += 1
        d = float(dF(x, y, max(len(col), 1))) if len(col) > 1 else 0.0
        same = d >= -1e-9
        if same != (u == v):
            return False
    return True


def bound_T4_event(cc, T, H, delta):
    """Smallest n0 for which Theorem 4 + 3b give failure <= delta/2 on the event statistic."""
    D, K = cc.D, cc.K
    kap = 1 - binom.cdf(D, H, .5)                       # worst state: all D + 1 go steps
    eps = 1 - kap
    pairs = K * K
    for n0 in range(4, 4000):
        split = binom.sf(c0(n0) - 1, 2 * n0, eps)
        U = upper_G(n0, n0, 1 / 3, 1)
        merge = merge_tail(n0, n0, np.array([kap, 0, eps]), np.array([0, kap, eps]), U)
        if pairs * (split + merge) <= delta / 2:
            return n0, kap
    return None, kap


def check_T4():
    print('\nTheorem 4: from scratch on CueCorridor, hold-out rows of one statistic, every pair of')
    print('  K states tested (what the induction reduces to). delta = 0.05. N in episodes; rate at')
    print('  N(Thm 4) must be >= 0.95. N(sim) is the first of N/16, N/4, N reaching 6/6.')
    print('  D  K  stat    H  n0  kappa   N(Thm 4)  rate there   N(sim)  ratio')
    ok = True
    delta = .05
    for D in (1, 2, 3):
        cc = CueCorridor(D)
        T = 4 * (D + 2) * 2
        for stat in ('event', 'window'):
            H = T // 2 if stat == 'event' else D + 1
            pass_hit = 1 - binom.cdf(D, T - H, .5)       # P(reach the deepest state in time)
            if stat == 'event':
                n0, kap = bound_T4_event(cc, T, H, delta)
                rate = 1.0
            else:
                n0, kap, rate = 4, 1.0, 2.0 ** (-H)
            p = pass_hit * .5 * rate                     # bit x reach x statistic applies
            Nth = int(np.ceil(max(4 * n0 / p, 16 * np.log(4 * cc.K / delta) / p)))
            R_ = 10
            at = sum(grow_ok(cc, *rows_of_stat(cc, T, Nth, np.random.default_rng(s + 7 * D), stat, H), n0)
                     for s in range(R_)) / R_
            sim = None
            for f in (1 / 16, 1 / 4, 1):
                N = max(int(Nth * f), 8)
                good = sum(grow_ok(cc, *rows_of_stat(cc, T, N, np.random.default_rng(99 * s + N), stat, H), n0)
                           for s in range(6))
                if good >= 6:
                    sim = N
                    break
            ok &= at >= .8
            print(f'  {D}  {cc.K:2d} {stat:7s} {H:2d} {n0:3d} {kap:.4f} {Nth:9d} {at:9.2f} {str(sim):>9s} '
                  f'{Nth / sim if sim else float("nan"):6.1f}')
    report('Theorem 4 success at N(Thm 4)', ok)


# ------------------------------------------------------------------ Proposition 6: Deb, RegORL
def check_lower():
    print('\nProposition 6: the 1/(d mu) statement against the two-point process (d = 1/4)')
    print('  Each of two histories has occupancy d; next bit is Bern(1/2) in M0, Bern(1/2 +- mu/2) in M1.')
    print('    mu   N(Õ(1/(d mu)))  P(detect M1)   false alarm    N(1/(d mu^2), corrected)  detect')
    d, delta, X = 0.25, 0.05, 2
    ok = True
    ell = np.log(8 / delta)
    for mu in (.3, .1, .03, .003):
        N1 = int(32 * ell / (d * mu))          # Deb Eq. (8) with the sqrt(H log) factor set to 1
        N2 = int(2 * 8 * np.log(4 * X / delta) / (d * mu ** 2))
        out = []
        for N in (N1, N2):
            z1 = RNG.binomial(N, d, size=100000)
            z2 = RNG.binomial(N, d, size=100000)
            thr = lambda m: np.sqrt(2 * np.log(4 * X / delta) / np.maximum(np.minimum(z1, z2), 1))
            for sign in (mu, 0.0):
                a = RNG.binomial(z1, .5 + sign / 2) / np.maximum(z1, 1)
                b = RNG.binomial(z2, .5 - sign / 2) / np.maximum(z2, 1)
                out.append((np.abs(a - b) >= thr(0)).mean())
        kl = -np.log(1 - mu ** 2)
        lb = np.log(1 / (4 * delta)) / (d * kl)
        print(f'  {mu:.3f} {N1:11d} {out[0]:13.3f} {out[1]:12.3f} {N2:20d} {out[2]:18.3f}'
              f'   any-test lower bound N >= {lb:.0f}')
        ok &= out[2] >= 1 - 2 * delta
        if mu <= .03:
            ok &= N1 < lb or out[0] < .9
    report('Prop 6 Õ(1/(d mu)) fails at small mu, 1/(d mu^2) works', ok,
           '(Deb Eq. 8 / RegORL Eq. 22 N against Deb Lemma 7 / RegORL Lemma 14 test)')
    # the same for the BMR rule
    print('  BMR rule on the same pair, m = 2: P(split at d N visits each)')
    for mu in (.3, .1, .03):
        N = int(2 * 2 * (np.log(1 / .05) + np.log(1 / mu)) / (d * J([.5 + mu / 2, .5 - mu / 2],
                                                                    [.5 - mu / 2, .5 + mu / 2])))
        n = int(N * d / 2)
        x, y = rows(n, np.array([.5 + mu / 2, .5 - mu / 2]), 100000), rows(n, np.array([.5 - mu / 2, .5 + mu / 2]), 100000)
        print(f'    mu {mu:.2f}: n {n}, J {J([.5 + mu / 2, .5 - mu / 2], [.5 - mu / 2, .5 + mu / 2]):.5f}, '
              f'mu^2/2 {mu ** 2 / 2:.5f}, P(split) {(dF(x, y, 2) < 0).mean():.3f}')


def check_rare():
    print('\nRemark: rare-event gap, event of probability mu in one row, 0 in the other')
    print('  mu     J        mu^2/2    n BMR 95%   n Hoeffding test 95%')
    ok = True
    for mu in (.1, .03, .01):
        p, q = np.array([mu, 1 - mu]), np.array([0.0, 1.0])
        grid = np.unique(np.round(np.logspace(1, 5.5, 46)).astype(int))
        nb, nh = None, None
        for n in grid:
            x, y = rows(n, p, 20000), rows(n, q, 20000)
            if nb is None and (dF(x, y, 2) < 0).mean() >= .95:
                nb = n
            if nh is None:
                t = np.sqrt(2 * np.log(4 * 2 / .05) / n)
                if (np.abs(x[:, 0] - y[:, 0]) / n >= t).mean() >= .95:
                    nh = n
        ok &= nb < nh
        print(f'  {mu:.2f} {J(p, q):.5f} {mu ** 2 / 2:.6f} {nb:8d} {nh:14d}')
    report('rare event: J ~ mu, so BMR needs ~1/mu where Hoeffding needs ~1/mu^2', ok)


# ------------------------------------------------------------------ Proposition 8
def check_edge():
    print('\nProposition 8: a rare edge truly enters A; pooled rows of A and B hold 20,000 visits each')
    print('  Bern(0.6) vs Bern(0.4). Pooled evidence is huge, yet only the n_e visits of the edge decide.')
    print('  n_e   dF(A, B) pooled  P(allowed to merge B)  best-test error  floor (Bretagnolle-Huber)  bound Thm 1b')
    pa, pb = np.array([.6, .4]), np.array([.4, .6])
    big = 20000
    A, B = rows(big, pa, 1)[0], rows(big, pb, 1)[0]
    kl = (pa * np.log(pa / pb)).sum()
    ok = True
    wrong = []
    for ne in (5, 20, 50, 150, 500, 2000):
        e = rows(ne, pa, 200000)
        d = dF(e, np.broadcast_to(B, e.shape), 2)
        allowed = (d >= 0).mean()
        llr = (e * np.log(pa / pb)).sum(1)
        err2 = (rows(ne, pb, 200000) * np.log(pa / pb)).sum(1)
        err = .5 * (llr < 0).mean() + .5 * (err2 > 0).mean()
        floor = .25 * np.exp(-ne * kl)
        U = upper_G(float(ne), float(big), .5, 2)
        b = merge_tail(ne, big, pa, pb, U)
        ok &= err >= floor - 2e-3 and allowed <= b + 3e-3
        wrong.append(allowed)
        print(f'  {ne:5d} {float(dF(A, B, 2)):14.1f} {allowed:18.3f} {err:17.3f} {floor:18.2e} {b:15.3f}')
    report('Prop 8 floor holds, and pooled evidence does not stop a rare edge', ok and wrong[1] > .5,
           f'P(merge with the wrong state) {wrong[1]:.2f} at n_e = 20 while pooled dF is {float(dF(A, B, 2)):.0f}')


# ------------------------------------------------------------------ Table for Section 5
def corridor_table():
    print('\nSection 5: exact numbers on CueCorridor(D), T = 4(D+2), delta = 0.05, 2 actions')
    print('  all N are episodes. layered = Deb setting; pooled = time-homogeneous, per-visit')
    print('  mu_X: single-symbol containment languages (X_111), min over layers and state pairs')
    print('  D  mu(Linf-p)  d_lay     mu_X    | Deb stated  corrected   L_X corr.  | d_pool p_pool  T3(U)   '
          'T4 event  T4 with Hoeffding')
    delta = .05
    for D in (1, 2, 3, 4, 5):
        cc = CueCorridor(D)
        T = 4 * (D + 2)
        vis, first, lay = cc.occupancy(T)
        d_l = lay[lay > 0].min()
        d_p = vis.min()
        mu = 2.0 ** (-D)                         # exact, see MERGE_PAC.md Section 5
        P = cc.P
        # L_X with X = {contains symbol y}: exact, over layers, for pairs of reachable states
        muX = np.inf
        for t in range(T):
            R = T - t
            S = [p for p in range(P) if lay[t, p, 0] > 0]
            rev = lambda p: 1 - binom.cdf(D - p, R, .5) if p <= D else 1.0   # reach last cell
            for (p1, b1), (p2, b2) in itertools.combinations([(p, b) for p in S for b in (0, 1)], 2):
                f1 = {cc.obs(b1, u) for u in range(p1 + 1, P)} | {cc.obs(b1, p1)}
                # P(contains y) by go-count only; equal futures are one layered state
                vals = []
                for y in (0, 1, 2, 3, 4):
                    pa, pb = 0.0, 0.0
                    for (pp, bb, name) in ((p1, b1, 'a'), (p2, b2, 'b')):
                        need = None
                        for u in range(pp, P):
                            if cc.obs(bb, u) == y:
                                need = u - pp
                                break
                        val = 0.0 if need is None else (1.0 if need == 0 else 1 - binom.cdf(need - 1, R, .5))
                        if name == 'a':
                            pa = val
                        else:
                            pb = val
                    vals.append(abs(pa - pb))
                if max(vals) > 0:
                    muX = min(muX, max(vals))
        Q, A, O, Rw = 2 * P * T, 2, 5, 1
        H = T
        ell = np.log(4 * 2 * Q * A * O / delta)
        stated = 32 * ell / (d_l * mu) * np.sqrt(H * np.log(4 * A * Rw * O))
        Mreq = 8 * (H * np.log(2 * A * Rw * O) + np.log(4 * 2 * Q * A * O / delta)) / mu ** 2
        corrected = 2 * Mreq / d_l
        cX = 5
        corrX = 2 * 8 * np.log(4 * cX * 2 * Q * A * O / delta) / (muX ** 2 * d_l)
        T4, He = 2 * T, T                                # Theorem 4 run at 2T, as in check_T4
        n0, kap = bound_T4_event(cc, T4, He, delta)
        pe = (1 - binom.cdf(D, T4 - He, .5)) * .5
        ours4 = max(4 * n0 / pe, 16 * np.log(4 * cc.K / delta) / pe)
        hoeff = 2 * 2 * 8 * np.log(4 * (D + 3) * 2 * cc.K * cc.K * A * O / delta) / (kap ** 2 * pe)
        pv = first.min()
        ours3 = max(2 * 4 / pv, 8 * np.log(cc.K * A / delta) / pv)
        print(f'  {D} {mu:9.4f} {d_l:9.2e} {muX:9.2e} | {stated:9.2e} {corrected:10.2e} {corrX:11.2e} |'
              f' {d_p:6.3f} {pv:6.3f} {ours3:7.0f} {ours4:7.0f} {hoeff:9.0f}')


# ------------------------------------------------------------------ --repo: question 4
# Not part of the default run (minutes, not seconds). Rule variants are patched in memory.
def bmr_variant(R, strong=3.0, mode='sum'):
    """bmr_rows with the veto as a choice. sum: as in kmaze.py; min: any single modality
    with dF < -strong (all that Theorem 1c needs); none: no veto."""
    X = R.X.copy()
    T = R.totals(X)
    a = R.alpha
    gX, gT = lg(a + X), lg(1 + T)
    ga, g1 = lg(a), lg(1.0)

    def row(i):
        per = ((lg(a + X[i] + X) + ga - gX[i] - gX) @ R.M - (lg(1 + T[i] + T) + g1 - gT[i] - gT))
        bad = {'sum': np.minimum(per, 0).sum(1) < -strong, 'min': per.min(1) < -strong,
               'none': np.zeros(len(per), bool)}[mode]
        return np.where(bad, -np.inf, per.sum(1))
    n = len(X)
    lab = np.arange(n)
    alive = np.ones(n, bool)
    D = np.array([row(i) for i in range(n)])
    np.fill_diagonal(D, -np.inf)
    while alive.sum() > 1:
        Mx = np.where(alive[:, None] & alive[None], D, -np.inf)
        i, j = np.unravel_index(np.argmax(Mx), Mx.shape)
        if Mx[i, j] < -1e-9:
            break
        X[i] += X[j]; T[i] += T[j]; alive[j] = False; lab[lab == j] = i
        gX[i], gT[i] = lg(a + X[i]), lg(1 + T[i])
        D[i] = D[:, i] = row(i); D[i, i] = -np.inf
    return np.unique(lab, return_inverse=True)[1]


def windows_fixed(items, obs, acts, L, nxt=None):
    """future_windows with event windows on a fixed horizon H = T // 2, from steps with H
    steps left, and an explicit 'none' outcome: the law no longer depends on the time left."""
    out, H = [], len(acts) // 2
    changes = [u for u in range(1, len(obs)) if obs[u] != obs[u - 1]]
    for t in range(len(acts)):
        for d in range(1, min(L, len(acts) - t) + 1):
            out.append((items[t], (tuple(acts[t:t + d]), tuple(obs[t + 1:t + d + 1]))))
        if nxt is not None and nxt[t + 1] >= 0:
            out.append((items[t], (('next', acts[t]), (obs[t + 1], int(nxt[t + 1])))))
        if t + H <= len(acts):
            ev = [obs[u] for u in changes if t < u <= t + H][:M.EVENTS]
            for d in range(1, M.EVENTS + 1):
                out.append((items[t], (('event', d), tuple(ev[:d]) if len(ev) >= d else ('none',))))
    return out


def predictive(env):
    """Exact predictive machine: rows[(q, a)] = {(obs, next q): prob}."""
    import merge_theory as MT
    rng = np.random.default_rng(0)
    start = {}
    for bits in itertools.product((0, 1), repeat=env.k):
        q = MT.key(env, 0, bits, MT.observe(env, 0, (False,) * env.k))
        start[q] = start.get(q, 0) + 0.5 ** env.k
    rows_, todo, done = {}, list(start), set()
    while todo:
        q = todo.pop()
        if q in done:
            continue
        done.add(q)
        loc, mask = q
        seen = tuple(b is not None for b in mask)
        fills = list(itertools.product(*[(b,) if b is not None else (0, 1) for b in mask]))
        for a in range(env.n_act):
            r = {}
            for bits in fills:
                l2 = env.move(loc, a)
                z = (env.emit(l2, bits, rng), MT.key(env, l2, bits, MT.observe(env, l2, seen)))
                r[z] = r.get(z, 0) + 1 / len(fills)
                todo.append(z[1])
            rows_[(q, a)] = r
    return start, rows_, sorted(done, key=str)


def obs_law(R, q, seq):
    front = {(q, ()): 1.0}
    for a in seq:
        nf = {}
        for (s, h), p in front.items():
            for (o, z), w in R[(s, a)].items():
                nf[(z, h + (o,))] = nf.get((z, h + (o,)), 0) + p * w
        front = nf
    return {h for _, h in front}


def repo_table():
    print('exact machines, uniform policy. evidence = (d_u + d_v) JS_w per episode, the smallest')
    print('over separated pairs, at the round they separate (one-step next-state evidence)')
    print('  env              K  d_min    r*  gamma  evidence  noisy rows  noisy windows   N*  N*d_min  N*evid')
    NSTAR = {'hub k=1': 500, 'star k=1': 2000, 'corridor D=1': 50, 'corridor D=2': 200,
             'corridor D=3': 200, 'corridor D=4': 500}
    envs = [('hub k=1', M.KMaze(1), 12), ('star k=1', M.KMaze(1, layout='star'), 12)]
    envs += [(f'corridor D={d}', M.Corridor(d), 10 + 3 * d) for d in (1, 2, 3, 4)]
    for name, env, T in envs:
        start, R, Q = predictive(env)
        A = env.n_act
        dist = dict(start)
        d = {(q, a): 0.0 for q in Q for a in range(A)}
        for _ in range(T):
            new = {}
            for q, p in dist.items():
                for a in range(A):
                    d[(q, a)] += p / A
                    for (o, z), w in R[(q, a)].items():
                        new[z] = new.get(z, 0) + p / A * w
            dist = new
        vis = [v for v in d.values() if v > 0]
        block = {q: 0 for q in Q}
        gam, ev, rounds = np.inf, np.inf, 0

        def law(q, a):
            out = {}
            for (o, z), w in R[(q, a)].items():
                out[(o, block[z])] = out.get((o, block[z]), 0) + w
            return out
        while True:
            sig = {q: (block[q], tuple(tuple(sorted(((str(k), round(v, 9)) for k, v in law(q, a).items())))
                                       for a in range(A))) for q in Q}
            ids = {}
            new = {q: ids.setdefault(sig[q], len(ids)) for q in Q}
            if len(ids) == len(set(block.values())):
                break
            for u, v in itertools.combinations(Q, 2):
                if block[u] == block[v] and new[u] != new[v]:
                    bj, be = 0.0, 0.0
                    for a in range(A):
                        lu, lv = law(u, a), law(v, a)
                        keys = sorted(set(lu) | set(lv), key=str)
                        pu, pv = [lu.get(k, 0) for k in keys], [lv.get(k, 0) for k in keys]
                        du, dv = d[(u, a)], d[(v, a)]
                        bj = max(bj, J(pu, pv))
                        if du + dv > 0:
                            be = max(be, (du + dv) * J(pu, pv, du / (du + dv)))
                    gam, ev = min(gam, bj), min(ev, be)
            block = new
            rounds += 1
        noisy_rows = sum(len({o for o, _ in R[(q, a)]}) > 1 for q in Q for a in range(A))
        noisy_w = max(sum(len(obs_law(R, q, seq)) > 1 for L in (1, 2, 3)
                          for seq in itertools.product(range(A), repeat=L)) for q in Q)
        tot_w = A + A ** 2 + A ** 3
        n = NSTAR[name]
        print(f'  {name:16s} {len(Q):2d} {min(vis):.4f} {rounds:3d} {gam:6.3f} {ev:9.4f} {noisy_rows:6d} '
              f'{noisy_w:8d}/{tot_w:<4d} {n:6d} {n * min(vis):7.1f} {n * ev:7.1f}')


def repo_k2(N=30000, T=20, L=3):
    """Hub k = 2, exact: what the true pooled rows hold for 'knows both' vs 'knows one' at an
    arm, against the same evidence at the scale of one edge. No learner is run."""
    env = M.KMaze(2)
    start, R, Q = predictive(env)
    A = env.n_act
    dist = dict(start)
    d = {(q, a): 0.0 for q in Q for a in range(A)}
    for _ in range(T):
        new = {}
        for q, p in dist.items():
            for a in range(A):
                d[(q, a)] += p / A
                for (o, z), w in R[(q, a)].items():
                    new[z] = new.get(z, 0) + p / A * w
        dist = new

    def law(q, seq):
        front = {(q, ()): 1.0}
        for a in seq:
            nf = {}
            for (s_, h), p in front.items():
                for (o, z), w in R[(s_, a)].items():
                    nf[(z, h + (o,))] = nf.get((z, h + (o,)), 0) + p * w
            front = nf
        res = {}
        for (s_, h), p in front.items():
            res[h] = res.get(h, 0) + p
        return res

    def evidence(nu, nv, u, v):
        """F- from expected counts, depth 1..L windows, and the best single modality."""
        Fneg, best, depth = 0.0, 0.0, None
        for ell in range(1, L + 1):
            for seq in itertools.product(range(A), repeat=ell):
                lu, lv = law(u, seq), law(v, seq)
                keys = sorted(set(lu) | set(lv))
                x = np.array([lu.get(k, 0) for k in keys]) * nu / A ** ell
                y = np.array([lv.get(k, 0) for k in keys]) * nv / A ** ell
                if len(keys) < 2 or lu == lv:
                    continue
                depth = depth or ell
                f = float(dF(x, y, len(keys)))
                Fneg += min(f, 0)
                best = min(best, f)
        return Fneg, best, depth
    arms = [q for q in Q if q[0] > 2]
    print(f'hub k=2, T={T}: {len(Q)} states; (state, action) visits per episode from {min(v for v in d.values() if v > 0):.4f}')
    print('  arm  u (knows both)  v (knows one)   visits u, v at N   first depth  F- pooled  best modality | '
          'back edge visits  F- at that n')
    seen = set()
    for u in arms:
        if None in u[1]:
            continue
        i = (u[0] - 1) % 2
        mask = tuple(b if j == i else None for j, b in enumerate(u[1]))
        v = (u[0], mask)
        if v not in Q or (u[0], mask) in seen:
            continue
        seen.add((u[0], mask))
        nu, nv = N * sum(d[(u, a)] for a in range(A)), N * sum(d[(v, a)] for a in range(A))
        F1, b1, dep = evidence(nu, nv, u, v)
        back = N * d[(u, A - 1)]
        F2, b2, _ = evidence(back, N * d[(v, A - 1)], u, v)
        print(f'  {env.loc_name(u[0]):3s} {str(u[1]):14s} {str(mask):14s} {nu:8.0f} {nv:8.0f} {str(dep):>6s} {F1:12.1f} {b1:9.1f} | '
              f'{back:8.0f} {F2:10.2f}')


def truth_labels(env, trajs):
    import merge_theory as MT
    ids, out = {}, []
    for obs, acts, true in trajs:
        seen = MT.observe(env, 0, (False,) * env.k)
        s = []
        for t in range(len(acts) + 1):
            loc, bits = true[t]
            seen = MT.observe(env, loc, seen)
            s.append(ids.setdefault(MT.key(env, loc, bits, seen), len(ids)))
        out.append(np.array(s))
    return out, ids


def repo_fixed_point():
    import functools
    print('\nshare of 8 seeds where one split round started at the truth creates no state')
    print('  env               N   sum veto  min veto  events off  fixed-horizon events')
    envs = [('hub k=1', M.KMaze(1), 12), ('star k=1', M.KMaze(1, layout='star'), 12)]
    envs += [(f'corridor D={d}', M.Corridor(d), 10 + 3 * d) for d in (1, 3)]
    orig_b, orig_w, orig_e = M.bmr_rows, M.future_windows, M.EVENTS
    conf = (('sum', orig_w, orig_e), ('min', orig_w, orig_e), ('sum', orig_w, 0), ('sum', windows_fixed, orig_e))
    for name, env, T in envs:
        for N in (500, 2000):
            res = []
            for mode, w, e in conf:
                M.bmr_rows, M.future_windows, M.EVENTS = functools.partial(bmr_variant, mode=mode), w, e
                hit = 0
                for s in range(8):
                    trajs = env.sample(N, T, 7000 * s + N)
                    S, ids = truth_labels(env, trajs)
                    hit += M.n_states(M.split_round(trajs, S, env.n_obs, 3)) == len(ids)
                res.append(hit / 8)
            M.bmr_rows, M.future_windows, M.EVENTS = orig_b, orig_w, orig_e
            print(f'  {name:16s} {N:5d} ' + ' '.join(f'{r:8.2f}' for r in res), flush=True)


def repo_variant(var):
    """Whole learner on the six environments under one rule (hours of CPU across rules is not
    needed: about 15 minutes each). var: base, noveto, minveto, minveto_fh."""
    import functools
    import merge_theory as MT
    NS = [50, 120, 300, 500, 800, 1200, 2000, 3000]
    M.bmr_rows = functools.partial(bmr_variant, mode={'base': 'sum', 'noveto': 'none', 'minveto': 'min',
                                                      'minveto_fh': 'min'}[var])
    if var.endswith('_fh'):
        M.future_windows = windows_fixed
    envs = [('hub k=1', M.KMaze(1), 12), ('star k=1', M.KMaze(1, layout='star'), 12)]
    envs += [(f'corridor D={d}', M.Corridor(d), 10 + 3 * d) for d in (1, 2, 3, 4)]
    for name, env, T in envs:
        d_min, truth, _ = MT.coverage(env, T)
        rate = []
        for n in NS:
            hits = 0
            for s in range(8):
                trajs = env.sample(n, T, 1000 * s + n)
                S, _ = M.learn_states(trajs, env.n_obs)
                hits += M.n_states(S) == truth and MT.exact(env, trajs, S)
            rate.append(hits / 8)
        ok = [n for n, r in zip(NS, rate) if r >= .75]
        print(var, name, rate, ok[0] if ok else None, flush=True)


if __name__ == '__main__':
    if '--repo' in sys.argv:
        repo_table()
        repo_k2()
        repo_fixed_point()
        sys.exit()
    if '--variant' in sys.argv:
        repo_variant(sys.argv[sys.argv.index('--variant') + 1])
        sys.exit()
    t0 = time.time()
    stages = (check_lemmas, check_split, check_merge, check_veto_merge, check_veto_split, check_T2,
              check_rate_delta, check_T3, check_near_det, check_T4, check_lower, check_rare, check_edge, corridor_table)
    for f in stages:
        t1 = time.time()
        f()
        print(f'  ({f.__name__}: {time.time() - t1:.1f}s)', flush=True)
    print('\nSummary')
    for name, ok, note in LOG:
        print(f'  {"PASS" if ok else "FAIL"}  {name}  {note}')
    print(f'{sum(o for _, o, _ in LOG)}/{len(LOG)} passed, {time.time() - t0:.0f}s')
