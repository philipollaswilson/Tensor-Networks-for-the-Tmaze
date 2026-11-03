"""Numerical checks for ACTIVE_PAC.md: PAC-style statements for active structure learning.

Every check is a small synthetic case. Tags match the document (A1 ... A17). A check prints
PASS or FAIL with its numbers; `Measured` checks have no theorem behind them and pass if the
stated direction holds. Run:  python src/active_pac_check.py [--only T1,T2,...]

The only code used from the learner is `kmaze.future_windows`, read-only, to test the real
event-window rule. Nothing here changes any other file.
"""
import itertools
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

import numpy as np
from scipy.special import digamma, gammaln, xlogy, zeta
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmaze import future_windows

RES = []
DELTA = 0.05


def check(tag, ok, detail):
    RES.append((tag, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: {detail}", flush=True)


# ---------------------------------------------------------------- Dirichlet quantities
def lndm(x, alpha):
    """Log sequence marginal of a row of counts; x has shape (..., m)."""
    x = np.asarray(x, float)
    m = x.shape[-1]
    a0 = alpha * m
    return (gammaln(a0) - gammaln(a0 + x.sum(-1))
            + (gammaln(alpha + x) - gammaln(alpha)).sum(-1))


def rho(x, alpha):
    """rho(x) = max_theta ln theta^x - ln DM(x), Lemma 1 of MERGE_PAC.md."""
    x = np.asarray(x, float)
    n = x.sum(-1, keepdims=True)
    return xlogy(x, x / np.maximum(n, 1)).sum(-1) - lndm(x, alpha)


def dfe(x, y, alpha):
    return lndm(np.asarray(x) + np.asarray(y), alpha) - lndm(x, alpha) - lndm(y, alpha)


def ent(p):
    return -xlogy(p, p).sum(-1)


def eig(c):
    """Exact I(o; theta) for one draw from Dirichlet(c). Works on (..., m); zero cells are ignored."""
    c = np.asarray(c, float)
    c0 = c.sum(-1, keepdims=True)
    p = c / c0
    return ent(p) - digamma(c0[..., 0] + 1) + (p * digamma(c + 1)).sum(-1)


def jlam_table(p, q, lam, ws=np.linspace(0, 1, 101)):
    """J_lambda(w) = min_theta [w c(p, theta) + (1 - w) c(q, theta)] / lambda for binary rows."""
    th = np.linspace(1e-4, 1 - 1e-4, 4001)

    def c(r):
        return -np.log(r ** (1 - lam) * th ** lam + (1 - r) ** (1 - lam) * (1 - th) ** lam)

    cp, cq = c(p), c(q)
    return ws, (ws[:, None] * cp + (1 - ws[:, None]) * cq).min(1) / lam


def stitched(v, alpha, eta=2.04, s=1.4, m=1.0):
    """Polynomial stitched sub-Gaussian boundary, Howard et al. 2021, Theorem 1 and Eq. (10)."""
    v = np.maximum(v, m)
    ell = s * np.log(np.log(eta * v / m) / np.log(eta)) + np.log(zeta(s) / alpha)
    return (eta ** 0.25 + eta ** -0.25) / np.sqrt(2) * np.sqrt(v * ell)


# ---------------------------------------------------------------- small unifilar machines
def machine(seed=1, Q=4, A=2, O=2):
    r = np.random.default_rng(seed)
    th = r.dirichlet(np.ones(O) * 3, size=(Q, A))
    th = 0.8 * th + 0.2 / O
    tau = r.integers(0, Q, size=(Q, A, O))
    return th, tau


def window_law(th, tau, q, L):
    """P_q^unif over (a1, o1, ..., aL, oL), indexed in base (A O)."""
    Q, A, O = th.shape
    law = np.array([1.0])
    states = np.array([q])
    for _ in range(L):
        nl, ns = [], []
        for w, s in zip(law, states):
            for a in range(A):
                for o in range(O):
                    nl.append(w * th[s, a, o] / A)
                    ns.append(tau[s, a, o])
        law, states = np.array(nl), np.array(ns)
    return law


def step(th, tau, q, a, r):
    o = (r.random(len(q)) > th[q, a, 0]).astype(int)   # binary observations
    return o, tau[q, a, o]


def uniform_windows(th, tau, q, L, r):
    code = np.zeros(len(q), np.int64)
    A, O = th.shape[1], th.shape[2]
    for _ in range(L):
        a = r.integers(0, A, len(q))
        o, q = step(th, tau, q, a, r)
        code = code * (A * O) + a * O + o
    return code


# ================================================================= T1: tail validity
def t1a_tail_exact(r):
    """A1: exact state, adaptive prefix, stopping-time start: window law equals the uniform law."""
    th, tau = machine(1)
    Q, L, T = 4, 3, 10
    cells = 4 ** L
    tab = np.zeros((Q, cells))
    bias = 0.5
    for batch in range(10):
        M = 40000
        q = np.zeros(M, int)
        ob = np.zeros(M, int)
        cnt = np.zeros(M, int)
        live = np.ones(M, bool)
        ones = 0
        for t in range(T):
            ix = np.flatnonzero(live)
            if len(ix) == 0:
                break
            # adaptive across steps (parity of the past) and across batches (bias from past data)
            a = np.where(r.random(len(ix)) < bias, (ob[ix] + cnt[ix]) % 2, r.integers(0, 2, len(ix)))
            o, q2 = step(th, tau, q[ix], a, r)
            q[ix], ob[ix] = q2, o
            cnt[ix] += o
            ones += o.sum()
            go = (o == 1) | (t >= 3)            # start time depends on the past only
            s = ix[go]
            if len(s):
                codes = uniform_windows(th, tau, q[s].copy(), L, r)
                np.add.at(tab, (q[s], codes), 1)
                live[s] = False
        bias = 0.3 + 0.6 * ones / max(ones + 1000, 1)
    ps, tvs = [], []
    for s in range(Q):
        n = tab[s].sum()
        if n < 2000:
            continue
        e = window_law(th, tau, s, L) * n
        x2 = ((tab[s] - e) ** 2 / e).sum()
        ps.append(chi2.sf(x2, cells - 1))
        tvs.append(0.5 * np.abs(tab[s] / n - e / n).sum())
    pmin = min(ps) * len(ps)
    check('T1a A1 tail law, exact state', pmin > 1e-3,
          f"{len(ps)} states, chi2 Bonferroni p = {min(pmin, 1):.3f}, TV to uniform law {max(tvs):.4f}, n = {int(tab.sum())}")


def t1b_closed_loop(r):
    """A1(b): action-conditioned rows need open-loop tails. Event-type rows need the uniform law."""
    th, tau = machine(2)
    q0, M = 1, 400000
    exact = np.zeros((4, 4))
    for a1, a2 in itertools.product(range(2), repeat=2):
        for o1, o2 in itertools.product(range(2), repeat=2):
            exact[a1 * 2 + a2, o1 * 2 + o2] = th[q0, a1, o1] * th[tau[q0, a1, o1], a2, o2]

    def rows(a1, policy):
        q = np.full(M, q0)
        o1, q = step(th, tau, q, a1, r)
        a2 = policy(o1)
        o2, _ = step(th, tau, q, a2, r)
        t = np.zeros((4, 4))
        np.add.at(t, (a1 * 2 + a2, o1 * 2 + o2), 1)
        return t

    a1 = r.integers(0, 2, M)
    closed = rows(a1, lambda o: np.where(r.random(M) < 0.9, o, 1 - o))
    tv_c = 0.5 * np.abs(closed / closed.sum(1, keepdims=True) - exact).sum(1).max()
    pair = r.choice(4, M, p=[0.7, 0.1, 0.1, 0.1])            # open loop, not uniform
    a1 = pair // 2
    open_ = rows(a1, lambda o: pair % 2)
    nt = open_.sum(1, keepdims=True)
    tv_o = 0.5 * np.abs(open_ / nt - exact).sum(1).max()
    x2 = ((open_ - nt * exact) ** 2 / (nt * exact)).sum()
    p = chi2.sf(x2, 12)
    marg_o = open_.sum(0) / open_.sum()
    marg_u = exact.mean(0)
    tv_m = 0.5 * np.abs(marg_o - marg_u).sum()
    check('T1b A1b open loop valid, closed loop not', tv_o < 0.01 and p > 1e-3 and tv_c > 0.05,
          f"TV of action-conditioned row: closed loop {tv_c:.3f}, open loop {tv_o:.4f} (chi2 p {p:.3f}); "
          f"action-marginal row under open non-uniform tail: TV {tv_m:.3f} from the uniform law")
    return tv_m


def t1c_impure(r):
    """A2: label merges two predictive states. Tail and one-step rows move with the prefix."""
    p_x, p_y, M = 0.8, 0.2, 400000

    def run(p_go_y):
        y = r.random(M) < p_go_y
        o = r.random(M) < np.where(y, p_y, p_x)
        return y.mean(), o.mean()

    w_base, f_base = run(0.02)
    w_dir, f_dir = run(1.0)
    pred_base = 0.02 * p_y + 0.98 * p_x
    ok = abs(f_base - pred_base) < 0.005 and abs(f_dir - p_y) < 0.005
    check('T1c A2 impure label, weights move', ok and abs(f_base - f_dir) > 0.3,
          f"p(o = 1 | label, a): base prefix {f_base:.3f} (formula {pred_base:.3f}), directed prefix {f_dir:.3f}. "
          f"Impurity under base 0.02, under directed 1.00. So one-step counts are not policy-free on an impure label")


def t1d_events(r):
    """A3(c): the real event window is censored by the episode end, so its law moves with the start."""
    T, R, n = 20, 200, 100

    def episode(t0):
        obs = [0] * (T + 1)
        if r.random() < 0.5:
            obs[t0 + 1] = 1
            for u in range(t0 + 2, T + 1):
                obs[u] = 1
        else:
            for u in range(t0 + 7, T + 1):
                obs[u] = 2
        return obs

    def row(t0):
        cnt = {1: 0, 2: 0}
        fixed = {1: 0, 2: 0, 'none': 0}
        for _ in range(n):
            obs = episode(t0)
            acts = [0] * T
            items = [('tail' if t == t0 else None) for t in range(T)]
            for it, key in future_windows(items, obs, acts, 3, tail=t0):
                if it == 'tail' and key[0] == ('event', 1):
                    cnt[key[1][0]] += 1
            ch = [u for u in range(t0 + 1, min(t0 + 5, T + 1)) if obs[u] != obs[u - 1]]
            fixed[obs[ch[0]] if ch else 'none'] += 1
        return np.array([cnt[1], cnt[2]]), np.array([fixed[1], fixed[2], fixed['none']])

    veto = veto_f = 0
    for _ in range(R):
        a, af = row(5)
        b, bf = row(16)
        veto += dfe(a, b, 0.5) < -3
        veto_f += dfe(af, bf, 1 / 3) < -3
    check('T1d A3c censored events split one state', veto / R > 0.9 and veto_f / R < 0.1,
          f"kmaze.future_windows, one true state, tail start t = 5 against t = 16 of T = 20, n = {n}: "
          f"P(dF < -3) = {veto / R:.2f}; fixed horizon 4 with value none: {veto_f / R:.2f}")


def t1e_stop(r):
    """A3(b): stopping inside the tail and keeping only windows that fit."""
    M, p = 400000, 0.5
    o1 = r.random(M) < p
    o2 = r.random(M) < p
    keep = ~o1                         # episode ends on o = 1, so a 2-window fits only if o1 = 0
    f_drop = o1[keep].mean()
    f_full = o1.mean()
    # the same with an end that is fixed before the window starts
    pre = r.random(M) < 0.5
    f_pre = o1[pre].mean()
    check('T1e A3b stop inside tail biases windows', f_drop == 0 and abs(f_pre - p) < 0.005,
          f"P(o1 = 1), truth {p}: all windows {f_full:.3f}, windows kept when the end depends on them {f_drop:.3f}, "
          f"end fixed in advance {f_pre:.3f}")


def t1f_refit(r):
    """A3(d): labels fitted on the data that tests them."""
    R, K, n = 2000, 32, 50
    x = r.binomial(n, 0.5, (R, K))
    x2 = r.binomial(n, 0.5, (R, K))
    order = np.argsort(x, 1)
    lo, hi = order[:, :K // 2], order[:, K // 2:]
    rows = lambda d, ix: np.take_along_axis(d, ix, 1).sum(1)
    n_half = n * K // 2
    a, b = rows(x, lo), rows(x, hi)
    same = dfe(np.c_[a, n_half - a], np.c_[b, n_half - b], 0.5)
    a2, b2 = rows(x2, lo), rows(x2, hi)
    held = dfe(np.c_[a2, n_half - a2], np.c_[b2, n_half - b2], 0.5)
    check('T1f A3d refit on same data', (same < -3).mean() > 0.9 and (held < -3).mean() < 0.05,
          f"{K} items of one true state split by their own rows: P(dF < -3) on the same data {(same < -3).mean():.2f}, "
          f"on held-out data {(held < -3).mean():.3f}")


def t1g_overlap(r):
    """A3(e): overlapping windows have the right marginal but are not independent draws."""
    R, n, L = 4000, 300, 3

    def rows(seq):
        c = seq[:, :-2] * 4 + seq[:, 1:-1] * 2 + seq[:, 2:]
        t = np.zeros((seq.shape[0], 8))
        np.add.at(t, (np.repeat(np.arange(seq.shape[0]), c.shape[1]), c.ravel()), 1)
        return t

    def nonoverlap(seq):
        c = seq[:, ::3] * 4 + seq[:, 1::3] * 2 + seq[:, 2::3]
        t = np.zeros((seq.shape[0], 8))
        np.add.at(t, (np.repeat(np.arange(seq.shape[0]), c.shape[1]), c.ravel()), 1)
        return t

    s1 = r.integers(0, 2, (R, n + 2))
    s2 = r.integers(0, 2, (R, n + 2))
    ov = dfe(rows(s1), rows(s2), 1 / 8)
    s3 = r.integers(0, 2, (R, 3 * n))
    s4 = r.integers(0, 2, (R, 3 * n))
    nv = dfe(nonoverlap(s3), nonoverlap(s4), 1 / 8)
    a, b = (ov < -3).mean(), (nv < -3).mean()
    var_o = rows(s1)[:, 0].var()
    var_n = nonoverlap(s3)[:, 0].var()
    check('T1g A3e overlap inflates variance', var_o > 1.2 * var_n,
          f"same number of windows n = {n}, equal laws: var(count of a cell) overlap {var_o:.1f}, non-overlap {var_n:.1f}; "
          f"P(dF < -3): overlap {a:.3f}, non-overlap {b:.3f}")


# ================================================================= T2: adaptive concentration
def t2a_ville(r):
    """A5, A7: the mixture martingale and the stitched band survive an adversary. A fixed-n test does not."""
    R, Tm = 1500, 1500
    for m, tag in ((2, 'm=2'), (4, 'm=4')):
        p = np.ones(m) / m
        alpha = 1 / m
        x = np.zeros((R, m))
        y = np.zeros((R, m))
        cr_m = np.zeros(R, bool)
        cr_v = np.zeros(R, bool)
        cr_h = np.zeros(R, bool)
        cr_s = np.zeros(R, bool)
        cr_z = np.zeros(R, bool)
        cum = np.cumsum(p)
        ix = np.arange(R)
        for t in range(Tm):
            n1, n2 = x.sum(1), y.sum(1)
            d1 = np.abs(x[:, 0] / np.maximum(n1, 1) - p[0])
            d2 = np.abs(y[:, 0] / np.maximum(n2, 1) - p[0])
            # keep the row that looks more extreme frozen, within a ratio of 3 of the counts
            pick_y = n2 < n1
            late = (n1 >= 10) & (n2 >= 10)
            pick_y = np.where(late & (d1 > d2) & (n2 < 3 * n1), True, pick_y)
            pick_y = np.where(late & (d2 > d1) & (n1 < 3 * n2), False, pick_y)
            o = (r.random(R)[:, None] > cum).sum(1)
            x[ix[~pick_y], o[~pick_y]] += 1
            y[ix[pick_y], o[pick_y]] += 1
            z = x + y
            lnm = lndm(x, alpha) + lndm(y, alpha) - (z * np.log(p)).sum(1)
            cr_m |= lnm >= np.log(1 / DELTA)
            cr_v |= -dfe(x, y, alpha) >= rho(z, alpha) + np.log(1 / DELTA)
            if m == 2:
                n1, n2 = np.maximum(x.sum(1), 1), np.maximum(y.sum(1), 1)
                gap = np.abs(x[:, 0] / n1 - y[:, 0] / n2)
                sd = np.sqrt((1 / n1 + 1 / n2) / 4)
                ok = (n1 >= 10) & (n2 >= 10)
                cr_z |= ok & (gap >= 1.96 * sd)
                cr_h |= ok & (gap >= sd * np.sqrt(2 * np.log(2 / DELTA)))
                cr_s |= gap >= stitched(n1 / 4, DELTA / 4) / n1 + stitched(n2 / 4, DELTA / 4) / n2
        check(f'T2a A5 anytime bounds, adversary, {tag}', cr_m.mean() <= DELTA and cr_v.mean() <= DELTA,
              f"P(ever ln M >= ln(1/delta)) = {cr_m.mean():.4f}, P(ever -dF >= rho(z) + ln(1/delta)) = {cr_v.mean():.4f}, delta {DELTA}")
        if m == 2:
            check('T2a A7 stitched against fixed-n tests', cr_s.mean() <= DELTA and cr_z.mean() > 2 * DELTA,
                  f"same adversary, P(ever cross): 5% z test {cr_z.mean():.3f}, fixed-n Hoeffding {cr_h.mean():.3f}, stitched {cr_s.mean():.4f}")


def t2b_merge(r):
    """A6: anytime wrong-merge bound at adaptive w."""
    R, lam = 600, 0.4
    for p, q, Tm in ((0.65, 0.35, 1400), (0.8, 0.2, 300)):
        ws, jt = jlam_table(p, q, lam)
        x = np.zeros((R, 2))
        y = np.zeros((R, 2))
        viol = np.zeros(R, bool)
        cert = np.full(R, -1)
        last = np.full(R, -1)
        for t in range(Tm):
            n1, n2 = x.sum(1), y.sum(1)
            f1 = x[:, 0] / np.maximum(n1, 1)
            f2 = y[:, 0] / np.maximum(n2, 1)
            pick_y = (f1 - f2 > p - q) ^ (t % 3 == 0)       # depends on the data
            o = r.random(R) > np.where(pick_y, q, p)
            ix = np.arange(R)
            x[ix[~pick_y], o[~pick_y].astype(int)] += 1
            y[ix[pick_y], o[pick_y].astype(int)] += 1
            z = x + y
            n = z.sum(1)
            w = x.sum(1) / n
            rhs = (rho(x, 0.5) + rho(y, 0.5) + (1 - lam) / lam * rho(z, 0.5) + np.log(1 / DELTA) / lam
                   - n * np.interp(w, ws, jt))
            d = dfe(x, y, 0.5)
            viol |= d > rhs
            sig = rho(z, 0.5) + np.log(1 / DELTA)
            newly = (cert < 0) & (rhs + sig < 0)
            cert[newly] = t + 1
            last[d >= -sig] = t + 1
        ok = viol.mean() <= DELTA and (cert > 0).all()
        ratio = np.median(cert[cert > 0]) / max(np.median(last), 1)
        check(f'T2b A6 anytime wrong merge, p={p}, q={q}', ok,
              f"P(ever dF > bound) = {viol.mean():.4f}; certified at n = {np.median(cert):.0f}, "
              f"rule last accepts at n = {np.median(last):.0f}, tightness {ratio:.1f}x")


def t2c_stack(r):
    """A4 and A8: first n0 rows do not see the policy. The fixed rule tested at every n is measured."""
    R = 8000
    for n0 in (30, 100, 300):
        x = np.zeros((R, 2))
        y = np.zeros((R, 2))
        ix = np.arange(R)
        for _ in range(2 * n0):
            n1, n2 = x.sum(1), y.sum(1)
            f1 = x[:, 0] / np.maximum(n1, 1)
            f2 = y[:, 0] / np.maximum(n2, 1)
            pick_y = ((np.abs(f1 - .5) > np.abs(f2 - .5)) & (n2 < n0)) | (n1 >= n0)
            o = (r.random(R) > 0.5).astype(int)
            x[ix[~pick_y], o[~pick_y]] += 1
            y[ix[pick_y], o[pick_y]] += 1
        a = (dfe(x, y, 0.5) < 0).mean()
        b1 = r.binomial(n0, 0.5, 200000)
        b2 = r.binomial(n0, 0.5, 200000)
        b = (dfe(np.c_[b1, n0 - b1], np.c_[b2, n0 - b2], 0.5) < 0).mean()
        s_ = np.sqrt(b * (1 - b) / R) * 3
        u = np.c_[r.binomial(n0, 0.5, 2 * R), np.zeros(2 * R)]
        v = np.c_[r.binomial(n0, 0.5, 2 * R), np.zeros(2 * R)]
        u[:, 1] = n0 - u[:, 0]
        v[:, 1] = n0 - v[:, 0]
        ever = dfe(u, v, 0.5) < 0
        stepn = max(1, n0 // 30)                       # tested every stepn draws per row
        for _ in range(7 * n0 // stepn):
            for w in (u, v):
                k = r.binomial(stepn, .5, 2 * R)
                w[:, 0] += k
                w[:, 1] += stepn - k
            ever |= dfe(u, v, 0.5) < 0
        check(f'T2c A4 stack rows, n0={n0}', abs(a - b) < s_ + 1e-3,
              f"P(dF < 0) at n1 = n2 = n0: adaptive reading {a:.4f}, iid rows {b:.4f}. "
              f"P(ever dF < 0 over n0..8n0) = {ever.mean():.4f}, ratio to fixed n {ever.mean() / max(b, 1e-4):.1f}x, "
              f"times sqrt(n0): {ever.mean() * np.sqrt(n0):.2f}")


def bisim(cnt, tau, alpha, dS):
    """Moore refinement, pair tests with an anytime threshold, components of allowed pairs."""
    K, A, O = cnt.shape
    part = np.zeros(K, int)
    for _ in range(K + 1):
        nb = part.max() + 1
        allow = np.eye(K, dtype=bool)
        for u, v in itertools.combinations(range(K), 2):
            if part[u] != part[v]:
                continue
            good = True
            for a in range(A):
                x = np.zeros(O * nb)
                y = np.zeros(O * nb)
                for o in range(O):
                    x[o * nb + part[tau[u, a, o]]] += cnt[u, a, o]
                    y[o * nb + part[tau[v, a, o]]] += cnt[v, a, o]
                al = 1 / (O * K)
                if dfe(x, y, al) < -(rho(x + y, al) + np.log(1 / dS)):
                    good = False
                    break
            allow[u, v] = allow[v, u] = good
        lab = -np.ones(K, int)
        c = 0
        for s in range(K):
            if lab[s] >= 0:
                continue
            stack, lab[s] = [s], c
            while stack:
                i = stack.pop()
                for j in np.flatnonzero(allow[i] & (lab < 0)):
                    lab[j] = c
                    stack.append(j)
            c += 1
        new = np.zeros(K, int)
        seen = {}
        for s in range(K):
            new[s] = seen.setdefault((part[s], lab[s]), len(seen))
        if (new == part).all() or len(seen) == nb:
            part = new
            break
        part = new
    return part


def t2d_recovery(r):
    """A9: exact recovery under adaptive collection on a ladder. The certificate uses realised counts."""
    p = np.array([.05, .95, .2, .8, .35, .65, .65])     # state 6 is a copy of state 5
    K, A, O, T = 7, 2, 2, 12
    tau = np.zeros((K, A, O), int)
    for v in range(K):
        tau[v, 1, :] = v + 1 if v < 5 else (6 if v == 5 else 5)
    lam = 0.4
    pairs = [(u, v) for u, v in itertools.combinations(range(K), 2) if p[u] != p[v]]
    dM = DELTA / (2 * len(pairs) * 2)
    dS = DELTA / (2 * A)
    tables = {pr: jlam_table(p[pr[0]], p[pr[1]], lam) for pr in pairs}
    al = 1 / (O * K)

    def cert(cnt, todo):
        left = []
        for pr in todo:
            u, v = pr
            done = False
            for a in range(A):
                x, y = cnt[u, a], cnt[v, a]
                n = x.sum() + y.sum()
                if x.sum() < 1 or y.sum() < 1:
                    continue
                ws, jt = tables[pr]
                need = (rho(x, al) + rho(y, al) + rho(x + y, al) / lam
                        + np.log(1 / dM) / lam + np.log(1 / dS))
                if n * np.interp(x.sum() / n, ws, jt) > need:
                    done = True
                    break
            if not done:
                left.append(pr)
        return left

    def uniform(cnt, M):
        v = np.zeros(M, int)
        for _ in range(T):
            a = r.integers(0, 2, M)
            o = (r.random(M) < p[v]).astype(int)
            np.add.at(cnt, (v, a, o), 1)
            v = tau[v, a, o]

    def directed(cnt, target, act):
        v = 0
        rs = r.random(T)
        for t in range(T):
            if v == target:
                a = act
            elif target >= 5 and v >= 5:           # toggle between the two top states
                a = 1
            else:
                a = 1 if v < target else 0
            o = int(rs[t] < p[v])
            cnt[v, a, o] += 1
            v = tau[v, a, o]

    def good(cnt):
        lab = bisim(cnt, tau, al, dS)
        return lab[5] == lab[6] and len(set(lab)) == 6

    grid = [50 * 2 ** k for k in range(13)]

    def run(mode, reps=6):
        Ns, wins = [], 0
        curve = {g: 0 for g in grid}
        for rep_i in range(reps):
            cnt = np.zeros((K, A, O))
            todo, N, stop = list(pairs), 0, None
            for g in grid:
                while N < g:
                    if mode == 'uniform':
                        uniform(cnt, g - N)
                        N = g
                    else:
                        for _ in range(min(25, g - N)):
                            if mode == 'count':
                                ix = np.unravel_index(np.argmin(cnt.sum(2)), (K, A))
                                directed(cnt, int(ix[0]), int(ix[1]))
                            elif todo:
                                u, v = min(todo, key=lambda pr: cnt[pr[0]].sum() + cnt[pr[1]].sum())
                                directed(cnt, u if cnt[u].sum() <= cnt[v].sum() else v, 1)
                        N += min(25, g - N)
                    todo = cert(cnt, todo)
                    if not todo and stop is None:
                        stop = N
                        wins += good(cnt)
                curve[g] += good(cnt)
                if stop is not None:
                    for later in grid:           # the rule is checked at grid points up to the stop only
                        if later > g:
                            curve[later] += 1
                    break
            Ns.append(stop if stop else np.inf)
        need = next((g for g in grid if curve[g] >= reps - 1), None)
        return float(np.median(Ns)), wins / reps, need

    res = {m: run(m) for m in ('uniform', 'count', 'pairs')}
    u, c, d = res['uniform'], res['count'], res['pairs']
    check('T2d A9 exact recovery at certificate', u[1] == 1 and c[1] == 1 and d[1] == 1 and np.isfinite(u[0]),
          f"ladder, 7 states. N at certificate: uniform {u[0]:.0f}, count-directed {c[0]:.0f}, pair-directed (knows gaps) {d[0]:.0f}; "
          f"gain {u[0] / c[0]:.1f}x and {u[0] / d[0]:.1f}x. Exact at certificate {u[1]:.2f}, {c[1]:.2f}, {d[1]:.2f}. "
          f"Same rule first right on the grid at N = {u[2]}, {c[2]}, {d[2]}: certificate loose by {u[0] / u[2]:.0f}x, {c[0] / c[2]:.0f}x")


def t2e_lower(r):
    """A10: no adaptive learner beats ln(1/(2.4 delta)) / KL per row; SPRT is near it."""
    R = 40000
    for g in (0.2, 0.5):
        p1, p0 = .5 + g / 2, .5 - g / 2
        kl = p1 * np.log(p1 / p0) + p0 * np.log(p0 / p1)
        for d in (1e-1, 1e-2, 1e-3):
            b = np.log((1 - d) / d)
            pos = np.zeros(R)
            n = np.zeros(R)
            live = np.ones(R, bool)
            inc = np.log(p1 / p0)
            while live.any():
                k = live.sum()
                pos[live] += np.where(r.random(k) < p1, inc, -inc)
                n[live] += 1
                live &= np.abs(pos) < b
            err = (pos < 0).mean()
            floor = np.log(1 / (2.4 * max(err, 1e-9))) / kl
            ok = n.mean() >= floor
            check(f'T2e A10 floor, g={g}, delta={d}', ok,
                  f"E[n] SPRT {n.mean():.1f}, realised error {err:.5f}, floor at that error {floor:.1f}, ratio {n.mean() / floor:.2f}")


def kl_tree(th, tau, u, v, L, chooser):
    """KL(P_u || P_v) over L steps under a policy on histories; chooser(h) gives action probs."""
    def rec(a_, b_, h, t):
        if t == L:
            return 0.0
        tot = 0.0
        pi = chooser(h)
        for a, w in enumerate(pi):
            if w == 0:
                continue
            for o in range(th.shape[2]):
                pu, pv = th[a_, a, o], th[b_, a, o]
                if pu == 0:
                    continue
                tot += w * pu * (np.log(pu / pv) + rec(tau[a_, a, o], tau[b_, a, o], h + ((a, o),), t + 1))
        return tot
    return rec(u, v, (), 0)


def kl_best(th, tau, u, v, L):
    """max over closed-loop policies: dynamic programming on the pair (u, v)."""
    A, O = th.shape[1], th.shape[2]
    V = np.zeros((th.shape[0], th.shape[0]))
    for _ in range(L):
        Q = np.zeros(V.shape + (A,))
        for s, w in itertools.product(range(th.shape[0]), repeat=2):
            for a in range(A):
                for o in range(O):
                    pu, pv = th[s, a, o], th[w, a, o]
                    if pu > 0:
                        Q[s, w, a] += pu * (np.log(pu / pv) + V[tau[s, a, o], tau[w, a, o]])
        V = Q.max(-1)
    return V[u, v]


def lock(D, A=2):
    """Two copies of a lock: only the right sequence of D actions reaches the distinguishing bit."""
    S = 2 * (D + 2)                                       # (copy, position), last = sink
    th = np.full((S, A, 2), 0.5)
    tau = np.zeros((S, A, 2), int)
    for c in range(2):
        for i in range(D + 2):
            s = c * (D + 2) + i
            sink = c * (D + 2) + D + 1
            for a in range(A):
                tau[s, a, :] = sink
                if i < D and a == 1:
                    tau[s, a, :] = s + 1
    g = 0.4
    for c, sign in ((0, 1), (1, -1)):
        s = c * (D + 2) + D
        th[s, 1, 1] = 0.5 + sign * g / 2                  # the bit shows under one action only
        th[s, 1, 0] = 0.5 - sign * g / 2
    return th, tau


def t2f_gain(r):
    """A11: KL^pi <= A^L KL^unif for any closed-loop pi, tight on a lock; hitting has the same cap."""
    worst = 0
    for s in range(120):
        th, tau = machine(100 + s)
        L = 3
        u, v = 0, 1
        uni = kl_tree(th, tau, u, v, L, lambda h: [.5, .5])
        best = kl_best(th, tau, u, v, L)
        worst = max(worst, best / max(uni, 1e-12))
    D = 3
    th, tau = lock(D)
    u_, v_ = 0, D + 2
    uni = kl_tree(th, tau, u_, v_, D + 1, lambda h: [.5, .5])
    best = kl_best(th, tau, u_, v_, D + 1)
    # hitting: enumerate on random machines
    hit_worst = 0
    for s in range(120):
        th2, tau2 = machine(300 + s)
        Dh, target = 3, 3

        def hit(q, t, pol, h):
            if q == target:
                return 1.0
            if t == Dh:
                return 0.0
            tot = 0
            for a, w in enumerate(pol(h)):
                if w:
                    for o in range(2):
                        tot += w * th2[q, a, o] * hit(tau2[q, a, o], t + 1, pol, h + ((a, o),))
            return tot

        pu = hit(0, 0, lambda h: [.5, .5], ())
        rs = np.random.default_rng(s)
        book = {}

        def pol(h):
            if h not in book:
                book[h] = [1, 0] if rs.random() < .5 else [0, 1]
            return book[h]

        pp = hit(0, 0, pol, ())
        if pu > 0:
            hit_worst = max(hit_worst, pp / pu)
    check('T2f A11 cap on distinguishability gain', worst <= 8 + 1e-9 and abs(best / uni - 2 ** (D + 1)) < 1e-6 and hit_worst <= 8 + 1e-9,
          f"120 random machines, L = 3: largest KL* / KL_unif = {worst:.2f} <= 8; lock D = {D}: "
          f"KL* / KL_unif = {best / uni:.3f}, exactly the cap 2^{D + 1}; largest hit ratio over random policies {hit_worst:.2f} <= 8")
    return best / uni


def t2g_lock(r):
    """A12: the form 1/(d mu^2) overstates the gain from a test: A^2L in the form, A^L in the information."""
    A, D, L, g = 2, 3, 3, 0.4
    p1, p0 = .5 + g / 2, .5 - g / 2
    kl = p1 * np.log(p1 / p0) + p0 * np.log(p0 / p1)
    R = 4000
    b = np.log((1 - DELTA) / DELTA)
    pos = np.zeros(R)
    n = np.zeros(R)
    live = np.ones(R, bool)
    while live.any():
        k = live.sum()
        pos[live] += np.where(r.random(k) < p1, 1, -1) * np.log(p1 / p0)
        n[live] += 1
        live &= np.abs(pos) < b
    rows = {'uniform prefix, uniform tail': A ** -(D + L), 'directed prefix, uniform tail': A ** -L,
            'directed prefix, open-loop test': 1.0}
    out, N = [], {}
    for k, d in rows.items():
        m = r.negative_binomial(n.astype(int), d)
        N[k] = (m + n).mean()
        out.append(f"{k}: {N[k]:.0f}")
    formal = {'uniform prefix, uniform tail': A ** (D + 2 * L), 'directed prefix, uniform tail': A ** (2 * L),
              'directed prefix, open-loop test': 1.0}
    g1 = N['uniform prefix, uniform tail'] / N['directed prefix, open-loop test']
    g2 = N['directed prefix, uniform tail'] / N['directed prefix, open-loop test']
    fg = formal['uniform prefix, uniform tail'] / formal['directed prefix, open-loop test']
    check('T2g A12 gains on a lock', abs(g1 / A ** (D + L) - 1) < 0.15 and abs(g2 / A ** L - 1) < 0.15,
          f"episodes to a decision: {'; '.join(out)}. Measured gain {g1:.0f}x (A^(D+L) = {A ** (D + L)}), "
          f"test alone {g2:.1f}x (A^L = {A ** L}); the form 1/(d mu^2) predicts {fg:.0f}x and {A ** (2 * L)}x")


def t2h_unknown_lock(r):
    """A11(f): planning repeats a known path. It cannot shorten the search for an unknown one."""
    A, R = 2, 200000
    rows = []
    for D, need in ((6, 50), (10, 50), (6, 5000)):
        p = A ** -D
        uni = (r.negative_binomial(need, p, R) + need).mean()
        act = (r.integers(1, A ** D + 1, R) + need - 1).mean()
        rows.append((D, need, uni, act, uni / act, min(2 * need, A ** D)))
    ok = all(g <= 1.02 * c for *_, g, c in rows)
    check('T2h A11f unknown lock caps the gain at min(2 n*, A^D)', ok,
          '; '.join(f"D = {d}, n* = {n}: uniform {u:.0f}, search then repeat {a:.0f}, gain {g:.1f}x, min(2 n*, A^D) = {c}"
                    for d, n, u, a, g, c in rows))


# ================================================================= T3: novelty
def t3a_eig(r):
    """A13(a): exact EIG against the two-term expansion."""
    err = []
    for n in (200, 2000, 20000):
        for m, s in ((2, 2), (4, 4), (8, 8), (8, 3), (8, 1), (2, 1), (16, 2)):
            al = 1 / m
            c = np.full(m, al)
            c[:s] += n / s
            ex = eig(c)
            ap = (s - 1) / (2 * c.sum()) + (m - s) * al / c.sum() * (np.log(1 / al) + digamma(1 + al))
            err.append(abs(ex - ap) / ex)
    naive = eig(np.array([2000 + .125, .125, .125, .125, .125, .125, .125, .125])) * 2 * 2000 / 7
    check('T3a A13a EIG expansion', max(err) < 0.1,
          f"largest relative error {max(err):.3f} on 21 cases; a deterministic row, m = 8, has {naive:.2f} of (m - 1)/(2n)")


def t3b_alloc(r):
    """A13(b): greedy EIG balances (s - 1) / n, whatever the gaps."""
    ss = [2, 3, 5, 8]
    M = 8
    c = np.full((4, M), 1 / M)
    masks = [np.arange(s) for s in ss]
    n = np.zeros(4)
    for t in range(4000):
        sc = eig(c)
        i = int(np.argmax(sc))
        o = r.choice(masks[i])
        c[i, o] += 1
        n[i] += 1
    al = 1 / M
    coef = (np.array(ss) - 1) / 2 + (M - np.array(ss)) * al * (np.log(1 / al) + digamma(1 + al))
    pred = coef / coef.sum()
    bare = (np.array(ss) - 1) / (np.array(ss) - 1).sum()
    got = n / n.sum()
    check('T3b A13b novelty allocates by a(s), not by gaps', np.abs(got - pred).max() < 0.02,
          f"shares {np.round(got, 3)}, a(s) shares {np.round(pred, 3)}; plain (s - 1) shares would be {np.round(bare, 3)}")


def route_sim(rng, A, B, policy, R=100, pool=None, max_steps=14000):
    """Edges with two possible targets, rows A[e], B[e]. Every policy has the same stopping rule and
    no policy is barred from a decided edge, except `elim`, whose whole point is that bar."""
    E, M = A.shape
    lab = rng.integers(0, 2, (R, E))
    Hs = ent(A), ent(B)
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.where((A > 0) & (B > 0), np.log(A) - np.log(B), 0.0)
    alpha = (A + B > 0) / np.maximum((A + B > 0).sum(1, keepdims=True), 1)
    c = np.broadcast_to(alpha, (R, E, M)).copy()
    L = np.zeros((R, E))
    n = np.zeros((R, E))
    d = DELTA / E
    thr = np.log((1 - d) / d)
    live = np.ones(R, bool)
    steps = np.zeros(R)
    gaps = ((A * lr).sum(1) + (B * -lr).sum(1)) / 2
    cum = np.cumsum(np.stack([B, A], 0), -1)
    ix = np.arange(R)
    ms = (alpha > 0).sum(1)
    for t in range(max_steps):
        if not live.any():
            break
        pi = 1 / (1 + np.exp(-np.clip(L, -300, 300)))
        if policy == 'count':
            s = -n
        elif policy == 'elim':
            s = np.where(np.abs(L) >= thr, -np.inf, -n)
        elif policy == 'bound':
            s = np.exp(-n * gaps) * (1 - np.exp(-gaps))
        elif policy == 'novelty':
            s = eig(c)
        elif policy == 'proxy':                   # src/active_k2.py: H(p_merge) / (n_e + 1)
            q = np.clip(pi, 1e-12, 1 - 1e-12)
            s = -(q * np.log(q) + (1 - q) * np.log(1 - q)) / (n + 1)
        else:
            mix = pi[..., None] * A + (1 - pi[..., None]) * B
            st = ent(mix) - pi * Hs[0] - (1 - pi) * Hs[1]
            if policy == 'struct':
                s = st
            elif policy == 'sum':                 # per-edge Dirichlet term: wrong conditioning
                s = st + eig(c)
            elif policy == 'chain':               # term 2 given the structure: the pooled target row
                s = st + (ms - 1) / (2 * pool)
        s = s + 1e-12 * rng.random(s.shape)
        e = s.argmax(1)
        el = lab[ix, e]
        o = np.minimum((rng.random(R)[:, None] > cum[el, e]).sum(1), M - 1)
        L[live, e[live]] += lr[e[live], o[live]]
        c[ix[live], e[live], o[live]] += 1
        n[live, e[live]] += 1
        steps[live] += 1
        live &= (np.abs(L) < thr).any(1)
    wrong = ((L > 0).astype(int) != lab).any(1).mean()
    return steps.mean(), wrong, n.mean(0), live.mean()


def bern_rows(gs, noise=None):
    E = len(gs)
    M = max(noise) * 2 if noise else 2
    A = np.zeros((E, M))
    B = np.zeros((E, M))
    for e, g in enumerate(gs):
        k = noise[e] if noise else 1
        A[e, :k] = (.5 + g / 2) / k
        A[e, k:2 * k] = (.5 - g / 2) / k
        B[e, :k] = (.5 - g / 2) / k
        B[e, k:2 * k] = (.5 + g / 2) / k
    return A, B


def t3c_compare(r):
    """A14, A15: equal gaps, unequal gaps, unequal noise, one stopping rule for all."""
    pool = 20000
    cases = {
        'equal gaps': bern_rows([.4] * 8),
        'one hard edge': bern_rows([.1] + [.9] * 7),
        'noisy rows': bern_rows([.4] * 8, noise=[1, 1, 1, 1, 8, 8, 8, 8]),
    }
    out = {}
    for name, (A, B) in cases.items():
        row = {}
        for pol in ('count', 'novelty', 'bound', 'elim', 'struct', 'proxy', 'sum', 'chain'):
            row[pol] = route_sim(np.random.default_rng(11), A, B, pol, pool=pool)
        out[name] = row
        steps = ', '.join(f"{k} {v[0]:.0f}" for k, v in row.items())
        errs = ', '.join(f"{k} {v[1]:.3f}" for k, v in row.items())
        print(f"      {name}: steps {steps}; error {errs}; unfinished {max(v[3] for v in row.values()):.3f}", flush=True)
    return out


def t3_checks(r):
    out = t3c_compare(r)
    eq = out['equal gaps']
    trio = [eq[k][0] for k in ('count', 'novelty', 'bound')]
    check('T3c A14 equal gaps and supports: novelty, count, static bound coincide',
          max(trio) / min(trio) < 1.1,
          f"steps count {trio[0]:.0f}, novelty {trio[1]:.0f}, static bound {trio[2]:.0f}")
    hard = out['one hard edge']
    check('T3d A14 novelty does not see gaps', hard['novelty'][0] > 3 * hard['struct'][0]
          and hard['bound'][0] < 1.5 * hard['struct'][0],
          "one hard edge, 7 easy: steps " + ', '.join(f"{k} {v[0]:.0f}" for k, v in hard.items())
          + f". Hard edge counts: novelty {hard['novelty'][2][0]:.0f}, struct {hard['struct'][2][0]:.0f}; "
          f"easy edge: novelty {hard['novelty'][2][1:].mean():.0f}, struct {hard['struct'][2][1:].mean():.1f}")
    no = out['noisy rows']
    check('T3e A14 novelty is drawn to outcomes that carry nothing',
          no['novelty'][2][4:].mean() > 3 * no['novelty'][2][:4].mean() and no['struct'][2][4:].mean() < 1.5 * no['struct'][2][:4].mean(),
          f"same gap, 16 against 2 outcomes: mean counts novelty {no['novelty'][2][:4].mean():.0f} / {no['novelty'][2][4:].mean():.0f}, "
          f"struct {no['struct'][2][:4].mean():.0f} / {no['struct'][2][4:].mean():.0f}; "
          f"steps novelty {no['novelty'][0]:.0f}, count {no['count'][0]:.0f}, struct {no['struct'][0]:.0f}")
    check('T3f A18 the entropy-per-visit proxy of active_k2 behaves like structure EIG',
          hard['proxy'][0] < 1.5 * hard['struct'][0] and no['proxy'][0] < 1.5 * no['struct'][0],
          f"steps, proxy against structure EIG: equal {eq['proxy'][0]:.0f}, {eq['struct'][0]:.0f}; "
          f"one hard {hard['proxy'][0]:.0f}, {hard['struct'][0]:.0f}; noisy {no['proxy'][0]:.0f}, {no['struct'][0]:.0f}")
    return out


def hitting(rng, gs, d, R=4000):
    """Evidence walks of the edges, to threshold ln((1 - d)/d). Returns tau, shape (R, E)."""
    thr = np.log((1 - d) / d)
    out = np.zeros((R, len(gs)))
    for e, g in enumerate(gs):
        p1, p0 = .5 + g / 2, .5 - g / 2
        inc = np.log(p1 / p0)
        pos = np.zeros(R)
        live = np.ones(R, bool)
        t = 0
        while live.any():
            k = live.sum()
            pos[live] += np.where(rng.random(k) < p1, inc, -inc)
            t += 1
            out[live, e] = t
            live &= np.abs(pos) < thr
    return out


def t4_ratio(out, r):
    hard = out['one hard edge']
    K = 8

    def kl(g):
        p1, p0 = .5 + g / 2, .5 - g / 2
        return (p1 - p0) * np.log(p1 / p0)

    floor = K * np.log(1 / (4 * DELTA)) / kl(.1)
    lim = K / kl(.1) / (1 / kl(.1) + 7 / kl(.9))
    check('T4a A15 evidence beats count on one hard edge', hard['count'][0] >= floor and hard['count'][0] > 3 * hard['struct'][0],
          f"K = {K}: count {hard['count'][0]:.0f} >= floor {floor:.0f}; elimination {hard['elim'][0]:.0f}, structure EIG {hard['struct'][0]:.0f}; "
          f"gain {hard['count'][0] / hard['elim'][0]:.1f}x; limit K max(1/KL) / sum(1/KL) = {lim:.1f}x")
    txt = []
    gains = []
    for d in (DELTA / 8, 1e-3, 1e-6, 1e-12):
        tau = hitting(np.random.default_rng(4), [.4] * 8, d)
        gains.append((8 * tau.max(1)).mean() / tau.sum(1).mean())
        txt.append(f"delta_e {d:.0e}: {gains[-1]:.2f}x")
    eq = out['equal gaps']
    sim = eq['count'][0] / eq['elim'][0]
    check('T4b A16 equal gaps: gain is max tau / mean tau, and falls towards 1', abs(sim / gains[0] - 1) < 0.15 and gains[-1] < gains[0],
          f"count {eq['count'][0]:.0f}, elimination {eq['elim'][0]:.0f}, struct {eq['struct'][0]:.0f}: gain {sim:.2f}x, identity predicts {gains[0]:.2f}x. "
          f"{'; '.join(txt)}")
    check('T4c A17 sum with term 2 given structure keeps the saving',
          hard['chain'][0] < 1.3 * hard['struct'][0],
          f"one hard edge: count {hard['count'][0]:.0f}, sum with (m - 1)/(2 n_e) {hard['sum'][0]:.0f}, "
          f"sum with term 2 given structure {hard['chain'][0]:.0f}, structure alone {hard['struct'][0]:.0f}. "
          f"Easy-edge counts at stop: sum {hard['sum'][2][1:].mean():.0f}, chain {hard['chain'][2][1:].mean():.1f}; "
          f"hard edge: sum {hard['sum'][2][0]:.0f}, chain {hard['chain'][2][0]:.0f}")


def t4d_discovery(r):
    """A16(c): an edge never seen has no merge to decide. Structure alone never visits it."""
    K = 6
    al = 0.5
    R = 200
    c = np.full((R, K, 2), al)
    door = K - 1
    first = np.full((R, 2), -1)
    for mode in (0, 1):
        c[:] = al
        n = np.zeros((R, K))
        rr = np.random.default_rng(3)
        for t in range(60):
            if mode == 0:
                s = eig(c)
            else:
                s = -n.copy()                     # visits among known edges, the door has no node
                s[:, door] = -np.inf
            e = (s + 1e-9 * rr.random(s.shape)).argmax(1)
            o = rr.integers(0, 2, R)
            c[np.arange(R), e, o] += 1
            n[np.arange(R), e] += 1
            newly = (e == door) & (first[:, mode] < 0)
            first[newly, mode] = t
    a = (first[:, 0] >= 0).mean()
    b = (first[:, 1] >= 0).mean()
    check('T4d A16c unseen edges need novelty', a == 1 and b == 0,
          f"door found within 60 steps: novelty {a:.2f}, structure on the known machine {b:.2f}")


def t4e_online(r):
    """A15, measured: unknown rows, merges decided online from pair posteriors."""
    p = np.array([.3, .3, .3, .7, .7, .7, .45, .6])
    K = len(p)
    truth = np.array([0, 0, 0, 1, 1, 1, 2, 3])
    iu, ju = np.triu_indices(K, 1)
    R, B = 30, 5                  # B draws of the chosen item per scoring round
    al = .5
    res = {}
    for pol in ('count', 'struct', 'sum', 'chain'):
        rr = np.random.default_rng(5)
        x = np.zeros((R, K, 2))
        steps = np.zeros(R)
        live = np.ones(R, bool)
        for t in range(3000):
            xi, xj = x[:, iu], x[:, ju]
            d = dfe(xi, xj, al)
            P = 1 / (1 + np.exp(-np.clip(d, -40, 40)))
            undec = (P > 0.1) & (P < 0.9)
            n = x.sum(2)
            if n.min() > 0:
                live = live & undec.any(1)
                if not live.any():
                    break
            if pol == 'count' or n.min() == 0:
                s = -n
            else:
                pm = (al + xi + xj) / (1 + (xi + xj).sum(2, keepdims=True))
                # which row is drawn: both directions are scored
                pi_ = (al + xi) / (1 + xi.sum(2, keepdims=True))
                pj_ = (al + xj) / (1 + xj.sum(2, keepdims=True))
                Pp = P[..., None]
                si = ent(Pp * pm + (1 - Pp) * pi_) - P * ent(pm) - (1 - P) * ent(pi_)
                sj = ent(Pp * pm + (1 - Pp) * pj_) - P * ent(pm) - (1 - P) * ent(pj_)
                st = np.zeros((R, K))
                np.add.at(st, (np.arange(R)[:, None], iu[None]), si)
                np.add.at(st, (np.arange(R)[:, None], ju[None]), sj)
                if pol == 'struct':
                    s = st
                elif pol == 'sum':
                    s = st + eig(x + al)
                else:
                    soft = n.copy()
                    np.add.at(soft, (np.arange(R)[:, None], iu[None]), P * n[:, ju])
                    np.add.at(soft, (np.arange(R)[:, None], ju[None]), P * n[:, iu])
                    s = st + 1 / (2 * np.maximum(soft, 1))
            e = (s + 1e-12 * rr.random(s.shape)).argmax(1)
            k = rr.binomial(B, p[e])
            lv = np.flatnonzero(live)
            x[lv, e[lv], 1] += k[lv]
            x[lv, e[lv], 0] += B - k[lv]
            steps[lv] += B
        d = dfe(x[:, iu], x[:, ju], al)
        same = truth[iu] == truth[ju]
        ok = ((d > 0) == same).all(1).mean()
        res[pol] = (float(np.median(steps)), float(ok))
    txt = ', '.join(f"{k} {v[0]:.0f} (exact {v[1]:.2f})" for k, v in res.items())
    check('T4e A15 online merges, measured', res['struct'][0] < res['count'][0],
          f"8 unknown rows, pair posterior outside (0.1, 0.9), {R} seeds: median steps {txt}; 2 s.e. on exact is about 0.2")
    return res


# ================================================================= T5: chain rule
def predictive_terms(x, y, al):
    """Sampling row x. Returns P(merge), I(o; S), I(o; theta | S), predictive rows."""
    d = dfe(x, y, al)
    P = 1 / (1 + np.exp(-d))
    z = x + y
    pm = (al + z) / (al * len(x) + z.sum())
    pd = (al + x) / (al * len(x) + x.sum())
    mix = P * pm + (1 - P) * pd
    iS = ent(mix) - P * ent(pm) - (1 - P) * ent(pd)
    iT = P * eig(z + al) + (1 - P) * eig(x + al)
    return P, iS, iT, pm, pd, mix


def t5a_identity(r):
    """A17(a): the sum is exact, by Monte Carlo on the joint. Also Bayes consistency of dF."""
    worst_mc = 0
    worst_bayes = 0
    M = 400000
    for trial in range(8):
        m = int(r.integers(2, 4))
        al = 1 / m
        x = r.integers(0, 12, m).astype(float)
        y = r.integers(0, 12, m).astype(float)
        if x.sum() == 0 or y.sum() == 0:
            continue
        P, iS, iT, pm, pd, mix = predictive_terms(x, y, al)
        s = r.random(M) < P
        th = np.where(s[:, None], r.dirichlet(x + y + al, M), r.dirichlet(x + al, M))
        o = (r.random(M)[:, None] > np.cumsum(th, 1)).sum(1).clip(0, m - 1)
        pmc = np.log(th[np.arange(M), o] / mix[o]).mean()
        se = np.log(th[np.arange(M), o] / mix[o]).std() / np.sqrt(M)
        worst_mc = max(worst_mc, abs(pmc - iS - iT) / max(se, 1e-9))
        for o2 in range(m):
            e = np.eye(m)[o2]
            d2 = dfe(x + e, y, al)
            worst_bayes = max(worst_bayes, abs(1 / (1 + np.exp(-d2)) - P * pm[o2] / mix[o2]))
    # data-dependent prior breaks it when a new column appears
    x = np.array([3., 2., 0.])
    y = np.array([2., 3., 0.])
    d0 = lndm(x[:2] + y[:2], .5) - lndm(x[:2], .5) - lndm(y[:2], .5)
    P0 = 1 / (1 + np.exp(-d0))
    e = np.array([0., 0., 1.])
    d1 = dfe(x + e, y, 1 / 3)
    pm0 = (1 / 3 + (x + y)[2]) / (1 + (x + y).sum())
    pd0 = (1 / 3 + x[2]) / (1 + x.sum())
    bayes = P0 * pm0 / (P0 * pm0 + (1 - P0) * pd0)
    gap = abs(1 / (1 + np.exp(-d1)) - bayes)
    check('T5a A17a chain rule exact on the joint', worst_mc < 4.5 and worst_bayes < 1e-9,
          f"8 tables, |sum - MC joint| at most {worst_mc:.2f} s.e.; posterior from dF equals Bayes rule to {worst_bayes:.1e}. "
          f"With prior 1/m' re-fitted on a new column the same step is off by {gap:.3f}")


def t5b_state(r):
    """A17(b): plug-in state term + conditional novelty is exact; both-conditioned overcounts."""
    worst_ident, top, worst_share = 0, 0, 0
    neg = 0
    M = 200000
    for trial in range(12):
        m = 2
        b = r.uniform(.2, .8)
        c = r.integers(1, 15, (2, m)).astype(float) + .5
        pb = c / c.sum(1, keepdims=True)
        mix = b * pb[0] + (1 - b) * pb[1]
        iq = ent(mix) - b * ent(pb[0]) - (1 - b) * ent(pb[1])                  # plug-in at the mean
        it = b * eig(c[0]) + (1 - b) * eig(c[1])
        th1, th2 = r.dirichlet(c[0], M), r.dirichlet(c[1], M)
        q = r.random(M) < b
        th = np.where(q[:, None], th1, th2)
        o = (r.random(M) > th[:, 0]).astype(int)
        lr = np.log(th[np.arange(M), o] / mix[o])
        joint, se = lr.mean(), lr.std() / np.sqrt(M)
        worst_ident = max(worst_ident, abs(joint - iq - it) / se)
        i_q_th = (ent(b * th1 + (1 - b) * th2) - b * ent(th1) - (1 - b) * ent(th2)).mean()
        ov = i_q_th - iq
        neg += ov < -3 * se
        top = max(top, ov)
        worst_share = max(worst_share, ov / joint)
    check('T5b A17b plug-in salience plus novelty', worst_ident < 4.5 and neg == 0,
          f"I(o;q | mean theta) + I(o;theta | q) matches the joint to {worst_ident:.2f} s.e.; using E_theta I(o;q | theta) "
          f"with the same novelty overcounts, never under ({neg} negative of 12): largest {top:.4f} nats, {worst_share:.0%} of the joint")


def t5c_map(r):
    """A17(c): MAP structure, and the per-edge Dirichlet term used when the merge is likely."""
    rows = []
    for x, y in ((np.array([6., 4.]), np.array([5., 5.])),
                 (np.array([40., 30.]), np.array([38., 32.])),
                 (np.array([400., 300.]), np.array([380., 320.])),
                 (np.array([30., 20.]), np.array([20., 30.])),
                 (np.array([60., 30.]), np.array([30., 60.]))):
        P, iS, iT, *_ = predictive_terms(x, y, .5)
        z = x + y
        ex = iS + iT
        mp = iS + (eig(z + .5) if P > .5 else eig(x + .5))
        ed = iS + eig(x + .5)
        rows.append(f"P {P:.3f}: exact {ex:.5f}, MAP err {abs(mp - ex) / ex:.0%}, per-edge term err {abs(ed - ex) / ex:.0%}, "
                    f"I_S share {iS / ex:.0%}")
    x, y = np.array([400., 300.]), np.array([380., 320.])
    P, iS, iT, *_ = predictive_terms(x, y, .5)
    ratio = eig(x + .5) / iT
    check('T5c A17c conditioning of the parameter term', ratio > 1.5,
          f"{' | '.join(rows)}. Per-edge over conditional term {ratio:.2f}x at P = {P:.2f}")


def t5d_pairs(r):
    """A17(d): pair formula against the exact partition posterior, three rows."""
    al = .5
    parts = [[[0, 1, 2]], [[0, 1], [2]], [[0, 2], [1]], [[1, 2], [0]], [[0], [1], [2]]]
    over, frac_gap = [], []
    sumpair_max = 0
    for trial in range(800):
        n = int(r.choice([6, 12, 30, 80]))
        base = r.uniform(.2, .8, 3) if trial % 2 else np.full(3, r.uniform(.2, .8))
        k = r.binomial(n, base)
        X = np.c_[k, n - k].astype(float)
        lw = []
        preds = []
        for pt in parts:
            lw.append(sum(lndm(X[b].sum(0), al) for b in pt))
            b = [bk for bk in pt if 0 in bk][0]
            pool = X[b].sum(0)
            preds.append((al + pool) / (1 + pool.sum()))
        w = np.exp(np.array(lw) - max(lw))
        w /= w.sum()
        preds = np.array(preds)
        mix = (w[:, None] * preds).sum(0)
        joint = ent(mix) - (w * ent(preds)).sum()
        # pair sum for a draw of row 0
        s = 0
        marg = []
        for j in (1, 2):
            _, iS, *_ = predictive_terms(X[0], X[j], al)
            s += iS
        d12 = dfe(X[0], X[1], al)
        pair = 1 / (1 + np.exp(-d12))
        exact = w[0] + w[1]
        frac_gap.append(abs(pair - exact))
        if joint > 1e-5:
            over.append(s / joint)
    over = np.array(over)
    check('T5d A17d pair sum against partition posterior', np.percentile(over, 90) > 1.1,
          f"three rows, 800 tables: sum of pair I(o;S) over exact I(o;partition): median {np.median(over):.2f}, "
          f"range {over.min():.2f} to {over.max():.2f}, 90th percentile {np.percentile(over, 90):.2f}; "
          f"|pair posterior - exact marginal| mean {np.mean(frac_gap):.3f}, max {np.max(frac_gap):.3f}. "
          f"Equal priors on 5 partitions give prior P(i ~ j) = 0.4, not 0.5")


def t5e_windows(r):
    """A17(e): summing depth 1, 2, 3 windows makes the posterior overconfident."""
    R, n = 40000, 30
    merged = r.random(R) < .5
    t1 = r.beta(.5, .5, R)
    t2 = np.where(merged, t1, r.beta(.5, .5, R))
    a = (r.random((R, n + 2)) < t1[:, None]).astype(int)
    b = (r.random((R, n + 2)) < t2[:, None]).astype(int)

    def row(s, d):
        if d == 1:
            c = s[:, :n]
        elif d == 2:
            c = s[:, :n] * 2 + s[:, 1:n + 1]
        else:
            c = s[:, :n] * 4 + s[:, 1:n + 1] * 2 + s[:, 2:n + 2]
        t = np.zeros((R, 2 ** d))
        np.add.at(t, (np.repeat(np.arange(R), n), c.ravel()), 1)
        return t

    ds = [dfe(row(a, d), row(b, d), 1 / 2 ** d) for d in (1, 2, 3)]

    def ece(F):
        P = 1 / (1 + np.exp(-np.clip(F, -30, 30)))
        bins = np.minimum((P * 10).astype(int), 9)
        e = 0
        for k in range(10):
            m = bins == k
            if m.sum():
                e += m.mean() * abs(P[m].mean() - merged[m].mean())
        return e

    e1, e3 = ece(ds[0]), ece(sum(ds))
    et = ece(sum(ds) / 3)
    check('T5e A17e overlapping windows, calibration', e3 > 1.5 * e1,
          f"expected calibration error: one modality {e1:.3f}, depths 1 + 2 + 3 summed {e3:.3f}, same divided by 3: {et:.3f}")


def t5f_plan(r):
    """A17(f): adding per-step gains in a plan without updating the belief overcounts past H(S)."""
    A = np.array([.7, .3])
    B = np.array([.3, .7])
    pi = .5
    per = ent(pi * A + (1 - pi) * B) - pi * ent(A) - (1 - pi) * ent(B)
    out = []
    for k in (1, 3, 9, 27):
        tot = 0
        for j in range(k + 1):
            la = gammaln(k + 1) - gammaln(j + 1) - gammaln(k - j + 1) + j * np.log(.7) + (k - j) * np.log(.3)
            lb = gammaln(k + 1) - gammaln(j + 1) - gammaln(k - j + 1) + j * np.log(.3) + (k - j) * np.log(.7)
            pa, pb = np.exp(la), np.exp(lb)
            mx = .5 * pa + .5 * pb
            if mx > 0:
                tot += .5 * pa * np.log(pa / mx) + .5 * pb * np.log(pb / mx)
        out.append((k, tot, k * per))
    txt = ', '.join(f"k = {k}: exact {t:.3f}, sum {s:.3f}" for k, t, s in out)
    check('T5f A17f plans must update beliefs', out[-1][2] > np.log(2) > out[-1][1],
          f"{txt}; H(S) = ln 2 = 0.693 caps the exact value")


def t5g_state_def(r):
    """A17(g): which state the term names. q_t gives 0, q_t+1 gives the entropy of a coin."""
    b = 0.5
    i_cur = 0.0
    i_next = -2 * b * np.log(b)
    nov = eig(np.array([50.5, 50.5]))
    check('T5g A17g state term on a unifilar machine', i_next > 100 * nov,
          f"a fair coin moves a known machine to one of two states: I(o;q_t) = {i_cur:.3f}, I(o;q_t+1) = {i_next:.3f} nats for ever, "
          f"against novelty {nov:.5f} at n = 100")


def t5h_late(r):
    """A16(d): on equal rows the structure term decays faster than novelty, so novelty leads late."""
    rows = []
    for n in (30, 300, 3000, 30000, 300000):
        vals = []
        for _ in range(1500):
            k1, k2 = r.binomial(n, .5, 2)
            x = np.array([k1, n - k1], float)
            y = np.array([k2, n - k2], float)
            _, iS, iT, *_ = predictive_terms(x, y, .5)
            vals.append(iS / iT)
        rows.append((n, float(np.median(vals))))
    check('T5h A16d structure term fades on equal rows', rows[-1][1] < rows[1][1] / 4,
          'median I(o;S) / I(o;theta | S): ' + ', '.join(f"n = {n}: {v:.4f}" for n, v in rows))


# =================================================================
GROUPS = {
    'T1': [t1a_tail_exact, t1b_closed_loop, t1c_impure, t1d_events, t1e_stop, t1f_refit, t1g_overlap],
    'T2': [t2a_ville, t2b_merge, t2c_stack, t2d_recovery, t2e_lower, t2f_gain, t2g_lock, t2h_unknown_lock],
    'T3': [t3a_eig, t3b_alloc],
    'T5': [t5a_identity, t5b_state, t5c_map, t5d_pairs, t5e_windows, t5f_plan, t5g_state_def, t5h_late],
}


def main():
    only = None
    if '--only' in sys.argv:
        only = sys.argv[sys.argv.index('--only') + 1].split(',')
    t0 = time.time()
    r = np.random.default_rng(2026)
    for name in ('T1', 'T2', 'T3'):
        if only and name not in only:
            continue
        print(f'--- {name}', flush=True)
        for f in GROUPS[name]:
            t1 = time.time()
            f(r)
            print(f'      ({f.__name__}: {time.time() - t1:.1f} s)', flush=True)
    if not only or 'T3' in only or 'T4' in only:
        print('--- T3/T4 routing', flush=True)
        t1 = time.time()
        out = t3_checks(r)
        t4_ratio(out, r)
        print(f'      (routing: {time.time() - t1:.1f} s)', flush=True)
        for f in (t4d_discovery, t4e_online):
            t1 = time.time()
            f(r)
            print(f'      ({f.__name__}: {time.time() - t1:.1f} s)', flush=True)
    if not only or 'T5' in only:
        print('--- T5', flush=True)
        for f in GROUPS['T5']:
            t1 = time.time()
            f(r)
            print(f'      ({f.__name__}: {time.time() - t1:.1f} s)', flush=True)
    bad = [t for t, ok, _ in RES if not ok]
    print(f'\n{len(RES) - len(bad)} of {len(RES)} checks pass in {time.time() - t0:.0f} s')
    for t in bad:
        print('FAILED:', t)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
