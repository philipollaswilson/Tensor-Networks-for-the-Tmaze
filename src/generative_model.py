"""From learned hidden states to a generative model (full pymdp T-maze).

Takes the BMR states of make_fig_hidden_states.py and builds, from the data only:

  flat     one hidden variable whose values are the learned (predictive) states.
           Edge-emitting: T[a, o][s, s'] = p(o | s, a) [s' = f(o)], start D = the
           state the empty history merges into. f(o) = the state of the histories
           that ended in o.
  mixture  a state whose prediction row is a convex mixture of the other rows is a
           belief, not a latent state. Tested by Bayes factor: row of state k
           explained by the fitted mixture vs. a free Dirichlet row.
  gen      generator over the remaining vertex states. A mixture state m is replaced
           by its weights w_m; on a transition into a mixture the hidden state
           persists when it is one of the mixture's components (inertia: nothing
           changes unless the data say so), else it spreads by w_m.
  factor   Hartmanis-Stearns: partitions of the states with the substitution
           property (every block maps into one block under every action); two of
           them with a trivial meet and product size = #states give a product of
           hidden variables, S = S1 x S2.

Checks: total variation to the true process at the data's horizon (two steps) and
one step beyond it (three steps, not in the data), via an exact simulator of the
same maze (verified against the data's leaf enumeration).
"""
import itertools
import math
import pathlib
import sys

import numpy as np
from scipy.optimize import nnls

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fig_hidden_states as HS  # noqa: E402
import seed_protocol as SP  # noqa: E402

NO = 24
ALPHA_O = 1.0 / NO
gammaln = np.vectorize(math.lgamma, otypes=[float])


def lnB(x):
    return gammaln(x).sum(-1) - gammaln(x.sum(-1))


def lnB_label(x):
    return lnB(x)


# ------------------------------------------------------------ the flat model
def flat_model(trajs):
    m = HS.learned_model(trajs)
    lab, K = m['lab'], m['K']
    nf = HS.full_counts_of(trajs)
    N = np.array([nf[lab == c].sum(0) for c in range(K)])          # (K, 4, 24)
    n0 = np.zeros((4, NO))                                           # empty history
    for (_, _, a1, o1, _, _) in trajs:
        n0[a1, SP.obs_index(*o1)] += 1
    # merge the empty history into a learned state (label projection, as the BMR)
    proj = lambda x: np.stack([np.bincount(HS.LAB, weights=r, minlength=HS.NL) for r in x])  # noqa: E731
    a = np.full(HS.NL, HS.ALPHA)
    dF = [float((lnB(a + proj(n0) + proj(N[c])) - lnB(a + proj(n0)) - lnB(a + proj(N[c]))
                 + lnB(a)).sum()) for c in range(K)]
    start = int(np.argmax(dF))
    assert dF[start] > 0, 'the empty history is a new state'
    f = {}
    for i, (_, _, o1) in enumerate(HS.HIST):
        if lab[i] >= 0:
            f.setdefault(o1, lab[i])
    P = N / np.maximum(N.sum(-1, keepdims=True), 1)
    T = np.zeros((4, NO, K, K))
    for o, s in f.items():
        T[:, o, :, s] = P[:, :, o].T
    D = np.eye(K)[start]
    return dict(m=m, K=K, N=N, P=P, T=T, D=D, f=f, start=start, start_dF=dF[start],
                k_of=m['k_of'])


# ------------------------------------------------------------ mixtures
def simplex_lsq(y, X):
    """min ||y - X^T w||, w >= 0, sum w = 1 (heavily weighted sum row)."""
    big = 1e3
    A = np.vstack([X.T, big * np.ones(len(X))])
    w, _ = nnls(A, np.concatenate([y, [big]]))
    return w / w.sum()


def mixture_test(fl):
    """Greedy: repeatedly remove the state best explained as a mixture of the rest."""
    N, P, K = fl['N'], fl['P'], fl['K']
    alive, weights, report = list(range(K)), {}, {}
    while True:
        best = None
        for k in alive:
            others = [j for j in alive if j != k]
            w = simplex_lsq(P[k].ravel(), P[others].reshape(len(others), -1))
            w = np.where(w > 0.02, w, 0)            # sparse: drop fitting noise
            w = w / w.sum()
            mix = np.einsum('j,jao->ao', w, P[others])
            with np.errstate(divide='ignore'):
                lmix = float((N[k] * np.log(np.where(N[k] > 0, mix, 1))).sum())
            a = np.full(NO, ALPHA_O)
            lfree = float((lnB(a + N[k]) - lnB(a)).sum())
            bf = lmix - lfree
            report.setdefault(k, (bf, dict(zip(others, w))))
            if best is None or bf > best[0]:
                best = (bf, k, dict(zip(others, w)))
        if best is None or best[0] <= 0:
            break
        bf, k, w = best
        report[k] = (bf, w)
        weights[k] = w
        alive.remove(k)
    return alive, weights, report


# ------------------------------------------------------------ the generator
def generator(fl, vertices, weights):
    V = list(vertices)
    ix = {k: i for i, k in enumerate(V)}
    W = np.zeros((fl['K'], len(V)))                 # every state as weights on vertices
    for k in range(fl['K']):
        if k in ix:
            W[k, ix[k]] = 1
        else:
            for j, w in weights[k].items():         # mixtures of vertices (one level here)
                W[k, ix[j]] += w
    G = np.zeros((4, NO, len(V), len(V)))
    for o, s in fl['f'].items():
        q = W[s]
        for i, k in enumerate(V):
            e = fl['P'][k, :, o]
            if q[i] > 1e-9:
                G[:, o, i, i] += e                  # inertia: the hidden state persists
            else:
                G[:, o, i, :] += e[:, None] * q[None]
    return dict(V=V, G=G, D=W[fl['start']], W=W)


# ------------------------------------------------------------ factorization
def partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


def sp_partitions(G, tol=1e-3):
    n = G.shape[-1]
    S = G.sum(1) > tol                              # (4, n, n) transition support
    out = []
    for p in partitions(list(range(n))):
        if len(p) in (1, n):
            continue
        blk = {s: b for b, B in enumerate(p) for s in B}
        if all(len({blk[j] for i in B for j in np.flatnonzero(S[a, i])}) <= 1
               for a in range(4) for B in p):
            out.append(sorted(sorted(B) for B in p))
    return out


def products(sps, n):
    out = []
    for p1, p2 in itertools.combinations(sps, 2):
        if len(p1) * len(p2) == n and all(len(set(a) & set(b)) == 1 for a in p1 for b in p2):
            out.append((p1, p2) if len(p1) <= len(p2) else (p2, p1))
    return out


def factor(gen, p1, p2):
    """B_f[a][b'|b] for each factor, the max error of B = B1 (x) B2, and the
    dependence of each observation modality (location, reward, cue bit) on each
    factor."""
    G = gen['G']
    Ta = G.sum(1)                                   # (4, n, n)
    n = Ta.shape[-1]
    b1 = {s: b for b, B in enumerate(p1) for s in B}
    b2 = {s: b for b, B in enumerate(p2) for s in B}
    Bs = []
    for blk, p in ((b1, p1), (b2, p2)):
        B = np.zeros((4, len(p), len(p)))
        for a in range(4):
            for i in range(n):
                for j in range(n):
                    B[a, blk[i], blk[j]] += Ta[a, i, j] / len(p[blk[i]])
        Bs.append(B)
    err = max(abs(Ta[a, i, j] - Bs[0][a, b1[i], b1[j]] * Bs[1][a, b2[i], b2[j]])
              for a in range(4) for i in range(n) for j in range(n))
    E = G.sum(-1)                                   # (4, 24, n): p(o | s, a)
    mods = [lambda o: o // 6, lambda o: (o // 2) % 3, lambda o: o % 2]
    sizes = [4, 3, 2]
    dep = np.zeros((3, 2))
    for m, (g, sz) in enumerate(zip(mods, sizes)):
        Em = np.zeros((4, sz, n))
        for o in range(NO):
            Em[:, g(o)] += E[:, o]
        for f, blk_other in enumerate((b2, b1)):    # vary factor f, hold the other fixed
            for i, j in itertools.combinations(range(n), 2):
                if blk_other[i] == blk_other[j]:
                    dep[m, f] = max(dep[m, f], 0.5 * np.abs(Em[:, :, i] - Em[:, :, j]).sum(1).max())
    return Bs, err, dep, (b1, b2)


# ------------------------------------------------------------ the true process
def true_joint(T):
    """p(o_1..o_T | a_1..a_T) of the maze, exact: context hidden, location from the
    actions, arms absorbing, the cue bit shows the context only at the cue."""
    def emit(loc, ctx):
        e = np.zeros(NO)
        r = 0 if loc in (0, 3) else (1 if (loc == 1) == (ctx == 0) else 2)
        for c in range(2):
            if loc == 3:
                e[SP.obs_index(loc, r, ctx)] = 1.0
            else:
                e[SP.obs_index(loc, r, c)] += 0.5
        return e
    out = np.zeros((4, NO) * T)
    for acts in itertools.product(range(4), repeat=T):
        for ctx in range(2):
            loc, p = 0, np.ones(())
            for a in acts:
                loc = loc if loc in (1, 2) else a
                p = np.multiply.outer(p, emit(loc, ctx))
            idx = tuple(x for a in acts for x in (a, slice(None)))
            out[idx] += 0.5 * p.reshape([NO] * T)
    return out


def check_simulator():
    q = SP.FULL.exact_joint(SP.FULL.enumerate_weighted())[0].sum(0) * 16   # (o0) summed
    p = true_joint(2)
    return float(np.abs(p - q).max())


def model_joint(T, D, steps):
    """p(o_1..o_steps | a_1..a_steps) of an edge-emitting model."""
    n = len(D)
    out = np.zeros((4, NO) * steps)
    for acts in itertools.product(range(4), repeat=steps):
        v = D[None]                                  # (paths, n)
        for a in acts:
            v = np.einsum('pi,oij->poj', v, T[a]).reshape(-1, n)
        idx = tuple(x for a in acts for x in (a, slice(None)))
        out[idx] = v.sum(-1).reshape([NO] * steps)
    return out


def tv_next(pm, pt, steps):
    """Mean TV of p(o_steps | past) between model and truth, past weighted by the
    truth, actions uniform."""
    tot = 0.0
    for acts in itertools.product(range(4), repeat=steps):
        idx = tuple(x for a in acts for x in (a, slice(None)))
        m, t = pm[idx].reshape(-1, NO), pt[idx].reshape(-1, NO)
        wt = t.sum(1)
        mc = m / np.where(m.sum(1, keepdims=True) > 0, m.sum(1, keepdims=True), 1)
        tc = t / np.where(wt[:, None] > 0, wt[:, None], 1)
        d = np.where(m.sum(1) > 0, 0.5 * np.abs(mc - tc).sum(1), 1.0)
        tot += (wt * d).sum()
    return tot / 4 ** steps


def build(trajs):
    fl = flat_model(trajs)
    vertices, weights, report = mixture_test(fl)
    gen = generator(fl, vertices, weights)
    sps = sp_partitions(gen['G'])
    prods = products(sps, len(vertices))
    fac = factor(gen, *prods[0]) if prods else None
    return dict(fl=fl, vertices=vertices, weights=weights, report=report, gen=gen,
                sps=sps, prods=prods, fac=fac)


def evaluate(res):
    fl, gen = res['fl'], res['gen']
    out = {}
    for steps in (2, 3):
        pt = true_joint(steps)
        for name, T, D in (('flat', fl['T'], fl['D']), ('gen', gen['G'], gen['D'])):
            out[name, steps] = tv_next(model_joint(T, D, steps), pt, steps)
            if steps == 2:
                out[name, 'joint'] = 0.5 * np.abs(model_joint(T, D, 2) - pt).sum() / 16
    return out


if __name__ == '__main__':
    print('simulator vs data enumeration, max abs diff:', check_simulator())
    res = build(HS.sample(HS.N_TRAIN, 0))
    fl = res['fl']
    name = lambda k: HS.STATES[fl['k_of'][k]][1]  # noqa: E731
    print('start merges into', name(fl['start']), f"dF {fl['start_dF']:.1f}")
    for k, (bf, w) in sorted(res['report'].items()):
        print(f'  {name(k):20s} lnBF {bf:10.1f}  ',
              {name(j): round(v, 3) for j, v in w.items() if v > 0.01})
    print('vertices', [name(k) for k in res['vertices']])
    print('D', np.round(res['gen']['D'], 3))
    print(len(res['sps']), 'SP partitions;', len(res['prods']), 'products')
    for p1, p2 in res['prods']:
        V = res['vertices']
        print('  ', [[name(V[i]) for i in B] for B in p1], 'x', [[name(V[i]) for i in B] for B in p2])
    Bs, err, dep, _ = res['fac']
    print('factorization error', err)
    print('B1 (stay)', np.round(Bs[0][0], 2)); print('B2', np.round(Bs[1], 2))
    print('modality dependence (loc, rew, cue) x (f1, f2)\n', np.round(dep, 3))
    print(evaluate(res))
