"""The structure-learning pipeline on any symbolic trajectory data, any history length.

Steps (as fig_structure_learning_math.png):
  1-2  count and predict: every history prefix h -> counts N(h, a, o) of the next step
  4    merge prefixes by Bayesian model reduction (Dirichlet, Perks prior 1/|O|)
  5    machine: s' = f(s, a, o) = the state of the prefix h a o, read from the data
  6    unmix: drop states whose row is a mixture of other rows (Bayes factor)
  7    generator over the remaining states (inertia only where the data are silent)
  8    factorize: partitions with the substitution property; pairs that form a product

Stress-test mazes (MazeEnv) with the T-maze's hidden context, no cue noise:
  pymdp        moves go straight to the named place, arms absorb       expect 6 = 2 x 3
  centre_only  arms reachable only from the centre (location matters)  expect 8 = 2 x 4
  three_ctx    a third context with no cheese anywhere                 expect no product
Run:  python src/structure_pipeline.py
"""
import itertools
from collections import defaultdict

import numpy as np
from scipy.optimize import nnls
from scipy.special import gammaln

LOC = ['C', 'R', 'L', 'Cue']
ACT = ['stay', 'right', 'left', 'cue']


# ------------------------------------------------------------ environments
class MazeEnv:
    def __init__(self, variant):
        self.variant = variant
        self.ctx = [0, 1, 2] if variant == 'three_ctx' else [0, 1]   # 0 cheese R, 1 L, 2 none
        self.obs = {}

    def o(self, loc, rew, cue):
        return self.obs.setdefault((loc, rew, cue), len(self.obs))

    def move(self, loc, a):
        if loc in (1, 2):
            return loc                                  # arms absorb
        if self.variant == 'centre_only' and loc == 3 and a in (1, 2):
            return 3                                    # arms only from the centre
        return a

    def emit(self, loc, ctx):
        rew = 0 if loc in (0, 3) else (1 if (loc == 1 and ctx == 0) or (loc == 2 and ctx == 1) else 2)
        return self.o(loc, rew, ctx if loc == 3 else -1)

    def sample(self, n, T, seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            ctx, loc = rng.choice(self.ctx), 0
            obs, acts, true = [self.emit(loc, ctx)], [], [(loc, ctx)]
            for _ in range(T):
                a = int(rng.integers(4))
                loc = self.move(loc, a)
                acts.append(a); obs.append(self.emit(loc, ctx)); true.append((loc, ctx))
            out.append((obs, acts, true))
        return out

    def name(self, o):
        loc, rew, cue = next(k for k, v in self.obs.items() if v == o)
        return LOC[loc] + ['', ' ch', ' sh'][rew] + ('' if cue < 0 else f'·{cue}')


# ------------------------------------------------------------ 1-2 counts
def prefix_counts(trajs, n_obs):
    N, truth = {}, defaultdict(lambda: defaultdict(int))
    for obs, acts, true in trajs:
        h = (obs[0],)
        for t, a in enumerate(acts):
            N.setdefault(h, np.zeros((4, n_obs)))[a, obs[t + 1]] += 1
            truth[h][true[t]] += 1
            h = h + (a, obs[t + 1])
    return N, truth


# ------------------------------------------------------------ 4 BMR
def lnB(x):
    return gammaln(x).sum(-1) - gammaln(x.sum(-1))


def bmr(C, alpha):
    """Greedy merges while the evidence rises. C: (n, A, O). Returns labels, dF trace."""
    C = C.astype(float).copy()
    n = len(C)
    a = np.full(C.shape[-1], alpha)
    alive = np.ones(n, bool)
    lab = np.arange(n)
    own = lnB(a + C).sum(-1)
    base = C.shape[1] * lnB(a)                         # one prior term per action

    def row(i):
        return (lnB(a + C[i][None] + C).sum(-1) - own[i] - own + base)

    D = np.array([row(i) for i in range(n)])
    np.fill_diagonal(D, -np.inf)
    trace = []
    while alive.sum() > 1:
        M = np.where(alive[:, None] & alive[None], D, -np.inf)
        i, j = np.unravel_index(np.argmax(M), M.shape)
        if M[i, j] <= 0:
            trace.append(M[i, j])
            break
        trace.append(M[i, j])
        C[i] += C[j]; alive[j] = False; lab[lab == j] = i
        own[i] = lnB(a + C[i]).sum()
        D[i] = D[:, i] = row(i); D[i, i] = -np.inf
    _, lab = np.unique(lab, return_inverse=True)
    return lab, trace


# ------------------------------------------------------------ 5 machine
def machine(N, n_obs, alpha):
    H = list(N)
    lab, trace = bmr(np.array([N[h] for h in H]), alpha)
    K = lab.max() + 1
    of = dict(zip(H, lab))
    C = np.zeros((K, 4, n_obs))
    for h, k in of.items():
        C[k] += N[h]
    P = C / np.maximum(C.sum(-1, keepdims=True), 1)
    votes = defaultdict(lambda: defaultdict(float))
    last = defaultdict(lambda: defaultdict(float))
    for h, k in of.items():
        for a in range(4):
            for o in np.flatnonzero(N[h][a]):
                child = h + (a, int(o))
                if child in of:
                    votes[k, a, int(o)][of[child]] += N[h][a, o]
                    last[int(o)][of[child]] += N[h][a, o]
    f = {key: max(v, key=v.get) for key, v in votes.items()}
    last = {o: max(v, key=v.get) for o, v in last.items()}
    start = next(of[h] for h in H if len(h) == 1)
    return dict(K=K, C=C, P=P, f=f, last=last, start=start, of=of, trace=trace)


# ------------------------------------------------------------ 6 unmix
def unmix(m, alpha):
    C, P, K = m['C'], m['P'], m['K']
    alive, weights, report = list(range(K)), {}, {}
    while len(alive) > 2:
        best = None
        for k in alive:
            others = [j for j in alive if j != k]
            X = P[others].reshape(len(others), -1)
            A = np.vstack([X.T, 1e3 * np.ones(len(others))])
            w, _ = nnls(A, np.concatenate([P[k].ravel(), [1e3]]))
            w = np.where(w > 0.02, w, 0); w /= w.sum()
            mix = np.einsum('j,jao->ao', w, P[others])
            with np.errstate(divide='ignore'):
                lmix = float((C[k] * np.log(np.where(C[k] > 0, mix, 1))).sum())
            lfree = float((lnB(alpha + C[k]) - lnB(np.full(C.shape[-1], alpha))).sum())
            bf = lmix - lfree
            report.setdefault(k, bf)
            if best is None or bf > best[0]:
                best = (bf, k, dict(zip(others, w)))
        if best[0] <= 0:
            break
        bf, k, w = best
        report[k] = bf
        weights[k] = {j: v for j, v in w.items() if v > 0}
        alive.remove(k)
    return alive, weights, report


# ------------------------------------------------------------ 7 generator
def generator(m, V, weights):
    ix = {k: i for i, k in enumerate(V)}
    W = np.zeros((m['K'], len(V)))

    def vec(k):
        if k in ix:
            return np.eye(len(V))[ix[k]]
        return sum(w * vec(j) for j, w in weights[k].items())
    for k in range(m['K']):
        W[k] = vec(k)
    n_obs = m['P'].shape[-1]
    G = np.zeros((4, n_obs, len(V), len(V)))
    silent = 0
    for i, k in enumerate(V):
        for a in range(4):
            for o in np.flatnonzero(m['P'][k, a] > 0):
                o = int(o)
                s = m['f'].get((k, a, o))
                if s is None:
                    s, silent = m['last'].get(o), silent + 1
                if s is None:
                    continue
                q = W[s]
                if q[i] > 1e-9 and q.max() < 1 - 1e-9:
                    G[a, o, i, i] += m['P'][k, a, o]          # inertia
                else:
                    G[a, o, i] += m['P'][k, a, o] * q
    return dict(G=G, D=W[m['start']], W=W, silent=silent)


# ------------------------------------------------------------ 8 factorize
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
    S = G.sum(1) > tol
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


def run(trajs, n_obs):
    alpha = 1.0 / n_obs
    N, truth = prefix_counts(trajs, n_obs)
    m = machine(N, n_obs, alpha)
    V, weights, report = unmix(m, alpha)
    gen = generator(m, V, weights)
    sps = sp_partitions(gen['G'])
    return dict(N=N, truth=truth, m=m, V=V, weights=weights, report=report, gen=gen, sps=sps,
                prods=products(sps, len(V)))


def describe(res, env):
    """Name each learned state by the true (location, context) of its member prefixes."""
    tot = defaultdict(lambda: defaultdict(int))
    for h, k in res['m']['of'].items():
        for st, c in res['truth'][h].items():
            tot[k][st] += c
    out = {}
    for k, d in tot.items():
        n = sum(d.values())
        locs = {LOC[l] for l, _ in d}
        ctx = defaultdict(float)
        for (_, c), v in d.items():
            ctx[c] += v / n
        out[k] = '/'.join(sorted(locs)) + ' ' + ','.join(f'{"RLN"[c]}{p:.2f}' if p < 0.99 else "RLN"[c]
                                                         for c, p in sorted(ctx.items()))
    return out


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    for variant in ('pymdp', 'centre_only', 'three_ctx'):
        for T in (2, 3):
            env = MazeEnv(variant)
            trajs = env.sample(20000, T, 0)
            res = run(trajs, len(env.obs))
            nm = describe(res, env)
            print(f'\n== {variant}, T = {T}: {len(res["N"])} prefixes -> {res["m"]["K"]} states -> '
                  f'{len(res["V"])} after unmixing; {len(res["sps"])} SP partitions, '
                  f'{len(res["prods"])} products; silent transitions {res["gen"]["silent"]}')
            for k in sorted(nm, key=lambda k: nm[k]):
                bf = res['report'].get(k, float('nan'))
                mix = res['weights'].get(k)
                print(f'   {nm[k]:22s} lnBF {bf:9.1f}' + (f'  = mix {{{", ".join(f"{nm[j]}: {w:.2f}" for j, w in mix.items())}}}' if mix else ''))
            for p1, p2 in res['prods']:
                V = res['V']
                print('   product:', [[nm[V[i]] for i in B] for B in p1], '\n          x',
                      [[nm[V[i]] for i in B] for B in p2])
