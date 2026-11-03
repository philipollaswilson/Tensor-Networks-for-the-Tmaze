"""Frozen lake: structure learning, the learned POMDP, EFE, and the factor test.

The pipeline of bmr_states.py -> learned_pomdp.py -> factor_test.py, run on the
frozen-lake MPS (frozenlake_train.py):

  1. histories h = (a1, o1, ..., a4, o4) cut at the bond before the last site;
     counts n_{h,a} over the next observation o5, model-faithful as in
     bmr_states.min_counts (learned p(o5 | h, a) times training visits).
  2. BMR groups the histories into predictive states (truth: the 9 cells).
  3. states -> POMDP: A[o, s] = the cell each state shows, B from pooled counts,
     D = the state showing S. Compared with the true transition table.
  4. sophisticated EFE to horizon 4 (the goal is 4 moves from S), on the learned
     and on the true model.
  5. factor test: do the learned states factor into two independent 3-valued
     variables, and is (row, column) the best such split?

Run from the repo root:  python src/frozenlake_pomdp.py [seed]
"""
import functools
import itertools
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bmr_states import lnB  # noqa: E402
from factor_test import cond_mi  # noqa: E402
from frozenlake_train import ACT, BATCH, GOAL, HOLE, NS, OBS, T, enumerate_weighted, true_B  # noqa: E402,F401
from learned_pomdp import infogain, risk  # noqa: E402
from observable_embedding import load  # noqa: E402

N_TRAJ = 10000
HORIZON = 4


# ------------------------------------------------------------ counts and BMR
def counts(seed, mode='model', name=None):
    """Histories over the first T-1 steps and their (a_T, o_T) counts."""
    Ts = load(name or f'frozenlake_s{seed}')
    leaves = enumerate_weighted(true_B())
    w = np.array([x[1] for x in leaves]); w /= w.sum()
    draws = np.random.default_rng(seed).choice(len(leaves), size=N_TRAJ, p=w)
    tab = {}
    for i in draws[:int(0.8 * N_TRAJ)]:
        idx = leaves[i][0]
        tab.setdefault(idx[:-2], np.zeros((len(ACT), NS)))[idx[-2], idx[-1]] += 1
    hists, cnts = [], []
    for h, n in sorted(tab.items()):
        if mode == 'model':
            b = Ts[0][:, h[0], h[1], :].sum(0)
            for k, Tk in enumerate(Ts[1:-1], start=1):
                b = b @ Tk[:, h[2 * k], h[2 * k + 1], :]
            p = np.abs(np.einsum('i,iao->ao', b, Ts[-1][..., 0])) ** 2
            p /= p.sum(1, keepdims=True)
            n = p * n.sum(1, keepdims=True)
        hists.append(h); cnts.append(n)
    return hists, cnts


def bmr(counts, alpha):
    """bmr_states.bmr with the pairwise dF cached (same greedy merges, O(n^2))."""
    cl = [[i] for i in range(len(counts))]
    C = [c.astype(float) for c in counts]
    a = np.full(counts[0].shape[1], alpha)
    base = lnB(a)

    def dF(x, y):
        return float((lnB(a + x + y) - lnB(a + x) - lnB(a + y) + base).sum())

    D = np.full((len(C), len(C)), -np.inf)
    for i, j in itertools.combinations(range(len(C)), 2):
        D[i, j] = dF(C[i], C[j])
    alive = list(range(len(C)))
    accepted = []
    while len(alive) > 1:
        sub = D[np.ix_(alive, alive)]
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        i, j = alive[i], alive[j]
        if D[i, j] <= 0:
            return [cl[k] for k in alive], accepted, float(D[i, j])
        accepted.append(float(D[i, j]))
        cl[i] += cl[j]; C[i] = C[i] + C[j]
        alive.remove(j); D[j, :] = D[:, j] = -np.inf
        for k in alive:
            if k != i:
                lo, hi = min(i, k), max(i, k)
                D[lo, hi] = dF(C[i], C[k])
    return [cl[k] for k in alive], accepted, None


def hist_name(h):
    return ' '.join(f'{ACT[a][0]}{o}' for a, o in zip(h[0::2], h[1::2]))


# ------------------------------------------------------------ the learned model
def learned_model(seed, alpha=0.1, name=None):
    hists, cnts = counts(seed, 'model', name)
    clusters, _, _ = bmr(cnts, alpha)
    state_of_obs, cells = {}, []
    for c in clusters:
        shown = {hists[i][-1] for i in c}
        assert len(shown) == 1, f'a learned state shows several cells: {sorted(shown)}'
        o = shown.pop()
        assert o not in state_of_obs, f'cell {OBS[o]} shown by two states'
        state_of_obs[o] = len(cells); cells.append(o)
    order = np.argsort(cells)                                   # states in cell order
    clusters = [clusters[k] for k in order]
    cells = [cells[k] for k in order]
    state_of_obs = {o: s for s, o in enumerate(cells)}
    nS = len(cells)

    A = np.zeros((NS, nS))
    for s, o in enumerate(cells):
        A[o, s] = 1
    B = np.zeros((nS, nS, len(ACT)))
    for k, c in enumerate(clusters):
        n = sum(cnts[i] for i in c)
        q = n / n.sum(1, keepdims=True)
        for a in range(len(ACT)):
            for o, s in state_of_obs.items():
                B[s, k, a] += q[a, o]
    B /= B.sum(0, keepdims=True)
    D = np.eye(nS)[state_of_obs[0]]
    return A, B, D, cells, clusters, hists


def true_model():
    return np.eye(NS), true_B(), np.eye(NS)[0]


def preferences(goal=2.0, hole=-4.0):
    C = np.zeros(NS)
    C[GOAL], C[HOLE] = goal, hole
    return C


# ------------------------------------------------------------ EFE, any horizon
def sophisticated(A, B, C, q, horizon, gamma=16.0):
    """G(a) at belief q, looking `horizon` steps ahead; later actions are drawn
    from sigma(-gamma G) and averaged, as in learned_pomdp.sophisticated."""
    @functools.lru_cache(maxsize=None)
    def G(key, h):
        q = np.frombuffer(key)
        out = []
        for a in range(B.shape[2]):
            q1 = B[:, :, a] @ q
            q_o = A @ q1
            g = risk(q_o, C) - infogain(A, q1)
            if h > 1:
                for o in np.flatnonzero(q_o > 1e-12):
                    post = A[o] * q1 / q_o[o]
                    nxt = G(np.round(post, 12).tobytes(), h - 1)
                    pi = np.exp(-gamma * (nxt - nxt.min())); pi /= pi.sum()
                    g += q_o[o] * float(pi @ nxt)
            out.append(g)
        return np.array(out)
    g = G(np.round(q, 12).tobytes(), horizon)
    post = np.exp(-gamma * (g - g.min())); post /= post.sum()
    return g, post


# ------------------------------------------------------------ factor test
def crosstalk(Tm, grid, n=3):
    """factor_test.crosstalk for an n x n grid: Tm[s', s], grid[s] = (x, y)."""
    J = np.zeros((n, n, n, n))
    for s, (x, y) in enumerate(grid):
        for t, (x2, y2) in enumerate(grid):
            J[x, y, x2, y2] += Tm[t, s] / len(grid)
    return (cond_mi(J, (2,), (1,), (0,)) + cond_mi(J, (3,), (0,), (1,))
            + cond_mi(J, (2,), (3,), (0, 1)))


def grids(n=3):
    """Every way to place n*n states on an n x n grid, up to relabelling the
    values of x and of y and swapping x with y."""
    seen = set()

    def partitions(items):
        if not items:
            yield []
            return
        first, rest = items[0], items[1:]
        for mates in itertools.combinations(rest, n - 1):
            left = [i for i in rest if i not in mates]
            for p in partitions(left):
                yield [(first,) + mates] + p

    for rows in partitions(list(range(n * n))):
        for perms in itertools.product(itertools.permutations(range(n)), repeat=n - 1):
            cols = [tuple([rows[0][c]] + [rows[r + 1][perms[r][c]] for r in range(n - 1)])
                    for c in range(n)]
            key = frozenset([frozenset(map(frozenset, rows)), frozenset(map(frozenset, cols))])
            if key in seen:
                continue
            seen.add(key)
            g = [None] * (n * n)
            for x, row in enumerate(rows):
                for s in row:
                    g[s] = [x, None]
            for y, col in enumerate(cols):
                for s in col:
                    g[s][1] = y
            yield [tuple(v) for v in g]


def factor_report(B, cells):
    rowcol = [(o // 3, o % 3) for o in cells]
    totals = []
    for g in grids():
        totals.append(sum(crosstalk(B[:, :, a], g) for a in range(len(ACT))))
    totals = np.array(totals)
    per_a = [crosstalk(B[:, :, a], rowcol) for a in range(len(ACT))]
    rank = int((totals < sum(per_a) - 1e-9).sum()) + 1
    return per_a, totals, rank


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hists, cnts = counts(seed)
    clusters, acc, rej = bmr(cnts, 0.1)
    print(f'Frozen lake, seed {seed}: {len(hists)} histories of {T - 1} steps')
    print(f'BMR (alpha 0.1): {len(clusters)} states (truth 9); weakest accepted dF '
          f'{min(acc):+.1f}, best refused dF {rej:+.1f}')

    A, B, D, cells, clusters, hists = learned_model(seed)
    Bt = true_B()[np.ix_(cells, cells)]
    print('\n  state | shows | histories | max_a TV(learned B, true B)')
    for s, (o, c) in enumerate(zip(cells, clusters)):
        tv = max(0.5 * np.abs(B[:, s, a] - Bt[:, s, a]).sum() for a in range(len(ACT)))
        print(f'  s{s}    | {OBS[o]:5s} | {len(c):9d} | {tv:.3f}')

    C = preferences()
    print(f'\nPreferences goal +2 / hole -4, sophisticated EFE, horizon {HORIZON}, gamma 16')
    gl, ql = sophisticated(A, B, C, D, HORIZON)
    At, Btrue, Dt = true_model()
    gt, qt = sophisticated(At, Btrue, C, Dt, HORIZON)
    print('  action | G (learned / true) | q(a) (learned / true)')
    for a in range(len(ACT)):
        print(f'  {ACT[a]:6s} | {gl[a]:7.3f} / {gt[a]:7.3f} | {ql[a]:.3f} / {qt[a]:.3f}')

    per_a, totals, rank = factor_report(B, cells)
    print('\nFactor test: x = row, y = column. Cross-talk in bits (0 = independent)')
    print('    ' + '   '.join(f'{ACT[a]}: {c:.3f}' for a, c in enumerate(per_a))
          + f'   total: {sum(per_a):.3f}')
    ref = [crosstalk(Btrue[:, :, a], [(o // 3, o % 3) for o in range(NS)]) for a in range(len(ACT))]
    print('    true model: ' + '   '.join(f'{ACT[a]}: {c:.3f}' for a, c in enumerate(ref))
          + f'   total: {sum(ref):.3f}')
    print(f'  row x column ranks {rank} of {len(totals)} ways to split the 9 states into a 3x3 grid'
          f' (best {totals.min():.3f}, median {np.median(totals):.3f} bits)')
