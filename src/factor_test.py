"""Do the learned states factor into two independent binary hidden variables?

Four states can be laid on a 2x2 grid (x, y) in three ways (pick which pairs of
states share x; the other pairing gives y). For each grid and action, the dynamics
factorize when x' depends only on x and y' only on y:
    p(x', y' | x, y, a) = p(x' | x, a) p(y' | y, a).
Cross-talk measures the violation, in bits, under a uniform distribution over states:
    I(x'; y | x, a) + I(y'; x | y, a) + I(x'; y' | x, y, a),
zero iff the grid factorizes under that action.

Run from the repo root:  python src/factor_test.py [seed]
"""
import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from learned_pomdp import ACT, learned_model  # noqa: E402


def short(lab):
    o = lab.strip('{}').split(', ')[0].split('/')[1]
    return {'Cheese': 'Cheese trap', 'Shock': 'Shock trap'}.get(o, f'Cue: {o[:-3]}')


def H(p):
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def cond_mi(J, a, b, c):
    """I(a; b | c) in bits for a joint J over named axes; a, b, c are tuples of axis ids."""
    def ent(axes):
        keep = tuple(sorted(axes))
        M = J.sum(tuple(i for i in range(J.ndim) if i not in keep))
        return H(M.ravel())
    return ent(a + c) + ent(b + c) - ent(a + b + c) - ent(c)


def crosstalk(T, grid):
    """T[s', s]: transition among the 4 states for one action; grid[s] = (x, y)."""
    J = np.zeros((2, 2, 2, 2))                     # axes: x, y, x', y'
    for s, (x, y) in enumerate(grid):
        for t, (x2, y2) in enumerate(grid):
            J[x, y, x2, y2] += T[t, s] / len(grid)
    return (cond_mi(J, (2,), (1,), (0,)) + cond_mi(J, (3,), (0,), (1,))
            + cond_mi(J, (2,), (3,), (0, 1)))


def grids(n=4):
    """The three ways to place 4 states on a 2x2 grid, up to relabelling."""
    pairings = [p for p in itertools.combinations(range(n), 2) if 0 in p]
    seen = []
    for px, py in itertools.permutations(pairings, 2):
        key = frozenset((px, py))
        if key in seen:
            continue
        seen.append(key)
        yield px, py, [(int(s in px), int(s in py)) for s in range(n)]


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    A, B, labels = learned_model(seed)
    S = [short(l) for l in labels[1:]]
    Bs = B[1:, 1:, :]                               # among the 4 learned states
    print(f'Minimal maze, seed {seed}. Cross-talk in bits (0 = two independent variables)\n')
    for px, py, g in grids(len(S)):
        name = lambda p: '{' + ', '.join(S[i] for i in p) + '} vs rest'
        ct = [crosstalk(Bs[:, :, a], g) for a in range(len(ACT))]
        print(f'  x = {name(px)}')
        print(f'  y = {name(py)}')
        print('    ' + '   '.join(f'{ACT[a]}: {c:.3f}' for a, c in enumerate(ct))
              + f'   total: {sum(ct):.3f}\n')
