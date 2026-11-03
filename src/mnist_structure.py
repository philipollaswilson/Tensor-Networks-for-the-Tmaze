"""Our structure learning on MNIST: images read as sequences of tiles, the label last.

Each image (32 x 32, preprocessed as DEM_MNIST_RGM) is cut into 4 x 4-pixel tiles; each
tile is one symbol: its first singular variate (SVD over training images) in 7 bins, as
spm_rgb2O with one mode. Tiles are read in raster order; the digit label is the last
symbol. The learner builds a unifilar machine layer by layer (one layer per position):
the candidates at layer t+1 are the (state, tile symbol) pairs seen at layer t, and BMR
merges candidates whose futures look alike. A candidate's "future" row is three Dirichlet
modalities: the next tile, the tile after it, and the label (so states keep what is
relevant to the digit as well as to the next pixels). Each layer's states form the hidden
variable at that position: the bond of a unifilar MPS.
Run:  python src/mnist_structure.py
"""
import numpy as np
from scipy.special import gammaln

import rgm_lite as RG

NB, ND = 7, 4


def lnB(x):
    return gammaln(x).sum(-1) - gammaln(x.sum(-1))


def tiles_fit(I):
    """SVD basis and bin edges per tile from training images; drops constant tiles."""
    T, H, W = I.shape
    basis = []
    for r in range(0, H, ND):
        for c in range(0, W, ND):
            Y = I[:, r:r + ND, c:c + ND].reshape(T, -1)
            mu = Y.mean(0)
            u, s, vt = np.linalg.svd(Y - mu, full_matrices=False)
            if s[0] < 1e-6:
                continue
            x = (Y - mu) @ vt[0]
            d = np.abs(x).max()
            basis.append((r, c, mu, vt[0], np.linspace(-d, d, NB)))
    return basis


def tiles_apply(I, basis):
    out = np.zeros((len(I), len(basis)), int)
    for k, (r, c, mu, v, a) in enumerate(basis):
        x = (I[:, r:r + ND, c:c + ND].reshape(len(I), -1) - mu) @ v
        out[:, k] = np.abs(x[:, None] - a[None]).argmin(1)
    return out


def merge_layer(rows, alpha):
    """Greedy BMR over candidates; rows: list of modality count arrays (n, K_m)."""
    n = len(rows[0])
    lab = np.arange(n)
    C = [r.astype(float).copy() for r in rows]
    a = [np.full(r.shape[1], al) for r, al in zip(rows, alpha)]
    alive = np.ones(n, bool)

    def ev(x, m):
        return lnB(a[m] + x)

    own = sum(ev(C[m], m) for m in range(len(C)))
    base = sum(lnB(a[m]) for m in range(len(C)))

    def row(i):
        return sum(ev(C[m][i][None] + C[m], m) for m in range(len(C))) - own[i] - own + base

    D = np.array([row(i) for i in range(n)])
    np.fill_diagonal(D, -np.inf)
    while alive.sum() > 1:
        M = np.where(alive[:, None] & alive[None], D, -np.inf)
        i, j = np.unravel_index(np.argmax(M), M.shape)
        if M[i, j] <= 0:
            break
        for m in range(len(C)):
            C[m][i] += C[m][j]
        alive[j] = False
        lab[lab == j] = i
        own[i] = sum(ev(C[m][i], m) for m in range(len(C)))
        D[i] = D[:, i] = row(i)
        D[i, i] = -np.inf
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def learn(X, y):
    """X: (N, P) tile symbols, y: labels. Returns the layered machine."""
    N, P = X.shape
    state = np.zeros(N, int)                       # layer 0: one state (empty history)
    layers = [dict(K=1, f={})]
    for t in range(P):
        cand = state * NB + X[:, t]                # (state, symbol) at this position
        keys, inv = np.unique(cand, return_inverse=True)
        rows = []
        for nxt, k in ((t + 1, NB), (t + 2, NB)):
            Rm = np.zeros((len(keys), k))
            if nxt < P:
                np.add.at(Rm, (inv, X[:, nxt]), 1)
            rows.append(Rm)
        Ry = np.zeros((len(keys), 10))
        np.add.at(Ry, (inv, y), 1)
        rows.append(Ry)
        lab = merge_layer(rows, [1 / NB, 1 / NB, 1 / 10])
        f = {int(c): int(l) for c, l in zip(keys, lab)}
        state = lab[inv]
        pl = np.zeros((lab.max() + 1, 10))
        np.add.at(pl, (state, y), 1)
        layers.append(dict(K=int(lab.max() + 1), f=f, label=pl, visits=pl.sum(1)))
    return layers


def run(layers, X):
    """Push images through the machine; unseen (state, symbol) pairs go to the most
    visited state reached by that symbol (else the most visited state)."""
    N, P = X.shape
    state = np.zeros(N, int)
    traj = [state.copy()]
    for t in range(P):
        L = layers[t + 1]
        by_sym = {}
        for c, l in L['f'].items():
            s = c % NB
            if s not in by_sym or L['visits'][l] > L['visits'][by_sym[s]]:
                by_sym[s] = l
        top = int(np.argmax(L['visits']))
        state = np.array([L['f'].get(int(s * NB + o), by_sym.get(int(o), top))
                          for s, o in zip(state, X[:, t])])
        traj.append(state.copy())
    return np.array(traj).T


def mutual_info(pl):
    p = pl / pl.sum()
    ps, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.nansum(p * np.log(p / (ps * py))))


if __name__ == '__main__':
    import time
    I, y = RG.mnist(12000)
    tr, te = slice(0, 10000), slice(10000, 12000)
    basis = tiles_fit(I[tr])
    Xtr, Xte = tiles_apply(I[tr], basis), tiles_apply(I[te], basis)
    t0 = time.time()
    layers = learn(Xtr, y[tr])
    print(f'{len(basis)} tile positions, learned in {time.time() - t0:.0f}s')
    print('states per position', [L['K'] for L in layers])
    pred = layers[-1]['label'].argmax(1)[run(layers, Xte)[:, -1]]
    print('test accuracy', (pred == y[te]).mean())
