"""Structure learning on images without sequences, tiles or a neighbourhood: a latent tree.

Pixels (MNIST, 14 x 14, binarised) are leaves; no order or locality is given. The learner
keeps a frontier of nodes, each with a state per training image, and repeats:
  1. join the two frontier nodes whose joint table has the largest Bayes factor for
     dependence over independence (Dirichlet evidence, Perks prior: total mass 1);
  2. the new node's candidate states are the observed (left state, right state) pairs;
     BMR merges candidates that predict the rest of the frontier alike (the m outside
     nodes with the largest dependence Bayes factor, each one Dirichlet modality).
So every node's states are classes of its subtree's configuration by what they say about
the complement: the bond of a tree tensor network across that cut. The same learner also
runs with fixed shapes (a raster chain = MPS, a k-d spatial tree = locality prior) for
comparison. Test likelihood by exact upward sum-product on the tree (smoothed tables).
Run:  python src/latent_tree.py
"""
import gzip
import pathlib

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import gammaln

from mnist_structure import merge_layer

ROOT = pathlib.Path(__file__).resolve().parents[1]
SIDE = 14


def lnB(x):
    return gammaln(x).sum(-1) - gammaln(x.sum(-1))


def load(n_train=10000, n_test=2000):
    """Raw MNIST, 2 x 2 average-pooled to 14 x 14, binarised at half intensity."""
    out = {}
    for name in ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'):
        with gzip.open(ROOT / 'data' / 'mnist' / name) as fh:
            out[name] = np.frombuffer(fh.read(), np.uint8)
    n = n_train + n_test
    X = out['train-images-idx3-ubyte.gz'][16:].reshape(-1, 28, 28)[:n].astype(float)
    X = X.reshape(n, SIDE, 2, SIDE, 2).mean((2, 4)) > 127.5
    y = out['train-labels-idx1-ubyte.gz'][8:][:n].astype(int)
    X = X.reshape(n, -1).astype(int)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def entropy(n):
    p = n[n > 0] / n.sum()
    return float(-(p * np.log(p)).sum())


def dep_bf(s, Ks, t, Kt):
    """ln Bayes factor: joint Dirichlet table vs. independent margins (Perks priors)."""
    n = np.bincount(s * Kt + t, minlength=Ks * Kt).astype(float)
    J = Ks * Kt
    joint = lnB(n + 1 / J) - lnB(np.full(J, 1 / J))
    ns, nt = n.reshape(Ks, Kt).sum(1), n.reshape(Ks, Kt).sum(0)
    ind = (lnB(ns + 1 / Ks) - lnB(np.full(Ks, 1 / Ks))
           + lnB(nt + 1 / Kt) - lnB(np.full(Kt, 1 / Kt)))
    return joint - ind


class Tree:
    """Nodes 0..P-1 are pixels; internal nodes are appended as they are joined."""

    def __init__(self, X, m=None, score='evidence'):
        self.X, self.m, self.score = X, m or len(X), score
        self.pix = list(np.flatnonzero(X.std(0) > 0))     # constant pixels carry nothing
        self.nodes = [dict(pix=[p], K=2, s=X[:, p].copy(), kids=None) for p in self.pix]
        self.frontier = list(range(len(self.nodes)))
        self.log = []                                     # frontier snapshots

    # --------------------------------------------------------------- one join
    def join(self, i, j):
        A, B = self.nodes[i], self.nodes[j]
        cand = A['s'] * B['K'] + B['s']
        keys, inv = np.unique(cand, return_inverse=True)
        out = [k for k in self.frontier if k not in (i, j)]
        rows, alpha = [], []
        if out:
            bf = np.array([dep_bf(inv, len(keys), self.nodes[k]['s'], self.nodes[k]['K'])
                           for k in out])
            for r in np.argsort(-bf)[:self.m]:
                if bf[r] <= 0:
                    break
                Nk = self.nodes[out[r]]
                R = np.zeros((len(keys), Nk['K']))
                np.add.at(R, (inv, Nk['s']), 1)
                rows.append(R)
                alpha.append(1 / Nk['K'])
        lab = merge_layer(rows, alpha) if rows else np.zeros(len(keys), int)
        K = int(lab.max() + 1)
        s = lab[inv]
        # tables for encoding and likelihood: counts n[c, a, b] (sparse), margins
        trip, cnt = np.unique(np.stack([s, A['s'], B['s']], 1), axis=0, return_counts=True)
        node = dict(pix=A['pix'] + B['pix'], K=K, s=s, kids=(i, j), trip=trip, cnt=cnt,
                    f={int(c): int(l) for c, l in zip(keys, lab)},
                    nc=np.bincount(s, minlength=K))
        node['pa'] = np.zeros((K, A['K'])); np.add.at(node['pa'], (trip[:, 0], trip[:, 1]), cnt)
        node['pb'] = np.zeros((K, B['K'])); np.add.at(node['pb'], (trip[:, 0], trip[:, 2]), cnt)
        self.nodes.append(node)
        k = len(self.nodes) - 1
        self.frontier = [x for x in self.frontier if x not in (i, j)] + [k]
        self.log.append(list(self.frontier))
        return k

    # --------------------------------------------------------------- shapes
    def learn(self, verbose=False):
        """Greedy: join the most dependent pair (largest dependence Bayes factor)."""
        S = {}
        for a in range(len(self.frontier)):
            for b in range(a + 1, len(self.frontier)):
                i, j = self.frontier[a], self.frontier[b]
                S[i, j] = self._bf(i, j)
        while len(self.frontier) > 1:
            (i, j) = max(S, key=S.get)
            k = self.join(i, j)
            S = {key: v for key, v in S.items() if i not in key and j not in key}
            for x in self.frontier[:-1]:
                S[x, k] = self._bf(x, k)
            if verbose and len(self.frontier) % 20 == 0:
                print(f'  frontier {len(self.frontier)}, last K {self.nodes[k]["K"]}')
        return self

    def _bf(self, i, j):
        A, B = self.nodes[i], self.nodes[j]
        bf = dep_bf(A['s'], A['K'], B['s'], B['K'])
        if self.score == 'redundancy':                # Bayes factor per nat of the smaller side
            h = [entropy(np.bincount(N['s'])) for N in (A, B)]
            return bf / (len(A['s']) * max(min(h), 1e-9))
        return bf

    def fixed(self, nested):
        """nested: tuple tree over pixel indices; joins lower subtrees first."""
        leaf = {p: n for n, p in enumerate(self.pix)}
        items = []

        def walk(t):
            if not isinstance(t, tuple):
                return leaf.get(t), 0
            parts = [walk(c) for c in t]
            parts = [p for p in parts if p[0] is not None]
            if not parts:
                return None, 0
            if len(parts) == 1:
                return parts[0]
            ref = ('pending', len(items))
            h = max(p[1] for p in parts) + 1
            items.append(([p[0] for p in parts], h))
            return ref, h
        walk(nested)
        ids = {}
        for n, (kids, h) in sorted(enumerate(items), key=lambda x: x[1][1]):
            kids = [ids[k[1]] if isinstance(k, tuple) else k for k in kids]
            ids[n] = self.join(*kids)
        return self

    # --------------------------------------------------------------- use
    def encode(self, X):
        """Deterministic states of new images; unseen pairs -> argmax p(c) p(a|c) p(b|c)."""
        S = {n: X[:, p] for n, p in enumerate(self.pix)}
        for k in range(len(self.pix), len(self.nodes)):
            N = self.nodes[k]
            i, j = N['kids']
            a, b = S[i], S[j]
            Kb = self.nodes[j]['K']
            sc = (np.log(N['nc'] + 1)[None]
                  + np.log((N['pa'] + 1 / N['pa'].shape[1]) / (N['nc'][:, None] + 1)).T[a]
                  + np.log((N['pb'] + 1 / N['pb'].shape[1]) / (N['nc'][:, None] + 1)).T[b])
            fb = sc.argmax(1)
            S[k] = np.array([N['f'].get(int(x * Kb + z), int(g)) for x, z, g in zip(a, b, fb)])
        return S

    def loglik(self, X):
        """log p(x) per image (nats) by upward sum-product on the latent tree with
        p(a, b | c) = (n_cab + 1/(Ka Kb)) / (n_c + 1)."""
        msg = {n: np.eye(2)[X[:, p]] for n, p in enumerate(self.pix)}
        logz = np.zeros(len(X))
        for k in range(len(self.pix), len(self.nodes)):
            N = self.nodes[k]
            i, j = N['kids']
            ma, mb = msg.pop(i), msg.pop(j)
            Ka, Kb = ma.shape[1], mb.shape[1]
            w = N['cnt'] / (N['nc'][N['trip'][:, 0]] + 1)
            W = csr_matrix((w, (np.arange(len(w)), N['trip'][:, 0])), shape=(len(w), N['K']))
            m = np.asarray((ma[:, N['trip'][:, 1]] * mb[:, N['trip'][:, 2]]) @ W)
            m +=(ma.sum(1) * mb.sum(1))[:, None] / (Ka * Kb * (N['nc'] + 1))[None]
            z = m.sum(1)
            logz += np.log(z)
            msg[k] = m / z[:, None]
        root = self.nodes[-1]
        pc = (root['nc'] + 1 / root['K']) / (root['nc'].sum() + 1)
        return logz + np.log(msg[len(self.nodes) - 1] @ pc)

    def readout(self, Str, ytr, Ste):
        """Digit from the two children of the root (the top cut): majority label per pair,
        naive Bayes over the two states for unseen pairs."""
        i, j = self.nodes[-1]['kids']
        Ka, Kb = self.nodes[i]['K'], self.nodes[j]['K']
        pair = Str[i] * Kb + Str[j]
        tab = {}
        for p_, yy in zip(pair, ytr):
            tab.setdefault(int(p_), np.zeros(10))[yy] += 1
        la = np.zeros((10, Ka)); np.add.at(la, (ytr, Str[i]), 1)
        lb = np.zeros((10, Kb)); np.add.at(lb, (ytr, Str[j]), 1)
        py = np.bincount(ytr, minlength=10) + 1
        nb = (np.log(py)[None] + np.log((la + 0.1) / (py[:, None] + 0.1 * Ka)).T[Ste[i]]
              + np.log((lb + 0.1) / (py[:, None] + 0.1 * Kb)).T[Ste[j]]).argmax(1)
        tp = Ste[i] * Kb + Ste[j]
        return np.array([tab[int(t)].argmax() if int(t) in tab else g for t, g in zip(tp, nb)])


# ------------------------------------------------------------------- reference shapes
def raster_chain():
    t = 0
    for p in range(1, SIDE * SIDE):
        t = (t, p)
    return t


def kd_tree(r0=0, r1=SIDE, c0=0, c1=SIDE):
    """Spatial binary tree: split the longer side in half (the locality prior)."""
    if (r1 - r0) * (c1 - c0) == 1:
        return r0 * SIDE + c0
    if r1 - r0 >= c1 - c0:
        h = (r0 + r1) // 2
        return (kd_tree(r0, h, c0, c1), kd_tree(h, r1, c0, c1))
    h = (c0 + c1) // 2
    return (kd_tree(r0, r1, c0, h), kd_tree(r0, r1, h, c1))


# ------------------------------------------------------------------- baselines
def independent_ll(Xtr, Xte, pix):
    p = (Xtr[:, pix].sum(0) + 0.5) / (len(Xtr) + 1)
    return (Xte[:, pix] * np.log(p) + (1 - Xte[:, pix]) * np.log(1 - p)).sum(1)


def chow_liu_ll(Xtr, Xte, pix):
    """Best visible tree over pixels (Chow & Liu): maximum spanning tree of pairwise MI."""
    from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
    Z = Xtr[:, pix].astype(float)
    n = len(Z)
    p11 = Z.T @ Z / n
    p1 = Z.mean(0)
    P = [p11, p1[:, None] - p11, p1[None] - p11, 1 - p1[:, None] - p1[None] + p11]
    q = [p1[:, None] * p1[None], p1[:, None] * (1 - p1[None]), (1 - p1[:, None]) * p1[None],
         (1 - p1[:, None]) * (1 - p1[None])]
    with np.errstate(divide='ignore', invalid='ignore'):
        mi = sum(np.nansum(np.where(a > 0, a * np.log(a / b), 0)[None], 0) for a, b in zip(P, q))
    np.fill_diagonal(mi, 0)
    T = minimum_spanning_tree(-(mi + 1e-9)).toarray()
    T = (T != 0) | (T.T != 0)
    order, parent = breadth_first_order(T, 0, directed=False)
    Zt = Xte[:, pix]
    ll = np.zeros(len(Zt))
    for v in order:
        u = parent[v]
        if u < 0:
            p = (Z[:, v].sum() + 0.5) / (n + 1)
            ll += np.where(Zt[:, v], np.log(p), np.log(1 - p))
            continue
        for val in (0, 1):
            sel = Z[:, u] == val
            p = (Z[sel, v].sum() + 0.5) / (sel.sum() + 1)
            on = Zt[:, u] == val
            ll[on] += np.where(Zt[on, v], np.log(p), np.log(1 - p))
    return ll


def mutual_info(s, K, y):
    J = np.zeros((K, 10)); np.add.at(J, (s, y), 1)
    p = J / J.sum()
    ps, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.nansum(p * np.log(p / (ps * py))))


def node_info(tree, Str, ytr):
    """I(node state ; digit) / ln 10 for every internal node (train)."""
    return np.array([mutual_info(Str[k], tree.nodes[k]['K'], ytr)
                     for k in range(len(tree.pix), len(tree.nodes))]) / np.log(10)


def node_acc(tree, k, Str, ytr, Ste, yte):
    """Digit from one node's state: majority training label of the state."""
    J = np.zeros((tree.nodes[k]['K'], 10)); np.add.at(J, (Str[k], ytr), 1)
    return (J.argmax(1)[Ste[k]] == yte).mean()


def evaluate(tree, Xtr, ytr, Xte, yte):
    Str, Ste = tree.encode(Xtr), tree.encode(Xte)
    internal = tree.nodes[len(tree.pix):]
    info = node_info(tree, Str, ytr)
    best = len(tree.pix) + int(info.argmax())          # chosen on training data
    return dict(
        best_acc=node_acc(tree, best, Str, ytr, Ste, yte),
        best_size=len(tree.nodes[best]['pix']), best_K=tree.nodes[best]['K'],
        nll_bits=-tree.loglik(Xte).mean() / np.log(2),
        nll_train_bits=-tree.loglik(Xtr[:2000]).mean() / np.log(2),
        acc=(tree.readout(Str, ytr, Ste) == yte).mean(),
        Kmax=max(N['K'] for N in internal),
        states=sum(N['K'] for N in internal),
        params=sum(len(N['cnt']) for N in internal),
        top_mi=max(mutual_info(N['s'], N['K'], ytr) for N in internal) / np.log(10))


if __name__ == '__main__':
    import time
    Xtr, ytr, Xte, yte = load()
    rows = {}
    for name, make in (('learned (evidence)', lambda: Tree(Xtr).learn()),
                       ('learned (redundancy)', lambda: Tree(Xtr, score='redundancy').learn()),
                       ('k-d (locality)', lambda: Tree(Xtr).fixed(kd_tree())),
                       ('raster chain (MPS)', lambda: Tree(Xtr).fixed(raster_chain()))):
        t0 = time.time()
        tree = make()
        r = evaluate(tree, Xtr, ytr, Xte, yte)
        rows[name] = r
        print(f'{name:20s} {time.time() - t0:5.0f}s', {k: round(float(v), 3) for k, v in r.items()})
    pix = Tree(Xtr).pix
    print('independent pixels  nll bits', -independent_ll(Xtr, Xte, pix).mean() / np.log(2))
    print('Chow-Liu tree       nll bits', -chow_liu_ll(Xtr, Xte, pix).mean() / np.log(2))
