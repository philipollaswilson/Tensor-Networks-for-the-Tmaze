"""RGM state generation in Python (after SPM's spm_rgb2O + spm_MB_structure_learning).

Stops at the states: no variational inference, no parametric learning, no classification.
For deterministic (one-hot) data, the state and path that spm_VB_XXX infers for a block
are the unique-pattern index at the block's first step and the path used on its first
transition, so they are read off directly.

  discretise(I)       tiles of nd x nd pixels (R frames stacked when R > 1); SVD over time
                      per tile; keep modes with s > s_max / su (at most mm); each singular
                      variate binned into nb levels. Streams = (tile, mode), located at the
                      tile centre. Simplification: non-overlapping tiles, no Gaussian window.
  structure_fast(O)   every unique column of a group's outcomes is a latent state; for
                      dt >= 2, transitions get a path index (the first free slot of the
                      previous state, as spm_structure_fast)
  rg_levels(O, L)     spm_space: group streams on a grid halved in each direction; one
                      model per group; next level's outcomes are each group's state and path
                      at the start of each block of dt steps; stop at one group.
"""
import numpy as np


def discretise(I, nd, nb, mm, su, R=1):
    """I: (T, H, W) floats. Returns O (streams, T') ints, L (streams, 2) locations."""
    T = R * (len(I) // R)
    I = I[:T]
    H, W = I.shape[1:]
    O, L, info = [], [], []
    for r in range(0, H, nd):
        for c in range(0, W, nd):
            Y = I[:, r:r + nd, c:c + nd].reshape(T // R, -1)
            Y = Y - Y.mean(0)
            u, s, vt = np.linalg.svd(Y, full_matrices=False)
            k = int(min(mm, (s > s[0] / su).sum())) if s[0] > 1e-9 else 0
            for m in range(k):
                x = u[:, m] * s[m]
                d = np.abs(x).max()
                a = np.linspace(-d, d, nb)
                O.append(np.abs(x[:, None] - a[None]).argmin(1))
                L.append((r + nd / 2, c + nd / 2))
                info.append((r, c, m))
    return np.array(O), np.array(L, float), info


def structure_fast(O, dt):
    """O: (streams, T). Returns state per time, number of states, path per transition."""
    _, j = np.unique(O.T, axis=0, return_inverse=True)
    order = {}                                         # stable relabel, as MATLAB 'stable'
    j = np.array([order.setdefault(x, len(order)) for x in j.ravel()])
    path = np.zeros(len(j), int)
    if dt >= 2:
        slots = {}                                     # state -> {next state: path}
        for t in range(len(j) - 1):
            nxt = slots.setdefault(j[t], {})
            if j[t + 1] not in nxt:
                nxt[j[t + 1]] = len(nxt)
            path[t] = nxt[j[t + 1]]
        path[-1] = path[-2] if len(j) > 1 else 0
    return j, len(order), path


def space_groups(L):
    U = np.unique(L, axis=0)
    ng = int(np.ceil(np.sqrt(len(U) / 4)))
    if ng <= 1:
        return [np.arange(len(L))]
    xs = np.linspace(L[:, 0].min(), L[:, 0].max(), ng)
    ys = np.linspace(L[:, 1].min(), L[:, 1].max(), ng)
    R = np.array([(x, y) for x in xs for y in ys])
    g = ((L[:, None] - R[None]) ** 2).sum(-1).argmin(1)
    _, first = np.unique(g, return_index=True)
    return [np.flatnonzero(g == u) for u in g[np.sort(first)]]


def rg_levels(O, L, dt):
    """Returns a list of levels; each: groups, states per group, paths per group."""
    levels = []
    while True:
        G = space_groups(L)
        lev = dict(groups=G, T=O.shape[1], states=[], paths=[], j=[], p=[])
        for g in G:
            j, ns, p = structure_fast(O[g], dt)
            lev['states'].append(ns); lev['paths'].append(int(p.max()) + 1)
            lev['j'].append(j); lev['p'].append(p)
        levels.append(lev)
        if len(G) == 1:
            return levels
        starts = np.arange(0, O.shape[1] - (O.shape[1] % max(dt, 1)), max(dt, 1))
        O = np.array([x for k in range(len(G)) for x in (lev['j'][k][starts], lev['p'][k][starts])])
        L = np.array([L[g].mean(0) for g in G for _ in range(2)])


# ------------------------------------------------------------ data
def mnist(n, seed=0):
    """First n MNIST training images, preprocessed as DEM_MNIST_RGM.m: 32 x 32, Gaussian
    smoothing (2 px), histogram equalisation exp(-(rank - n)^2 / (24 n))."""
    import gzip
    import pathlib
    import urllib.request
    from scipy.ndimage import gaussian_filter, zoom
    root = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'mnist'
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'):
        f = root / name
        if not f.exists():
            urllib.request.urlretrieve('https://storage.googleapis.com/cvdf-datasets/mnist/' + name, f)
        with gzip.open(f) as fh:
            out[name] = np.frombuffer(fh.read(), np.uint8)
    X = out['train-images-idx3-ubyte.gz'][16:].reshape(-1, 28, 28)[:n].astype(float)
    y = out['train-labels-idx1-ubyte.gz'][8:][:n]
    I = []
    for x in X:
        t = gaussian_filter(zoom(x, 32 / 28, order=1), 2)
        i = np.argsort(t.ravel())
        npx = t.size
        e = np.empty(npx)
        e[i] = np.exp(-((np.arange(1, npx + 1) - npx) ** 2) / (24 * npx))
        I.append(np.clip(e.reshape(32, 32), 0, 1))
    return np.array(I), y


def bird(T=192, size=64, wing_period=8, bob_period=3):
    """A synthetic flapping bird: wing angle cycles with period 8, the body bobs with
    period 3 (independent). Ground truth: wing phase x bob phase, 24 distinct frames."""
    yy, xx = np.mgrid[:size, :size]
    frames, truth = [], []
    for t in range(T):
        w, b = t % wing_period, t % bob_period
        cy = size / 2 + [-4, 0, 4][b]
        cx = size / 2
        img = (((xx - cx) / 9) ** 2 + ((yy - cy) / 4) ** 2 <= 1).astype(float)
        ang = np.deg2rad(50 * np.cos(2 * np.pi * w / wing_period))
        for side in (-1, 1):
            for r in np.linspace(4, 24, 60):
                x = cx + side * r * np.cos(ang)
                y = cy - r * np.sin(ang)
                img[((xx - x) ** 2 + (yy - y) ** 2) <= 2.2 ** 2] = 1
        frames.append(img)
        truth.append((w, b))
    return np.array(frames), np.array(truth)
