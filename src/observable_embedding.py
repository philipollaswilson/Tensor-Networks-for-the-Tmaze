"""Gauge- and phase-invariant hidden-state recovery from an MPS bond.

Each history h gives a bond vector b_h. Every future Born probability is linear
in R_h = b_h b_h^dagger:  p(f | h) = sum_jk R_h[j,k] E_f[j,k],  E_f = M_f M_f^dagger,
where M_f is the future part of the network contracted with future f.

The L2 distance between two histories' future distributions,
    d(i,j)^2 = sum_f (p_i(f) - p_j(f))^2 = (r_i - r_j)^T S conj(r_i - r_j),
needs only S = sum_f vec(E_f) vec(E_f)^dagger, built right-to-left by the
recursion S <- sum_{a,label} K S K^dagger with K = sum_{o in label} A_ao (x) conj(A_ao)
(a four-copy contraction; cost polynomial in the bond dimension, linear in length,
no enumeration of futures). rank(S) is the dimension of the observable subspace;
directions of R outside it (unobservable bond components, irrelevant phases)
drop out. Modalities are marginalised by grouping observations into labels.

Run from the repo root:  python src/observable_embedding.py
"""
import itertools
import pathlib
import sys

import numpy as np
import torch

R = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R), str(R / 'src')]
import seed_protocol as SP  # noqa: E402

FULL, MIN, oi = SP.FULL, SP.MIN, SP.obs_index


# ---------------------------------------------------------------- core
def site_ops(T, labels):
    """Group a site tensor T[l, a, o, r] into K operators, one per (action, label)."""
    Ks = []
    for a in range(T.shape[1]):
        for lab in sorted(set(labels)):
            K = sum(np.kron(T[:, a, o, :], T[:, a, o, :].conj())
                    for o in range(T.shape[2]) if labels[o] == lab)
            Ks.append(K)
    return Ks


def future_gram(sites, labels):
    """S at the bond left of sites[0], and the total-mass vector e_tot."""
    S = np.ones((1, 1), complex)
    e_tot = np.ones(1, complex)
    for T in reversed(sites):
        Ks = site_ops(T, labels)
        S = sum(K @ S @ K.conj().T for K in Ks)
        e_tot = sum(K @ e_tot for K in Ks)
    return S, e_tot


def embed(bs, S, e_tot, tol=1e-10):
    """Embedding y_h with |y_i - y_j| = L2 distance of future distributions."""
    w, U = np.linalg.eigh(S)
    keep = w > tol * w.max()
    L = U[:, keep] * np.sqrt(w[keep])
    ys = []
    for b in bs:
        r = np.outer(b, b.conj()).ravel()
        r = r / (r @ e_tot).real                       # normalise to a distribution
        ys.append(L.conj().T @ r.conj())
    return np.array(ys), int(keep.sum())


def brute_future_probs(b, sites, labels):
    """Enumerate futures explicitly (for checking only)."""
    out = {}
    nlab = sorted(set(labels))
    for f in itertools.product(*[[(a, o) for a in range(T.shape[1]) for o in range(T.shape[2])]
                                 for T in sites]):
        M = np.eye(sites[0].shape[0])
        for T, (a, o) in zip(sites, f):
            M = M @ T[:, a, o, :]
        key = tuple((a, labels[o]) for a, o in f)
        out[key] = out.get(key, 0.0) + abs(b @ M[:, 0]) ** 2
    p = np.array([out[k] for k in sorted(out)])
    return p / p.sum()


def average_linkage(Y):
    """Full average-linkage dendrogram on Euclidean distances; merge heights."""
    D = np.linalg.norm(Y[:, None] - Y[None], axis=-1)
    cl = [[i] for i in range(len(Y))]
    hs, parts = [], [[c[:] for c in cl]]
    while len(cl) > 1:
        best = min(((D[np.ix_(cl[i], cl[j])].mean(), i, j)
                    for i in range(len(cl)) for j in range(i + 1, len(cl))))
        d, i, j = best
        hs.append(d)
        cl[i] += cl.pop(j)
        parts.append([c[:] for c in cl])
    return np.array(hs), parts


def gap_k(hs, n):
    """Number of clusters at the largest jump between consecutive merge heights."""
    gaps = np.diff(np.concatenate([[0.0], hs]))
    m = int(np.argmax(gaps))                          # merge m is the first refused
    return n - m, hs[m - 1] if m else 0.0, hs[m]


def wavefunction_fidelity(bs, sites):
    """Fidelity of the future wavefunctions (gauge-invariant amplitude overlap)."""
    psis = []
    for b in bs:
        v = []
        for f in itertools.product(*[[(a, o) for a in range(T.shape[1]) for o in range(T.shape[2])]
                                     for T in sites]):
            M = np.eye(sites[0].shape[0])
            for T, (a, o) in zip(sites, f):
                M = M @ T[:, a, o, :]
            v.append(b @ M[:, 0])
        v = np.array(v)
        psis.append(v / np.linalg.norm(v))
    return [p for p in psis]


# ---------------------------------------------------------------- models
def load(name):
    mps = torch.load(R / 'Saved_Models' / f'{name}.pt', weights_only=False)
    return [m.detach().cpu().numpy() for m in mps.matrices]


def full_histories(T1, T2):
    q = FULL.exact_joint(FULL.enumerate_weighted())
    P = ['ctr', 'R', 'L', 'cue']; RW = ['-', 'ch', 'sh']
    hs, bs = [], []
    for o1 in (oi(0, 0, 0), oi(0, 0, 1)):
        for a2 in range(4):
            for o2 in range(24):
                if q[0, o1, a2, o2].sum() < 1e-12:
                    continue
                b = np.einsum('i,ij->j', T1[0, 0, o1, :], T2[:, a2, o2, :])
                if np.linalg.norm(b) < 1e-9:
                    continue
                hs.append(f'{P[o2 // 6]}/{RW[(o2 // 2) % 3]}/c{o2 % 2}/s{o1 % 2}')
                bs.append(b)
    return hs, bs


def min_histories(T1):
    q = MIN.exact_joint()
    hs, bs = [], []
    for a1 in range(3):
        for o2 in range(4):
            if q[a1, o2].sum() < 1e-9:
                continue
            hs.append(f'{MIN.ACT[a1]}/{MIN.OBS[o2]}')
            bs.append(T1[:, a1, o2, :].sum(0))
    return hs, bs


LAB_FULL = {'pos+rew': [3 * (o // 6) + (o // 2) % 3 for o in range(24)],
            'all':     list(range(24)),
            'rew':     [(o // 2) % 3 for o in range(24)]}


def site_ops_action(T, labels, a):
    return [sum(np.kron(T[:, a, o, :], T[:, a, o, :].conj())
                for o in range(T.shape[2]) if labels[o] == lab) for lab in sorted(set(labels))]


def embed_conditional(bs, sites, labels, tol=1e-10):
    """Action-conditioned embedding: one block per future action sequence, each
    normalised to p(obs | h, actions); |y_i - y_j|^2 = mean over action sequences
    of the L2^2 distance. Enumerates action sequences (|A|^H), never observations."""
    blocks, rank = [], 0
    for acts in itertools.product(*[range(T.shape[1]) for T in sites]):
        S = np.ones((1, 1), complex); e = np.ones(1, complex)
        for T, a in zip(reversed(sites), reversed(acts)):
            Ks = site_ops_action(T, labels, a)
            S = sum(K @ S @ K.conj().T for K in Ks)
            e = sum(K @ e for K in Ks)
        Y, r = embed(bs, S, e, tol)
        blocks.append(Y); rank = max(rank, r)
    return np.concatenate(blocks, 1) / np.sqrt(len(blocks)), rank


def report(tag, hist, bs, sites, labels, truth, conditional=True):
    if conditional:
        Y, rank = embed_conditional(bs, sites, labels)
    else:
        S, e = future_gram(sites, labels)
        Y, rank = embed(bs, S, e)
    hs, parts = average_linkage(Y)
    k, below, above = gap_k(hs, len(Y))
    # fidelity baselines: raw bond rays, and future wavefunctions
    rays = [b / np.linalg.norm(b) for b in bs]
    lab_raw, _ = SP.cluster_rays(rays, 0.5)
    lab_wf, _ = SP.cluster_rays(wavefunction_fidelity(bs, sites), 0.5)
    print(f'{tag:28s} obs-rank {rank:3d} | new: k={k} (last merge {below:.4f}, '
          f'next {above:.4f}, ratio {above / max(below, 1e-12):7.1f}) | '
          f'raw-fid {lab_raw.max() + 1} | wf-fid {lab_wf.max() + 1} | truth {truth}')
    return parts[len(Y) - k], hist


if __name__ == '__main__':
    # 1. correctness: recursion vs brute force, incl. a 2-site future (full maze, bond 1)
    T1, T2, T3 = load('seedprot_full_s0')
    lab = LAB_FULL['pos+rew']
    for sites, bs in [([T3], full_histories(T1, T2)[1][:6]),
                      ([T2, T3], [T1[0, 0, oi(0, 0, c), :] for c in (0, 1)])]:
        S, e = future_gram(sites, lab)
        Y, _ = embed(bs, S, e)
        P = [brute_future_probs(b, sites, lab) for b in bs]
        err = max(abs(np.linalg.norm(Y[i] - Y[j]) - np.linalg.norm(P[i] - P[j]))
                  for i in range(len(bs)) for j in range(len(bs)))
        print(f'check {len(sites)}-site future: max |d_embed - d_brute| = {err:.2e}')

    # 2. minimal maze, all seeds
    print()
    for s in range(5):
        p = R / 'Saved_Models' / f'seedprot_minimal_s{s}.pt'
        if not p.exists():
            continue
        T1, T2 = load(f'seedprot_minimal_s{s}')
        h, bs = min_histories(T1)
        report(f'minimal s{s}', h, bs, [T2], list(range(4)), 4)

    # 3. full maze, three seeds, three choices of observed modalities
    for name, truth in [('pos+rew', 7), ('all', 7), ('rew', 5)]:
        print()
        for s in range(3):
            T1, T2, T3 = load(f'seedprot_full_s{s}')
            h, bs = full_histories(T1, T2)
            part, hist = report(f'full s{s} [{name}]', h, bs, [T3], LAB_FULL[name], truth)
            if s == 0:
                for c in part:
                    print('     ', sorted({hist[i].rsplit('/', 2)[0] for i in c}))
