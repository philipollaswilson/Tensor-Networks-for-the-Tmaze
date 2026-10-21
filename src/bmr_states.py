"""Bayesian model reduction over predictive states: threshold-free state count.

Each history h has, for every next action a, a categorical over next-observation
labels with a Dirichlet(alpha) prior. Two candidate states are merged when the
log evidence of one shared Dirichlet exceeds that of two separate ones:
    dF = sum_a [lnB(alpha+n_i+n_j) - lnB(alpha+n_i) - lnB(alpha+n_j) + lnB(alpha)] > 0.
Greedy agglomeration merges the best pair while dF > 0; no distance threshold.

Counts n_{h,a} are model-faithful: the learned model's p(label | h, a) times the
number of training visits N_{h,a} (option 'model'), or the raw training counts
(option 'data'). Run from the repo root:  python src/bmr_states.py
"""
import sys
from collections import Counter

import math

import numpy as np

gammaln = np.vectorize(math.lgamma, otypes=[float])

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from observable_embedding import FULL, MIN, oi, load, LAB_FULL  # noqa: E402


def lnB(x):
    return gammaln(x).sum(-1) - gammaln(x.sum(-1))


def bmr(counts, alpha):
    """counts: list of (n_actions, n_labels) arrays. Returns partition and dF trace."""
    cl = [[i] for i in range(len(counts))]
    C = [c.astype(float) for c in counts]
    a = np.full(counts[0].shape[1], alpha)
    base = lnB(a)

    def dF(x, y):
        return float((lnB(a + x + y) - lnB(a + x) - lnB(a + y) + base).sum())

    accepted = []
    while len(cl) > 1:
        best = max((dF(C[i], C[j]), i, j) for i in range(len(cl)) for j in range(i + 1, len(cl)))
        d, i, j = best
        if d <= 0:
            return cl, accepted, d
        accepted.append(d)
        cl[i] += cl.pop(j)
        C[i] = C[i] + C.pop(j)
    return cl, accepted, None


# ------------------------------------------------------------ full maze
def full_counts(seed, labels, mode):
    T1, T2, T3 = load(f'seedprot_full_s{seed}')
    leaves = FULL.enumerate_weighted()
    w = np.array([x[1] for x in leaves], float); w /= w.sum()
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(leaves), size=4096, p=w)
    train = [leaves[i][0] for i in draws[:int(0.8 * 4096)]]
    nlab = max(labels) + 1
    tab = {}
    for (_, (_, _, c0), a1, o1, a2, o2) in train:
        key = (c0, a1, oi(*o1))
        tab.setdefault(key, np.zeros((4, nlab)))[a2, labels[oi(*o2)]] += 1
    names, counts = [], []
    P = ['ctr', 'R', 'L', 'cue']; RW = ['-', 'ch', 'sh']
    for (c0, a1, o1), n in sorted(tab.items()):
        if mode == 'model':
            b = np.einsum('i,ij->j', T1[0, 0, oi(0, 0, c0), :], T2[:, a1, o1, :])
            amp = np.abs(np.einsum('j,jaq->aq', b, T3[:, :, :, 0])) ** 2
            p = np.zeros((4, nlab))
            for o in range(24):
                p[:, labels[o]] += amp[:, o]
            p /= p.sum(1, keepdims=True)
            n = p * n.sum(1, keepdims=True)
        p1, r1, cc = o1 // 6, (o1 // 2) % 3, o1 % 2
        names.append(f'{P[p1]}/{RW[r1]}' + (f'/c{cc}' if p1 == 3 else ''))
        counts.append(n)
    return names, counts


# ------------------------------------------------------------ minimal maze
def min_counts(seed, mode):
    T1, T2 = load(f'seedprot_minimal_s{seed}')
    seqs = MIN.enumerate_sequences()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(seqs), size=900)
    tab = {}
    for a1, o2, a2, o3 in (seqs[i] for i in draws[:720]):
        tab.setdefault((a1, o2), np.zeros((3, 4)))[a2, o3] += 1
    names, counts = [], []
    for (a1, o2), n in sorted(tab.items()):
        if mode == 'model':
            b = T1[:, a1, o2, :].sum(0)
            p = np.abs(np.einsum('i,ibp->bp', b, T2[:, :, :, 0])) ** 2
            p /= p.sum(1, keepdims=True)
            n = p * n.sum(1, keepdims=True)
        names.append(f'{MIN.ACT[a1]}/{MIN.OBS[o2]}')
        counts.append(n)
    return names, counts


def show(tag, names, counts, alpha, verbose=False):
    cl, acc, rej = bmr(counts, alpha)
    worst_acc = min(acc) if acc else float('nan')
    print(f'  {tag:22s} alpha={alpha:<4} k={len(cl)}  weakest accepted merge dF={worst_acc:+.1f}  '
          f'best refused merge dF={rej:+.1f}')
    if verbose:
        for c in cl:
            print('       ', sorted({names[i] for i in c}))


if __name__ == '__main__':
    for mode in ('model', 'data'):
        print(f'\n=== counts from: {mode}')
        print(' minimal maze (truth 4)')
        for s in range(5):
            names, counts = min_counts(s, mode)
            for alpha in (0.1, 1.0, 10.0):
                show(f'minimal s{s}', names, counts, alpha, verbose=(s == 1 and alpha == 1.0))
        for lab, truth in (('pos+rew', 7), ('rew', 5)):
            print(f' full maze [{lab}] (truth {truth})')
            for s in range(3):
                names, counts = full_counts(s, LAB_FULL[lab], mode)
                for alpha in (0.1, 1.0, 10.0):
                    show(f'full s{s}', names, counts, alpha, verbose=(s == 0 and alpha == 1.0))
