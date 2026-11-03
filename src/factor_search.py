"""Polynomial search for factorizations of a learned state machine.

A partition of the states has the substitution property (SP) if, for every action, the
successors of each block lie in one block. Every SP partition is a join of the closures
of single pairs (the smallest SP partition merging i and j), so the lattice is generated
by n(n-1)/2 closures, each O(n A) union-find passes, instead of enumerating Bell(n)
partitions. Two SP partitions with a trivial meet and |p1| |p2| = n give S = S1 x S2.

Also: predictive states of a single long stream (no episodes) from length-k contexts.
"""
import itertools

import numpy as np

from structure_pipeline import bmr


def _close(parent, succ):
    """Merge until every block's successors (per action) lie in one block."""
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    changed = True
    while changed:
        changed = False
        for a in range(len(succ)):
            target = {}
            for i, nxt in enumerate(succ[a]):
                r = find(i)
                for j in nxt:
                    rj = find(j)
                    if r not in target:
                        target[r] = rj
                    elif find(target[r]) != rj:
                        x, y = find(target[r]), rj
                        parent[max(x, y)] = min(x, y)
                        changed = True
    return tuple(find(i) for i in range(len(parent)))


def canon(labels):
    m = {}
    return tuple(m.setdefault(x, len(m)) for x in labels)


def pair_closures(succ):
    n = len(succ[0])
    out = set()
    for i, j in itertools.combinations(range(n), 2):
        parent = list(range(n))
        parent[j] = i
        out.add(canon(_close(parent, succ)))
    return out


def join(p, q, succ):
    n = len(p)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for lab in (p, q):
        first = {}
        for i, b in enumerate(lab):
            if b in first:
                x, y = find(first[b]), find(i)
                if x != y:
                    parent[max(x, y)] = min(x, y)
            else:
                first[b] = i
    return canon(_close(parent, succ))


def sp_lattice(succ, max_rounds=3):
    """All SP partitions reachable as joins of pair closures (complete for small lattices)."""
    n = len(succ[0])
    sps = pair_closures(succ)
    frontier = set(sps)
    for _ in range(max_rounds):
        new = {join(p, q, succ) for p in frontier for q in sps} - sps
        if not new:
            break
        sps |= new
        frontier = new
    return {p for p in sps if 1 < len(set(p)) < n}


def products(sps, n):
    out = []
    sps = sorted(sps, key=lambda p: len(set(p)))
    for p, q in itertools.combinations(sps, 2):
        kp, kq = len(set(p)), len(set(q))
        if kp * kq == n and len(set(zip(p, q))) == n:
            out.append((p, q))
    return out


def factor_matrices(T, p):
    """Block transition matrix B[a][b'|b] of an SP partition (deterministic or averaged)."""
    k = len(set(p))
    B = np.zeros((len(T), k, k))
    for a in range(len(T)):
        for i in range(len(p)):
            for j in np.flatnonzero(T[a][i] > 0):
                B[a, p[i], p[j]] += T[a][i, j]
        B[a] /= np.maximum(B[a].sum(1, keepdims=True), 1e-12)
    return B


# ------------------------------------------------------------ predictive states of a stream
def stream_states(x, k=1, alpha=None):
    """x: symbol sequence. Contexts = last k symbols; BMR merges contexts that predict the
    next symbol alike; returns state per context, transition matrix T[s, s'] (one action)."""
    sym = {s: i for i, s in enumerate(dict.fromkeys(x))}
    x = [sym[s] for s in x]
    no = len(sym)
    alpha = alpha or 1.0 / no
    ctx = {}
    for t in range(k - 1, len(x) - 1):
        c = tuple(x[t - k + 1:t + 1])
        ctx.setdefault(c, np.zeros((1, no)))[0, x[t + 1]] += 1
    keys = list(ctx)
    lab, _ = bmr(np.array([ctx[c] for c in keys]), alpha)
    of = dict(zip(keys, lab))
    # refine to a unifilar machine: merged contexts must lead (per next symbol) to the same
    # state; one-step BMR alone can merge contexts whose later futures differ
    while True:
        sig = {c: (of[c],) + tuple(sorted((o, of.get(c[1:] + (o,), -1))
                                          for o in np.flatnonzero(ctx[c][0])))
               for c in keys}
        new = {c: i for i, c in enumerate(dict.fromkeys(sig.values()))}
        relab = {c: new[sig[c]] for c in keys}
        if len(new) == len(set(of.values())):
            break
        of = relab
    K = max(of.values()) + 1
    T = np.zeros((K, K))
    for t in range(k - 1, len(x) - 1):
        c, c2 = tuple(x[t - k + 1:t + 1]), tuple(x[t - k + 2:t + 2])
        if c2 in of:
            T[of[c], of[c2]] += 1
    T /= np.maximum(T.sum(1, keepdims=True), 1)
    return of, K, T
