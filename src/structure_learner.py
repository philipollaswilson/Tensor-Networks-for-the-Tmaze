"""Online structure learning with one shared classical tensor, for any symbolic task.

The model and its learning rules are those of frozenlake_sl.py, written for any number
of actions NA and observation symbols NO, with vectorized evidence so that it runs with
hundreds of states:
    M^{a,o}[s', s] = p(o | s, a) * [s' = f(s, a, o)]
a uniform non-negative MPS / input-driven unifilar HMM. Every step one Dirichlet count;
a state splits on the incoming edge whose arrivals predict a different future (Bayes
factor); at the end of an episode, BMR merges with a cascade keep f deterministic.

Planning tables (the same as frozenlake_sl.py, for a preference vector over symbols):
  efe_table   sophisticated EFE with the novelty (parameter information gain) term
  psrl_table  posterior sampling from the tensor's Dirichlet counts (knob-free)
"""
import numpy as np
from scipy.special import digamma, gammaln

RESET = -1


def lnB(x):
    """Log multivariate beta over the last axis, summed over the rest (keeps leading axes
    when called on a stack of count tables with keep=n)."""
    return (gammaln(x).sum(-1) - gammaln(x.sum(-1))).sum(-1)


class Machine:
    def __init__(self, na, no, alpha=0.1, k_max=None):
        self.na, self.no, self.alpha = na, no, alpha
        self.k_max = k_max                     # memory budget (None: grow freely)
        self.K = 1
        self.f = {}                            # (s, a, o) -> s'
        self.start = {}                        # o0 -> s
        self.episodes = []                     # [o0, [(a, o), ...]]
        self.replay()

    def dF(self, x, y):
        """Log evidence for one shared Dirichlet over two count tables; > 0 means merge.
        Broadcasts over leading axes."""
        a = self.alpha
        return lnB(a + x + y) - lnB(a + x) - lnB(a + y) + lnB(np.full(x.shape[-2:], a))

    # n[s] has NA + 1 rows: row 0 counts the observation s was entered with (the A side),
    # rows 1.. count the next observation per action (the B side).
    def replay(self):
        self.n = np.zeros((self.K, self.na + 1, self.no))
        self.ctx = [dict() for _ in range(self.K)]
        for o0, steps in self.episodes:
            s, e = self.enter(o0)
            for a, o in steps:
                s, e = self.count(s, e, a, o)

    def arrive(self, s, e):
        self.n[s, 0, e[2]] += 1
        self.ctx[s].setdefault(e, np.zeros((self.na + 1, self.no)))[0, e[2]] += 1

    def enter(self, o0):
        if o0 not in self.start:
            self.start[o0] = self.default_target(o0) if self.n.sum() else 0
        e = (RESET, RESET, o0)
        self.arrive(self.start[o0], e)
        return self.start[o0], e

    def count(self, s, e, a, o):
        self.n[s, 1 + a, o] += 1
        self.ctx[s][e][1 + a, o] += 1
        if (s, a, o) not in self.f:
            self.f[(s, a, o)] = self.default_target(o)
        t, e2 = self.f[(s, a, o)], (s, a, o)
        self.arrive(t, e2)
        return t, e2

    def full(self):
        return self.k_max is not None and self.K >= self.k_max

    def default_target(self, o):
        """The state most often entered with o, else a new state (if the budget allows)."""
        if self.n[:, 0, o].max() > 0 or self.full():
            return int(np.argmax(self.n[:, 0, o] + 1e-9 * self.n[:, 0].sum(-1)))
        self.K += 1
        self.n = np.concatenate([self.n, np.zeros((1, self.na + 1, self.no))])
        self.ctx.append({})
        return self.K - 1

    def locate(self, o0, steps):
        s, e = self.start[o0], (RESET, RESET, o0)
        for a, o in steps:
            s, e = self.f[(s, a, o)], (s, a, o)
        return s, e

    # ---- grow on surprise, by whole-machine evidence
    def evidence(self, n):
        a = self.alpha
        return float((lnB(a + n) - lnB(np.full(n.shape[1:], a))).sum())

    def fast_counts(self, f, start, K):
        n = np.zeros((K, self.na + 1, self.no))
        for o0, steps in self.episodes:
            s = start[o0]
            n[s, 0, o0] += 1
            for a, o in steps:
                n[s, 1 + a, o] += 1
                s = f[(s, a, o)]
                n[s, 0, o] += 1
        return n

    def try_split_loop(self, s, e):
        """Split s on incoming edge e. Two candidates: the copy keeps its own self-loops
        (it remembers how it was entered while the observation stays the same) or returns
        to s after one step (as in frozenlake_sl.py). The better one is accepted if the
        evidence of the whole machine, recounted exactly, goes up."""
        if self.full() or len(self.ctx[s]) < 2:
            return False
        new, best, F = self.K, None, self.evidence(self.n)
        for keep_loops in (True, False):
            f, start = dict(self.f), dict(self.start)
            if e[0] == RESET:
                start[e[2]] = new
            else:
                f[e] = new
            for (s0, a, o), t in self.f.items():
                if s0 == s:
                    f[(new, a, o)] = new if (keep_loops and t == s) else t
            F_new = self.evidence(self.fast_counts(f, start, self.K + 1))
            if F_new > F:
                best, F = (f, start), F_new
        if best is None:
            return False
        self.f, self.start = best
        self.K += 1
        self.replay()
        return True

    # ---- grow on surprise (one-step version, as in frozenlake_sl.py)
    def try_split(self, s):
        if len(self.ctx[s]) < 2 or self.full():
            return False
        edges = list(self.ctx[s])
        ne = np.stack([self.ctx[s][e] for e in edges])
        rest = self.n[s][None] - ne
        ok = (ne.sum((1, 2)) > 0) & (rest.sum((1, 2)) > 0)
        if not ok.any():
            return False
        d = np.where(ok, -self.dF(ne, rest), -np.inf)
        k = int(np.argmax(d))
        if d[k] <= 0:
            return False
        best_e = edges[k]
        new = self.K
        self.K += 1
        if best_e[0] == RESET:
            self.start[best_e[2]] = new
        else:
            self.f[best_e] = new
        for (s0, a, o), t in list(self.f.items()):
            if s0 == s:
                self.f[(new, a, o)] = t
        self.replay()
        return True

    # ---- shrink by BMR
    def try_merge(self, max_merges=10_000):
        merged = 0
        while merged < max_merges:
            iu, ju = np.triu_indices(self.K, 1)
            if len(iu) == 0:
                break
            d = self.dF(self.n[iu], self.n[ju])
            order = np.argsort(-d)
            for k in order:
                if d[k] <= 0:
                    return merged
                root = self.cascade(int(iu[k]), int(ju[k]))
                if root is not None:
                    self.relabel(root)
                    merged += 1
                    break
            else:
                break
        return merged

    def cascade(self, i, j):
        root = list(range(self.K))

        def find(x):
            while root[x] != x:
                x = root[x]
            return x
        n = {s: self.n[s].copy() for s in range(self.K)}
        out = {s: {} for s in range(self.K)}
        for (s0, a, o), t in self.f.items():
            out[s0][(a, o)] = t
        queue = [(i, j)]
        while queue:
            x, y = map(find, queue.pop())
            if x == y:
                continue
            if self.dF(n[x], n[y]) <= 0:
                return None
            root[y] = x
            n[x] = n[x] + n[y]
            for k, t in out[y].items():
                if k in out[x]:
                    queue.append((out[x][k], t))
                else:
                    out[x][k] = t
        return [find(s) for s in range(self.K)]

    def relabel(self, root):
        ids = {r: k for k, r in enumerate(sorted(set(root)))}
        m = [ids[root[s]] for s in range(self.K)]
        f = {}
        for (s0, a, o), t in self.f.items():
            f.setdefault((m[s0], a, o), m[t])
        self.f = f
        self.start = {o0: m[s] for o0, s in self.start.items()}
        self.K = len(ids)
        self.replay()

    # ---- the generative model
    def p_obs(self):
        c = self.n[:, 1:] + self.alpha
        return c / c.sum(-1, keepdims=True)

    def novelty(self):
        """Expected information gain about p(o | s, a) from one more observation: the
        exact Dirichlet mutual information H[E p] - E H[p] (bounded by ln NO). The
        0.5 * sum p (1/c - 1/C) approximation of frozenlake_sl.py diverges for the small
        per-symbol priors used with large alphabets."""
        c = self.n[:, 1:] + self.alpha
        C = c.sum(-1, keepdims=True)
        p = c / C
        H_mean = -(p * np.log(p)).sum(-1)
        EH = digamma(C[..., 0] + 1) - (p * digamma(c + 1)).sum(-1)
        return H_mean - EH

    def tried(self):
        return self.n[:, 1:].sum(-1)

    def next_table(self):
        nxt = np.zeros((self.K, self.na, self.no), int)
        nxt[...] = np.arange(self.K)[:, None, None]      # unseen (s, a, o): stay (unused)
        for (s, a, o), t in self.f.items():
            nxt[s, a, o] = t
        return nxt


# ------------------------------------------------------------ planning
def risk(C):
    """Risk per observation symbol for log-preferences C."""
    return -(C - np.log(np.exp(C).sum()))


def efe_table(p, nxt, r, nov, horizon, gamma):
    """Sophisticated EFE by DP over known states: G(s, a) = E_o[risk + V(s')] - novelty."""
    V = np.zeros(p.shape[0])
    for _ in range(horizon):
        G = (p * (r + V[nxt])).sum(-1) - nov
        pi = np.exp(-gamma * (G - G.min(1, keepdims=True)))
        pi /= pi.sum(1, keepdims=True)
        V = (pi * G).sum(1)
    return G


def psrl_table(counts, alpha, nxt, reward, horizon, rng):
    """Finite-horizon Q for one Dirichlet posterior sample of p(o | s, a)."""
    p = rng.gamma(counts + alpha)
    p /= p.sum(-1, keepdims=True)
    Qh = np.zeros((horizon + 1,) + p.shape[:2])
    for h in range(1, horizon + 1):
        Qh[h] = (p * (reward + Qh[h - 1].max(1)[nxt])).sum(-1)
    return Qh
