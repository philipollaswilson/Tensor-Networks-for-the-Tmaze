"""The clock (clock_born.py) against current methods, across training set sizes.

One representative per family, all scored in held-out bits/symbol above the true model:
  born       instrument, bond chosen from BORN_D
  spectral   weighted automaton from the SVD of the Hankel matrix, rank r: signed, no
             gradient descent, may give negative probabilities (clipped)
  polish     spectral at the chosen rank, then Adam on the signed model (spectral + SGD)
  hmm        Baum-Welch from random starts, states chosen from HMM_D
  hmm_warm   Baum-Welch from the age chain: a hint about the clock, reported apart
  cssr       causal-state splitting (Shalizi and Klinkner) without the determinization
             step; prediction by longest matching history
  count      the BMR learner of kmaze.py
  gru        recurrent network, size and learning rate chosen from a grid
  transformer causal transformer, same grid rule
Same rules for all: every method trains on the same 90% of the sequences, and chooses its
size and settings on the same 10%. Two processes, so each side has a home ground: the
clock (a Born model) and a random classical HMM with 6 states.

Run:  python src/clock_compare.py --seed 0 --process clock|hmm
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from scipy.stats import chi2_contingency

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import clock_born as C  # noqa: E402

OUT = C.OUT
NS = [100, 300, 1000, 2000]
RANKS = [2, 3, 4, 5, 6, 8]
BASIS = 5
BORN_D = [1, 2, 3, 4, 6]
HMM_D = [2, 4, 6, 8, 16, 24, 32]
WARM_D = [8, 16, 24, 32]
GRU_H = [4, 8, 16, 32]
TF_W = [16, 32, 64]
LRS = [0.001, 0.005]
POLISH_LRS = [3e-4, 2e-3]
NEURAL_STEPS = 3000
POLISH_STEPS = 800
BORN_RESTARTS = 2
HMM_RESTARTS = 4
CSSR_LAMBDA, CSSR_ALPHA = 24, 0.001
P_MIN = 1e-4
FDT = torch.float64


class Clock:
    name = 'clock'
    sample = staticmethod(C.sample)
    nll = staticmethod(C.true_nll)

    def hazard(self):
        return C.born_hazard(*C.clock())


class Classical:
    """Random 6-state Mealy HMM, fixed across seeds; not unifilar."""
    name = 'hmm'
    D = 6

    def __init__(self):
        rng = np.random.default_rng(12345)
        self.K = rng.dirichlet(np.full(2 * self.D, 0.2), self.D).reshape(self.D, 2, self.D)
        self.pi = np.full(self.D, 1 / self.D)

    def sample(self, n, T, seed):
        rng = np.random.default_rng(seed)
        s = rng.choice(self.D, n, p=self.pi)
        cum = self.K.reshape(self.D, -1).cumsum(1)
        x = np.zeros((n, T), int)
        for t in range(T):
            i = (rng.random((n, 1)) > cum[s]).sum(1).clip(max=2 * self.D - 1)
            x[:, t], s = i // self.D, i % self.D
        return x

    def nll(self, x):
        return C.bits(C.hmm_forward(torch.tensor(self.K), torch.tensor(self.pi),
                                    torch.as_tensor(x)), x)

    def hazard(self):
        return C.hmm_hazard(torch.tensor(self.K), torch.tensor(self.pi))


def split(x, seed):
    """Train and validation parts of a training set (10% validation)."""
    idx = np.random.default_rng(seed).permutation(len(x))
    k = max(10, len(x) // 10)
    return x[idx[k:]], x[idx[:k]]


# ------------------------------------------------------------------- spectral
def strings(L):
    return [tuple(int(b) for b in format(i, f'0{n}b')) for n in range(L + 1)
            for i in range(2 ** n)] if L else [()]


def substring_counts(x, maxlen):
    """Mean frequency of each string over positions, for lengths up to maxlen."""
    f = {(): 1.0}
    n, T = x.shape
    for ln in range(1, maxlen + 1):
        w = 2 ** np.arange(ln)[::-1]
        codes = np.stack([x[:, t:t + ln] @ w for t in range(T - ln + 1)]).ravel()
        c = np.bincount(codes, minlength=2 ** ln) / codes.size
        for i in range(2 ** ln):
            f[tuple(int(b) for b in format(i, f'0{ln}b'))] = c[i]
    return f


def prefix_counts(x, maxlen):
    f = {(): 1.0}
    for ln in range(1, maxlen + 1):
        w = 2 ** np.arange(ln)[::-1]
        c = np.bincount(x[:, :ln] @ w, minlength=2 ** ln) / len(x)
        for i in range(2 ** ln):
            f[tuple(int(b) for b in format(i, f'0{ln}b'))] = c[i]
    return f


def fit_spectral(x, rank):
    """Weighted automaton: (start, A0, A1, stop). Substring statistics give the transition
    operators; the start vector comes from the prefixes, the only place that sees t = 0."""
    B = strings(BASIS)
    f = substring_counts(x, 2 * BASIS + 1)
    g = prefix_counts(x, BASIS)
    H = np.array([[f[u + v] for v in B] for u in B])
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    Fp = np.diag(1 / s[:rank]) @ U[:, :rank].T
    Bp = Vt[:rank].T
    A = [Fp @ np.array([[f[u + (o,) + v] for v in B] for u in B]) @ Bp for o in (0, 1)]
    start = np.array([g[v] for v in B]) @ Bp
    stop = Fp @ np.array([f[u] for u in B])
    return start, np.stack(A), stop


def wfa_nll(m, x):
    start, A, stop = m
    s = np.tile(start, (len(x), 1))
    ll = 0.0
    for t in range(x.shape[1]):
        n = s @ A                                         # (2, B, r)
        p = np.maximum(n @ stop, P_MIN)                   # (2, B)
        p = p / p.sum(0)
        o = x[:, t]
        idx = np.arange(len(x))
        ll += np.log(p[o, idx]).sum()
        k = n[o, idx]
        d = k @ stop
        d = np.where(np.abs(d) < 1e-9, 1e-9, d)
        s = k / d[:, None]
    return C.bits(ll, x)


def wfa_hazard(m):
    start, A, stop = m
    s, out = start, []
    for _ in range(C.HAZARD_N):
        p = np.maximum([s @ A[0] @ stop, s @ A[1] @ stop], P_MIN)
        out.append(float(p[1] / p.sum()))
        k = s @ A[0]
        s = k / (k @ stop)
    return out


def wfa_loglik(start, A, stop, x):
    s = start.expand(x.shape[0], -1)
    ll = 0.0
    for t in range(x.shape[1]):
        n = s @ A                                         # (2, B, r)
        p = (n @ stop).clamp_min(P_MIN)
        p = p / p.sum(0)
        o = x[:, t]
        ll = ll + torch.log(p[o, torch.arange(len(o))]).sum()
        k = n[o, torch.arange(len(o))]
        d = k @ stop
        d = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        s = k / d[:, None]
    return ll


def fit_polish(m, xtr, xva, lr):
    start, A, stop = [torch.tensor(v, dtype=FDT, requires_grad=True) for v in m]
    opt = torch.optim.Adam([start, A, stop], lr=lr)
    xtr, xva = torch.as_tensor(xtr), torch.as_tensor(xva)
    best, keep = np.inf, None
    for step in range(POLISH_STEPS + 1):
        if step % 25 == 0:
            with torch.no_grad():
                v = -float(wfa_loglik(start, A, stop, xva))
            if v < best:
                best, keep = v, [t.detach().numpy().copy() for t in (start, A, stop)]
        opt.zero_grad()
        (-wfa_loglik(start, A, stop, xtr)).backward()
        opt.step()
    return tuple(keep)


# ------------------------------------------------------------------- Born, HMM
def fit_born(xtr, xva, xte):
    """Bond and restart chosen on validation."""
    x, v = torch.as_tensor(xtr), torch.as_tensor(xva)
    best = None
    C.BORN_STEPS = 1000
    for D in BORN_D:
        for r in range(BORN_RESTARTS):
            m = C.Born(D, r)
            opt = torch.optim.Adam(m.parameters(), lr=0.03)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, C.BORN_STEPS, eta_min=1e-3)
            for _ in range(C.BORN_STEPS):
                opt.zero_grad()
                (-m.loglik(x)).backward()
                opt.step()
                sched.step()
            with torch.no_grad():
                va = float(m.loglik(v))
            if best is None or va > best[0]:
                best = (va, m, D)
    _, m, D = best
    with torch.no_grad():
        te = C.bits(float(m.loglik(torch.as_tensor(xte))), xte)
        K0 = m.kraus()[0].numpy()
    ev = np.linalg.eigvals(np.kron(K0, K0.conj()))
    return te, m.hazard(), ev, D


def chain_init(D, x):
    """Mealy HMM form of the age chain, mixed with noise so EM can leave it."""
    Cn = np.full((D, 2), 0.5)
    np.add.at(Cn, (C.ages(x, D), x), 1)
    P = Cn / Cn.sum(1, keepdims=True)
    K = np.full((D, 2, D), 1e-2 / D)
    for s in range(D):
        K[s, 0, min(s + 1, D - 1)] += P[s, 0]
        K[s, 1, 0] += P[s, 1]
    K = torch.tensor(K)
    pi = torch.zeros(D, dtype=FDT) + 1e-3
    pi[0] = 1
    return K / K.sum((1, 2), keepdim=True), pi / pi.sum()


def fit_hmm_cold(D, xtr, xva, xte, seed):
    """Random starts, best on validation."""
    x, v = torch.as_tensor(xtr), torch.as_tensor(xva)
    best = None
    for r in range(HMM_RESTARTS):
        g = torch.Generator().manual_seed(1000 * seed + r)
        K = torch.rand((D, 2, D), generator=g, dtype=FDT) + 0.1
        K = K / K.sum((1, 2), keepdim=True)
        pi = torch.ones(D, dtype=FDT) / D
        for _ in range(C.EM_STEPS):
            K, pi = C.hmm_em_step(K, pi, x)
        va = C.hmm_forward(K, pi, v)
        if best is None or va > best[0]:
            best = (va, K, pi)
    va, K, pi = best
    return C.bits(C.hmm_forward(K, pi, torch.as_tensor(xte)), xte), -va, C.hmm_hazard(K, pi)


def fit_hmm(D, xtr, xva, xte):
    K, pi = chain_init(D, xtr)
    x = torch.as_tensor(xtr)
    best, keep = -np.inf, (K, pi)
    for step in range(C.EM_STEPS + 1):
        if step % 25 == 0:
            v = C.hmm_forward(K, pi, torch.as_tensor(xva))
            if v > best:
                best, keep = v, (K, pi)
        K, pi = C.hmm_em_step(K, pi, x)
    K, pi = keep
    return C.bits(C.hmm_forward(K, pi, torch.as_tensor(xte)), xte), -best, C.hmm_hazard(K, pi)


# ------------------------------------------------------------------- CSSR
def cssr_counts(x, lam):
    """Next-symbol counts per history. Each sequence starts as if after a 1."""
    c = {}
    for r in x:
        s = '1' * lam + ''.join(map(str, r))
        for t in range(lam, len(s)):
            for L in range(lam + 1):
                c.setdefault(s[t - L:t], [0, 0])[int(s[t])] += 1
    return c


def same_dist(a, b):
    a, b = np.array(a, float), np.array(b, float)
    if min(a.sum(), b.sum()) == 0 or (a + b).min() == 0:
        return 1.0
    return chi2_contingency(np.stack([a, b]), correction=False)[1]


def fit_cssr(xtr, xte):
    c = cssr_counts(xtr, CSSR_LAMBDA)
    states = [{'hist': {''}, 'n': list(c[''])}]
    for L in range(CSSR_LAMBDA):
        for st in list(states):
            for h in [h for h in st['hist'] if len(h) == L]:
                for a in '01':
                    ext = a + h
                    if ext not in c:
                        continue
                    if same_dist(c[ext], st['n']) > CSSR_ALPHA:
                        home = st
                    else:
                        ps = [same_dist(c[ext], t['n']) for t in states]
                        home = states[int(np.argmax(ps))] if max(ps) > CSSR_ALPHA else None
                        if home is None:
                            home = {'hist': set(), 'n': [0, 0]}
                            states.append(home)
                    home['hist'].add(ext)
                    home['n'] = [u + v for u, v in zip(home['n'], c[ext])]
    of = {h: i for i, st in enumerate(states) for h in st['hist']}
    P = np.array([(np.array(st['n']) + 0.5) / (sum(st['n']) + 1) for st in states])

    def predict(h):
        for L in range(min(len(h), CSSR_LAMBDA), -1, -1):
            s = of.get(h[len(h) - L:])
            if s is not None:
                return P[s]
        return P[0]
    ll = 0.0
    for r in xte:
        h = '1' * CSSR_LAMBDA
        for o in r:
            ll += np.log(predict(h)[o])
            h = h[1:] + str(int(o))
    hz, h = [], '1' * CSSR_LAMBDA
    for _ in range(C.HAZARD_N):
        hz.append(float(predict(h)[1]))
        h = h[1:] + '0'
    return C.bits(ll, xte), len(states), hz


# ------------------------------------------------------------------- neural
class GRU(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = torch.nn.GRU(2, h, batch_first=True)
        self.out = torch.nn.Linear(h, 2)

    def forward(self, inp):
        return self.out(self.net(inp)[0])


class Transformer(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.emb = torch.nn.Linear(2, w)
        self.pos = torch.nn.Parameter(torch.zeros(1, C.T_SEQ + C.HAZARD_N, w))
        layer = torch.nn.TransformerEncoderLayer(w, 2, 2 * w, dropout=0.0, batch_first=True)
        self.net = torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out = torch.nn.Linear(w, 2)

    def forward(self, inp):
        n = inp.shape[1]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(n)
        return self.out(self.net(self.emb(inp) + self.pos[:, :n], mask=mask))


def inputs(x):
    """One-hot of the previous symbol; the start reads as a 1 (the reset)."""
    prev = torch.cat([torch.ones(len(x), 1, dtype=torch.long), x[:, :-1]], 1)
    return torch.nn.functional.one_hot(prev, 2).float()


def nats(m, x):
    return torch.nn.functional.cross_entropy(m(inputs(x)).transpose(1, 2), x, reduction='sum')


def fit_neural(make, sizes, xtr, xva, xte, seed):
    """Best size on validation; returns test bits, chosen size, parameters, hazard."""
    tr, va, te = [torch.as_tensor(v) for v in (xtr, xva, xte)]
    best = None
    for h in sizes:
        for lr in LRS:
            torch.manual_seed(seed)
            m = make(h)
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            low, keep, stale = np.inf, None, 0
            for step in range(NEURAL_STEPS):
                if step % 25 == 0:
                    with torch.no_grad():
                        v = float(nats(m, va))
                    if v < low:
                        low, stale = v, 0
                        keep = {k: t.clone() for k, t in m.state_dict().items()}
                    else:
                        stale += 1
                        if stale >= 12:
                            break
                opt.zero_grad()
                nats(m, tr).backward()
                opt.step()
            m.load_state_dict(keep)
            if best is None or low < best[0]:
                best = (low, h, m)
    _, h, m = best
    with torch.no_grad():
        t = C.bits(-float(nats(m, te)), xte)
        z = torch.nn.functional.one_hot(torch.tensor([[1] + [0] * (C.HAZARD_N - 1)]), 2).float()
        hz = torch.softmax(m(z), -1)[0, :, 1].tolist()
    return t, h, sum(p.numel() for p in m.parameters()), hz


# ------------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, nargs='*', default=NS)
    ap.add_argument('--process', choices=['clock', 'hmm'], default='clock')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    proc = Clock() if args.process == 'clock' else Classical()
    Xall = proc.sample(C.N_SEQ, C.T_SEQ, 2 * args.seed)
    Xte = proc.sample(C.N_SEQ, C.T_SEQ, 2 * args.seed + 1)
    truth = proc.nll(Xte)
    print('true nll test %.4f bits/symbol' % truth, flush=True)
    path = OUT / ('fair_%s_s%d.json' % (proc.name, args.seed))
    res = {'seed': args.seed, 'process': proc.name, 'true': truth,
           'true_hazard': proc.hazard(), 'runs': {}}

    def record(n, name, entry, t0):
        entry['gap'] = entry.pop('test') - truth
        entry['time'] = time.time() - t0
        res['runs'].setdefault(str(n), {})[name] = entry
        path.write_text(json.dumps(res))
        print('N=%d %-12s gap %+.5f %s (%.0fs)' % (
            n, name, entry['gap'],
            {k: v for k, v in entry.items() if k in ('size', 'params')}, entry['time']),
            flush=True)

    for n in args.n:
        Xtr, Xva = split(Xall[:n], args.seed)

        t0 = time.time()
        te, hz, ev, D = fit_born(Xtr, Xva, Xte)
        record(n, 'born', {'test': te, 'size': D, 'params': 3 * D * D + 2 * D - 2,
                           'hazard': hz, 'eig_abs': np.abs(ev).tolist(),
                           'eig_angle': np.angle(ev).tolist()}, t0)

        t0 = time.time()
        ms = {r: fit_spectral(Xtr, r) for r in RANKS}
        for r, m in ms.items():
            res['runs'].setdefault(str(n), {})['spectral_r%d' % r] = {
                'val': wfa_nll(m, Xva), 'gap': wfa_nll(m, Xte) - truth}
        r = min(ms, key=lambda q: wfa_nll(ms[q], Xva))
        ev = np.linalg.eigvals(ms[3][1][0])
        record(n, 'spectral', {'test': wfa_nll(ms[r], Xte), 'size': r,
                               'params': 2 * r * r + 2 * r, 'hazard': wfa_hazard(ms[r]),
                               'eig3_abs': np.abs(ev).tolist(),
                               'eig3_angle': np.angle(ev).tolist()}, t0)

        t0 = time.time()
        m = min((fit_polish(ms[r], Xtr, Xva, lr) for lr in POLISH_LRS),
                key=lambda q: wfa_nll(q, Xva))
        ev = np.linalg.eigvals(m[1][0])
        record(n, 'polish', {'test': wfa_nll(m, Xte), 'size': r, 'params': 2 * r * r + 2 * r,
                             'hazard': wfa_hazard(m), 'eig_abs': np.abs(ev).tolist(),
                             'eig_angle': np.angle(ev).tolist()}, t0)

        t0 = time.time()
        fits = [(D,) + fit_hmm_cold(D, Xtr, Xva, Xte, args.seed) for D in HMM_D]
        D, te, _, hz = min(fits, key=lambda f: f[2])
        record(n, 'hmm', {'test': te, 'size': D, 'params': 2 * D * D - 1, 'hazard': hz,
                          'by_size': {str(f[0]): f[1] - truth for f in fits}}, t0)

        t0 = time.time()
        fits = [(D,) + fit_hmm(D, Xtr, Xva, Xte) for D in WARM_D]
        D, te, _, hz = min(fits, key=lambda f: f[2])
        record(n, 'hmm_warm', {'test': te, 'size': D, 'params': 2 * D * D - 1, 'hazard': hz,
                               'by_size': {str(f[0]): f[1] - truth for f in fits}}, t0)

        t0 = time.time()
        te, K, hz = fit_cssr(Xtr, Xte)
        record(n, 'cssr', {'test': te, 'size': K, 'params': K, 'hazard': hz}, t0)

        t0 = time.time()
        K, _, te, hz = C.fit_learned(Xtr, Xte)
        record(n, 'count', {'test': te, 'size': K, 'params': K, 'hazard': hz}, t0)

        t0 = time.time()
        te, h, p, hz = fit_neural(GRU, GRU_H, Xtr, Xva, Xte, args.seed)
        record(n, 'gru', {'test': te, 'size': h, 'params': p, 'hazard': hz}, t0)

        t0 = time.time()
        te, h, p, hz = fit_neural(Transformer, TF_W, Xtr, Xva, Xte, args.seed)
        record(n, 'transformer', {'test': te, 'size': h, 'params': p, 'hazard': hz}, t0)


if __name__ == '__main__':
    main()
