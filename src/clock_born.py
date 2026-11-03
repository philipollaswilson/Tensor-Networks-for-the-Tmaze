"""Positive control (A1c): a process where a Born bond of 2 is exact and no finite HMM is.

Clock: a qubit instrument with outcomes 0, 1. K0 = R(theta) diag(1, s) rotates and damps,
K1 = sqrt(1 - s^2) |0><1| fires and resets to |0>. Runs of 0 are a damped rotation, so the
hazard p(1 | 0^n) oscillates and never settles. The map on density matrices for outcome 0
has eigenvalues s, s e^(+-2i phi) with cos(phi) = 3/5. All three share one modulus and
phi/pi is irrational (Niven, also for 2 phi: cos(2 phi) = -7/25), whereas Perron-Frobenius
only allows roots of unity on the top circle of a non-negative matrix. So the run-length
law has no finite positive realization: no HMM of any finite size is exact.

Models, all fitted to the same sequences and scored on the same held-out sequences, as
bits/symbol above the true model:
  born      uniform complex MPS read as an instrument, bond D (isometric, Adam)
  hmm       non-negative MPS = Mealy HMM, K_o[s, s'] >= 0, bond D (Baum-Welch)
  chain     unifilar: state = zeros since the last 1, capped at K - 1 (closed form); the
            best a state-from-history model can do here at K states
  learned   the count-and-BMR learner of kmaze.py (--unifilar), with no size given
Also the singular values of the data's past-future Hankel matrix, and the hazard each
model implies.

Run:  python src/clock_born.py --seed 0 [--unifilar]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / 'results' / 'clock'
S, COS_PHI = 0.81, 0.6
N_SEQ, T_SEQ = 2000, 60
BORN_D = [1, 2, 3, 4]
HMM_D = [1, 2, 3, 4, 6, 8, 12, 16]
CHAIN_K = [1, 2, 4, 8, 12, 16, 24, 32, 48]
RESTARTS = 4
BORN_STEPS = 1500
EM_STEPS = 300
HAZARD_N = 40
LOG2 = np.log(2)
CDT = torch.complex128


# ------------------------------------------------------------------- the process
def clock():
    """Kraus operators (2, 2, 2) and the start state."""
    cos_t = 2 * np.sqrt(S) * COS_PHI / (1 + S)
    sin_t = np.sqrt(1 - cos_t ** 2)
    K0 = np.array([[cos_t, -sin_t], [sin_t, cos_t]]) @ np.diag([1, S])
    K1 = np.sqrt(1 - S ** 2) * np.array([[0, 1.0], [0, 0]])
    return np.stack([K0, K1]), np.array([1.0, 0])


def superop_spectrum(K0):
    return np.linalg.eigvals(np.kron(K0, K0))


def sample(n, T, seed):
    K, psi0 = clock()
    rng = np.random.default_rng(seed)
    psi = np.tile(psi0, (n, 1))
    x = np.zeros((n, T), int)
    for t in range(T):
        phi = psi @ K.transpose(0, 2, 1)                  # (2, n, 2)
        p = (phi ** 2).sum(-1)
        o = (rng.random(n) < p[1] / p.sum(0)).astype(int)
        phi = phi[o, np.arange(n)]
        psi = phi / np.linalg.norm(phi, axis=1, keepdims=True)
        x[:, t] = o
    return x


def hankel_singular_values(x, L=5, top=8):
    """Singular values of the matrix p(past of length L, future of length L), pooled over
    positions after the first L."""
    w = 2 ** np.arange(L)
    H = np.zeros((2 ** L, 2 ** L))
    for t in range(L, x.shape[1] - L + 1):
        np.add.at(H, (x[:, t - L:t] @ w, x[:, t:t + L] @ w), 1)
    return (np.linalg.svd(H / H.sum(), compute_uv=False)[:top]).tolist()


# ------------------------------------------------------------------- Born
class Born(torch.nn.Module):
    def __init__(self, D, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.D = D
        self.W = torch.nn.Parameter(torch.randn((2 * D, D), generator=g, dtype=CDT))
        self.psi0 = torch.nn.Parameter(torch.randn(D, generator=g, dtype=CDT))

    def kraus(self):
        evals, U = torch.linalg.eigh(self.W.mH @ self.W)
        V = self.W @ (U @ torch.diag(evals.clamp_min(1e-12).rsqrt().to(CDT)) @ U.mH)
        return V.reshape(2, self.D, self.D)

    def loglik(self, x):
        """Total log p(x), nats. x (B, T)."""
        K = self.kraus()
        psi = (self.psi0 / self.psi0.norm()).expand(x.shape[0], self.D)
        ll = 0.0
        for t in range(x.shape[1]):
            phi = psi @ K.transpose(1, 2)                 # (2, B, D)
            phi = torch.where(x[:, t, None] == 0, phi[0], phi[1])
            p = (phi.abs() ** 2).sum(-1)
            ll = ll + torch.log(p + 1e-30).sum()
            psi = phi / p.clamp_min(1e-30).sqrt()[:, None]
        return ll

    def hazard(self):
        with torch.no_grad():
            K = self.kraus().numpy()
            psi = (self.psi0 / self.psi0.norm()).numpy()
        return born_hazard(K, psi)


def born_hazard(K, psi):
    out = []
    for _ in range(HAZARD_N):
        p0, p1 = [np.linalg.norm(k @ psi) ** 2 for k in K]
        out.append(float(p1 / (p0 + p1)))
        psi = K[0] @ psi
        psi = psi / np.linalg.norm(psi)
    return out


def fit_born(D, xtr, xte, seed):
    best = None
    for r in range(RESTARTS):
        m = Born(D, 1000 * seed + r)
        opt = torch.optim.Adam(m.parameters(), lr=0.03)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, BORN_STEPS, eta_min=1e-3)
        for _ in range(BORN_STEPS):
            opt.zero_grad()
            loss = -m.loglik(xtr)
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            tr = float(m.loglik(xtr))
        if best is None or tr > best[0]:
            best = (tr, m)
    tr, m = best
    with torch.no_grad():
        te = float(m.loglik(xte))
    return bits(tr, xtr), bits(te, xte), m.hazard()


# ------------------------------------------------------------------- HMM
def hmm_forward(K, pi, x):
    a = pi.expand(x.shape[0], -1)
    ll = 0.0
    for t in range(x.shape[1]):
        a = torch.where(x[:, t, None] == 0, a @ K[:, 0], a @ K[:, 1])
        c = a.sum(-1, keepdim=True)
        ll = ll + torch.log(c).sum()
        a = a / c
    return float(ll)


def hmm_em_step(K, pi, x):
    """One Baum-Welch step on a Mealy HMM with scaling; returns new K, pi."""
    D = K.shape[0]
    alphas, cs = [pi.expand(x.shape[0], -1)], []
    for t in range(x.shape[1]):
        a = torch.where(x[:, t, None] == 0, alphas[-1] @ K[:, 0], alphas[-1] @ K[:, 1])
        c = a.sum(-1, keepdim=True)
        cs.append(c)
        alphas.append(a / c)
    C = torch.full_like(K, 1e-3)
    b = torch.ones(x.shape[0], D, dtype=K.dtype)
    for t in range(x.shape[1] - 1, -1, -1):
        for o in (0, 1):
            m = x[:, t] == o
            if m.any():
                C[:, o] += K[:, o] * (alphas[t][m].T @ (b[m] / cs[t][m]))
        b = torch.where(x[:, t, None] == 0, b @ K[:, 0].T, b @ K[:, 1].T) / cs[t]
    pi2 = (alphas[0] * b).sum(0)
    return C / C.sum((1, 2), keepdim=True), pi2 / pi2.sum()


def hmm_hazard(K, pi):
    a, out = pi.numpy(), []
    K = K.numpy()
    for _ in range(HAZARD_N):
        a0, a1 = a @ K[:, 0], a @ K[:, 1]
        out.append(float(a1.sum() / (a0.sum() + a1.sum())))
        a = a0 / a0.sum()
    return out


def fit_hmm(D, xtr, xte, seed):
    best = None
    for r in range(RESTARTS):
        g = torch.Generator().manual_seed(1000 * seed + r)
        K = torch.rand((D, 2, D), generator=g, dtype=torch.float64) + 0.1
        K = K / K.sum((1, 2), keepdim=True)
        pi = torch.ones(D, dtype=torch.float64) / D
        for _ in range(EM_STEPS):
            K, pi = hmm_em_step(K, pi, xtr)
        tr = hmm_forward(K, pi, xtr)
        if best is None or tr > best[0]:
            best = (tr, K, pi)
    tr, K, pi = best
    return bits(tr, xtr), bits(hmm_forward(K, pi, xte), xte), hmm_hazard(K, pi)


# ------------------------------------------------------------------- unifilar
def ages(x, K):
    """State at every step: zeros since the last 1 (start counts as a 1), capped at K - 1."""
    n, T = x.shape
    s = np.zeros((n, T), int)
    for t in range(1, T):
        s[:, t] = np.where(x[:, t - 1] == 1, 0, np.minimum(s[:, t - 1] + 1, K - 1))
    return s


def fit_chain(K, xtr, xte):
    C = np.full((K, 2), 0.5)
    np.add.at(C, (ages(xtr, K), xtr), 1)
    P = C / C.sum(1, keepdims=True)
    tr, te = [float(np.log(P[ages(x, K), x]).sum()) for x in (xtr, xte)]
    return bits(tr, xtr), bits(te, xte), [float(P[min(n, K - 1), 1]) for n in range(HAZARD_N)]


def fit_learned(xtr, xte):
    """kmaze.py's learner on the clock: one action, the reset shown as a leading 1."""
    import kmaze as M
    def as_trajs(x):
        return [([1] + [int(v) for v in r], [0] * len(r), None) for r in x]
    trajs = as_trajs(xtr)
    S_, _ = M.learn_states(trajs, 2)
    K = int(max(s.max() for s in S_)) + 1
    C = np.full((K, 2), 0.5)
    succ = {}
    for (obs, acts, _), s in zip(trajs, S_):
        for t in range(len(acts)):
            C[s[t], obs[t + 1]] += 1
            if s[t + 1] >= 0:
                succ.setdefault((int(s[t]), obs[t + 1]), {}).setdefault(int(s[t + 1]), 0)
                succ[int(s[t]), obs[t + 1]][int(s[t + 1])] += 1
    P = C / C.sum(1, keepdims=True)
    next_of = {k: max(v, key=v.get) for k, v in succ.items()}
    start = np.bincount([int(s[0]) for s in S_]).argmax()

    def loglik(x):
        ll = 0.0
        for r in x:
            s = start
            for o in r:
                ll += np.log(P[s, o])
                s = next_of.get((s, int(o)), s)
        return float(ll)
    hazard, s = [], start
    for _ in range(HAZARD_N):
        hazard.append(float(P[s, 1]))
        s = next_of.get((s, 0), s)
    return K, bits(loglik(xtr), xtr), bits(loglik(xte), xte), hazard


# ------------------------------------------------------------------- run
def bits(ll, x):
    """Negative log-likelihood in bits per symbol."""
    x = np.asarray(x)
    return float(-ll / (x.size * LOG2))


def true_nll(x):
    K, psi0 = clock()
    psi = np.tile(psi0, (len(x), 1))
    ll = 0.0
    for t in range(x.shape[1]):
        phi = psi @ K.transpose(0, 2, 1)
        phi = phi[x[:, t], np.arange(len(x))]
        p = (phi ** 2).sum(-1)
        ll += np.log(p).sum()
        psi = phi / np.sqrt(p)[:, None]
    return bits(ll, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--unifilar', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (('learned_s%d.json' if args.unifilar else 'seed%d.json') % args.seed)
    torch.set_num_threads(2)

    Xtr = sample(N_SEQ, T_SEQ, 2 * args.seed)
    Xte = sample(N_SEQ, T_SEQ, 2 * args.seed + 1)
    xtr, xte = torch.as_tensor(Xtr), torch.as_tensor(Xte)
    K_true, psi0 = clock()
    res = {'seed': args.seed, 'n': N_SEQ, 'T': T_SEQ,
           'true': {'train': true_nll(Xtr), 'test': true_nll(Xte),
                    'hazard': born_hazard(K_true, psi0)}}
    print('true nll train %.4f test %.4f bits/symbol' % (res['true']['train'], res['true']['test']))

    if args.unifilar:
        t0 = time.time()
        K, tr, te, hz = fit_learned(Xtr, Xte)
        res['learned'] = {'K': K, 'train': tr, 'test': te, 'hazard': hz, 'time': time.time() - t0}
        print('learned K=%d test gap %+.4f (%.0fs)' % (K, te - res['true']['test'], time.time() - t0))
        path.write_text(json.dumps(res))
        return

    ev = superop_spectrum(K_true[0])
    print('superoperator |eig|', np.round(np.abs(ev), 4), 'angle/pi',
          np.round(np.angle(ev) / np.pi, 4))
    res['hankel'] = hankel_singular_values(Xtr)
    print('hankel sv', np.round(res['hankel'], 5))
    res['born'], res['hmm'], res['chain'] = {}, {}, {}
    for K in CHAIN_K:
        tr, te, hz = fit_chain(K, Xtr, Xte)
        res['chain'][K] = {'train': tr, 'test': te, 'hazard': hz}
        print('chain K=%d test gap %+.4f' % (K, te - res['true']['test']), flush=True)
    for name, ds, fit in (('hmm', HMM_D, fit_hmm), ('born', BORN_D, fit_born)):
        for D in ds:
            t0 = time.time()
            tr, te, hz = fit(D, xtr, xte, args.seed)
            res[name][D] = {'train': tr, 'test': te, 'hazard': hz, 'time': time.time() - t0}
            print('%s D=%d test gap %+.4f (%.0fs)'
                  % (name, D, te - res['true']['test'], time.time() - t0), flush=True)
            path.write_text(json.dumps(res))


if __name__ == '__main__':
    main()
