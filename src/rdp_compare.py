"""Fair comparison of structure learners on the RDP benchmark set (rdp_envs.py).

Rules, fixed before any result was seen:
  data      uniform random behaviour. All methods train on the same 90% of the episodes and
            choose settings on the same 10%; ones with nothing to choose ignore that 10%.
  test      2,000 fresh episodes per seed. Score: nats per step above the exact Bayes-optimal
            log-loss of the true model (belief_logloss), and state count against the exact
            truth (n_pred; n_layered for learners tied to the time step).
  settings  ours at the defaults of kmaze.learn_states, unchanged after seeing results;
            AdaCT-H at delta = 0.05; FlexFringe at its stock alergia.ini. These three share
            rdp_baselines.Machine, so they differ only through their states.
  neural    GRU: hidden size and learning rate chosen on validation, early stopping.
  HMM       action-conditional, cold random starts, states chosen on validation. No hints.
  away      Tiger, where the state is never certain, so merging should fail.
  budget    15 minutes per structure-learner fit, the same for all; running out is reported
            as that method's result. Added after a test run showed ours slows on Tiger.

Run:  python src/rdp_compare.py [--workers 12]
"""
import argparse
import ctypes
import itertools
import json
import multiprocessing as mp
import os
import pathlib
import queue
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import kmaze as M  # noqa: E402
import rdp_baselines as B  # noqa: E402
import rdp_envs as E  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / 'results' / 'rdp'
NS = [300, 1000, 3000, 10000, 30000]
SEEDS = 5
TEST = 2000
HMM_D = [4, 8, 16, 32]
HMM_STEPS, HMM_RESTARTS = 100, 2
GRU_H = [16, 64]
GRU_LR = [1e-3, 5e-3]
GRU_STEPS, GRU_BATCH = 2000, 512
BUDGET = 900


def ours(trajs, A, O):
    """kmaze.learn_states, as a Machine: successors by majority vote, start by first obs."""
    S, rounds = M.learn_states(trajs, O)
    votes = {}
    for (obs, acts, _), s in zip(trajs, S):
        votes.setdefault(('start', A * O + obs[0]), {}).setdefault(int(s[0]), 0)
        votes['start', A * O + obs[0]][int(s[0])] += 1
        for t, a in enumerate(acts):
            if s[t + 1] >= 0:
                v = votes.setdefault((int(s[t]), a * O + obs[t + 1]), {})
                v[int(s[t + 1])] = v.get(int(s[t + 1]), 0) + 1
    delta = {e: max(v, key=v.get) for e, v in votes.items()}
    return B.Machine(delta, 'start', M.n_states(S), trajs, A, O, rounds=rounds)


# ------------------------------------------------------------------- action-conditional HMM
def arrays(trajs):
    return np.array([t[0] for t in trajs]), np.array([t[1] for t in trajs])


def hmm_pass(K, pi, x, a, A, O, em=True):
    """Scaled forward (and backward when em) on K[a, s, o, s']. Returns log-likelihood and the
    new parameters."""
    N, T = a.shape
    sym = torch.as_tensor(a * O + x[:, 1:])
    alpha, cs = [pi[torch.as_tensor(x[:, 0])]], []
    for t in range(T):
        nxt = torch.zeros_like(alpha[-1])
        for s in sym[:, t].unique().tolist():
            m = sym[:, t] == s
            nxt[m] = alpha[-1][m] @ K[s // O, :, s % O]
        c = nxt.sum(1, keepdim=True).clamp_min(1e-300)
        cs.append(c)
        alpha.append(nxt / c)
    ll = float(sum(torch.log(c).sum() for c in cs))
    if not em:
        return ll
    C = torch.full_like(K, 1e-3)
    b = torch.ones_like(alpha[0])
    for t in range(T - 1, -1, -1):
        nb = torch.zeros_like(b)
        for s in sym[:, t].unique().tolist():
            m = sym[:, t] == s
            ka = K[s // O, :, s % O]
            C[s // O, :, s % O] += ka * (alpha[t][m].T @ (b[m] / cs[t][m]))
            nb[m] = b[m] @ ka.T
        b = nb / cs[t]
    gamma = alpha[0] * b
    P = torch.full_like(pi, 1e-3)
    P.index_add_(0, torch.as_tensor(x[:, 0]), gamma)
    return ll, C / C.sum((2, 3), keepdim=True), P / P.sum(1, keepdim=True)


def fit_hmm(Xtr, Xva, Xte, A, O, seed):
    (xt, at), (xv, av), (xe, ae) = arrays(Xtr), arrays(Xva), arrays(Xte)
    best = None
    for D in HMM_D:
        for r in range(HMM_RESTARTS):
            g = torch.Generator().manual_seed(1000 * seed + 10 * D + r)
            K = torch.rand((A, D, O, D), generator=g, dtype=torch.float64) + 0.1
            K = K / K.sum((2, 3), keepdim=True)
            pi = torch.rand((O, D), generator=g, dtype=torch.float64) + 0.1
            pi = pi / pi.sum(1, keepdim=True)
            for _ in range(HMM_STEPS):
                _, K, pi = hmm_pass(K, pi, xt, at, A, O)
            v = hmm_pass(K, pi, xv, av, A, O, em=False)
            if best is None or v > best[0]:
                best = (v, D, K, pi)
    _, D, K, pi = best
    return -hmm_pass(K, pi, xe, ae, A, O, em=False) / ae.size, D


# ------------------------------------------------------------------- GRU
class GRU(torch.nn.Module):
    def __init__(self, A, O, h):
        super().__init__()
        self.A, self.O = A, O
        self.net = torch.nn.GRU(A + O, h, batch_first=True)
        self.out = torch.nn.Linear(h, O)

    def forward(self, x, a):
        one = torch.nn.functional.one_hot
        return self.out(self.net(torch.cat([one(x[:, :-1], self.O), one(a, self.A)], -1).float())[0])

    def loss(self, x, a):
        return torch.nn.functional.cross_entropy(self(x, a).transpose(1, 2), x[:, 1:])


def fit_gru(Xtr, Xva, Xte, A, O, seed):
    (xt, at), (xv, av), (xe, ae) = [tuple(torch.as_tensor(v) for v in arrays(s)) for s in (Xtr, Xva, Xte)]
    rng, best = np.random.default_rng(seed), None
    for h, lr in itertools.product(GRU_H, GRU_LR):
        torch.manual_seed(seed)
        m = GRU(A, O, h)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        low, keep, stale = np.inf, None, 0
        for step in range(GRU_STEPS):
            if step % 25 == 0:
                with torch.no_grad():
                    v = float(m.loss(xv, av))
                if v < low:
                    low, stale, keep = v, 0, {k: t.clone() for k, t in m.state_dict().items()}
                else:
                    stale += 1
                    if stale >= 12:
                        break
            i = rng.integers(len(xt), size=min(GRU_BATCH, len(xt)))
            opt.zero_grad()
            m.loss(xt[i], at[i]).backward()
            opt.step()
        if best is None or low < best[0]:
            m.load_state_dict(keep)
            best = (low, h, lr, m)
    _, h, lr, m = best
    with torch.no_grad():
        return float(m.loss(xe, ae)), h, lr


# ------------------------------------------------------------------- run
FITS = {'ours': ours, 'adact': B.adact_h, 'flexfringe': B.flexfringe}


def fit_one(q, name, Xtr, Xte, A, O):
    try:
        torch.set_num_threads(1)
        m = FITS[name](Xtr, A, O)
        loss, kept = m.logloss(Xte)
        q.put(dict(loss=loss, kept=kept, states=m.n_states))
    except Exception as e:
        q.put(dict(error=repr(e)))


def cell(env_name, n, seed):
    path = OUT / f'{env_name}_n{n}_s{seed}.json'
    if path.exists():
        return path.name
    torch.set_num_threads(1)
    env = E.ENVS[env_name]()
    T = E.PAPER_H[env_name]
    A, O = env.n_act, env.n_obs
    data = env.sample(n, T, 7919 * seed + n)
    k = max(10, n // 10)
    Xtr, Xva = data[k:], data[:k]
    Xte = env.sample(TEST, T, 104729 + seed)
    floor = env.belief_logloss(Xte)
    res = dict(env=env_name, n=n, seed=seed, T=T, floor=floor, methods={})

    def save(name, entry, t0):
        entry['time'] = time.time() - t0
        entry['gap'] = entry['loss'] - floor
        res['methods'][name] = entry
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(res))
        return tmp

    ctx = mp.get_context('spawn')
    for name in FITS:
        t0 = time.time()
        q = ctx.Queue()
        p = ctx.Process(target=fit_one, args=(q, name, Xtr, Xte, A, O))
        p.start()
        try:
            out = q.get(timeout=BUDGET)
        except queue.Empty:
            p.terminate()
            out = dict(error='timeout')
        p.join()
        if 'error' in out:                        # a failure is a result, not a crash
            res['methods'][name] = dict(out, time=time.time() - t0)
        else:
            save(name, out, t0)
    t0 = time.time()
    loss, D = fit_hmm(Xtr, Xva, Xte, A, O, seed)
    tmp = save('hmm', dict(loss=loss, states=D), t0)
    t0 = time.time()
    loss, h, lr = fit_gru(Xtr, Xva, Xte, A, O, seed)
    tmp = save('gru', dict(loss=loss, states=h, lr=lr), t0)
    os.replace(tmp, path)
    return path.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.platform == 'win32':                   # no system sleep while this process runs
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    jobs =[(e, n, s) for n in NS for e in E.ENVS for s in range(SEEDS)]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        fs = {ex.submit(cell, *j): j for j in jobs}
        for i, f in enumerate(as_completed(fs)):
            try:
                print(f'{i + 1}/{len(jobs)} {f.result()} ({time.time() - t0:.0f}s)', flush=True)
            except Exception as e:
                print(f'FAILED {fs[f]}: {e!r}', flush=True)


if __name__ == '__main__':
    main()
