"""Online Born-machine (quantum) agent on position-only Mountain Car.

Model: a uniform complex MPS read as a quantum instrument (experiment A2). For each
action a there are Kraus operators K_{a,o} (D x D, complex) with sum_o K^dag K = I, so
    p(o | psi, a) = ||K_{a,o} psi||^2,      psi' = K_{a,o} psi / ||K_{a,o} psi||
is a proper conditional p(o | history, a) at every step and the model streams forever.
An extra "reset" instrument emits the first observation of an episode from a learned
start state. The bond D is the memory budget: beliefs are pure states in C^D.
Constraint: each action's stacked operator V_a = [K_{a,o}]_o (NO*D x D) is an isometry,
by polar projection V = W (W^dag W)^(-1/2) of an unconstrained W.

Learning every step: one Adam step on the log-likelihood of a minibatch of stored
episodes plus the current one, for each of ENS ensemble members (different inits and
bootstrap weights). The ensemble stands in for the parameter posterior (B2): there are
no conjugate counts for a Born machine.

Planning (sophisticated EFE, same preferences as the classical agent):
  * beliefs reached so far are covered by a set of nodes (a new node when a belief is
    farther than DELTA in mean infidelity from all nodes, across members);
  * for each node and action, p(o | node, a) is the ensemble mean, and each successor
    belief is mapped to its nearest node, giving a finite graph;
  * novelty(node, a) = ensemble disagreement about the next observation (Jensen-Shannon,
    the expected information gain about which member is right), the counterpart of the
    Dirichlet novelty term;
  * EFE by DP on the graph over the steps left; the action is chosen from the exact
    current belief, one step ahead of the graph.

Run from the repo root:
  python src/mountaincar_born.py --D 8 [--seed 0 --episodes 300 --moving]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mountaincar_env as E  # noqa: E402
from mountaincar_sl import GAMMA, preferences  # noqa: E402
from structure_learner import risk  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
ENS = 5
LR = 0.02
BATCH = 16
WINDOW = 50                       # truncated BPTT window
DELTA = 0.05
MAX_NODES = 600
REBUILD = 10                      # steps between graph rebuilds
P_MIN = 0.01                      # successors kept per (node, a)
CDT = torch.complex128


class BornEnsemble(torch.nn.Module):
    def __init__(self, D, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        shape = (ENS, E.NA + 1, E.NO * D, D)
        self.W = torch.nn.Parameter(torch.randn(shape, generator=g, dtype=torch.float64)
                                    + 1j * torch.randn(shape, generator=g, dtype=torch.float64))
        self.psi0 = torch.nn.Parameter(torch.randn((ENS, D), generator=g, dtype=CDT))
        self.D = D

    def kraus(self):
        """(ENS, NA+1, NO, D, D) Kraus operators of isometric instruments."""
        W = self.W
        G = W.mH @ W
        evals, U = torch.linalg.eigh(G)
        inv_sqrt = U @ torch.diag_embed(evals.clamp_min(1e-12).rsqrt().to(CDT)) @ U.mH
        V = W @ inv_sqrt
        return V.reshape(ENS, E.NA + 1, E.NO, self.D, self.D)

    def start(self):
        return self.psi0 / self.psi0.norm(dim=-1, keepdim=True)

    def loglik(self, K, o0, acts, obs, mask, weights, t0=0, length=None):
        """Sum of log p over a batch of episodes for every member. o0 (B,), acts/obs
        (B, T), mask (B, T); weights (ENS, B) bootstrap weights. Truncated BPTT: the
        belief at t0 is carried in without gradient and only steps t0..t0+length count."""
        B = o0.shape[0]
        T = acts.shape[1] if length is None else min(acts.shape[1], t0 + length)
        with torch.set_grad_enabled(t0 == 0 and torch.is_grad_enabled()):
            Kp = K if t0 == 0 else K.detach()
            psi = self.start()[:, None, :].expand(ENS, B, self.D)
            if t0 > 0:
                psi = psi.detach()
            phi = (Kp[:, E.NA][:, o0] @ psi[..., None])[..., 0]
            p = (phi.abs() ** 2).sum(-1)
            ll = torch.log(p + 1e-30) if t0 == 0 else torch.zeros_like(p)
            psi = phi / p.sqrt()[..., None]
            for t in range(t0):
                phi = (Kp[:, acts[:, t], obs[:, t]] @ psi[..., None])[..., 0]
                n = phi.norm(dim=-1, keepdim=True).clamp_min(1e-30)
                psi = torch.where(mask[None, :, t, None] > 0, phi / n, psi)
        for t in range(t0, T):
            Kt = K[:, acts[:, t], obs[:, t]]                     # (ENS, B, D, D)
            phi = (Kt @ psi[..., None])[..., 0]
            p = (phi.abs() ** 2).sum(-1)
            m = mask[:, t]
            ll = ll + m * torch.log(p + 1e-30)
            psi = torch.where(m[None, :, None] > 0, phi / p.clamp_min(1e-30).sqrt()[..., None], psi)
        return (weights * ll).sum(1)


class BornAgent:
    def __init__(self, D, seed):
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.model = BornEnsemble(D, seed)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.episodes = []                    # (o0, [(a, o)])
        self.boot = []                        # per-episode bootstrap weights (ENS,)
        self.r = risk(preferences())
        self.explore = True
        self.nodes = None
        self.D = D

    # ---------------------------------------------------------- learning
    def batch(self, idx):
        T = max(1, max(len(self.episodes[i][1]) for i in idx))
        o0 = torch.tensor([self.episodes[i][0] for i in idx])
        acts = torch.zeros((len(idx), T), dtype=torch.long)
        obs = torch.zeros((len(idx), T), dtype=torch.long)
        mask = torch.zeros((len(idx), T), dtype=torch.float64)
        for b, i in enumerate(idx):
            steps = self.episodes[i][1]
            if steps:
                acts[b, :len(steps)] = torch.tensor([a for a, _ in steps])
                obs[b, :len(steps)] = torch.tensor([o for _, o in steps])
                mask[b, :len(steps)] = 1
        return o0, acts, obs, mask

    def train_step(self):
        n = len(self.episodes)
        idx = list(self.rng.choice(n - 1, size=min(BATCH - 1, n - 1), replace=False)) if n > 1 else []
        idx = idx + [n - 1]
        o0, acts, obs, mask = self.batch(idx)
        w = torch.tensor(np.stack([self.boot[i] for i in idx], 1))
        K = self.model.kraus()
        T = acts.shape[1]
        t0 = int(self.rng.integers(0, max(1, T - WINDOW + 1))) if T > WINDOW else 0
        ll = self.model.loglik(K, o0, acts, obs, mask, w, t0, WINDOW)
        loss = -ll.sum() / max(1.0, mask[:, t0:t0 + WINDOW].sum().item())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    # ---------------------------------------------------------- beliefs
    @torch.no_grad()
    def all_beliefs(self, K):
        """Beliefs (ENS, D) after every prefix of every stored episode."""
        out = []
        idx = list(range(len(self.episodes)))
        o0, acts, obs, mask = self.batch(idx)
        B = len(idx)
        psi = self.model.start()[:, None, :].expand(ENS, B, self.D)
        phi = (K[:, E.NA][:, o0] @ psi[..., None])[..., 0]
        psi = phi / phi.norm(dim=-1, keepdim=True)
        out.append(psi.permute(1, 0, 2))
        for t in range(acts.shape[1]):
            phi = (K[:, acts[:, t], obs[:, t]] @ psi[..., None])[..., 0]
            psi = torch.where(mask[None, :, t, None] > 0, phi / phi.norm(dim=-1, keepdim=True).clamp_min(1e-30), psi)
            out.append(psi.permute(1, 0, 2)[mask[:, t] > 0])
        return torch.cat(out)                                      # (N, ENS, D)

    @staticmethod
    def infid(X, Y):
        """Mean over members of 1 - |<x|y>|^2, for X (N, ENS, D), Y (M, ENS, D)."""
        ov = torch.einsum('nkd,mkd->nmk', X.conj(), Y).abs() ** 2
        return 1 - ov.mean(-1)

    @torch.no_grad()
    def rebuild(self):
        K = self.model.kraus()
        self.K = K
        bel = self.all_beliefs(K)
        # greedy cover of visited beliefs (keeps the cover small and spread out)
        sel = torch.randperm(len(bel), generator=torch.Generator().manual_seed(len(bel)))[:4000]
        nodes = [bel[sel[0]]]
        for i in sel[1:]:
            if len(nodes) >= MAX_NODES:
                break
            if self.infid(bel[i][None], torch.stack(nodes)).min() > DELTA:
                nodes.append(bel[i])
        N = torch.stack(nodes)                                    # (N, ENS, D)
        # successors
        Ka = K[:, :E.NA]                                          # (ENS, NA, NO, D, D)
        phi = torch.einsum('kaoij,nkj->naoki', Ka, N)              # (N, NA, NO, ENS, D)
        pm = (phi.abs() ** 2).sum(-1)                              # (N, NA, NO, ENS)
        p = pm.mean(-1)
        p = p / p.sum(-1, keepdim=True)
        nxt = torch.arange(len(N))[:, None, None].expand(p.shape).clone()
        keep = p > P_MIN
        if keep.any():
            ii = keep.nonzero(as_tuple=True)
            succ = phi[ii] / (pm[ii].sqrt()[..., None] + 1e-30)     # (S, ENS, D)
            nxt[ii] = self.infid(succ, N).argmin(1)
        pk = pm / pm.sum(2, keepdim=True)                          # per-member p(o | node, a)
        H = lambda q: -(q * torch.log(q + 1e-30)).sum(2)
        jsd = H(pk.mean(-1, keepdim=True))[..., 0] - H(pk).mean(-1)
        self.N, self.p, self.nxt, self.nov = N, p.numpy(), nxt.numpy(), jsd.numpy()

    @torch.no_grad()
    def advance(self, a, o):
        phi = (self.K[:, a, o] @ self.psi[..., None])[..., 0]
        self.psi = phi / phi.norm(dim=-1, keepdim=True).clamp_min(1e-30)

    # ---------------------------------------------------------- planning
    def values(self, horizon, nov):
        V = np.zeros(len(self.p))
        for _ in range(horizon):
            G = (self.p * (self.r + V[self.nxt])).sum(-1) - nov
            pi = np.exp(-GAMMA * (G - G.min(1, keepdims=True)))
            pi /= pi.sum(1, keepdims=True)
            V = (pi * G).sum(1)
        return V

    @torch.no_grad()
    def act(self, t):
        if self.t_since >= REBUILD or self.nodes is None:
            self.rebuild()
            self.nodes, self.t_since = True, 0
            self.V_cache = {}
            self.psi = self.replay_current()
        h = E.T_MAX - t - 1
        nov = self.nov if self.explore else 0.0
        if h not in self.V_cache:
            self.V_cache = {h: self.values(h, nov)}
        V = self.V_cache[h]
        phi = torch.einsum('kaoij,kj->aoki', self.K[:, :E.NA], self.psi)   # (NA, NO, ENS, D)
        pm = (phi.abs() ** 2).sum(-1)
        p = (pm.mean(-1) / pm.mean(-1).sum(-1, keepdim=True)).numpy()      # (NA, NO)
        G = np.zeros(E.NA)
        for a in range(E.NA):
            os = np.flatnonzero(p[a] > P_MIN)
            succ = phi[a, os] / (pm[a, os].sqrt()[..., None] + 1e-30)
            j = self.infid(succ, self.N).argmin(1).numpy()
            G[a] = (p[a, os] * (self.r[os] + V[j])).sum() / p[a, os].sum()
            if self.explore:
                pk = pm[a] / pm[a].sum(0, keepdim=True)
                G[a] -= float(-(pk.mean(-1) * torch.log(pk.mean(-1) + 1e-30)).sum()
                              + (pk * torch.log(pk + 1e-30)).sum(0).mean())
        if not self.explore:
            return int(self.rng.choice(np.flatnonzero(G == G.min())))
        pi = np.exp(-GAMMA * (G - G.min()))
        return int(self.rng.choice(E.NA, p=pi / pi.sum()))

    # ---------------------------------------------------------- episode interface
    def begin(self, o0, info):
        self.episodes.append((o0, []))
        self.boot.append(self.rng.exponential(size=ENS))          # Bayesian bootstrap
        self.t_since = REBUILD
        if self.explore:
            self.train_step()
        K = self.model.kraus().detach() if self.nodes is None else self.K
        self.K = K
        with torch.no_grad():
            phi = (K[:, E.NA, o0] @ self.model.start()[..., None])[..., 0]
            self.psi = phi / phi.norm(dim=-1, keepdim=True)

    def observe(self, a, o, info):
        self.episodes[-1][1].append((a, o))
        if self.explore:
            self.train_step()                                     # learn every step
        self.t_since += 1
        if self.t_since >= REBUILD:
            self.rebuild()
            self.t_since, self.V_cache = 0, {}
            self.psi = self.replay_current()
        else:
            self.advance(a, o)

    @torch.no_grad()
    def replay_current(self):
        o0, steps = self.episodes[-1]
        K = self.K
        phi = (K[:, E.NA, o0] @ self.model.start()[..., None])[..., 0]
        psi = phi / phi.norm(dim=-1, keepdim=True)
        for a, o in steps:
            phi = (K[:, a, o] @ psi[..., None])[..., 0]
            psi = phi / phi.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        return psi

    def end(self):
        if not self.explore:
            self.episodes.pop()
            self.boot.pop()

    def __deepcopy__(self, memo):
        """Evaluation copies share the trained model (no learning during evaluation)."""
        cp = object.__new__(BornAgent)
        cp.__dict__.update(self.__dict__)
        cp.episodes = list(self.episodes)
        cp.boot = list(self.boot)
        cp.rng = np.random.default_rng(int(self.rng.integers(1 << 31)))
        return cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--moving', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    agent = BornAgent(args.D, args.seed)
    name = f'born_D{args.D}'
    t0 = time.time()

    def extra(ag):
        return {'n_nodes': int(len(ag.N)) if ag.nodes else 0, 'seconds': time.time() - t0}
    log = E.run_agent(agent, name, args.seed, args.episodes, args.moving, extra)
    out = REPO / 'results' / ('mountaincar_moving' if args.moving else 'mountaincar')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'{name}_s{args.seed}.json', 'w') as fh:
        json.dump(log, fh, indent=1)
    R = [r['return'] for r in log]
    print(f"{name:10s} seed {args.seed}: return last 20 {np.mean(R[-20:]):.1f}, "
          f"regret {sum(r['regret'] for r in log):.0f}, first flag at "
          f"{next((i for i, x in enumerate(R) if x > -200), None)}, {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
