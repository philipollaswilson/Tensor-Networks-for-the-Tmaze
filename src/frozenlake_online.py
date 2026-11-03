"""Online frozen lake: act, grow the MPS, re-learn the states, plan, repeat.

Nothing about the task length is fixed in advance:
  * episodes run in the real environment and end when it says so (hole or goal),
    or at a time limit T_MAX, as gymnasium's TimeLimit wrapper does (it uses 100
    for 4x4; we use a smaller limit because the chain has one site per step).
    After termination the remaining sites repeat the terminal observation (the
    hole and goal are absorbing, as in the env's own transition table). A shared
    End token would not do: a history ending in the hole and one ending in the
    goal would then have the same future, and BMR rightly merges them, so the
    agent could not tell the hole from the goal. Episodes cut by the time limit
    already fill the chain and are never padded.
  * site 0 is (no action, start observation), as in mpstwo/frozenlake.py.
  * the MPS starts with 2 sites and a site is appended whenever an episode is
    longer than the chain. A new site is a unit-norm product tensor of bond 1,
    which keeps the right-canonical form; two-site training grows the bond.
  * structure learning uses every history prefix at every bond (not one cut):
    model-faithful counts p_MPS(o | h, a) x visits, then BMR (bmr_states).
  * the POMDP is built from the BMR clusters without assuming they are pure:
    A[o, k] = share of k's histories that end in o; B maps the next observation
    to states through p(state | observation).
  * the agent plans with sophisticated EFE (risk - state info gain - novelty);
    novelty is the active-learning (parameter information gain) term on B's
    Dirichlet counts. The counts are updated after every step, so B and the
    novelty change within an episode; the state space is re-learned between
    rounds. Before each round's training, add_support mixes in a small uniform
    state so that new sequences never start with zero amplitude.

Run from the repo root:
  python src/frozenlake_online.py --policy efe    [--rounds 10 --episodes 40]
  python src/frozenlake_online.py --policy random
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
import minimal_tmaze_train  # noqa: E402,F401  (installs the import stubs)
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv  # noqa: E402

from mpstwo.model.mpstwo import MPSTwo                   # noqa: E402
from mpstwo.model.optimizers import SGD                  # noqa: E402
from mpstwo.utils.mapping import MultiOneHotMap          # noqa: E402
from mpstwo.utils.operations import contract             # noqa: E402
from frozenlake_train import ACT, DESC, GOAL, HOLE, true_B  # noqa: E402
from learned_pomdp import infogain, risk                 # noqa: E402

NO, NA = 9, len(ACT)                     # observation = cell
OBS = [f'{DESC[c // 3][c % 3]}{c}' for c in range(NO)]
T_MAX = 10
HORIZON = 5
ALPHA = 0.1                              # BMR Dirichlet prior, as in learned_pomdp
DTYPE = torch.complex128


# ------------------------------------------------------------ environment
def run_episode(env, choose, rng):
    """[(0, o0), (a1, o1), ...] until termination or T_MAX steps."""
    o, _ = env.reset()
    seq = [(0, int(o))]
    state = choose(None, int(o))
    for _ in range(T_MAX):
        a = choose(state, None) if state is not None else int(rng.integers(NA))
        o, _, term, _, _ = env.step(a)
        seq.append((a, int(o)))
        if term:
            break
        if state is not None:                              # once lost, explore at random
            state = choose((state, a), int(o))
    return seq


# ------------------------------------------------------------ the growing MPS
def append_sites(mps, n):
    for _ in range(n):
        # random, not constant: identical constant sites make the two-site merges
        # exactly low rank and the SVD then cancels some sequences to psi == 0
        u = torch.randn((1, NA, NO, 1), dtype=DTYPE)
        u /= torch.linalg.norm(u)
        mps.matrices.append(u)
        mps.physical_legs += 1


def add_support(mps, eps=0.05):
    """psi <- psi + eps * (uniform product state), as a direct sum on every bond
    (each bond grows by 1). New data can contain sequences the current model
    gives zero amplitude, and the gradient divides by psi; this gives every
    sequence support before training, and training then sharpens it again."""
    u = torch.full((NA, NO), (NA * NO) ** -0.5, dtype=DTYPE)
    Ts, L = mps.matrices, len(mps.matrices)
    out = []
    for k, T in enumerate(Ts):
        l, _, _, r = T.shape
        nl = l if k == 0 else l + 1
        nr = r if k == L - 1 else r + 1
        N = torch.zeros((nl, NA, NO, nr), dtype=DTYPE)
        N[:l, :, :, :r] = T
        N[nl - 1, :, :, nr - 1] += (eps if k == 0 else 1.0) * u
        out.append(N)
    mps.matrices = out
    mps.right_canonical()


def padded(episodes, pads, L):
    out = []
    for ep, pad in zip(episodes, pads):
        out.append(ep + [(int(a), ep[-1][1]) for a in pad[:L - len(ep)]])
    return out


class ClippedSGD(SGD):
    """Two-site cumulant SGD with the gradient norm capped at `clip` times the norm
    of the merged tensor. A sequence the model barely supports contributes
    psi'/psi, which is huge; unclipped, one step is dominated by it, the
    relative-cutoff SVD then discards the rest, the bonds collapse and other
    sequences fall to psi == 0."""
    def __init__(self, model, lr, clip=5.0):
        super().__init__(model, lr)
        self.clip = clip

    def _sgd_cumulants(self, m1, m2, obs, act, left, right, going_right):
        merged = contract(m1, m2)
        g = self._gradient(merged, obs, act, left, right)
        cap = self.clip * torch.linalg.norm(merged)
        n = torch.linalg.norm(g)
        if n > cap:
            g = g * (cap / n)
        return merged - self.lr * 2 * g


def train(mps, seqs, updates, batch=512):
    """Warm-started two-site sweeps (mps.update, the step MPSTrainer runs) on all
    data so far: the whole buffer per update, or a random `batch` of it once it is
    larger. With small minibatches, sequences absent from a batch are truncated
    away and come back with psi == 0, which the cumulant gradient cannot take."""
    obs_map, act_map = MultiOneHotMap([NO]), MultiOneHotMap([NA])
    O = torch.stack([obs_map(torch.tensor([[o] for _, o in s])) for s in seqs]).type(DTYPE)
    A = torch.stack([act_map(torch.tensor([[a] for a, _ in s])) for s in seqs]).type(DTYPE)
    opt = ClippedSGD(mps, lr=1e-2)
    with torch.inference_mode():
        for _ in range(updates):
            idx = torch.randperm(len(seqs))[:batch]
            mps.update(O[idx], A[idx], opt)
        return float(-mps.get_prob(O, A).log().mean())       # train NLL, nats


# ------------------------------------------------------------ histories and counts
def history_counts(mps, seqs):
    """Every prefix h ending before the last site: model counts, data counts, visits."""
    Ts = [m.detach().cpu().numpy() for m in mps.matrices]
    L = len(Ts)
    E = [None] * L                                       # right environment after site k
    E[L - 1] = np.ones((1, 1), complex)
    for k in range(L - 1, 0, -1):
        E[k - 1] = np.einsum('iaoj,jk,laok->il', Ts[k], E[k], Ts[k].conj())
    data = {}
    for s in seqs:
        for t in range(L - 1):
            h = tuple(s[:t + 1])
            data.setdefault(h, np.zeros((NA, NO)))[s[t + 1]] += 1
    hists, model, dcounts = [], [], []
    vec = {}
    for h in sorted(data, key=len):
        a, o = h[-1]
        b = Ts[0][0, a, o, :] if len(h) == 1 else vec[h[:-1]] @ Ts[len(h) - 1][:, a, o, :]
        vec[h] = b
        v = np.einsum('i,iaoj->aoj', b, Ts[len(h)])
        p = np.einsum('aoj,jk,aok->ao', v, E[len(h)], v.conj()).real.clip(0)
        p /= np.maximum(p.sum(1, keepdims=True), 1e-300)
        n = data[h]
        hists.append(h); model.append(p * n.sum(1, keepdims=True)); dcounts.append(n)
    return hists, np.array(model), np.array(dcounts)


def lnB(x):
    return (torch.lgamma(x).sum(-1) - torch.lgamma(x.sum(-1))).sum(-1)


def bmr(counts, alpha=ALPHA):
    """bmr_states.bmr (greedy merge while dF > 0), with cached pairwise dF and
    per-row best partners so it scales to thousands of histories."""
    X = torch.tensor(counts, dtype=torch.float64)
    n = len(X)
    a = alpha
    base = float(lnB(torch.full(X.shape[1:], a, dtype=torch.float64)))
    own = lnB(a + X)
    D = torch.full((n, n), -np.inf, dtype=torch.float64)
    for i in range(n):
        if i + 1 < n:
            D[i, i + 1:] = lnB(a + X[i] + X[i + 1:]) - own[i] - own[i + 1:] + base
    D = torch.maximum(D, D.T)
    alive = torch.ones(n, dtype=torch.bool)
    clusters = {i: [i] for i in range(n)}
    best_v, best_j = D.max(1)
    while alive.sum() > 1:
        i = int(torch.argmax(torch.where(alive, best_v, -np.inf)))
        j = int(best_j[i])
        if best_v[i] <= 0:
            break
        clusters[i] += clusters.pop(j)
        X[i] += X[j]; alive[j] = False
        D[j, :] = D[:, j] = -np.inf
        own[i] = lnB(a + X[i])
        row = lnB(a + X[i] + X) - own[i] - own + base
        row[~alive] = -np.inf; row[i] = -np.inf
        D[i, :] = row; D[:, i] = row
        best_v[i], best_j[i] = row.max(0)
        stale = alive & ((best_j == i) | (best_j == j))
        stale[i] = False
        for r in torch.nonzero(stale).flatten().tolist():
            best_v[r], best_j[r] = D[r].max(0)
        better = alive & (row > best_v)
        best_v[better] = row[better]; best_j[better] = i
    return [clusters[i] for i in sorted(clusters)]


# ------------------------------------------------------------ the learned POMDP
class Learned:
    def __init__(self, hists, model, dcounts, clusters):
        nS = len(clusters)
        w = dcounts.sum((1, 2))                               # visits per history
        last = np.array([h[-1][1] for h in hists])
        self.A = np.zeros((NO, nS))
        for k, c in enumerate(clusters):
            np.add.at(self.A[:, k], last[c], w[c])
        self.state_given_obs = self.A / np.maximum(self.A.sum(1, keepdims=True), 1e-300)
        self.A /= self.A.sum(0, keepdims=True)
        self.nm = np.array([model[c].sum(0) for c in clusters])   # (k, a, o)
        self.nd = np.array([dcounts[c].sum(0) for c in clusters])
        start = [k for k, c in enumerate(clusters) if any(len(hists[i]) == 1 for i in c)]
        self.D = np.zeros(nS); self.D[start] = 1; self.D /= self.D.sum()
        self.clusters, self.nS = clusters, nS
        self._derive()

    def _derive(self):
        nS = self.nS
        cm = np.einsum('kao,os->ska', self.nm, self.state_given_obs)
        self.B = (cm + ALPHA / nS) / (cm + ALPHA / nS).sum(0, keepdims=True)
        cd = np.einsum('kao,os->ska', self.nd, self.state_given_obs) + 1.0
        nov = 0.5 * (1 / cd - 1 / cd.sum(0, keepdims=True))
        self.novelty = np.einsum('ska,ska->ka', self.B, nov)  # (k, a)
        self.tried = self.nd.sum(-1)                          # data counts (k, a)
        self.G = None                                         # EFE table, filled by plan()

    def update(self, q, a, o):
        """Dirichlet update of B after one transition: belief q, action a, next obs o.
        The state space itself is re-learned only between rounds (MPS + BMR)."""
        self.nm[:, a, o] += q
        self.nd[:, a, o] += q
        self._derive()

    def cell_of_state(self):
        return [int(np.argmax(self.A[:, k])) for k in range(self.nS)]


def preferences():
    C = np.zeros(NO)
    C[GOAL], C[HOLE] = 2.0, -4.0
    return C


def softmin(G, gamma):
    p = np.exp(-gamma * (G - G.min(-1, keepdims=True)))
    return p / p.sum(-1, keepdims=True)


def state_efe(M, C, horizon=HORIZON, gamma=16.0):
    """Sophisticated EFE G[s, a] (risk - state info gain - novelty), recursing over
    next states rather than beliefs: each step assumes the next state is revealed.
    That is exact when the learned states are pure (fully observed lake) and keeps
    the cost at O(horizon x states^2 x actions); the belief tree grows ~25x per step
    while early clusters are impure."""
    G1 = np.array([[risk(M.A @ M.B[:, s, a], C) - infogain(M.A, M.B[:, s, a]) - M.novelty[s, a]
                    for a in range(NA)] for s in range(M.nS)])
    G = G1
    for _ in range(horizon - 1):
        V = (softmin(G, gamma) * G).sum(1)                  # E_{a'~sigma(-gamma G)} G(s', a')
        G = G1 + np.einsum('tsa,t->sa', M.B, V)
    return G


def plan(M, C, q, gamma=16.0):
    if M.G is None:
        M.G = state_efe(M, C)
    return softmin(q @ M.G, gamma)


def efe_chooser(M, C, rng):
    def choose(state, o):
        if o is None:                                      # pick an action from belief
            p = plan(M, C, state)
            return int(rng.choice(NA, p=p))
        if state is None:                                  # first observation
            q = M.D.copy()
        else:
            (q, a) = state
            M.update(q, a, o)                              # learn B from this step
            q = M.B[:, :, a] @ q
        post = M.A[o] * q
        if post.sum() < 1e-12:                             # surprise: trust the observation
            post = M.A[o].copy()
        if post.sum() < 1e-12:                             # a cell no state shows yet
            return None
        return post / post.sum()
    return choose


def random_chooser(rng):
    def choose(state, o):
        return int(rng.integers(NA)) if o is None else 0
    return choose


# ------------------------------------------------------------ evaluation
def evaluate(M):
    cells = M.cell_of_state()
    purity = M.A.max(0)
    Bt = true_B()
    tvs = []
    for k in range(M.nS):
        if purity[k] < 0.95:
            continue
        for a in range(NA):
            if M.tried[k, a] == 0:
                continue
            pred = np.zeros(NO)
            for s2 in range(M.nS):
                pred[cells[s2]] += M.B[s2, k, a]
            tvs.append(0.5 * np.abs(pred - Bt[:, cells[k], a]).sum())
    return {'n_states': M.nS, 'cells': [OBS[c] for c in cells],
            'min_purity': float(purity.min()),
            'B_TV_mean': float(np.mean(tvs)) if tvs else None,
            'B_TV_max': float(np.max(tvs)) if tvs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', choices=['efe', 'random'], default='efe')
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--episodes', type=int, default=40)
    ap.add_argument('--updates', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    env = FrozenLakeEnv(desc=DESC, is_slippery=False)
    env.reset(seed=args.seed)
    C = preferences()
    out = REPO / 'results' / 'frozenlake_online'
    out.mkdir(parents=True, exist_ok=True)
    tag = f'frozenlake_online_{args.policy}_s{args.seed}'

    mps = MPSTwo(2, feature_dim_obs=NO, feature_dim_act=NA, bond_dim=4,
                 init_mode='random', max_bond=12, cutoff=0.01, dtype=DTYPE)
    episodes, pads, log, M = [], [], [], None
    for r in range(args.rounds):
        t0 = time.time()
        choose = random_chooser(rng) if (M is None or args.policy == 'random') else efe_chooser(M, C, rng)
        new = [run_episode(env, choose, rng) for _ in range(args.episodes)]
        episodes += new
        pads += [rng.integers(NA, size=T_MAX + 1) for _ in new]

        grow = max(len(e) for e in episodes) - mps.physical_legs
        if grow > 0:
            append_sites(mps, grow)
        seqs = padded(episodes, pads, mps.physical_legs)
        if r:
            add_support(mps)
        nll = train(mps, seqs, args.updates)

        hists, model, dcounts = history_counts(mps, seqs)
        clusters = bmr(model)
        M = Learned(hists, model, dcounts, clusters)
        ends = [e[-1][1] for e in new]
        tried = {(o, a) for e in episodes for (_, o), (a, _) in zip(e[:-1], e[1:])}
        rec = {'round': r, 'policy': args.policy, 'episodes_total': len(episodes),
               'sites': mps.physical_legs, 'bond_dims': mps.bond_dim[:-1],
               'goal_rate': float(np.mean([o == GOAL for o in ends])),
               'hole_rate': float(np.mean([o == HOLE for o in ends])),
               'mean_steps': float(np.mean([len(e) - 1 for e in new])),
               'cells_seen': len({o for e in episodes for _, o in e}),
               'cell_action_pairs_tried': len(tried),
               'n_histories': len(hists), **evaluate(M),
               'train_nll': nll, 'seconds': time.time() - t0}
        log.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != 'bond_dims'}))
        with open(out / f'{args.policy}_s{args.seed}.json', 'w') as f:
            json.dump(log, f, indent=1)
    torch.save(mps, REPO / 'Saved_Models' / f'{tag}.pt')


if __name__ == '__main__':
    main()
