"""Online structure learning on the small frozen lake with one shared classical tensor.

No task knowledge is built in:
  * the agent observes what the env emits each step: the cell and the reward, as one
    symbol o = (cell, reward). Its only preference is for reward (the same task signal
    RL gets); nothing says that the hole is bad.
  * episodes are not cut at termination: every episode runs the full time limit
    (gymnasium's 100 steps for FrozenLake 4x4) and, after falling in the hole or
    reaching the goal, the agent stays stuck there. Hole and goal become ordinary
    learned states with self-loops. Being stuck in the hole is bad only because no
    reward can follow.

The model is a uniform non-negative MPS (the same tensor at every time step):
    M^{a,o}[s', s] = p(o | s, a) * [s' = f(s, a, o)]
an input-driven unifilar HMM: the observation emitted on a transition fixes the next
state, so the state is a function of the history (a predictive state, as BMR finds
offline). The bond dimension is the number of states and does not grow with time.

Learning, every step:
  * parameters: one Dirichlet count on p(o | s, a) for the transition just made, and
    one on the observation the next state was entered with. Every Dirichlet uses the
    same prior concentration ALPHA = 0.1 (as in bmr_states).
  * start small: the machine starts with one state; a never-seen transition points at
    the state most often entered with the same observation, or at a new state if the
    observation is new.
  * grow on surprise: for the state just left, compare arrivals through each incoming
    edge with all its other arrivals (entry observation and what followed). If one edge
    differs, the Bayes factor favours two states over one (the reverse of the BMR
    merge) and the state splits: the bond grows by 1.
  * shrink: at the end of each episode, BMR merges pairs of states whose evidence
    agrees, cascading to their successors so f stays deterministic.
After a structure change the stored transitions are replayed through the new machine
so that counts stay exact.

Planning: sophisticated EFE over the learned states, horizon HORIZON:
    G(s, a) = sum_o p(o|s,a) [risk(o) + future] - novelty(s, a)
novelty = expected information gain about the tensor's own Dirichlet parameters (the
active-learning term). State information gain is zero: the state is known exactly.

Run from the repo root:
  python src/frozenlake_sl.py --policy efe|efe_nonovelty|random [--episodes 150 --seed 0]
"""
import argparse
import itertools
import json
import math
import pathlib

import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

REPO = pathlib.Path(__file__).resolve().parents[1]
DESC = ['SFF', 'FHF', 'FFG']
ACT = ['Left', 'Down', 'Right', 'Up']
NC, NA = 9, 4                  # the env's observation and action spaces
NO = 2 * NC                    # observation symbol = (cell, reward in {0, 1})
OBS = [f'{DESC[c // 3][c % 3]}{c}' + ('+r' if r else '') for r in (0, 1) for c in range(NC)]
T_MAX = 100                    # gymnasium's TimeLimit for FrozenLake 4x4
HORIZON = 10
ALPHA = 0.1                    # one Dirichlet prior for everything, as in bmr_states
REWARD_PREF = 2.0              # log-preference for an observation that carries reward
GAMMA = 16.0
RESET = -1                     # incoming edge of an episode's first state: (RESET, RESET, o0)


def symbol(cell, reward):
    return int(cell) + NC * int(reward > 0)


def lnB(x):
    return float((np.array([math.lgamma(v) for v in x.ravel()]).reshape(x.shape).sum(-1)
                  - np.array([math.lgamma(v) for v in x.sum(-1)])).sum())


def dF_merge(x, y, a=ALPHA):
    """Log evidence for one shared Dirichlet over two (bmr_states.bmr); > 0 means merge."""
    base = np.full(x.shape, a)
    return lnB(base + x + y) - lnB(base + x) - lnB(base + y) + lnB(base)


# ------------------------------------------------------------ the machine
class Machine:
    def __init__(self):
        self.K = 1
        self.f = {}                       # (s, a, o) -> s'
        self.start = {}                   # o0 -> s
        self.episodes = []                # [o0, [(a, o), ...]]
        self.replay()

    # n[s] has NA + 1 rows: row 0 counts the observation s was entered with (the A
    # side), rows 1.. count the next observation per action (the B side). Split and
    # merge tests use both, so states that show different observations need evidence
    # before they merge.
    def replay(self):
        """Recompute all counts by running stored transitions through the machine."""
        self.n = np.zeros((self.K, NA + 1, NO))
        self.ctx = [dict() for _ in range(self.K)]   # s -> incoming edge -> counts
        for o0, steps in self.episodes:
            s, e = self.enter(o0)
            for a, o in steps:
                s, e = self.count(s, e, a, o)

    def arrive(self, s, e):
        self.n[s, 0, e[2]] += 1
        self.ctx[s].setdefault(e, np.zeros((NA + 1, NO)))[0, e[2]] += 1

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

    def default_target(self, o):
        """First guess for a never-seen transition: the state most often entered with
        observation o, or a new state if there is none. Splits refine it later."""
        if self.n[:, 0, o].max() > 0:
            return int(np.argmax(self.n[:, 0, o]))
        self.K += 1
        self.n = np.concatenate([self.n, np.zeros((1, NA + 1, NO))])
        self.ctx.append({})
        return self.K - 1

    def locate(self, o0, steps):
        """Current state and incoming edge after an episode prefix, without counting."""
        s, e = self.start[o0], (RESET, RESET, o0)
        for a, o in steps:
            s, e = self.f[(s, a, o)], (s, a, o)
        return s, e

    # ---- grow on surprise
    def try_split(self, s):
        """Split s on the incoming edge whose arrivals differ most from the rest."""
        best, best_e = 0.0, None
        if len(self.ctx[s]) < 2:
            return False
        for e, ne in self.ctx[s].items():
            rest = self.n[s] - ne
            if ne.sum() == 0 or rest.sum() == 0:
                continue
            d = -dF_merge(ne, rest)
            if d > best:
                best, best_e = d, e
        if best_e is None:
            return False
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
    def try_merge(self):
        """Greedy BMR merges (best dF first), each cascading to successors."""
        merged = False
        while True:
            pairs = sorted(((dF_merge(self.n[i], self.n[j]), i, j)
                            for i, j in itertools.combinations(range(self.K), 2)), reverse=True)
            for d, i, j in pairs:
                if d <= 0:
                    return merged
                root = self.cascade(i, j)
                if root is not None:
                    self.relabel(root)
                    merged = True
                    break
            else:
                return merged

    def cascade(self, i, j):
        """Merge i and j and, to keep f deterministic, every pair of successors they
        reach by the same (a, o); all-or-nothing on the BMR test."""
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
            if dF_merge(n[x], n[y]) <= 0:
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
        c = self.n[:, 1:] + ALPHA
        return c / c.sum(-1, keepdims=True)                # p(o | s, a)

    def novelty(self):
        c = self.n[:, 1:] + ALPHA
        p = c / c.sum(-1, keepdims=True)
        return 0.5 * (p * (1 / c - 1 / c.sum(-1, keepdims=True))).sum(-1)   # (s, a)

    def tried(self):
        return self.n[:, 1:].sum(-1)                        # (s, a) data counts

    def shown(self):
        """Each state's observation: the symbol it was most often entered with."""
        w = self.n[:, 0]
        out = [int(np.argmax(r)) if r.sum() else -1 for r in w]
        purity = [float(r.max() / r.sum()) if r.sum() else 1.0 for r in w]
        return out, purity


# ------------------------------------------------------------ planning
def risk_per_obs():
    C = np.zeros(NO)
    C[NC:] = REWARD_PREF                                    # any symbol carrying reward
    return -(C - np.log(np.exp(C).sum()))


def efe_table(M, novelty_weight=1.0, horizon=HORIZON, gamma=GAMMA):
    r = risk_per_obs()
    p = M.p_obs()
    nov = novelty_weight * M.novelty()
    nxt = np.zeros((M.K, NA, NO), int)
    for (s, a, o), t in M.f.items():
        nxt[s, a, o] = t
    V = np.zeros(M.K)
    for _ in range(horizon):
        G = (p * (r + V[nxt])).sum(-1) - nov
        pi = np.exp(-gamma * (G - G.min(1, keepdims=True)))
        pi /= pi.sum(1, keepdims=True)
        V = (pi * G).sum(1)
    return G


def psrl_table(M, rng, horizon=T_MAX):
    """Posterior sampling on the learned machine: draw p(o | s, a) from the tensor's
    Dirichlet posterior and maximise expected reward over the remaining steps. With a
    preference for reward only, this is EFE's risk term at infinite precision with the
    epistemic terms replaced by sampling: no exploration-exploitation weight."""
    p = rng.gamma(M.n[:, 1:] + ALPHA)
    p /= p.sum(-1, keepdims=True)
    reward = (np.arange(NO) >= NC).astype(float)
    nxt = np.zeros((M.K, NA, NO), int)
    for (s, a, o), t in M.f.items():
        nxt[s, a, o] = t
    Qh = np.zeros((horizon + 1, M.K, NA))
    for h in range(1, horizon + 1):
        Qh[h] = (p * (reward + Qh[h - 1].max(1)[nxt])).sum(-1)
    return Qh


# ------------------------------------------------------------ evaluation
def true_next():
    """True p(next symbol | cell, a) from the env's own table (stuck in hole/goal)."""
    P = FrozenLakeEnv(desc=DESC, is_slippery=False).P
    T = np.zeros((NC, NA, NO))
    for c in range(NC):
        for a in range(NA):
            for prob, c2, rew, _ in P[c][a]:
                T[c, a, symbol(c2, rew)] += prob
    return T


def evaluate(M, Ttrue, seen):
    shown, purity = M.shown()
    p = M.p_obs()
    tvs = [0.5 * np.abs(p[s, a] - Ttrue[o % NC, a]).sum()
           for s, o in enumerate(shown) if o >= 0 and purity[s] == 1.0
           for a in range(NA) if M.tried()[s, a] > 0]
    # minimal machine: one pure state per observation symbol seen
    exact = (M.K == len(seen) and sorted(shown) == sorted(seen) and min(purity) == 1.0)
    return {'n_states': M.K, 'shown': [OBS[o] if o >= 0 else '?' for o in shown],
            'min_purity': min(purity), 'exact_structure': bool(exact),
            'B_TV_mean': float(np.mean(tvs)) if tvs else None}


def outcome(steps):
    cells = [o % NC for _, o in steps]
    if any(o >= NC for _, o in steps):
        return 'goal'
    return 'hole' if 4 in cells else 'timeout'


def run(policy, seed, episodes):
    """One agent from scratch; returns the per-episode log and the final machine."""
    rng = np.random.default_rng(seed)
    env = FrozenLakeEnv(desc=DESC, is_slippery=False)
    env.reset(seed=seed)
    Ttrue = true_next()
    M = Machine()
    log, seen = [], set()
    for ep in range(episodes):
        c0, _ = env.reset()
        o0 = symbol(c0, 0)
        seen.add(o0)
        M.episodes.append((o0, []))
        s, e = M.enter(o0)
        splits = 0
        Qh = psrl_table(M, rng) if policy == 'psrl_learned' else None
        for t in range(T_MAX):                            # never cut at termination
            if policy == 'random':
                a = int(rng.integers(NA))
            elif policy == 'psrl_learned':
                if Qh.shape[1] != M.K:                    # a new observation added a state
                    Qh = psrl_table(M, rng)
                q = Qh[T_MAX - t, s]
                a = int(rng.choice(np.flatnonzero(q == q.max())))
            else:
                G = efe_table(M, 0.0 if policy == 'efe_nonovelty' else 1.0)
                pi = np.exp(-GAMMA * (G[s] - G[s].min())); pi /= pi.sum()
                a = int(rng.choice(NA, p=pi))
            c, rew, _, _, _ = env.step(a)
            o = symbol(c, rew)
            seen.add(o)
            M.episodes[-1][1].append((a, o))
            s_prev = s
            s, e = M.count(s, e, a, o)                    # per-step parameter learning
            if M.try_split(s_prev):                       # grow on surprise
                splits += 1
                s, e = M.locate(o0, M.episodes[-1][1])
                if policy == 'psrl_learned':              # the state space changed
                    Qh = psrl_table(M, rng)
        merged = M.try_merge()
        s_steps = M.episodes[-1][1]
        log.append({'episode': ep, 'policy': policy, 'seed': seed,
                    'result': outcome(s_steps), 'reward': sum(o >= NC for _, o in s_steps),
                    'splits': splits, 'merged': merged,
                    'state_action_pairs_tried': int((M.tried() > 0).sum()),
                    **evaluate(M, Ttrue, seen)})
    return log, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', choices=['efe', 'efe_nonovelty', 'psrl_learned', 'random'], default='efe')
    ap.add_argument('--episodes', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    log, _ = run(args.policy, args.seed, args.episodes)
    out = REPO / 'results' / 'frozenlake_sl'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'{args.policy}_s{args.seed}.json', 'w') as fh:
        json.dump(log, fh, indent=1)
    last = log[-1]
    first_exact = next((r['episode'] for r in log if r['exact_structure']), None)
    goals = [r['result'] == 'goal' for r in log]
    print(f"{args.policy:14s} seed {args.seed}: states {last['n_states']} "
          f"exact {last['exact_structure']} (first at episode {first_exact}), "
          f"goal rate last 50 {np.mean(goals[-50:]):.2f}, holes total "
          f"{sum(r['result'] == 'hole' for r in log)}, B TV {last['B_TV_mean']}, "
          f"shown {last['shown']}")


if __name__ == '__main__':
    main()
