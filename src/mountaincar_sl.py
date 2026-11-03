"""Online structure learning on position-only Mountain Car (classical tensor).

The agent sees only the binned position and the reward (mountaincar_env.py); velocity
must be recovered as learned states. Model, learning and planning are those of the
frozen lake (frozenlake_sl.py) via structure_learner.py: a uniform non-negative MPS /
unifilar HMM that counts every step, splits states on surprise and merges them by BMR.
Its only preference is for reward-0 symbols (being at the flag).

  efe           sophisticated EFE + novelty, horizon = steps left in the episode
  psrl_learned  posterior sampling from the tensor's Dirichlet counts (knob-free)
  random        uniform actions
`--k_max D` caps the number of states (the memory budget D, for the Born comparison).

Run from the repo root:
  python src/mountaincar_sl.py --policy efe [--seed 0 --episodes 100 --moving]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mountaincar_env as E  # noqa: E402
from structure_learner import Machine, efe_table, psrl_table, risk  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
ALPHA = 1.0 / E.NO              # Perks prior: total prior mass 1 per (state, action)
REWARD_PREF = 2.0
GAMMA = 16.0
SURPRISE = np.log(2.0)          # test a split when an observation was worse than a coin flip


def preferences():
    C = np.zeros(E.NO)
    C[E.NB:] = REWARD_PREF                       # symbols that carry reward 0 (at the flag)
    return C


class StructureAgent:
    def __init__(self, policy, seed, k_max=None):
        self.policy, self.rng = policy, np.random.default_rng(seed)
        self.M = Machine(E.NA, E.NO, ALPHA, k_max)
        self.r = risk(preferences())
        self.reward = (np.arange(E.NO) >= E.NB).astype(float) - 1.0   # gym's reward
        self.explore = True

    def plan_psrl(self):
        M = self.M
        if self.explore:
            self.Qh = psrl_table(M.n[:, 1:], ALPHA, M.next_table(), self.reward, E.T_MAX, self.rng)
        else:                                     # evaluation: posterior mean
            p, nxt = M.p_obs(), M.next_table()
            self.Qh = np.zeros((E.T_MAX + 1, M.K, E.NA))
            for h in range(1, E.T_MAX + 1):
                self.Qh[h] = (p * (self.reward + self.Qh[h - 1].max(1)[nxt])).sum(-1)

    def begin(self, o0, info):
        self.o0 = o0
        self.M.episodes.append((o0, []))
        self.s, self.e = self.M.enter(o0)
        self.splits = 0
        if self.policy == 'psrl_learned':
            self.plan_psrl()

    def act(self, t):
        M = self.M
        if self.policy == 'random':
            return int(self.rng.integers(E.NA))
        if self.policy == 'psrl_learned':
            if self.Qh.shape[1] != M.K:
                self.plan_psrl()
            q = self.Qh[E.T_MAX - t, self.s]
            return int(self.rng.choice(np.flatnonzero(q == q.max())))
        nov = M.novelty() if self.explore else 0.0
        G = efe_table(M.p_obs(), M.next_table(), self.r, nov, E.T_MAX - t, GAMMA)[self.s]
        if not self.explore:
            return int(self.rng.choice(np.flatnonzero(G == G.min())))
        pi = np.exp(-GAMMA * (G - G.min()))
        return int(self.rng.choice(E.NA, p=pi / pi.sum()))

    def observe(self, a, o, info):
        M = self.M
        M.episodes[-1][1].append((a, o))
        s_prev, e_prev = self.s, self.e
        surprise = -np.log(M.p_obs()[s_prev, a, o])
        self.s, self.e = M.count(self.s, self.e, a, o)
        if surprise > SURPRISE and M.try_split_loop(s_prev, e_prev):
            self.splits += 1
            self.s, self.e = M.locate(self.o0, M.episodes[-1][1])
            if self.policy == 'psrl_learned':
                self.plan_psrl()

    def end(self):
        self.merged = self.M.try_merge()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', choices=['efe', 'psrl_learned', 'random'], default='efe')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k_max', type=int, default=None)
    ap.add_argument('--moving', action='store_true')
    args = ap.parse_args()
    agent = StructureAgent(args.policy, args.seed, args.k_max)
    name = args.policy + (f'_k{args.k_max}' if args.k_max else '')
    t0 = time.time()

    def extra(ag):
        return {'n_states': ag.M.K, 'splits': ag.splits, 'merged': ag.merged,
                'seconds': time.time() - t0}
    log = E.run_agent(agent, name, args.seed, args.episodes, args.moving, extra)
    out = REPO / 'results' / ('mountaincar_moving' if args.moving else 'mountaincar')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'{name}_s{args.seed}.json', 'w') as fh:
        json.dump(log, fh, indent=1)
    R = [r['return'] for r in log]
    print(f"{name:14s} seed {args.seed}: return last 20 {np.mean(R[-20:]):.1f}, "
          f"regret {sum(r['regret'] for r in log):.0f}, states {log[-1]['n_states']}, "
          f"{time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
