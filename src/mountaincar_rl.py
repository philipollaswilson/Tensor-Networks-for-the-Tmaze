"""RL baselines for position-only Mountain Car, on the stream mountaincar_sl.py sees.

  psrl_state   PSRL (Osband et al. 2013) told the true state: (x, v) on a 20 x 20 grid
               plus the reward bit, Dirichlet(1/S) over the S next states (Perks prior,
               total mass 1, as our learner; Dirichlet(0.1) over 800 states adds 80
               pseudo-counts and never reached the flag in 500 episodes). Knob-free.
  psrl_obs     PSRL on the observation symbol alone (binned position + reward): the
               same input as our learner, without memory.
  qlearning    tabular Q-learning told the true state (20 x 20 grid), epsilon-greedy
               (alpha 0.1, gamma 0.99, epsilon 0.1, Q initialised to 0, which is
               optimistic under reward -1 per step).
The reward is gym's (-1 per step until the flag, 0 afterwards).

Run from the repo root:  python src/mountaincar_rl.py --policy psrl_state [--seed 0 --moving]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mountaincar_env as E  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
GX, GV = 20, 20


def grid_state(info):
    x, v, done = info
    ix = min(int((x - E.LO) / (E.HI - E.LO) * GX), GX - 1)
    iv = min(int((v + 0.07) / 0.14 * GV), GV - 1)
    return ix + GX * iv + GX * GV * int(done)


class TabularAgent:
    def __init__(self, policy, seed):
        self.policy, self.rng = policy, np.random.default_rng(seed)
        self.S = 2 * GX * GV if policy != 'psrl_obs' else E.NO
        self.prior = 1.0 / self.S             # Perks prior, as our learner (total mass 1)
        self.N = np.zeros((self.S, E.NA, self.S))
        self.Q = np.zeros((self.S, E.NA))
        self.explore = True
        if policy == 'psrl_obs':
            self.reward = (np.arange(self.S) >= E.NB).astype(float) - 1.0
        else:
            self.reward = (np.arange(self.S) >= GX * GV).astype(float) - 1.0

    def state(self, o, info):
        return o if self.policy == 'psrl_obs' else grid_state(info)

    def plan(self):
        if self.explore:
            P = self.rng.gamma(self.prior + self.N)
        else:
            P = self.prior + self.N
        P /= P.sum(-1, keepdims=True)
        self.Qh = np.zeros((E.T_MAX + 1, self.S, E.NA))
        for h in range(1, E.T_MAX + 1):
            V = self.Qh[h - 1].max(1)
            self.Qh[h] = P @ (self.reward + V)

    def begin(self, o0, info):
        self.s = self.state(o0, info)
        if self.policy.startswith('psrl'):
            self.plan()

    def act(self, t):
        if self.policy == 'qlearning':
            if self.explore and self.rng.random() < 0.1:
                return int(self.rng.integers(E.NA))
            q = self.Q[self.s]
        else:
            q = self.Qh[E.T_MAX - t, self.s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def observe(self, a, o, info):
        s2 = self.state(o, info)
        if self.policy == 'qlearning':
            r = self.reward[s2]
            self.Q[self.s, a] += 0.1 * (r + 0.99 * self.Q[s2].max() - self.Q[self.s, a])
        else:
            self.N[self.s, a, s2] += 1
        self.s = s2

    def end(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', choices=['psrl_state', 'psrl_obs', 'qlearning'], default='psrl_state')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--moving', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    log = E.run_agent(TabularAgent(args.policy, args.seed), args.policy, args.seed,
                      args.episodes, args.moving, lambda ag: {'seconds': time.time() - t0})
    out = REPO / 'results' / ('mountaincar_moving' if args.moving else 'mountaincar')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'{args.policy}_s{args.seed}.json', 'w') as fh:
        json.dump(log, fh, indent=1)
    R = [r['return'] for r in log]
    print(f"{args.policy:12s} seed {args.seed}: return last 20 {np.mean(R[-20:]):.1f}, "
          f"regret {sum(r['regret'] for r in log):.0f}, first flag at "
          f"{next((i for i, x in enumerate(R) if x > -200), None)}, {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
