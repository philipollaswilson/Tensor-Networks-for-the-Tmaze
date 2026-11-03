"""RL baselines for the small frozen lake, on exactly the stream frozenlake_sl.py sees.

Same map, seeds, 100-step episodes never cut at termination (stuck in the hole or the
goal afterwards), same reward. Unlike the structure-learning agent, both baselines are
told that the cell is the state (a tabular Markov state is given, not learned).

  qlearning  tabular Q-learning, epsilon-greedy (alpha 0.1, gamma 0.99, epsilon 0.1):
             the standard model-free baseline.
  psrl       posterior sampling RL (Osband, Russo & Van Roy 2013): Dirichlet(1) over
             next cells and Beta(1, 1) over reward per (cell, action, next cell); each
             episode samples one MDP and follows its optimal finite-horizon policy.
  opsrl      optimistic posterior sampling (Tiapkin et al., NeurIPS 2022, Algorithm 1):
             the prior is one pseudo-count towards an absorbing pseudo-state paying
             r0 = 2 per step; each episode draws J = 10 transition samples per (cell,
             action) from Dir(counts / kappa) with kappa = 1 and backs up the max over
             samples. Reported to match PSRL and beat UCBVI/RLSVI; minimax-optimal regret.
             The paper's theory uses stage-dependent transitions and much larger
             J, kappa, n0; we pool over steps (the lake is stationary), as PSRL does here.
  Neither posterior-sampling method has an exploration-exploitation weight: exploration
  comes from sampling the posterior (and, in OPSRL, from the optimistic prior).

Run from the repo root:  python src/frozenlake_rl.py --policy qlearning|psrl|opsrl [--seed 0]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from frozenlake_sl import DESC, NA, NC, REPO, T_MAX, outcome, symbol  # noqa: E402


def greedy(q, rng):
    return int(rng.choice(np.flatnonzero(q == q.max())))


def opsrl_q(N, R1, rng, J=10, kappa=1.0, n0=1.0, r0=2.0):
    """Optimistic backward induction over J posterior samples (OPSRL, eq. 1)."""
    alpha = np.concatenate([N, np.full(N.shape[:2] + (1,), n0)], -1) / kappa
    W = rng.gamma(alpha[None].repeat(J, 0))
    W /= W.sum(-1, keepdims=True)                         # (J, cell, a, next cell + s0)
    Rhat = np.where(N > 0, R1 / np.maximum(N, 1), 0.0)
    Qh = np.zeros((T_MAX + 1, NC, NA))
    for h in range(1, T_MAX + 1):
        V = Qh[h - 1].max(1)
        backup = (W[..., :NC] * (Rhat + V)[None]).sum(-1) + W[..., NC] * r0 * h
        Qh[h] = backup.max(0)
    return Qh


PRIOR = {'psrl': 1.0, 'psrl_sparse': 0.1,        # Dirichlet concentration per next cell
         'opsrl': 1.0, 'opsrl_sparse': 0.1}        # pseudo-count n0 to the optimistic state


def run(policy, seed, episodes, lr=0.1, discount=0.99, eps=0.1):
    rng = np.random.default_rng(seed)
    name, prior = policy, PRIOR.get(policy)
    policy = policy.replace('_sparse', '')
    env = FrozenLakeEnv(desc=DESC, is_slippery=False)
    env.reset(seed=seed)
    Q = np.zeros((NC, NA))
    N = np.zeros((NC, NA, NC))                    # PSRL transition counts
    R1 = np.zeros((NC, NA, NC))                   # PSRL rewarded transitions
    log = []
    for ep in range(episodes):
        c, _ = env.reset()
        if policy == 'psrl':                       # sample an MDP, plan the whole episode
            P = np.array([[rng.dirichlet(prior + N[x, a]) for a in range(NA)] for x in range(NC)])
            Rs = rng.beta(prior + R1, prior + N - R1)
            Qh = np.zeros((T_MAX + 1, NC, NA))
            for h in range(1, T_MAX + 1):
                Qh[h] = (P * (Rs + Qh[h - 1].max(1)[None, None, :])).sum(-1)
        elif policy == 'opsrl':
            Qh = opsrl_q(N, R1, rng, n0=prior)
        steps = []
        for t in range(T_MAX):                     # never cut at termination
            if policy == 'qlearning':
                a = int(rng.integers(NA)) if rng.random() < eps else greedy(Q[c], rng)
            else:
                a = greedy(Qh[T_MAX - t, c], rng)
            c2, rew, _, _, _ = env.step(a)
            if policy == 'qlearning':
                Q[c, a] += lr * (rew + discount * Q[c2].max() - Q[c, a])
            else:
                N[c, a, c2] += 1
                R1[c, a, c2] += rew
            steps.append((a, symbol(c2, rew)))
            c = c2
        log.append({'episode': ep, 'policy': name, 'seed': seed, 'result': outcome(steps),
                    'reward': sum(o >= NC for _, o in steps)})
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', choices=['qlearning', 'psrl', 'psrl_sparse', 'opsrl', 'opsrl_sparse'], default='psrl')
    ap.add_argument('--episodes', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    log = run(args.policy, args.seed, args.episodes)
    out = REPO / 'results' / 'frozenlake_sl'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'{args.policy}_s{args.seed}.json', 'w') as fh:
        json.dump(log, fh, indent=1)
    goals = [r['result'] == 'goal' for r in log]
    first = next((i for i, g in enumerate(goals) if g), None)
    print(f"{args.policy:10s} seed {args.seed}: first goal at episode {first}, goal rate "
          f"last 50 {np.mean(goals[-50:]):.2f}, holes total "
          f"{sum(r['result'] == 'hole' for r in log)}")


if __name__ == '__main__':
    main()
