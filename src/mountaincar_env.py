"""Partially observed Mountain Car (position only) as a symbolic stream, plus scoring.

Physics and constants are gymnasium's MountainCar-v0 (force 0.001, gravity 0.0025,
|v| <= 0.07, x in [-1.2, 0.6], inelastic left wall, 200-step limit, reward -1 per step,
start x0 ~ U[-0.6, -0.4], v0 = 0). Changes, all applied identically to every agent:
  * velocity is hidden: the agent sees only the position, binned into NB = 18 bins of
    width 0.1 (the standard position-only POMDP variant of recurrent-RL papers);
  * no terminal flag: after reaching the flag the car stays there for the rest of the
    200 steps and the reward is 0 instead of -1 (gym's return is unchanged: the episode
    return is -(steps to the flag), or -200).
Observation symbol o = bin + NB * [reward == 0], so the agent sees the reward too.
  * moving flag (optional): the flag is on the right hilltop (gym's x >= 0.5, v >= 0)
    or on the left (x <= -1.1); `flag_schedule` switches it every SWITCH episodes.

Scoring follows RL practice: regret per episode against a near-optimal controller
(value iteration on a fine (x, v) grid, run on the true continuous dynamics from the
same start), evaluation return with exploration off, and IQM over seeds.
"""
import copy
import math

import numpy as np

NB = 60
LO, HI = -1.2, 0.6
NA, NO = 3, 2 * NB
T_MAX = 200
SWITCH = 50                      # moving-flag condition: episodes per flag side
ACT = ['push left', 'none', 'push right']


def to_bin(x):
    return min(int((x - LO) / (HI - LO) * NB), NB - 1)


def symbol(x, reward):
    return to_bin(x) + NB * int(reward == 0)


def physics(x, v, a):
    v = np.clip(v + (a - 1) * 0.001 + np.cos(3 * x) * (-0.0025), -0.07, 0.07)
    x = np.clip(x + v, LO, HI)
    v = np.where((x == LO) & (v < 0), 0.0, v)
    return x, v


def at_flag(x, v, flag):
    return (x >= 0.5) & (v >= 0) if flag == 'right' else x <= -1.1


def flag_schedule(moving, episode):
    return 'left' if moving and (episode // SWITCH) % 2 == 1 else 'right'


class MountainCarPO:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)       # env randomness only: start states

    def reset(self, flag='right', x0=None):
        self.flag = flag
        self.x = float(self.rng.uniform(-0.6, -0.4)) if x0 is None else x0
        self.v, self.done = 0.0, False
        return symbol(self.x, -1)

    def step(self, a):
        if self.done:
            return symbol(self.x, 0), 0.0
        x, v = physics(self.x, self.v, a)
        self.x, self.v = float(x), float(v)
        self.done = bool(at_flag(self.x, self.v, self.flag))
        return symbol(self.x, -1), -1.0


# ------------------------------------------------------------ near-optimal controller
_GRID = {}


def oracle_policy(flag, nx=1801, nv=561):
    """Greedy policy of value iteration (steps to flag) on an nx x nv grid. Returns from
    the evaluation starts change by <= 4 steps on a 4x finer grid."""
    if flag in _GRID:
        return _GRID[flag]
    xs, vs = np.linspace(LO, HI, nx), np.linspace(-0.07, 0.07, nv)
    X, Vv = np.meshgrid(xs, vs, indexing='ij')
    nxt = []
    for a in range(NA):
        x2, v2 = physics(X, Vv, a)
        ix = np.clip(np.rint((x2 - LO) / (HI - LO) * (nx - 1)).astype(int), 0, nx - 1)
        iv = np.clip(np.rint((v2 + 0.07) / 0.14 * (nv - 1)).astype(int), 0, nv - 1)
        nxt.append((ix, iv, at_flag(x2, v2, flag)))
    J = np.zeros((nx, nv))
    for _ in range(T_MAX + 50):
        Q = np.stack([np.where(g, 1.0, 1.0 + J[ix, iv]) for ix, iv, g in nxt])
        J = Q.min(0)
    _GRID[flag] = (xs, vs, Q)
    return _GRID[flag]


def oracle_return(x0, flag):
    xs, vs, Q = oracle_policy(flag)
    x, v = x0, 0.0
    for t in range(T_MAX):
        ix = int(np.rint((x - LO) / (HI - LO) * (len(xs) - 1)))
        iv = int(np.rint((v + 0.07) / 0.14 * (len(vs) - 1)))
        x, v = physics(x, v, int(np.argmin(Q[:, ix, iv])))
        x, v = float(x), float(v)
        if at_flag(x, v, flag):
            return -(t + 1)
    return -T_MAX


def episode_return(rewards):
    return float(sum(rewards))


EVAL_STARTS = list(np.linspace(-0.6, -0.4, 5))     # fixed evaluation starts
EVAL_EVERY = 10


def play(agent, env, flag, x0=None):
    """One 200-step episode. Agents see symbols; `info` (true x, v) is passed for the
    baselines that are told the state, and ignored by the learners."""
    o0 = env.reset(flag, x0)
    env.x0 = env.x
    agent.begin(o0, (env.x, env.v, env.done))
    rewards = []
    for t in range(T_MAX):
        a = agent.act(t)
        o, r = env.step(a)
        agent.observe(a, o, (env.x, env.v, env.done))
        rewards.append(r)
    agent.end()
    return episode_return(rewards)


def run_agent(agent, name, seed, episodes, moving=False, log_extra=None):
    """Online run with per-episode regret and periodic evaluation (exploration off, on
    a copy of the agent, from the fixed EVAL_STARTS)."""
    env, eval_env = MountainCarPO(seed), MountainCarPO(seed + 10_000)
    log = []
    for ep in range(episodes):
        flag = flag_schedule(moving, ep)
        R = play(agent, env, flag)
        rec = {'episode': ep, 'policy': name, 'seed': seed, 'flag': flag, 'return': R}
        rec['optimal'] = oracle_return(env.x0, flag)
        rec['regret'] = rec['optimal'] - R
        if (ep + 1) % EVAL_EVERY == 0:
            evals = []
            for x0 in EVAL_STARTS:
                ag = copy.deepcopy(agent)
                ag.explore = False
                evals.append(play(ag, eval_env, flag, x0))
            rec['eval_return'] = float(np.mean(evals))
        if log_extra:
            rec.update(log_extra(agent))
        log.append(rec)
    return log


def iqm(x):
    """Interquartile mean (rliable): mean of the middle 50% of values."""
    x = np.sort(np.asarray(x, float).ravel())
    n = len(x)
    lo, hi = int(math.floor(0.25 * n)), int(math.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def iqm_ci(per_seed, reps=2000, rng=None):
    """Stratified bootstrap 95% CI of the IQM over seeds."""
    rng = rng or np.random.default_rng(0)
    per_seed = np.asarray(per_seed, float)
    boots = [iqm(per_seed[rng.integers(len(per_seed), size=len(per_seed))]) for _ in range(reps)]
    return iqm(per_seed), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
