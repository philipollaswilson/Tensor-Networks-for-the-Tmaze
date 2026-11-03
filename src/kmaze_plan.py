"""Planning with the learned model vs. model-free baselines on the same offline data.

Star k-secret maze (kmaze.py). Reward: +1 per step in an arm with cheese, -4 in an empty
arm, 0 elsewhere (with -1, guessing an arm and switching after a miss averages 10.5 and
beats reading the cue; with -4 it averages 8.5, so the cue is worth visiting). Every agent gets the same random-policy episodes and is scored on fresh
episodes in the real maze (undiscounted return over H steps).
  ours     the learned unifilar machine; its states are sufficient for prediction, so value
           iteration on it is planning in belief space
  window W fitted Q-iteration on the last W (action, observation) pairs (W = 1 memoryless)
  oracle   H - 2: cue, back, correct arm, stay
Unvisited (state, action) pairs have value 0 for every agent, and every agent's estimates
carry the same prior: one pseudo-observation of value 0.
Run:  python src/kmaze_plan.py
"""
from collections import defaultdict

import numpy as np

import kmaze as K

GAMMA = 0.9


def reward(env, o):
    if hasattr(env, 'reward'):
        return env.reward(o)
    loc, v = divmod(o, 2)
    if loc <= env.k:
        return 0.0
    return 1.0 if v else -4.0


def value_iteration(m, env, iters=200):
    """Posterior-mean transitions: n(o) / (n + 1); the remaining 1 / (n + 1) (prior mass
    1, as in structure learning) goes to an unknown outcome of value 0. Without it a
    rarely visited state trusts a handful of lucky samples."""
    Kn, A, O = m['P1'].shape
    R = np.array([reward(env, o) for o in range(O)])
    P = m['C1'] / (m['C1'].sum(-1, keepdims=True) + 1)
    Q = np.zeros((Kn, A))
    for _ in range(iters):
        V = Q.max(1)
        Qn = np.zeros_like(Q)
        for (s, a, o), s2 in m['f'].items():
            Qn[s, a] += P[s, a, o] * (R[o] + GAMMA * V[s2])
        Q = Qn
    return Q


def fqi(trajs, env, W, iters=200):
    """Tabular fitted Q-iteration on windows of the last W (action, observation) pairs."""
    def key(obs, acts, t):
        lo = max(0, t - W + 1)
        return tuple(obs[lo:t + 1]) + tuple(acts[lo:t])
    data = defaultdict(list)
    for obs, acts, _ in trajs:
        for t, a in enumerate(acts):
            data[key(obs, acts, t), a].append((reward(env, obs[t + 1]), key(obs, acts, t + 1)))
    Q = defaultdict(float)
    for _ in range(iters):
        V = defaultdict(float)
        for (x, a), q in Q.items():
            V[x] = max(V[x], q) if x in V else q
        # same prior as ours: one pseudo-observation of value 0
        Q = defaultdict(float, {(x, a): sum(r + GAMMA * V[x2] for r, x2 in ys) / (len(ys) + 1)
                                for (x, a), ys in data.items()})
    return Q, key


def evaluate(env, act, H, n=2000, seed=1):
    """act(obs, acts) -> action. Returns the mean undiscounted return."""
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n):
        bits = tuple(int(b) for b in rng.integers(2, size=env.k))
        loc = 0
        obs, acts = [env.emit(loc, bits, rng)], []
        for _ in range(H):
            a = act(obs, acts, rng)
            loc = env.move(loc, a)
            acts.append(a); obs.append(env.emit(loc, bits, rng))
            total += reward(env, obs[-1])
    return total / n


def ours_policy(m, env):
    Q = value_iteration(m, env)

    def act(obs, acts, rng):
        s = m['starts'].get(obs[0], m['start'])
        for t, a in enumerate(acts):
            s = m['f'].get((s, a, obs[t + 1]), s)
        q = Q[s]
        return int(rng.choice(np.flatnonzero(q == q.max())))
    return act


def window_policy(trajs, env, W):
    Q, key = fqi(trajs, env, W)

    def act(obs, acts, rng):
        x = key(obs, acts, len(acts))
        q = np.array([Q.get((x, a), 0.0) for a in range(env.n_act)])
        return int(rng.choice(np.flatnonzero(q == q.max())))
    return act


def compare(k=1, budgets=(25, 50, 100, 200, 500, 1000, 3000), T=12, H=12, seeds=(0, 1, 2),
            windows=(1, 2, 3), verbose=True):
    env = K.KMaze(k, layout='star')
    out = defaultdict(lambda: defaultdict(list))
    for n in budgets:
        for seed in seeds:
            trajs = env.sample(n, T, seed=100 + seed)
            S, _ = K.learn_states(trajs, env.n_obs, 3)
            m = K.machine(trajs, S, env.n_act, env.n_obs)
            out['ours'][n].append(evaluate(env, ours_policy(m, env), H, seed=seed))
            out['ours_K'][n].append(m['K'])
            for W in windows:
                out[f'window {W}'][n].append(evaluate(env, window_policy(trajs, env, W), H,
                                                      seed=seed))
        if verbose:
            print(f'n={n:5d}: ' + ', '.join(f'{name} {np.mean(v[n]):.2f}'
                                           for name, v in out.items()), flush=True)
    return {name: {n: list(v) for n, v in d.items()} for name, d in out.items()}, H - 2


def compare_corridor(Ds=(2, 4, 8), budgets=(100, 300, 1000, 3000), seeds=(0, 1, 2),
                     verbose=True):
    """Long-corridor T-maze: a window agent needs W >= D + 2 to carry the cue to the
    junction. Optimal return H - D - 1 = 11 (H = D + 12): D + 1 steps forward, turn."""
    out = {}
    for D in Ds:
        env = K.Corridor(D)
        H, T = D + 12, 3 * D + 12
        windows = sorted({1, 2, 4, 8, D + 2})
        res = defaultdict(lambda: defaultdict(list))
        for n in budgets:
            for seed in seeds:
                trajs = env.sample(n, T, seed=100 + seed)
                S, _ = K.learn_states(trajs, env.n_obs, 3)
                m = K.machine(trajs, S, env.n_act, env.n_obs)
                res['ours'][n].append(evaluate(env, ours_policy(m, env), H, n=500, seed=seed))
                res['ours_K'][n].append(m['K'])
                for W in windows:
                    res[f'window {W}'][n].append(
                        evaluate(env, window_policy(trajs, env, W), H, n=500, seed=seed))
            if verbose:
                print(f'D={D} n={n:5d}: ' + ', '.join(f'{k} {np.mean(v[n]):.2f}'
                                                    for k, v in res.items()), flush=True)
        out[D] = dict(res={k: {n: list(v) for n, v in d.items()} for k, d in res.items()},
                      oracle=H - D - 1, truth_K=2 * (D + 4))
    return out


if __name__ == '__main__':
    import json
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / 'results'
    root.mkdir(exist_ok=True)
    res, oracle = compare()
    json.dump(dict(res=res, oracle=oracle), open(root / 'kmaze_plan_star.json', 'w'))
    out = compare_corridor()
    json.dump({str(D): v for D, v in out.items()}, open(root / 'kmaze_plan_corridor.json', 'w'))
