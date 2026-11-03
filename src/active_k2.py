"""Active structure learning on the k = 2 maze: can exploration fix what uniform data can't?

Rules, fixed before any result was seen. Same learner (kmaze.learn_states, defaults), same
first batch, same step budget for all arms; truth is used only to score, and by the oracle.
  loop     learn states on all data; score actions; collect until the next checkpoint
  episode  20 steps: 12 planned, then 8 uniform random (the tail). Multi-step and event windows
           come from tails only; one-step counts come from every step (kmaze.Episode).
           The uniform arm has tail = 0, so it keeps windows at every step.
  scores   in nats, no weights, re-planned every REPLAN episodes with updated counts
    uniform    no planning
    novelty    Dirichlet expected information gain about p(o | s, a)
    structure  for each edge (s, a, o), the entropy of the undecided merge with the other edges
               into its target, spread over its visits: H(p_merge) / (n_e + 1), with
               p_merge = 1 / (1 + exp(-margin)) and margin the distance to the learner's decision
    combined   novelty + structure
    oracle     novelty on the true predictive states and true transitions: an upper limit
  score    steps to exact recovery (45 states, 99% of visits one-to-one); share of visits on the
           right state; number of learned states owning two true states
Outcomes: arm 3 or 4 beats arm 2 with non-overlapping seed ranges: structure exploration is its
own effect. Arm 2 beats arm 1: exploration helps. Oracle fails: the problem is the learner.
Merge doubt is measured for edges into one state, so it can't see copies of one state in two
targets; that is the failure k = 2 shows least.

Progress is saved at every checkpoint (<arm>_s<seed>.pkl), and a rerun resumes from there.
Run:  python src/active_k2.py [--workers 4] [--arms novelty oracle] [--seeds 0 1]
"""
import argparse
import collections
import ctypes
import itertools
import json
import pathlib
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.special import digamma, expit, gammaln

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import kmaze as M  # noqa: E402
import merge_theory as MT  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / 'results' / 'active_k2'
ARMS = ['uniform', 'novelty', 'structure', 'combined', 'oracle']
K_SECRETS, T, PREFIX, L = 2, 20, 12, 3
CHECKS = [1000, 2000, 4000, 8000, 16000, 32000]
SEEDS = 5
REPLAN = 50
STRONG = 3.0                # the veto of kmaze.bmr_rows


def novelty(C, O):
    """Expected information gain about p(o | s, a) from one more sample; C (K, A, O)."""
    alpha = C + 1.0 / O
    a0 = alpha.sum(-1, keepdims=True)
    p = alpha / a0
    h_mean = -(p * np.log(p)).sum(-1)
    e_h = -(p * (digamma(alpha + 1) - digamma(a0 + 1))).sum(-1)
    return np.maximum(h_mean - e_h, 0)


def merge_entropy(trajs, S, O):
    """Entropy, in nats, of the most undecided merge of each edge (s, a, o) with the other edges
    into the same state. margin > 0 means the learner's test would merge them."""
    windows = collections.defaultdict(list)
    for ep, s in zip(trajs, S):
        obs, acts, _ = ep
        ed = [None] + [(int(s[t - 1]), acts[t - 1], obs[t], int(s[t]))
                       for t in range(1, len(acts) + 1)]
        for t, key in M.future_windows(list(range(len(acts) + 1)), obs, acts, L,
                                       tail=M.tail_of(ep)):
            if ed[t] is not None and ed[t][0] >= 0 and ed[t][3] >= 0:
                windows[ed[t][3]].append((ed[t], key))
    H = {}
    for wins in windows.values():
        items = list(dict.fromkeys(e for e, _ in wins))
        if len(items) < 2:
            continue
        R = M.Rows(items, wins, O, L)
        a, X, Tot = R.alpha, R.X, R.totals(R.X)
        gX, gT, ga = gammaln(a + X), gammaln(1 + Tot), gammaln(a)
        for i, j in itertools.combinations(range(len(items)), 2):
            per = ((gammaln(a + X[i] + X[j]) + ga - gX[i] - gX[j]) @ R.M
                   - (gammaln(1 + Tot[i] + Tot[j]) - gT[i] - gT[j]))
            margin = min(np.minimum(per, 0).sum() + STRONG, per.sum())
            p = np.clip(expit(margin), 1e-12, 1 - 1e-12)
            h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
            for e in (items[i], items[j]):
                H[e[:3]] = max(H.get(e[:3], 0.0), h)
    return H


class Model:
    """Unifilar machine: counts C[s, a, o], successor nxt[s, a, o] (-1 unknown)."""

    def __init__(self, C, nxt, start, H=None):
        self.C, self.nxt, self.start, self.H = C, nxt, start, H

    def values(self, arm):
        K, A, O = self.C.shape
        v = np.zeros((K, A))
        if arm in ('novelty', 'combined', 'oracle'):
            v += novelty(self.C, O)
        if arm in ('structure', 'combined'):
            n = self.C.sum(-1, keepdims=True)
            p = (self.C + 1.0 / O) / (n + 1)
            for (s, a, o), h in self.H.items():
                v[s, a] += p[s, a, o] * h / (self.C[s, a, o] + 1)
        return v

    def plan(self, arm, rng):
        """Greedy table pi[h, s]: best action with h steps left of the prefix."""
        K, A, O = self.C.shape
        P = (self.C + 1e-3) / (self.C + 1e-3).sum(-1, keepdims=True)
        v = self.values(arm) + 1e-12 * rng.random((K, A))
        V, pi = np.zeros(K), np.zeros((PREFIX + 1, K), int)
        known = self.nxt >= 0
        for h in range(1, PREFIX + 1):
            Q = v + (P * np.where(known, V[np.maximum(self.nxt, 0)], 0.0)).sum(-1)
            pi[h] = Q.argmax(1)
            V = Q.max(1)
        return pi


def learned_model(trajs, S, env):
    K = int(max(s.max() for s in S)) + 1
    A, O = env.n_act, env.n_obs
    C, votes, first = np.zeros((K, A, O)), collections.defaultdict(collections.Counter), {}
    for (obs, acts, _), s in zip(trajs, S):
        first.setdefault(obs[0], collections.Counter())[int(s[0])] += 1
        for t, a in enumerate(acts):
            if s[t] >= 0:
                C[s[t], a, obs[t + 1]] += 1
                if s[t + 1] >= 0:
                    votes[int(s[t]), a, obs[t + 1]][int(s[t + 1])] += 1
    nxt = -np.ones((K, A, O), int)
    for (s, a, o), v in votes.items():
        nxt[s, a, o] = v.most_common(1)[0][0]
    start = {o: v.most_common(1)[0][0] for o, v in first.items()}
    return Model(C, nxt, start, merge_entropy(trajs, S, O))


def true_model(env):
    """Predictive states of the maze, by breadth-first search from the start."""
    bits = list(itertools.product((0, 1), repeat=env.k))
    start = frozenset((0, b, MT.observe(env, 0, (False,) * env.k)) for b in bits)
    ix, todo, trans = {start: 0}, [start], {}
    while todo:
        B = todo.pop()
        for a in range(env.n_act):
            by = collections.defaultdict(set)
            for loc, b, seen in B:
                l2 = env.move(loc, a)
                by[env.emit(l2, b, None)].add((l2, b, MT.observe(env, l2, seen)))
            for o, z in by.items():
                z = frozenset(z)
                if z not in ix:
                    ix[z] = len(ix)
                    todo.append(z)
                trans[ix[B], a, o] = ix[z]
    nxt = -np.ones((len(ix), env.n_act, env.n_obs), int)
    for e, j in trans.items():
        nxt[e] = j
    o0 = env.emit(0, bits[0], None)
    return nxt, {o0: 0}


def collect(env, model, arm, n, rng):
    """n episodes: planned prefix on the model, uniform tail. model.C is updated as we go."""
    out, A = [], env.n_act
    for i in range(n):
        if i % REPLAN == 0:
            pi = model.plan(arm, rng)
        bits = tuple(int(b) for b in rng.integers(2, size=env.k))
        loc = 0
        obs, acts, true = [env.emit(loc, bits, rng)], [], [(loc, bits)]
        s = model.start.get(obs[0], -1)
        for t in range(T):
            a = int(pi[PREFIX - t, s]) if t < PREFIX and s >= 0 else int(rng.integers(A))
            loc = env.move(loc, a)
            o = env.emit(loc, bits, rng)
            acts.append(a); obs.append(o); true.append((loc, bits))
            if s >= 0:
                model.C[s, a, o] += 1
                s = int(model.nxt[s, a, o])
        ep = M.Episode((obs, acts, true))
        ep.tail = PREFIX
        out.append(ep)
    return out


def score(env, trajs, S, truth):
    own = collections.defaultdict(collections.Counter)
    for (obs, acts, true), st in zip(trajs, S):
        seen = MT.observe(env, 0, (False,) * env.k)
        for t in range(len(acts) + 1):
            loc, bits = true[t]
            seen = MT.observe(env, loc, seen)
            if st[t] >= 0:
                own[MT.key(env, loc, bits, seen)][int(st[t])] += 1
    tot = sum(sum(v.values()) for v in own.values())
    owners = collections.Counter(v.most_common(1)[0][0] for v in own.values())
    K = M.n_states(S)
    return dict(K=K, share=sum(max(v.values()) for v in own.values()) / tot,
                seen=len(own), merged=sum(c > 1 for c in owners.values()),
                exact=bool(K == truth and MT.exact(env, trajs, S)))


def save_state(saved, i, trajs, res, rng):
    tmp = saved.with_suffix('.pkl.tmp')
    tmp.write_bytes(pickle.dumps(dict(i=i, trajs=trajs, res=res, rng=rng.bit_generator.state)))
    tmp.replace(saved)


def run(arm, seed, checks=None):
    checks = checks or CHECKS
    path, saved = OUT / f'{arm}_s{seed}.json', OUT / f'{arm}_s{seed}.pkl'
    if path.exists() and path.stat().st_size > 2:          # '{}' marks a skipped run
        return path.name
    env = M.KMaze(K_SECRETS)
    _, truth, _ = MT.coverage(env, T)
    rng = np.random.default_rng(1000 * seed + 5 + ARMS.index(arm))
    trajs = env.sample(checks[0], T, 1000 * seed + 7)            # the same first batch per seed
    res, first = dict(arm=arm, seed=seed, checks=[]), 0
    if saved.exists():                                     # resume at the last checkpoint
        st = pickle.loads(saved.read_bytes())
        first, trajs, res = st['i'], st['trajs'], st['res']
        rng.bit_generator.state = st['rng']
    if arm == 'oracle':
        nxt, start = true_model(env)
    t0 = time.time()
    for i, n in enumerate(checks):
        if i < first:
            continue
        S, rounds = M.learn_states(trajs, env.n_obs)
        res['checks'].append(dict(score(env, trajs, S, truth), n=n, steps=n * T, rounds=rounds,
                                  time=time.time() - t0))
        (OUT / f'{arm}_s{seed}.tmp').write_text(json.dumps(res))
        if i + 1 == len(checks):
            break
        m = checks[i + 1] - n
        if arm == 'uniform':
            trajs = trajs + env.sample(m, T, 1000 * seed + 11 + i)
            save_state(saved, i + 1, trajs, res, rng)
            continue
        if arm == 'oracle':
            C = np.zeros(nxt.shape)
            model = Model(C, nxt, start)
            for (obs, acts, _) in trajs:                         # true-state counts so far
                s = start[obs[0]]
                for a, o in zip(acts, obs[1:]):
                    C[s, a, o] += 1
                    s = nxt[s, a, o]
        else:
            model = learned_model(trajs, S, env)
        trajs = trajs + collect(env, model, arm, m, rng)
        save_state(saved, i + 1, trajs, res, rng)
    (OUT / f'{arm}_s{seed}.tmp').replace(path)
    return path.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--arms', nargs='*', default=ARMS)
    ap.add_argument('--seeds', type=int, nargs='*', default=list(range(SEEDS)))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.platform == 'win32':                   # no system sleep while this process runs
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    jobs = [(a, s) for s in args.seeds for a in args.arms]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        fs = {ex.submit(run, *j): j for j in jobs}
        for i, f in enumerate(as_completed(fs)):
            try:
                print(f'{i + 1}/{len(jobs)} {f.result()} ({time.time() - t0:.0f}s)', flush=True)
            except Exception as e:
                print(f'FAILED {fs[f]}: {e!r}', flush=True)


if __name__ == '__main__':
    main()
