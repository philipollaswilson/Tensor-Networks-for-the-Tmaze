"""From learned states to an active inference generative model, tested by acting.

Learner: kmaze.learn_states at its defaults, on uniform random episodes (same seeds and 90%
split as rdp_compare.py). It never sees the truth. The truth is used only for the optimum, the
floor, the oracle control and the purity column of show.

Forms. Both start from an edge tensor E[a, o, s, s'], counts of p(o, s' | s, a):
  pred    predictive states: s' = f(s, a, o) by majority vote, episodes re-routed through f
  vertex  kmaze.unmix + kmaze.generator: E = n(s, a) G[a, o, s, s'], visits to mixture states
          are not counted (generator reads only the vertices' rows)
  oracle  control, truth only: the true hidden state of every step, same code from here on.
          What the planner loses with perfect states

A = p(o | s). Learned machines are edge-emitting: reward arrives on the edge, and two edges with
different rewards can enter one state, so p(o | s) alone would mix them. Moore lift: a state z is
(s, tag), tag = a set of observations entering s. Each observation starts as its own tag; two
merge when p(o | incoming (s, a)) has the same ratio on every incoming edge (Beta-binomial Bayes
factor, Jeffreys prior, ties merge as in kmaze.bmr_rows). Then o is independent of the past given
z, A is stochastic only where the data say so, and A B reproduces E.

Dirichlet. a[z, o], b[a, s, z'], d[z]: the counts, tied over the tags of s. Perks prior, mass 1:
over z's tag for a, over all Z + 1 outcomes for b and d. Posterior means A, B, D. The mass of
outcomes never seen from (s, a) is lumped into one unknown state (index Z), which emits an
unknown symbol (index O), is absorbing while planning and has C = 0: it is what the model cannot
say, the same device as the unknown outcome of kmaze_plan.py. In the filter that mass is spread by
state occupancy, so a surprise re-syncs on the real observation. Leaves: learn_states leaves some
episode ends at -1, so an edge can have a known observation and no known successor. It enters leaf
K + o, one per observation, whose actions were never tried. Leaves are not counted in K.

Expected free energy, per action, from belief q, with q' = q B[a], p(o) = q' A:
  G(a) = - sum_o p(o) C(o)                                          risk (expected log preference)
         - [H(p(o)) - sum_z q'(z) H(A[z])]                          state information gain
         - [sum_z q(z) I(b[a, z]) + sum_z q'(z) I(a[z])]            parameter novelty (option)
         + sum_o p(o) sum_a' pi(a' | q'_o) G(a' | q'_o)             sophisticated recursion
  C(o) = LAMBDA * r(o) nats, with r the environment's reward function (the only task information
  the agent gets); ln P(o) = C(o) + const. Epistemic terms are mutual informations, in nats, added
  with weight 1 (chain rule: I(s, theta; o) = I(s; o) + I(theta; o | s)). I of a Dirichlet row is
  exact: H[mean] - E H = H[m] - psi(a0 + 1) + sum_k m_k psi(a_k + 1). Parameter novelty is
  conditioned on s, and the model stays frozen during the test.

Constants, fixed before any return was seen: LAMBDA = 4 nats per unit reward (one value for all
environments; 1 and 16 are run as sensitivity), GAMMA = 16 (policy precision, inside the tree and
for the sampled action), DEPTH = 8 (capped by the steps left; the agent knows H), EPS = 1e-3
(outcomes and beliefs below it are pruned).
Conditions: risk (no epistemic term: expected utility), efe (+ state), efe+nov (+ parameters).

Results (results/gen_model/table.txt). Planner check: on true states with 100k episodes it reaches
the optimum in all six environments. n = 10000: Corridor, Cheese, Mini-hall at the optimum, Cookie
0.96 of 1. Failures: T-maze 1.5 of 4 at every n (the cues merge at t = 0, so the junction is a
coin flip); Mini-hall n = 1000 0.49 of 2.35 ((B, N) and (C, S) merged); Tiger no model (budget
1800 s at n = 300, 1000, 10000; at n = 100, 113 states from 900 steps and rollouts cost 2.5 s per
episode: unknown mass 45%, wide beliefs, few memo hits, so eval is not run on it by default).
State information gain breaks ties (Cookie seed 1, 0.63 -> 1.00) and can trap the agent: a state
that forgot a miss re-offers the same information for ever (Cookie seed 2, 0.99 -> 0.87).

Run:  python src/gen_model.py learn [--workers 3] [--budget 1800] [--only tiger] [--ns 1000]
      python src/gen_model.py eval [--workers 3]
      python src/gen_model.py table
      python src/gen_model.py show tmaze [--n 10000]
"""
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import pickle
import queue
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from fractions import Fraction as F

for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')

import numpy as np  # noqa: E402
from scipy.special import betaln, digamma  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import kmaze as M  # noqa: E402
import rdp_envs as E  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / 'results' / 'gen_model'
NS = [1000, 10000]
SEEDS = 3
TEST = 2000
LAMBDA, GAMMA, DEPTH, EPS = 4.0, 16.0, 8, 1e-3
CONDS = {'risk': (0, 0), 'efe': (1, 0), 'efe+nov': (1, 1)}


# ------------------------------------------------------------------- data and learner
def dataset(name, n, seed):
    """Same draw and same 90% as rdp_compare.cell. The 10% is unused: nothing here is chosen."""
    env = E.ENVS[name]()
    T = E.PAPER_H[name]
    data = env.sample(n, T, 7919 * seed + n)
    return env, T, data[max(10, n // 10):]


def fit_one(q, trajs, O):
    try:
        q.put(M.learn_states(trajs, O))
    except Exception as e:                        # a failure is a result, not a crash
        q.put(e)


def learn_cell(name, n, seed, budget):
    """Learn in a child process under a budget. Saves S, or the failure."""
    path = OUT / f'S_{name}_n{n}_s{seed}.pkl'
    if path.exists():
        return path.name
    env, _, trajs = dataset(name, n, seed)
    ctx = mp.get_context('spawn')
    q, t0 = ctx.Queue(), time.time()
    p = ctx.Process(target=fit_one, args=(q, trajs, env.n_obs))
    p.start()
    try:
        out = q.get(timeout=budget)
    except queue.Empty:
        out = TimeoutError(f'budget {budget}s')
    p.terminate(); p.join()
    res = dict(error=repr(out)) if isinstance(out, Exception) else dict(S=out[0], rounds=out[1])
    res['time'] = time.time() - t0
    path.write_bytes(pickle.dumps(res))
    return f'{path.name} {res["time"]:.0f}s ' + (res['error'] if 'error' in res else f'{M.n_states(res["S"])} states')


# ------------------------------------------------------------------- edge tensors
def successors(trajs, S):
    """f[s, a, o] and f['start', o0]: majority vote, as rdp_compare.ours."""
    votes = defaultdict(Counter)
    for (obs, acts, _), s in zip(trajs, S):
        if s[0] >= 0:
            votes['start', obs[0]][int(s[0])] += 1
        for t, a in enumerate(acts):
            if s[t] >= 0 and s[t + 1] >= 0:
                votes[int(s[t]), a, obs[t + 1]][int(s[t + 1])] += 1
    return {k: v.most_common(1)[0][0] for k, v in votes.items()}


def pred_edges(trajs, S):
    """Counts E[a, o, s, s'], E0[o, s], lost[a, s] on the routed episodes: the counts the agent's
    own walk through f would see. An edge with no known successor keeps its observation: it goes
    to the leaf K + o. (learn_states leaves episode ends at -1, and rewards can arrive only there:
    T-maze's +4 is always the last step.)"""
    f = successors(trajs, S)
    K = int(max(s.max() for s in S)) + 1
    Ec, E0, lost = Counter(), Counter(), Counter()
    for obs, acts, _ in trajs:
        s = f.get(('start', obs[0]))
        if s is None:
            continue
        E0[obs[0], s] += 1
        for a, o in zip(acts, obs[1:]):
            n = f.get((s, a, o))
            if n is None:
                Ec[a, o, s, K + o] += 1
                break
            Ec[a, o, s, n] += 1
            s = n
    return Ec, E0, lost, K


def vertex_edges(trajs, S, A, O):
    """kmaze.unmix and kmaze.generator. machine cannot read -1 (Rows has no such item), so each
    episode is cut at its first -1 for this step; edges with no successor go to leaves, as in
    pred_edges."""
    tr, Sc = [], []
    for (obs, acts, true), s in zip(trajs, S):
        cut = int(np.argmax(s < 0)) if (s < 0).any() else len(s)
        if cut:
            tr.append((obs[:cut], acts[:cut - 1], true[:cut]))
            Sc.append(s[:cut])
    m = M.machine(tr, Sc, A, O)
    V, weights, report = M.unmix(m)
    gen = M.generator(m, V, weights, A, O)
    Ec, E0, lost = Counter(), Counter(), Counter()
    n = m['C1'].sum(-1)
    for a, o, i, j in zip(*np.nonzero(gen['G'] > 1e-12)):
        Ec[int(a), int(o), int(i), int(j)] = n[V[i], a] * gen['G'][a, o, i, j]
    for a in range(A):
        for i, k in enumerate(V):
            lost[a, i] = max(n[k, a] - n[k, a] * gen['G'][a, :, i].sum(), 0.0)
    for (obs, _, _), s in zip(tr, Sc):
        for j in np.flatnonzero(gen['W'][s[0]] > 1e-12):
            E0[obs[0], int(j)] += gen['W'][s[0], j]
    ix = {k: i for i, k in enumerate(V)}
    for (obs, acts, _), s in zip(trajs, S):
        for t, a in enumerate(acts):
            k = int(s[t])
            if k < 0:
                break
            if k in ix and (s[t + 1] < 0 or (k, a, obs[t + 1]) not in m['f']):
                Ec[a, obs[t + 1], ix[k], len(V) + obs[t + 1]] += 1
    return Ec, E0, lost, len(V), dict(V=V, weights=weights, report=report, m=m, gen=gen)


def oracle_edges(trajs):
    """Control: the edge tensor of the true hidden states. Truth, so evaluation only."""
    ids, Ec, E0, lost = {}, Counter(), Counter(), Counter()
    for obs, acts, true in trajs:
        s = [ids.setdefault(x, len(ids)) for x in true]
        E0[obs[0], s[0]] += 1
        for t, a in enumerate(acts):
            Ec[a, obs[t + 1], s[t], s[t + 1]] += 1
    return Ec, E0, lost, len(ids)


# ------------------------------------------------------------------- A, B, D
def tags(rows):
    """Group the observations entering one state. rows = {incoming edge: {o: count}}. Pairs of
    groups merge, best first, while one shared ratio has at least the evidence of one per edge."""
    cols = sorted({o for r in rows.values() for o in r})
    T = np.array([[r.get(o, 0.0) for o in cols] for r in rows.values()])
    G = [[i] for i in range(len(cols))]
    while len(G) > 1:
        best, pair = -1e-9, None
        for i in range(len(G)):
            for j in range(i):
                x, y = T[:, G[i]].sum(1), T[:, G[j]].sum(1)
                dF = (betaln(.5 + x.sum(), .5 + y.sum()) - betaln(.5, .5)
                      - (betaln(.5 + x, .5 + y) - betaln(.5, .5)).sum())
                if dF >= best:
                    best, pair = dF, (i, j)
        if pair is None:
            break
        G[pair[1]] += G.pop(pair[0])
    return [tuple(sorted(cols[i] for i in g)) for g in G]


def dirichlet_info(c):
    """I(theta; outcome) of Dirichlet rows c (last axis), exact, in nats."""
    a0 = c.sum(-1, keepdims=True)
    m = c / a0
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.where(m > 0, m * np.log(m), 0.0).sum(-1)
    return h - digamma(a0[..., 0] + 1) + (m * digamma(c + 1)).sum(-1)


def entropy(p):
    return -np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0.0).sum(-1)


class Model:
    """Discrete POMDP on lifted states: A[z, o], B[a, z, z'], D[z], index Z = unknown, symbol O =
    unknown outcome. a, b, d: Dirichlet counts. base[z] = s, tag[z] = observations of z."""

    def __init__(self, Ec, E0, lost, K, A_, O):
        rows = defaultdict(lambda: defaultdict(Counter))
        for (a, o, i, j), c in Ec.items():
            if c > 0:
                rows[j][a, i][o] += c
        for (o, j), c in E0.items():
            rows[j]['start'][o] += c
        self.base, self.tag, zof = [], [], {}
        for j in sorted(rows):
            for g in tags(rows[j]):
                for o in g:
                    zof[j, o] = len(self.base)
                self.base.append(j); self.tag.append(g)
        Z = len(self.base)
        self.K, self.Z, self.n_act, self.O = K, Z, A_, O
        a, b, d = np.zeros((Z, O)), np.zeros((A_, K, Z + 1)), np.zeros(Z + 1)
        for (ac, o, i, j), c in Ec.items():
            if c > 0:
                a[zof[j, o], o] += c; b[ac, i, zof[j, o]] += c
        for (o, j), c in E0.items():
            a[zof[j, o], o] += c; d[zof[j, o]] += c
        for (ac, i), c in lost.items():
            b[ac, i, Z] += c
        self.a, self.b, self.d = a, b, d
        al = 1 / (Z + 1)
        base = np.array(self.base)
        Bs = (b > 0) * (b + al) / (b.sum(-1, keepdims=True) + 1)
        Bs[..., Z] += 1 - Bs.sum(-1)
        self.B = np.zeros((A_, Z + 1, Z + 1))
        self.B[:, :Z] = Bs[:, base]
        self.B[:, Z, Z] = 1
        self.Nb = np.zeros((A_, Z + 1))
        self.Nb[:, :Z] = dirichlet_info(b + al)[:, base]
        self.A = np.zeros((Z + 1, O + 1))
        self.A[Z, O] = 1
        self.Na = np.zeros(Z + 1)
        for z, g in enumerate(self.tag):
            c = a[z, list(g)] + 1 / len(g)
            self.A[z, list(g)] = c / c.sum()
            self.Na[z] = dirichlet_info(c)
        self.D = (d > 0) * (d + al) / (d.sum() + 1)
        self.D[Z] += 1 - self.D.sum()
        self.HA = entropy(self.A)
        self.occ = a.sum(1) / a.sum()
        self.N = a.sum()

    def known(self, pred):
        """Spread the unknown mass by occupancy (filter only)."""
        return pred[:self.Z] + pred[self.Z] * self.occ

    def filt(self, pred, o):
        p = self.known(pred)
        post = p * self.A[:self.Z, o]
        post = post if post.sum() > 0 else p      # never seen anywhere: keep the prediction
        post = np.where(post / post.sum() >= EPS, post, 0)
        return np.append(post / post.sum(), 0.0)

    def start(self, o):
        return self.filt(self.D, o)

    def step(self, q, a, o):
        return self.filt(q @ self.B[a], o)

    def logloss(self, trajs):
        """Held-out nats per step, filtered; a pseudo-count of 1 / O keeps the log finite."""
        tot, n = 0.0, 0
        for obs, acts, _ in trajs:
            q = self.start(obs[0])
            for a, o in zip(acts, obs[1:]):
                pred = q @ self.B[a]
                p = self.known(pred) @ self.A[:self.Z, o]
                tot -= np.log((p * self.N + 1 / self.O) / (self.N + 1))
                n += 1
                q = self.filt(pred, o)
        return tot / n


# ------------------------------------------------------------------- planning
class Planner:
    """Depth-limited sophisticated EFE, memoised on (belief, depth). state, novelty are switches
    (0 or 1), not weights."""

    def __init__(self, m, C, state=1, novelty=0):
        self.m, self.state, self.novelty = m, state, novelty
        self.C = np.append(C, 0.0)                # unknown outcome: no preference
        self.memo = {}

    def pi(self, G):
        p = np.exp(-GAMMA * (G - G.min()))
        return p / p.sum()

    def G(self, q, d):
        key = (np.round(q, 4).tobytes(), d)
        if key in self.memo:
            return self.memo[key]
        m = self.m
        out = np.zeros(m.n_act)
        for a in range(m.n_act):
            pred = q @ m.B[a]
            po = pred @ m.A
            g = -po @ self.C
            if self.state:
                g -= entropy(po) - pred @ m.HA
            if self.novelty:
                g -= q @ m.Nb[a] + pred @ m.Na
            if d > 1:
                keep = np.flatnonzero(po >= EPS)
                w = po[keep] / po[keep].sum()
                for o, p in zip(keep, w):
                    q2 = pred * m.A[:, o] / po[o]
                    q2 = np.where(q2 >= EPS, q2, 0)
                    Gn = self.G(q2 / q2.sum(), d - 1)
                    g += p * (self.pi(Gn) @ Gn)
            out[a] = g
        self.memo[key] = out
        return out

    def act(self, q, d, rng):
        return int(rng.choice(self.m.n_act, p=self.pi(self.G(q, d))))


def rollout(env, act, H, n, seed):
    """Mean return per episode (and its per-episode sd) in the true environment; act(ctx, o, t,
    rng) sees only the action it took and the observation."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for e in range(n):
        s, o = env.draw(None, None, rng)
        ctx = act(None, None, o, 0, rng)
        for t in range(H):
            s, o = env.draw(s, ctx['a'], rng)
            out[e] += env.reward(o)
            if t < H - 1:
                ctx = act(ctx, ctx['a'], o, t + 1, rng)
    return float(out.mean()), float(out.std())


def agent(m, plan, H):
    def act(ctx, a, o, t, rng):
        q = m.start(o) if ctx is None else m.step(ctx['q'], a, o)
        return dict(q=q, a=plan.act(q, min(DEPTH, H - t), rng))
    return act


def random_agent(A):
    return lambda ctx, a, o, t, rng: dict(a=int(rng.integers(A)))


def optimum(env, H):
    """Exact finite-horizon value on the reachable beliefs of the true model, in Fractions."""
    memo = {}

    def V(b, t):
        if t == 0:
            return F(0)
        key = (frozenset(b.items()), t)
        if key not in memo:
            by = defaultdict(F)
            for (a, sym), (p, nb) in env.children(b).items():
                by[a] += p * (env.reward(sym) + V(nb, t - 1))
            memo[key] = max(by.values())
        return memo[key]
    return float(sum(p * (env.reward(sym) + V(b, H)) for (_, sym), (p, b) in env.children(None).items()))


# ------------------------------------------------------------------- evaluation
def eval_cell(name, n, seed):
    path = OUT / f'R_{name}_n{n}_s{seed}.json'
    if path.exists():
        return path.name
    env, H, trajs = dataset(name, n, seed)
    A, O = env.n_act, env.n_obs
    Cr = np.array([float(env.reward(o)) for o in range(O)])
    Xte = env.sample(TEST, H, 104729 + seed)
    res = dict(env=name, n=n, seed=seed, H=H, optimum=optimum(env, H), floor=env.belief_logloss(Xte),
               random=rollout(env, random_agent(A), H, TEST, 104729 + seed)[0],
               n_pred=env.n_pred, forms={})
    fit = pickle.loads((OUT / f'S_{name}_n{n}_s{seed}.pkl').read_bytes())
    res['learn_time'] = fit['time']
    forms = {'oracle': oracle_edges(trajs)}
    if 'S' in fit:
        res['rounds'] = fit['rounds']
        forms['pred'] = pred_edges(trajs, fit['S'])
        try:
            forms['vertex'] = vertex_edges(trajs, fit['S'], A, O)[:4]
        except Exception as e:
            res['forms']['vertex'] = dict(error=repr(e))
    else:
        res['error'] = fit['error']
    for form, (Ec, E0, lost, K) in forms.items():
        t0 = time.time()
        m = Model(Ec, E0, lost, K + O, A, O)
        r = dict(K=K, Z=m.Z, n_edges=len(Ec), gap=m.logloss(Xte) - res['floor'],
                 stochastic_A=int(sum(len(g) > 1 for g in m.tag)), returns={}, sd={}, plans={})
        for c, (st, nov) in CONDS.items():
            p = Planner(m, LAMBDA * Cr, st, nov)
            r['returns'][c], r['sd'][c] = rollout(env, agent(m, p, H), H, TEST, 104729 + seed)
            r['plans'][c] = len(p.memo)
        for lam in (1.0, 16.0):
            p = Planner(m, lam * Cr, 1, 0)
            r['returns'][f'efe l={lam:g}'] = rollout(env, agent(m, p, H), H, TEST, 104729 + seed)[0]
        r['time'] = time.time() - t0
        res['forms'][form] = r
    path.write_text(json.dumps(res))
    return path.name


def table():
    rows = defaultdict(list)
    for p in sorted(OUT.glob('R_*.json')):
        r = json.loads(p.read_text())
        rows[r['env'], r['n']].append(r)
    keys = ['risk', 'efe', 'efe+nov', 'efe l=1', 'efe l=16']
    print('env n | optimum random | form K Z gap | ' + ' '.join(keys) + ' | per seed efe')
    for (env, n), rs in sorted(rows.items()):
        print(f'{env} {n}: opt {rs[0]["optimum"]:.3f} random {np.mean([r["random"] for r in rs]):.3f} '
              f'n_pred {rs[0]["n_pred"]} seeds {len(rs)} learn {[round(r["learn_time"]) for r in rs]}'
              + ''.join(f' ERR {r["error"]}' for r in rs if 'error' in r))
        for f in ('pred', 'vertex', 'oracle'):
            ok = [r['forms'][f] for r in rs if f in r['forms'] and 'returns' in r['forms'][f]]
            for r in rs:
                if 'error' in r['forms'].get(f, {}):
                    print(f'   {f}: {r["forms"][f]["error"]}')
            if ok:
                print(f'   {f:6s} K {[x["K"] for x in ok]} Z {[x["Z"] for x in ok]} '
                      f'gap {np.mean([x["gap"] for x in ok]):.4f} | '
                      + ' '.join(f'{np.mean([x["returns"][k] for x in ok]):.3f}' for k in keys)
                      + f' | {[round(x["returns"]["efe"], 2) for x in ok]}')


# ------------------------------------------------------------------- readable states
NAMES = {'tmaze': (lambda c: ('cue:goal-S', 'cue:goal-N', 'corridor', 'junction', 'end')[c], 'NSEW'),
         'cookie': (lambda c: ('white', 'red', 'blue', 'green', 'blue+cookie', 'green+cookie')[c],
                    ('left', 'right', 'up', 'down', 'press', 'eat'))}


def show(name, n=10000, seed=0, top=3):
    """Every lifted predictive state: what enters it, what it predicts, which histories map to it.
    The last column (truth) is for reading only and was not used."""
    fit = pickle.loads((OUT / f'S_{name}_n{n}_s{seed}.pkl').read_bytes())
    env, _, trajs = dataset(name, n, seed)
    A, O = env.n_act, env.n_obs
    Ec, E0, lost, K = pred_edges(trajs, fit['S'])
    m = Model(Ec, E0, lost, K + O, A, O)
    code, acts = NAMES.get(name, (lambda c: f'c{c}', [str(a) for a in range(A)]))
    nr = len(env.rewards)

    def ob(o):
        r = env.reward(o)
        return code(o // nr) + (f' r={r:+g}' if r else '')

    f = successors(trajs, fit['S'])
    zof = {(b, o): z for z, (b, g) in enumerate(zip(m.base, m.tag)) for o in g}
    hist, truth = defaultdict(Counter), defaultdict(Counter)
    for obs, ac, true in trajs:
        s = f.get(('start', obs[0]))
        h = ob(obs[0])
        if s is None:
            continue
        hist[zof[s, obs[0]]][h] += 1; truth[zof[s, obs[0]]][true[0]] += 1
        for t, (a, o) in enumerate(zip(ac, obs[1:])):
            s = f.get((s, a, o))
            if s is None:
                break
            h += f' -{acts[a]}> {ob(o)}'
            hist[zof[s, o]][h] += 1; truth[zof[s, o]][true[t + 1]] += 1
    print(f'{name} n={n} seed {seed}: {K} predictive states, {m.Z} lifted states, truth n_pred '
          f'{env.n_pred}. D on: ' + ', '.join(f'z{z} {m.D[z]:.2f}' for z in np.flatnonzero(m.D[:-1] > .01)))
    for z in np.argsort([-sum(hist[i].values()) for i in range(m.Z)]):
        tot = sum(hist[z].values())
        emit = ', '.join(f'{ob(o)} {m.A[z, o]:.2f}' for o in m.tag[z])
        print(f'\nz{z} (s{m.base[z]}), {tot} visits. A: {emit}')
        for a in range(A):
            nz = m.b[a, m.base[z]].sum()
            if nz == 0:
                print(f'   {acts[a]:>6}: never tried here')
                continue
            top_o = Counter()
            for g, zz in zip(m.tag, range(m.Z)):
                pz = m.B[a, z, zz]
                if pz > .01:
                    top_o[f'z{zz} [{", ".join(ob(o) for o in g)}]'] = pz
            print(f'   {acts[a]:>6}: ' + ', '.join(f'{k} {v:.2f}' for k, v in top_o.most_common(4))
                  + f'  (n={nz:.0f})')
        for h, c in hist[z].most_common(top):
            print(f'   history x{c}: {h}')
        st = truth[z].most_common(2)
        print('   truth, not used: ' + ', '.join(f'{s} {c / tot:.0%}' for s, c in st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['learn', 'eval', 'table', 'show'])
    ap.add_argument('env', nargs='?', default='tmaze')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--budget', type=int, default=1800)
    ap.add_argument('--only', default=None)
    ap.add_argument('--ns', type=int, nargs='*', default=NS)
    ap.add_argument('--n', type=int, default=10000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding='utf-8')
    if args.cmd == 'table':
        return table()
    if args.cmd == 'show':
        return show(args.env, args.n)
    envs = [args.only] if args.only else list(E.ENVS)
    jobs = sorted(((e, n, s) for n in args.ns for e in envs for s in range(SEEDS)),
                  key=lambda j: (j[0] == 'tiger', j[1]))      # Tiger's fits may run out of budget
    t0 = time.time()
    if args.cmd == 'learn':
        ex = ThreadPoolExecutor(args.workers)     # threads wait; each fit is its own process
        fs = {ex.submit(learn_cell, *j, args.budget): j for j in jobs}
    else:
        jobs = [j for j in jobs if (OUT / f'S_{j[0]}_n{j[1]}_s{j[2]}.pkl').exists()]
        ex = ProcessPoolExecutor(args.workers)
        fs = {ex.submit(eval_cell, *j): j for j in jobs}
    for i, f in enumerate(as_completed(fs)):
        try:
            print(f'{i + 1}/{len(jobs)} {f.result()} ({time.time() - t0:.0f}s)', flush=True)
        except Exception as e:
            print(f'FAILED {fs[f]}: {e!r}', flush=True)


if __name__ == '__main__':
    main()
