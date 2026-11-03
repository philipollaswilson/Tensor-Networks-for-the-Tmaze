"""k-secret maze: structure learning where memory, factorization and unmixing all matter.

C = centre; for each secret i: a cue room Q_i (shows bit i) and two arms L_i, R_i (cheese
in L_i if bit i = 0, in R_i if bit i = 1). Layout 'hub': go_i (C -> Q_i), left, right
(Q_i -> L_i / R_i), back (arm -> Q_i, Q_i -> C). Layout 'star' (T-maze-like): from C, one
action per room; back returns to C. Anything else stays put.
Hidden state = location (3k + 1) x k bits. The bits never change; location follows actions
regardless of the bits, so the true model is a product of k + 1 factors.

Learner (offline, from random-policy episodes; nothing about the maze is given):
  states  iterative refinement: the state after a step is g(previous state, action, obs).
          Route all episodes through the current states, pool for every edge (s, a, o) the
          counts of the next L steps (one Dirichlet modality per action sequence, outcome =
          the observation sequence; Perks prior over the outcomes observed for that action
          sequence), merge edges by BMR; merged edges are the new states.
          Repeat until stable. Knowledge travels one step further per round; unifilar at
          the fixed point. L = 3: a difference must show within 3 steps to count.
          (Red-blue merging on the history tree, as ALERGIA, failed here: each specific
          history that knows two secrets is rare, so no node holds enough evidence.)
  unmix   drop states whose L-step prediction rows are mixtures of others (Bayes factor)
  gen     generator over the remaining (vertex) states, D = mixture weights of the start
  factor  SP partitions via pair closures (factor_search.py); k-way products
Run:  python src/kmaze.py
"""
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import nnls
from scipy.special import gammaln

import factor_search as FS


# ------------------------------------------------------------------- environment
class KMaze:
    """layout 'hub': arms hang off their cue room (no memory needed to act).
    layout 'star': every room hangs off the centre, as in the T-maze; to use a secret it
    must be remembered from the cue room, through the centre, to the arm."""

    def __init__(self, k, noise=0.0, layout='hub'):
        self.k, self.noise, self.layout = k, noise, layout
        self.n_loc = 3 * k + 1
        # hub: go_1..go_k, left, right, back; star: cue_i, left_i, right_i, back
        self.n_act = k + 3 if layout == 'hub' else 3 * k + 1
        self.n_obs = 2 * self.n_loc               # (location, one bit of content)

    def move(self, loc, a):
        k = self.k
        if self.layout == 'star':
            if loc == 0:
                return a + 1 if a < 3 * k else 0     # cue_i -> Q_i, left_i -> L_i, ...
            return 0 if a == 3 * k else loc          # back -> C; anything else stays
        if loc == 0:
            return 1 + a if a < k else 0
        if loc <= k:                              # cue room Q_i
            if a == k:
                return k + loc                    # L_i
            if a == k + 1:
                return 2 * k + loc                # R_i
            return 0 if a == k + 2 else loc
        if a == k + 2:                            # arm -> its cue room
            return (loc - 1) % k + 1
        return loc

    def emit(self, loc, bits, rng):
        k = self.k
        if loc == 0:
            v = 0
        elif loc <= k:
            v = bits[loc - 1]
            if self.noise and rng.random() < self.noise:
                v = 1 - v
        else:
            i = (loc - 1) % k
            v = int((loc <= 2 * k) == (bits[i] == 0))   # 1 = cheese
        return 2 * loc + v

    def sample(self, n, T, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            bits = tuple(int(b) for b in rng.integers(2, size=self.k))
            loc = 0
            obs, acts, true = [self.emit(loc, bits, rng)], [], [(loc, bits)]
            for _ in range(T):
                a = int(rng.integers(self.n_act))
                loc = self.move(loc, a)
                acts.append(a); obs.append(self.emit(loc, bits, rng)); true.append((loc, bits))
            out.append((obs, acts, true))
        return out

    def loc_name(self, loc):
        k = self.k
        return 'C' if loc == 0 else f'Q{loc}' if loc <= k else \
            f'L{loc - k}' if loc <= 2 * k else f'R{loc - 2 * k}'


class Corridor:
    """Long-corridor T-maze (memory length D): the start cell shows the secret; then D
    corridor cells that all look alike, a junction, and two arms (cheese in L if the
    secret is 0, in R if 1). Actions: forward, back, left, right (left / right only at the
    junction; back from an arm returns to the junction; anything else stays put).
    Hidden state = position (D + 4) x secret (2)."""

    def __init__(self, D):
        self.D, self.k = D, 1
        self.n_loc = D + 4
        self.n_act = 4
        self.n_obs = 10                           # 2 x (cue, corridor, junction, L, R)

    def move(self, loc, a):
        D = self.D
        J = D + 1
        if loc <= D:
            return loc + 1 if a == 0 else (loc - 1 if a == 1 and loc > 0 else loc)
        if loc == J:
            return {1: D, 2: J + 1, 3: J + 2}.get(a, J)
        return J if a == 1 else loc               # arms

    def emit(self, loc, bits, rng):
        D = self.D
        if loc == 0:
            return bits[0]
        if loc <= D:
            return 2
        if loc == D + 1:
            return 4
        cheese = (loc == D + 2) == (bits[0] == 0)
        return (6 if loc == D + 2 else 8) + int(cheese)

    def reward(self, o):
        if o < 6:
            return 0.0
        return 1.0 if o % 2 else -4.0

    sample = KMaze.sample

    def loc_name(self, loc):
        D = self.D
        return 'S' if loc == 0 else f'c{loc}' if loc <= D else 'J' if loc == D + 1 else \
            'L' if loc == D + 2 else 'R'


# ------------------------------------------------------------------- L-step rows
class Rows:
    """Counts of the next L steps for a set of items: columns are (action sequence,
    observation sequence); each action sequence is one Dirichlet modality whose Perks
    prior spreads mass 1 over all O^L observation sequences."""

    def __init__(self, items, windows, O, L):
        self.items = items
        cols, mods = {}, {}
        for _, key in windows:
            if key not in cols:
                cols[key] = len(cols)
                mods.setdefault(key[0], len(mods))
        ix = {e: i for i, e in enumerate(items)}
        X = np.zeros((len(items), len(cols)))
        for e, key in windows:
            X[ix[e], cols[key]] += 1
        self.X = X
        self.mod = np.array([mods[key[0]] for key in cols])
        M = np.zeros((len(cols), len(mods))); M[np.arange(len(cols)), self.mod] = 1
        self.M = M
        # Perks prior over the outcomes seen for that action sequence (mass 1 per modality);
        # spreading it over all O^L sequences makes every fresh row cost ~L ln O nats
        self.alpha = 1.0 / M.sum(0)[self.mod]

    def totals(self, X):
        return X @ self.M


def bmr_rows(R, strong=3.0):
    """Greedy BMR over items. Per modality m, dF_m(i, j) = ln p(one row) - ln p(two rows).
    Merging everywhere is weighed against sharing only where the rows agree, so a pair is
    vetoed when the modalities that disagree hold strong evidence against sharing:
    sum_m min(dF_m, 0) < -3 (Bayes factor 20). Allowed pairs are ranked by sum_m dF_m."""
    X = R.X.copy()
    T = R.totals(X)
    a = R.alpha
    gX, gT = gammaln(a + X), gammaln(1 + T)
    ga, g1 = gammaln(a), gammaln(1.0)

    def row(i):
        per = ((gammaln(a + X[i] + X) + ga - gX[i] - gX) @ R.M
               - (gammaln(1 + T[i] + T) + g1 - gT[i] - gT))
        return np.where(np.minimum(per, 0).sum(1) < -strong, -np.inf, per.sum(1))

    n = len(X)
    lab = np.arange(n)
    alive = np.ones(n, bool)
    D = np.array([row(i) for i in range(n)])
    np.fill_diagonal(D, -np.inf)
    while alive.sum() > 1:
        Mx = np.where(alive[:, None] & alive[None], D, -np.inf)
        i, j = np.unravel_index(np.argmax(Mx), Mx.shape)
        if Mx[i, j] < -1e-9:     # ties merge: equal evidence, fewer states
            break
        X[i] += X[j]; T[i] += T[j]; alive[j] = False; lab[lab == j] = i
        gX[i], gT[i] = gammaln(a + X[i]), gammaln(1 + T[i])
        D[i] = D[:, i] = row(i); D[i, i] = -np.inf
    _, lab = np.unique(lab, return_inverse=True)
    return lab


# ------------------------------------------------------------------- state learning
EVENTS = 3   # event windows: the next 1..EVENTS observation changes (0 = off)


class Episode(tuple):
    """(obs, acts, true) with tail = the step from which actions are uniform random. A planned
    prefix keeps one-step counts, whose law is p(o | s, a) under any policy; statistics of
    longer futures are only valid from the tail on, since they marginalise the policy."""
    tail = 0


def tail_of(ep):
    return getattr(ep, 'tail', 0)


def future_windows(items, obs, acts, L, nxt=None, tail=0):
    """(item at t, (actions t..t+d-1, observations t+1..t+d)) for d = 1..L: short action
    sequences pool the samples that longer ones split (the evidence across depths overlaps,
    which the veto and the ranking both see). With nxt (the current state at every step),
    also one modality per action whose outcome is (observation, next state): differences
    the next state already carries then show after one step, with fully pooled counts.
    Event windows: the first d observations after t that differ from the one before them
    (actions and run lengths dropped), counted only if the d-th change falls inside the
    episode. The policy is random and history-independent, so this is a statistic of the
    future like any other; it sees a difference however many look-alike steps come first."""
    out = []
    changes = [u for u in range(1, len(obs)) if obs[u] != obs[u - 1]]
    for t in range(len(acts)):
        for d in range(1, (min(L, len(acts) - t) if t >= tail else 1) + 1):
            out.append((items[t], (tuple(acts[t:t + d]), tuple(obs[t + 1:t + d + 1]))))
        if nxt is not None and nxt[t + 1] >= 0:
            out.append((items[t], (('next', acts[t]), (obs[t + 1], int(nxt[t + 1])))))
        if t < tail:
            continue
        ev = [obs[u] for u in changes if u > t][:EVENTS]
        for d in range(1, len(ev) + 1):
            out.append((items[t], (('event', d), tuple(ev[:d]))))
    return out


def split_round(trajs, S, O, L):
    """One split step: the edges (previous state, a, o) entering each state are grouped by
    BMR on their futures (with the next state as a modality); each group becomes a state.
    Edges seen only at episode ends have no future and are marked -1."""
    windows, edge_of = [], []
    for ep, s in zip(trajs, S):
        obs, acts, _ = ep
        ed = [('start', int(s[0]))] + [(int(s[t - 1]), acts[t - 1], obs[t], int(s[t]))
                                       for t in range(1, len(acts) + 1)]
        edge_of.append(ed)
        windows += future_windows(ed, obs, acts, L, nxt=s, tail=tail_of(ep))
    by_target = defaultdict(list)
    for e, w in windows:
        by_target[e[-1]].append((e, w))
    lab = {}
    for tgt, wins in by_target.items():
        items = list(dict.fromkeys(e for e, _ in wins))
        g = bmr_rows(Rows(items, wins, O, L)) if len(items) > 1 else np.zeros(1, int)
        for e, x in zip(items, g):
            lab[e] = (tgt, int(x))
    keys = {}
    return [np.array([keys.setdefault(lab[e], len(keys)) if e in lab else -1 for e in ed])
            for ed in edge_of]


def n_states(S):
    return len(set(np.concatenate(S)) - {-1})


def relabel(S):
    """Consecutive state numbers, keeping -1 (end point with no known successor)."""
    ids = np.unique(np.concatenate(S))
    ids = ids[ids >= 0]
    new = -np.ones(int(ids.max()) + 2, int)
    new[ids] = np.arange(len(ids))
    return [new[s] for s in S]


def partition_key(S):
    """Hash of the partition of visited steps. Rounds stop when a partition repeats, either
    at once (settled) or later (a cycle). An equal count is not enough: a false state can
    survive at a constant count while its edges are repaired one layer per round."""
    a = np.concatenate(S)
    _, first, inv = np.unique(a, return_index=True, return_inverse=True)
    return hash(np.argsort(np.argsort(first))[inv].tobytes())


def learn_states(trajs, O, L=3, max_rounds=40, verbose=False):
    """Three stages.
    1. Split, then minimize (BMR on L-step futures, majority successors, re-route), until
       the state count holds. Keeps the machine small; loses distinctions that only the
       next state can show (rare states, weak L-step evidence).
    2. One split round on that consistent machine, so the next-state modality can carry
       those distinctions back (split-only rounds repeated without minimizing multiply
       duplicate states every round).
    3. Bisimulation: group by L-step futures, then regroup with "which group comes next"
       until the grouping is a fixed point. Stages 2-3 repeat until the count holds."""
    S = [np.array(obs) for obs, _, _ in trajs]
    seen = {partition_key(S)}
    for rnd in range(max_rounds):
        S = minimize(trajs, split_round(trajs, S, O, L), O, L)
        if verbose:
            print(f'   stage 1 round {rnd + 1}: {n_states(S)} states')
        if partition_key(S) in seen:
            break
        seen.add(partition_key(S))
    rounds = rnd + 1
    # stage 2 and 3, alternated: one split round (without minimizing, it multiplies
    # duplicates every round), then bisimulation, until the count holds
    seen = {partition_key(S)}
    for rnd in range(max_rounds):
        S = fill_ends(trajs, split_round(trajs, S, O, L))
        if verbose:
            print(f'   stage 2 round {rnd + 1}: split to {n_states(S)} states')
        S = bisimulate(trajs, S, O, L, verbose)
        if partition_key(S) in seen:
            break
        seen.add(partition_key(S))
    return S, rounds + rnd + 1


def fill_ends(trajs, S):
    """Give end points (-1) the successor the data show for (previous state, a, o). An edge
    never seen elsewhere stays -1: a guess files the visit under the wrong state, and those
    visits seed chains of duplicates that grow with the data."""
    votes = defaultdict(Counter)
    for (obs, acts, _), s in zip(trajs, S):
        for t, a in enumerate(acts):
            if s[t] >= 0 and s[t + 1] >= 0:
                votes[int(s[t]), a, obs[t + 1]][int(s[t + 1])] += 1
    out = []
    for (obs, acts, _), s in zip(trajs, S):
        s = s.copy()
        for t, a in enumerate(acts):
            if s[t + 1] < 0:
                v = votes.get((int(s[t]), a, obs[t + 1]))
                if v:
                    s[t + 1] = v.most_common(1)[0][0]
        out.append(s)
    return relabel(out)


def bisimulate(trajs, S, O, L, verbose=False, max_iter=20):
    """Group states by BMR on L-step futures; then add one modality per action whose outcome
    is (observation, group of the next state) and regroup, until the grouping repeats."""
    K = n_states(S)
    windows, trans = [], Counter()
    for ep, s in zip(trajs, S):
        obs, acts, _ = ep
        windows += [w for w in future_windows([int(x) for x in s], obs, acts, L, tail=tail_of(ep))
                    if w[0] >= 0]
        for t, a in enumerate(acts):
            if s[t] >= 0 and s[t + 1] >= 0:
                trans[int(s[t]), a, obs[t + 1], int(s[t + 1])] += 1
    items = list(range(K))
    block = bmr_rows(Rows(items, windows, O, L))
    for it in range(max_iter):
        nxt = [(i, (('next', a), (o, int(block[j])))) for (i, a, o, j), c in trans.items()
               for _ in range(c)]
        new = bmr_rows(Rows(items, windows + nxt, O, L))
        if verbose:
            print(f'   stage 3 iteration {it + 1}: {len(set(new))} groups')
        if len(set(zip(block, new))) == len(set(block)) == len(set(new)):
            break
        block = new
    _, block = np.unique(block, return_inverse=True)
    return [np.where(s < 0, -1, block[s]) for s in S]


def minimize(trajs, S, O, L):
    """Minimize the machine S defines: group states by BMR on their pooled L-step futures,
    give each group the majority successor of its members under every (action, obs), and
    re-route all episodes through the grouped machine. Returns the new state sequences."""
    K = int(max(s.max() for s in S)) + 1
    windows, votes = [], defaultdict(Counter)
    S = [np.asarray(s) for s in S]
    for ep, s in zip(trajs, S):
        obs, acts, _ = ep
        windows += [w for w in future_windows([int(x) for x in s], obs, acts, L, tail=tail_of(ep))
                    if w[0] >= 0]
        for t, a in enumerate(acts):
            if s[t] >= 0 and s[t + 1] >= 0:
                votes[int(s[t]), a, obs[t + 1]][int(s[t + 1])] += 1
    seen = {e for e, _ in windows}
    items = [i for i in range(K) if i in seen]
    # (adding "which group comes next" here, with old labels or iterated to a fixed point,
    # over-splits: stale successors make duplicate states look different)
    block = dict(zip(items, bmr_rows(Rows(items, windows, O, L))))
    for i in range(K):
        block.setdefault(i, max(block.values(), default=-1) + 1)
    # a block's successor under (a, o): the majority over its members' transitions, so
    # stale pointers (a state that was split off late) are outvoted
    bvotes = defaultdict(Counter)
    for (i, a, o), v in votes.items():
        for j, c in v.items():
            bvotes[block[i], a, o][block[j]] += c
    # (redirecting only on a clear majority brings the duplicate chains back; with sparse
    # data a group can mix situations, and the majority then makes a confident error)
    bsucc = {key: v.most_common(1)[0][0] for key, v in bvotes.items()}
    # re-route every episode through the block machine from the start
    out = []
    for (obs, acts, _), s in zip(trajs, S):
        b = [block[int(s[0])]]
        for t, a in enumerate(acts):
            x = int(s[t + 1])
            nxt = bsucc.get((b[-1], a, obs[t + 1]))
            b.append(nxt if nxt is not None else (block[x] if x >= 0 else -1))
        out.append(np.array(b))
    return relabel(out)


def machine(trajs, S, A, O, L=3):
    K = int(max(s.max() for s in S)) + 1
    C1 = np.zeros((K, A, O))
    votes = defaultdict(Counter)
    truth = [Counter() for _ in range(K)]
    windows = []
    for (obs, acts, true), s in zip(trajs, S):
        for t, a in enumerate(acts):
            C1[s[t], a, obs[t + 1]] += 1
            if s[t + 1] >= 0:
                votes[int(s[t]), a, obs[t + 1]][int(s[t + 1])] += 1
            truth[s[t]][true[t]] += 1
        windows += [w for w in future_windows([int(x) for x in s], obs, acts, L) if w[0] >= 0]
    f ={key: v.most_common(1)[0][0] for key, v in votes.items()}
    conflicts = sum(sum(v.values()) - v.most_common(1)[0][1] for v in votes.values())
    RL = Rows(list(range(K)), windows, O, L)
    P1 = C1 / np.maximum(C1.sum(-1, keepdims=True), 1)
    first = defaultdict(Counter)
    for (obs, _, _), s in zip(trajs, S):
        first[obs[0]][int(s[0])] += 1
    starts = {o: v.most_common(1)[0][0] for o, v in first.items()}
    start = int(S[0][0])
    return dict(K=K, C1=C1, P1=P1, f=f, truth=truth, RL=RL, start=start, starts=starts,
                conflicts=conflicts)


def unmix(m, tol=0.02):
    """Drop states whose L-step rows are a convex mixture of other states' rows."""
    R = m['RL']
    X, mod, a = R.X, R.mod, R.alpha
    T = R.totals(X)
    P = X / np.maximum(T[:, mod], 1)
    used = T > 0
    alive, weights, report = list(range(m['K'])), {}, {}
    while len(alive) > 2:
        best = None
        for k in alive:
            ok = [j for j in alive if j != k and used[j][used[k]].all()]
            if len(ok) < 2:
                continue
            cols = used[k][mod]
            Am = np.vstack([P[ok][:, cols].T, 1e3 * np.ones(len(ok))])
            w, _ = nnls(Am, np.concatenate([P[k][cols], [1e3]]))
            w = np.where(w > tol, w, 0)
            if (w > 0).sum() < 2:
                continue
            w /= w.sum()
            mix = w @ P[ok][:, cols]
            x = X[k][cols]
            with np.errstate(divide='ignore'):
                lmix = float((x * np.log(np.where(x > 0, mix, 1))).sum())
            lfree = float((gammaln(a[cols] + x) - gammaln(a[cols])).sum()
                          - (gammaln(1 + T[k][used[k]]) - gammaln(1.0)).sum())
            bf = lmix - lfree
            if best is None or bf > best[0]:
                best = (bf, k, {j: v for j, v in zip(ok, w) if v > 0})
        if best is None or best[0] <= 0:
            break
        bf, k, w = best
        report[k] = bf
        weights[k] = w
        alive.remove(k)
    return alive, weights, report


def generator(m, V, weights, A, O):
    ix = {k: i for i, k in enumerate(V)}
    W = np.zeros((m['K'], len(V)))

    def vec(k):
        if k in ix:
            return np.eye(len(V))[ix[k]]
        return sum(w * vec(j) for j, w in weights[k].items())
    for k in range(m['K']):
        W[k] = vec(k)
    G = np.zeros((A, O, len(V), len(V)))
    for i, k in enumerate(V):
        for a in range(A):
            for o in np.flatnonzero(m['P1'][k, a] > 0):
                s = m['f'].get((k, a, int(o)))
                if s is None:
                    continue
                q = W[s]
                if q[i] > 1e-9 and q.max() < 1 - 1e-9:
                    G[a, o, i, i] += m['P1'][k, a, o]          # inertia
                else:
                    G[a, o, i] += m['P1'][k, a, o] * q
    return dict(G=G, D=W[m['start']], W=W)


# ------------------------------------------------------------------- factors
def kway_products(sps, n, max_factors=6):
    """Sets of SP partitions whose joint labels are all distinct and sizes multiply to n."""
    sps = sorted(sps, key=lambda p: len(set(p)))
    size = [len(set(p)) for p in sps]
    best = []

    def dfs(start, chosen, prod):
        nonlocal best
        if prod == n:
            if len(chosen) > len(best):
                best = list(chosen)
            return
        if len(chosen) >= max_factors:
            return
        for i in range(start, len(sps)):
            if n % (prod * size[i]):
                continue
            joint = set(zip(*[sps[j] for j in chosen + [i]]))
            if len(joint) == prod * size[i]:
                dfs(i + 1, chosen + [i], prod * size[i])
    dfs(0, [], 1)
    return [sps[i] for i in best]


def run(env, trajs, L=3, verbose=False):
    A, O = env.n_act, env.n_obs
    S, rounds = learn_states(trajs, O, L, verbose=verbose)
    m = machine(trajs, S, A, O, L)
    V, weights, report = unmix(m)
    gen = generator(m, V, weights, A, O)
    Sg = gen['G'].sum(1) > 1e-6
    succ = [[set(np.flatnonzero(Sg[a, i])) for i in range(len(V))] for a in range(A)]
    sps = FS.sp_lattice(succ)
    prod = kway_products(sps, len(V))
    return dict(m=m, V=V, weights=weights, report=report, gen=gen, sps=sps, prod=prod,
                rounds=rounds)


def truth_of(res, i):
    """Dominant true (loc, bits) of vertex i and its purity."""
    t = res['m']['truth'][res['V'][i]]
    (st, c), n = t.most_common(1)[0], sum(t.values())
    return st, c / n


def check_factors(res, env):
    """Match each learned factor to location or a bit, by ARI against the vertices' truth."""
    from make_fig_hidden_states import ari
    st = [truth_of(res, i)[0] for i in range(len(res['V']))]
    targets = {'loc': [s[0] for s in st]}
    for b in range(env.k):
        targets[f'bit{b + 1}'] = [s[1][b] for s in st]
    out = []
    for p in res['prod']:
        scores = {name: ari(np.array(p), np.array(t)) for name, t in targets.items()}
        name = max(scores, key=scores.get)
        out.append((len(set(p)), name, round(scores[name], 3)))
    return out


if __name__ == '__main__':
    import sys
    import time
    sys.stdout.reconfigure(encoding='utf-8')
    for k, n, T in ((1, 3000, 12), (2, 10000, 16), (3, 30000, 20)):
        env = KMaze(k)
        t0 = time.time()
        res = run(env, env.sample(n, T, seed=0), verbose=True)
        V = res['V']
        pur = np.mean([truth_of(res, i)[1] for i in range(len(V))])
        print(f'k={k}: {n} episodes x {T} steps, {time.time() - t0:.0f}s, {res["rounds"]} rounds: '
              f'{res["m"]["K"]} predictive states -> {len(V)} vertices '
              f'(truth {env.n_loc * 2 ** k}), purity {pur:.3f}, conflicts {res["m"]["conflicts"]}, '
              f'{len(res["sps"])} SP partitions')
        print('   factors (size, matches, ARI):', check_factors(res, env))
