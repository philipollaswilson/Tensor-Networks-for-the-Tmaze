"""RDP / POMDP benchmark domains of Deb, Cipollone, Jonsson, Ronca, Talebi, 'Tractable Offline
Learning of Regular Decision Processes' (arXiv 2409.02747v1), plus Tiger.

Same interface as kmaze.KMaze: n_act, n_obs, sample(n, T, seed) -> list of (obs, acts, true),
obs length T + 1, acts length T, true = hidden state per step (for scoring only, no learner may
read it). Behaviour policy: uniform over actions (the paper gives none).

Reward. A paper trace symbol is a o / r, so here an observation is the pair (o, r), encoded
o = code * len(rewards) + index of r, and reward(o) reads r back. n_obs = |O| x |R|, the same
alphabet size as the ARO of the paper's threshold; impossible pairs just never occur.

Truth (exact rational arithmetic on the reachable beliefs, never from data):
  n_pred        minimal predictive-state count: Moore-minimised reachable belief graph, after o_0,
                cyclic. None if unbounded, then pred_note says why
  n_layered(T)  the same with states tied to time and futures cut at T, including q_0 and the
                final layer. This, not n_pred, is what AdaCT-H should be compared with
  belief_logloss(trajs)  Bayes-optimal held-out nats per step (the floor of any predictor)

Table: FIG = read off a figure of the paper. ASSUMED = neither paper nor classic source fixes it.

env       | from the paper                        | from the classic source  | ASSUMED / conflicts
----------|---------------------------------------|--------------------------|---------------------
TMaze     | N,S,E,W; cues 110 (goal N, Example 2) | Bakker 2001: N, S, E, W  | N=4 from Table 1 H=5 and
          | / 011, corridor 101, junction 010;    | maze, wall bumps stay    | FIG 3 (four E moves then
          | +4 right at junction, -1 otherwise    | put                      | N/S); Figure 1 says N=10,
          | and episode ends; -1 standing still;  |                          | Example 2 says 13 cells.
          | H=5, unrestricted actions (Table 1).  |                          | FIG 5: going back to S
          | FIG 5: return to S shows 101          |                          | shows 101, not the cue
          | (cue_once=True)                       |                          | (text: 'at the initial
          |                                       |                          | position'). FIG 3 shows 010
          |                                       |                          | on the N/S step; the text
          |                                       |                          | says at the junction: I
          |                                       |                          | follow the text. ASSUMED:
          |                                       |                          | E and W at the junction
          |                                       |                          | are 'otherwise' (-1, end);
          |                                       |                          | end state with a fifth
          |                                       |                          | code 000, padded to T,
          |                                       |                          | reward 0. Rewards +4, -1
          |                                       |                          | break the paper's R in
          |                                       |                          | [0,1], kept as printed
Cookie    | rooms blue, white, green, red; button | Icarte 2019: no cookie   | FIG 4b: white hub, red up,
          | in red makes a cookie in blue or      | at start; pressing with  | blue left, green right,
          | green at 1/2; moves left, right, up,  | a cookie out replaces    | start white. Figure's room
          | down, press, eat; +1 for eating; 6    | it                       | is orange, its button green
          | observations; H=9 (Table 1)           |                          | (text: red room). Figure's
          |                                       |                          | cookie taken as an example
          |                                       |                          | only. ASSUMED: press outside
          |                                       |                          | red and eat with no cookie
          |                                       |                          | do nothing; no step cost;
          |                                       |                          | the cookie shows only from
          |                                       |                          | inside its room
Corridor  | 2 x m grid; a0, a1 go to row 0, 1 of  | Ronca & De Giacomo 2021  | FIG 2 decoded: m=6, p0=1,
          | next column; enemy with p_i^0, p_i^1, | Example 1 (2m states,    | p1=0, start row uniform,
          | swapped at every encounter; +1 for    | p symbolic, no numbers)  | no enemy in column 0. All
          | avoiding; wraps; obs (i, j, e); H=5   |                          | edges are deterministic,
          |                                       |                          | so the text's probabilities
          |                                       |                          | are 0 and 1 in that run;
          |                                       |                          | p0, p1 are arguments.
          |                                       |                          | ASSUMED: no reward in
          |                                       |                          | column 0 (no enemy to
          |                                       |                          | avoid); p same per column
Cheese    | 10 states, 6 obs, 4 actions; +1 at    | McCallum; Cassandra's    | Goal is not a state (to get
          | goal; reset uniform on non-goal; H=6  | cheese.95.POMDP: maze,   | 10 and 6): entering it
          |                                       | obs, uniform start       | yields the obs of the new
          |                                       |                          | cell with r=1 on one step.
          |                                       |                          | Classic has 11 states, 7
          |                                       |                          | obs and a step inside goal
MiniHall  | 12 states, 3 rooms, 4 orientations, 3 | Littman, Cassandra,      | Obs 0-5 per orientation:
          | actions, +1 at star, reset, 6 obs;    | Kaelbling 1995 via       | FIG 4a, not the file (which
          | FIG 4a: obs per room and orientation  | mini-hall2.POMDP:        | has 9 obs). Forward at (B,
          | H=15                                  | forward, right, left,    | S) stays, per Fig 4a; the
          |                                       | uniform start            | file jumps to (A, N), which
          |                                       |                          | looks like a typo. Goal
          |                                       |                          | handled as in Cheese
Tiger     | not in the paper                      | Kaelbling, Littman,      | o_0 uniform and carrying
          |                                       | Cassandra 1998 tiger.95: | r=0 (classic has no o_0),
          |                                       | listen acc. 0.85; -1;    | so R has 4 values. Opening
          |                                       | +10, -100; open resets   | continues the episode, as
          |                                       | tiger, uniform obs       | the classic file does

Not in the paper for any domain: dataset size, behaviour policy, delta, FlexFringe settings.

Checks against the paper. Agree: Corridor layered 11 (Table 1, Figure 2), T-maze 'east' layered 12
(Figure 3), optimal return T-maze 4.0 and Cookie 1.0 (Table 1). Do not agree: T-maze 'ew' layered
21 vs 19 in Figure 5 (which has no W edge at S at t = 0, so its action set is not this one);
Corridor optimal return 5.0 vs r = 1.0 (T-maze's r is a sum, so unexplained). Table 1's Q vs
n_layered(H): T-maze 18 / 29 / 104 vs 21, Cookie 91 / 220 / 116 vs 77, Cheese 326 / 669 / 1158 vs
74, Mini-hall 5134 / 897 / - vs 236 (restricted / FlexFringe / CMS). Its learners are not minimal
there or its domains differ from these; the paper does not say which.
"""
from fractions import Fraction as F

import numpy as np


class TabEnv:
    """Subclass: start() and step(s, a) give lists of (prob, next state, code, reward)."""
    rewards = (0,)
    unbounded = False
    pred_note = 'exact: finite reachable beliefs'

    def enc(self, code, r):
        return code * len(self.rewards) + self.rewards.index(r)

    def reward(self, o):
        return self.rewards[o % len(self.rewards)]

    def legal(self, s):
        return range(self.n_act)

    def outcomes(self, s, a):
        """(cumulative probs, list of (next state, symbol)), cached."""
        key = (s, a)
        if key not in self._tab:
            rows = self.start() if s is None else self.step(s, a)
            self._tab[key] = (np.cumsum([float(r[0]) for r in rows]),
                              [(r[1], self.enc(r[2], r[3])) for r in rows])
        return self._tab[key]

    def draw(self, s, a, rng):
        cum, out = self.outcomes(s, a)
        return out[min(int(np.searchsorted(cum, rng.random())), len(out) - 1)]

    def sample(self, n, T, seed=0):
        rng = np.random.default_rng(seed)
        res = []
        for _ in range(n):
            s, o = self.draw(None, None, rng)
            obs, acts, true = [o], [], [s]
            for _ in range(T):
                leg = self.legal(s)
                a = int(leg[rng.integers(len(leg))])
                s, o = self.draw(s, a, rng)
                acts.append(a); obs.append(o); true.append(s)
            res.append((obs, acts, true))
        return res

    # ---------------------------------------------------------------- exact beliefs
    def children(self, b):
        """Belief b -> {(a, sym): (P(sym | b, a), next belief)}, all in Fractions."""
        mass = {}
        for s, p in (b or {None: F(1)}).items():
            for a in (self.legal(s) if s is not None else [None]):
                for q, s2, c, r in (self.start() if s is None else self.step(s, a)):
                    d = mass.setdefault((a, self.enc(c, r)), {})
                    d[s2] = d.get(s2, 0) + p * F(q)
        out = {}
        for key, d in mass.items():
            tot = sum(d.values())
            out[key] = (tot, {s: v / tot for s, v in d.items()})
        return out

    def belief_graph(self, depth=None, cap=200000):
        """Reachable beliefs. edges[i] = {(a, sym): (p, j)}; reach[t] = beliefs after o_t;
        first[i] = first t reached. Node 0 is q_0, before o_0. depth=None: to closure."""
        keys, bel, edges, first = {None: 0}, [None], [{}], [-1]
        reach, frontier, t = [], {0}, 0
        while frontier and (depth is None or t <= depth + 1):
            nxt = set()
            for i in sorted(frontier):
                if edges[i]:
                    nxt.update(j for _, j in edges[i].values())
                    continue
                for key, (p, b) in self.children(bel[i]).items():
                    k = frozenset(b.items())
                    if k not in keys:
                        keys[k] = len(bel)
                        bel.append(b); edges.append({}); first.append(t)
                        if len(bel) > cap:
                            raise OverflowError(f'more than {cap} beliefs')
                    edges[i][key] = (p, keys[k])
                    nxt.add(keys[k])
            if depth is not None and t == depth + 1:
                break
            reach.append(nxt)
            frontier = nxt if depth is not None or any(not edges[j] for j in nxt) else set()
            t += 1
        return edges, reach, first

    @property
    def n_pred(self):
        if self.unbounded:
            return None
        if self._n_pred is None:
            edges, _, _ = self.belief_graph()
            nodes = range(1, len(edges))
            cls = dict.fromkeys(nodes, 0)
            while True:
                sig = {i: tuple(sorted((a, o, p, cls[j]) for (a, o), (p, j) in edges[i].items()))
                       for i in nodes}
                ids = {}
                new = {i: ids.setdefault(sig[i], len(ids)) for i in nodes}
                if len(ids) == len(set(cls.values())):
                    break
                cls = new
            self._n_pred = len(ids)
        return self._n_pred

    def n_layered(self, T):
        """Minimal layered RDP for horizon H = T: 1 (q_0) + classes per layer, Q_{H+1} included."""
        edges, reach, first = self.belief_graph(depth=T)
        sig = {i: 0 for i in range(len(edges))}
        per_k = [sig]
        for k in range(1, T + 1):
            ids, cur = {}, {}
            for i in range(len(edges)):
                if first[i] <= T - k:
                    v = tuple(sorted((a, o, p, per_k[-1][j]) for (a, o), (p, j) in edges[i].items()))
                    cur[i] = ids.setdefault(v, len(ids))
            per_k.append(cur)
        return 1 + sum(len({per_k[T - t][i] for i in reach[t]}) for t in range(T + 1))

    def belief_logloss(self, trajs):
        """Mean -log P(o_{t+1} | history, a_t) of the true model, float belief filter."""
        tot, n = 0.0, 0
        for obs, acts, _ in trajs:
            b = {}
            cum, out = self.outcomes(None, None)
            for i, (s, o) in enumerate(out):
                if o == obs[0]:
                    b[s] = b.get(s, 0.0) + (cum[i] - (cum[i - 1] if i else 0.0))
            z = sum(b.values())
            b = {s: v / z for s, v in b.items()}
            for a, o in zip(acts, obs[1:]):
                nb = {}
                for s, v in b.items():
                    cum, out = self.outcomes(s, a)
                    for i, (s2, y) in enumerate(out):
                        if y == o:
                            nb[s2] = nb.get(s2, 0.0) + v * (cum[i] - (cum[i - 1] if i else 0.0))
                z = sum(nb.values())
                tot -= np.log(z); n += 1
                b = {s: v / z for s, v in nb.items()}
        return tot / n


class TMaze(TabEnv):
    """Cells 0 (S) .. N (junction). State (x, g) with g = 0 goal north, 1 goal south, or 'end'.
    actions='full' (Table 1), 'east' (Figure 3: corridor E only, junction N, S), 'ew' (Figure 5:
    corridor E, W)."""
    rewards = (0, -1, 4)
    CUES, CORR, JUNC, END = (1, 0), 2, 3, 4     # codes 011 = 0, 110 = 1, 101, 010, 000

    def __init__(self, N=4, actions='full', cue_once=True):
        self.N, self.actions, self.cue_once = N, actions, cue_once
        self.n_act, self.n_codes = 4, 5
        self.n_obs = self.n_codes * len(self.rewards)
        self.names = 'NSEW'
        self._tab, self._n_pred = {}, None

    def start(self):
        return [(F(1, 2), (0, g), self.CUES[g], 0) for g in (0, 1)]

    def legal(self, s):
        if s == 'end' or self.actions == 'full':
            return range(4)
        if s[0] == self.N:
            return (0, 1)
        return (2,) if self.actions == 'east' else (2, 3)

    def code(self, x, g):
        if x == self.N:
            return self.JUNC
        return self.CUES[g] if x == 0 and not self.cue_once else self.CORR

    def step(self, s, a):
        if s == 'end':
            return [(1, 'end', self.END, 0)]
        x, g = s
        if x == self.N:
            return [(1, 'end', self.END, 4 if a == g else -1)]
        if a == 2:
            return [(1, (x + 1, g), self.code(x + 1, g), 0)]
        if a == 3 and x > 0:
            return [(1, (x - 1, g), self.code(x - 1, g), 0)]
        return [(1, s, self.code(x, g), -1)]    # stood still


class Cookie(TabEnv):
    """Rooms white 0, red 1 (up), blue 2 (left), green 3 (right). State (room, cookie room or 0).
    Actions left, right, up, down, press, eat. Codes: room, 4 blue + cookie, 5 green + cookie."""
    rewards = (0, 1)
    MOVE = {(0, 0): 2, (0, 1): 3, (0, 2): 1, (2, 1): 0, (3, 0): 0, (1, 3): 0}

    def __init__(self):
        self.n_act, self.n_codes = 6, 6
        self.n_obs = self.n_codes * len(self.rewards)
        self._tab, self._n_pred = {}, None

    def code(self, room, c):
        return room + 2 if room == c and room >= 2 else room

    def start(self):
        return [(1, (0, 0), 0, 0)]

    def step(self, s, a):
        room, c = s
        if a < 4:
            room = self.MOVE.get((room, a), room)
            return [(1, (room, c), self.code(room, c), 0)]
        if a == 4 and room == 1:
            return [(F(1, 2), (1, x), 1, 0) for x in (2, 3)]
        if a == 5 and c and room == c:
            return [(1, (room, 0), room, 1)]
        return [(1, s, self.code(room, c), 0)]


class Corridor(TabEnv):
    """State (row, column, swapped). Action = next row. Enemy at (row, column) with p0 / p1, the
    two swapped when swapped = 1; none in column 0. Code = (row, column, enemy)."""
    rewards = (0, 1)

    def __init__(self, m=6, p0=F(1), p1=F(0)):
        self.m, self.p0, self.p1 = m, F(p0), F(p1)
        self.n_act, self.n_codes = 2, 4 * m
        self.n_obs = self.n_codes * len(self.rewards)
        self._tab, self._n_pred = {}, None

    def code(self, i, j, e):
        return (2 * i + e) * self.m + j

    def start(self):
        return [(F(1, 2), (i, 0, 0), self.code(i, 0, 0), 0) for i in (0, 1)]

    def step(self, s, a):
        _, col, sw = s
        col = (col + 1) % self.m
        if col == 0:
            return [(1, (a, 0, sw), self.code(a, 0, 0), 0)]
        p = self.p1 if bool(a) != bool(sw) else self.p0
        rows = [(1 - p, (a, col, sw), self.code(a, col, 0), 1), (p, (a, col, 1 - sw), self.code(a, col, 1), 0)]
        return [r for r in rows if r[0] > 0]


class GoalMaze(TabEnv):
    """Deterministic maze with aliased observations and a goal that is not a state: entering
    it gives r = 1 together with the obs of a uniform restart cell. Subclass sets cells,
    self.obs_of[cell], and move(cell, a) returning a cell or 'goal'."""
    rewards = (0, 1)

    def start(self):
        return [(F(1, len(self.cells)), c, self.obs_of[c], 0) for c in self.cells]

    def step(self, s, a):
        c = self.move(s, a)
        if c == 'goal':
            return [(F(1, len(self.cells)), x, self.obs_of[x], 1) for x in self.cells]
        return [(1, c, self.obs_of[c], 0)]


class Cheese(GoalMaze):
    """Numbering of cheese.95.POMDP: top row 0-4, then 5, 6, 7 below 0, 2, 4, then 8, 9 below
    5, 7; goal below 6. Actions N, S, E, W."""

    def __init__(self):
        self.cells = tuple(range(10))
        self.obs_of = (0, 1, 2, 1, 3, 4, 4, 4, 5, 5)
        self.n_act, self.n_codes = 4, 6
        self.n_obs = self.n_codes * len(self.rewards)
        self.names = 'NSEW'
        self._tab, self._n_pred = {}, None

    def move(self, c, a):
        if a == 0:
            return {5: 0, 6: 2, 7: 4, 8: 5, 9: 7}.get(c, c)
        if a == 1:
            return {0: 5, 2: 6, 4: 7, 5: 8, 6: 'goal', 7: 9}.get(c, c)
        if c <= 4:
            return min(c + 1, 4) if a == 2 else max(c - 1, 0)
        return c


class MiniHall(GoalMaze):
    """Rooms A 0, B 1, C 2; heading N 0, E 1, S 2, W 3; state (room, heading). Actions forward,
    turn right, turn left. Goal below C. Observations of Figure 4a."""
    OBS = {0: (0, 1, 2, 3), 1: (4, 5, 3, 1), 2: (3, 1, 4, 5)}
    DOOR = {(0, 2): (1, 2), (1, 0): (0, 0), (1, 1): (2, 1), (2, 3): (1, 3), (2, 2): 'goal'}

    def __init__(self):
        self.cells = tuple((r, d) for r in range(3) for d in range(4))
        self.obs_of = {(r, d): self.OBS[r][d] for r, d in self.cells}
        self.n_act, self.n_codes = 3, 6
        self.n_obs = self.n_codes * len(self.rewards)
        self._tab, self._n_pred = {}, None

    def move(self, c, a):
        r, d = c
        if a == 0:
            return self.DOOR.get(c, c)
        return (r, (d + (1 if a == 1 else -1)) % 4)


class Tiger(TabEnv):
    """State 0 tiger left, 1 right. Actions listen, open left, open right. Code 0 hear left, 1
    hear right."""
    rewards = (0, -1, 10, -100)
    unbounded = True
    pred_note = ('unbounded: the belief is a function of k = #hear-left - #hear-right since the '
                 'last opening, every integer k is reachable, and P(hear left | k) is strictly '
                 'monotonic in k, so all k are predictively distinct. n_layered(T) is finite')

    def __init__(self, acc=F(17, 20)):
        self.acc = F(acc)
        self.n_act, self.n_codes = 3, 2
        self.n_obs = self.n_codes * len(self.rewards)
        self._tab, self._n_pred = {}, None

    def start(self):
        return [(F(1, 4), s, c, 0) for s in (0, 1) for c in (0, 1)]

    def step(self, s, a):
        if a == 0:
            return [(self.acc, s, s, -1), (1 - self.acc, s, 1 - s, -1)]
        r = -100 if a - 1 == s else 10
        return [(F(1, 4), x, c, r) for x in (0, 1) for c in (0, 1)]


ENVS = {'tmaze': TMaze, 'cookie': Cookie, 'corridor': Corridor, 'cheese': Cheese,
        'minihall': MiniHall, 'tiger': Tiger}
PAPER_H = {'tmaze': 5, 'cookie': 9, 'corridor': 5, 'cheese': 6, 'minihall': 15, 'tiger': 10}  # Table 1; tiger: mine
