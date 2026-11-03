"""RDP baselines on the trajectory format of kmaze / rdp_envs: list of (obs, acts, true). true is
never read here. Each learner gets only trajs and the alphabet sizes A, O and returns a Machine.

Trace symbols as in Deb et al. (arXiv 2409.02747v1): e = a_0 o_0 ... a_H o_H with a_0 a dummy,
H = T, a_t = acts[t - 1], o_t = obs[t], symbol = a * O + o. The paper's reward is inside o (see
rdp_envs), so here R = 1 and O = |O| x |R|: the threshold's ARO is unchanged.

adact_h     Appendix A of the paper, on the plain L-infinity prefix distance (not the CMS or the
            restricted languages X_{1,1,1} of the paper's Table 1). Choices the pseudocode leaves
            open:
              delta = 0.05 (the paper prints no value)
              candidates are visited by decreasing count, ties by (state, symbol), and 'element in
              Similar' is the first promoted (the paper: any, loop order unstated)
              candidates with no data are skipped: they would merge anywhere (empty Z makes the
              threshold infinite) and change no count; their edge stays undefined
              t in TestDistinct is the loop index. The pseudocode's header says traces of length
              H - t + 1, but at loop t the suffixes have length H - t, which is also the exponent
flexfringe  the release binary of tudelft-cda-lab/FlexFringe in lib/flexfringe, run with the
            project's stock ini/alergia.ini (a copy sits next to the binary), unchanged. Atomic
            symbols plus a termination symbol, as the paper describes. The paper does not give
            its settings
Machine     counts P(o | state, a) by routing the training trajs. Both learners share it, so the
            scores differ only through the states. Perks prior, total mass 1 per (state, a); an
            unseen (state, a) or a symbol with no edge falls back to the pooled P(o | a), and the
            walk stays lost to the end of the episode. Layered machines cannot go past H.
"""
import json
import os
import subprocess
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF_DIR = os.path.join(ROOT, 'lib', 'flexfringe')
FF_EXE = os.path.join(FF_DIR, 'flexfringe-x64-windows.exe')
FF_INI = os.path.join(FF_DIR, 'alergia.ini')


class Machine:
    """Deterministic automaton on symbols a * O + o, a = A the dummy. delta[(q, sym)] = q'."""

    def __init__(self, delta, start, n_states, trajs, A, O, **info):
        self.delta, self.start, self.n_states, self.A, self.O = delta, start, n_states, A, O
        self.info = info
        self.counts, self.pool = {}, np.zeros((A, O))
        for obs, acts, _ in trajs:
            q = delta.get((start, A * O + obs[0]))
            for a, o in zip(acts, obs[1:]):
                self.pool[a, o] += 1
                if q is None:
                    continue
                self.counts.setdefault(q, np.zeros((A, O)))[a, o] += 1
                q = delta.get((q, a * O + o))

    def probs(self, q, a):
        c = self.counts.get(q) if q is not None else None
        c = c[a] if c is not None and c[a].sum() else self.pool[a]
        return (c + 1 / self.O) / (c.sum() + 1)

    def state_after(self, obs, acts):
        """State after obs[0], acts[0], obs[1], ...; None when lost."""
        q = self.delta.get((self.start, self.A * self.O + obs[0]))
        for a, o in zip(acts, obs[1:]):
            q = None if q is None else self.delta.get((q, a * self.O + o))
        return q

    def predict(self, obs, acts):
        """P(next obs | obs[:t + 1], acts[:t + 1]): acts has the same length as obs, its last
        entry being the action just taken."""
        return self.probs(self.state_after(obs, acts[:-1]), acts[-1])

    def logloss(self, trajs):
        """(mean -log P(o_{t+1} | history, a_t) in nats, fraction of steps not lost)."""
        tot, n, kept = 0.0, 0, 0
        for obs, acts, _ in trajs:
            q = self.delta.get((self.start, self.A * self.O + obs[0]))
            for a, o in zip(acts, obs[1:]):
                tot -= np.log(self.probs(q, a)[o])
                n += 1; kept += q is not None
                q = None if q is None else self.delta.get((q, a * self.O + o))
        return tot / n, kept / n


# ------------------------------------------------------------------------------ AdaCT-H
def symbols(trajs, A, O):
    """N x (H + 1) array of symbols e_0 .. e_H."""
    return np.array([[A * O + obs[0]] + [a * O + o for a, o in zip(acts, obs[1:])]
                     for obs, acts, _ in trajs])


def threshold(t, H, n, A, R, O, delta):
    """Right side of TestDistinct, n = min(|Z1|, |Z2|), in logs so that (ARO)^(H-t) can't overflow."""
    return np.sqrt(2 * (np.log(8 / delta) + (H - t) * np.log(A * R * O)) / n)


def linf_prefix(z1, z2):
    """L-infinity prefix distance of two lists of equal-length suffix tuples, by definition:
    max over prefixes e (any length) of |p1(e*) - p2(e*)|. Slow, used to check the fast one."""
    d = 0.0
    for u in range(len(z1[0]) + 1 if z1 else 0):
        c1, c2 = {}, {}
        for z, c in ((z1, c1), (z2, c2)):
            for e in z:
                c[e[:u]] = c.get(e[:u], 0) + 1 / len(z)
        d = max(d, *(abs(c1.get(e, 0) - c2.get(e, 0)) for e in {*c1, *c2}))
    return d


class Prefixes:
    """Ids of all prefixes of the suffixes e_{t:H}, per length; per-state counts on demand."""

    def __init__(self, E, t):
        self.ids, cur = [], np.zeros(len(E), dtype=np.int64)
        K = int(E.max()) + 1
        for j in range(t, E.shape[1]):
            cur = np.unique(cur * K + E[:, j], return_inverse=True)[1].astype(np.int64)
            self.ids.append(cur)
        self.buf = np.zeros(len(E))

    def counts(self, u, idx):
        return np.unique(self.ids[u][idx], return_counts=True)

    def at_least(self, Z1, Z2, thr):
        """L-infinity prefix distance of two index sets >= thr? Stops at the first witness."""
        n1, n2, buf = len(Z1), len(Z2), self.buf
        for u in range(len(self.ids)):
            i1, c1 = self.counts(u, Z1)
            i2, c2 = self.counts(u, Z2)
            buf[i1] += c1 / n1
            buf[i2] -= c2 / n2
            d = max(np.abs(buf[i1]).max(), np.abs(buf[i2]).max())
            buf[i1] = 0; buf[i2] = 0
            if d >= thr:
                return True
        return False


def adact_h(trajs, A, O, delta=0.05, R=1):
    """Appendix A. Returns a Machine with n_states = |Q_0 u ... u Q_{H+1}| and info['layers'] the
    layer sizes. States are (layer, index)."""
    E = symbols(trajs, A, O)
    N, H = len(E), E.shape[1] - 1
    Z = [np.arange(N)]                                    # Z(q) for q in Q_t
    tau, sizes = {}, [1]
    for t in range(H + 1):
        groups = {}                                       # candidates q a o
        for q, idx in enumerate(Z):
            for s in np.unique(E[idx, t]):
                groups[q, int(s)] = idx[E[idx, t] == s]
        cand = sorted(groups, key=lambda c: (-len(groups[c]), c))
        pre = Prefixes(E, t + 1) if t < H else None
        Q = [groups[cand[0]]]                             # most common candidate is promoted
        tau[(t, cand[0][0]), cand[0][1]] = (t + 1, 0)
        for c in cand[1:]:
            Zc = groups[c]
            home = None
            for j, Zj in enumerate(Q):                    # TestDistinct(t, Zc, Zj, delta)
                thr = threshold(t, H, min(len(Zc), len(Zj)), A, R, O, delta)
                if pre is None or thr > 1 or not pre.at_least(Zc, Zj, thr):
                    home = j                              # 'not distinct'; L-inf is at most 1
                    break
            if home is None:
                Q.append(Zc)
                home = len(Q) - 1
            else:
                Q[home] = np.concatenate((Q[home], Zc))
            tau[(t, c[0]), c[1]] = (t + 1, home)
        Z = Q
        sizes.append(len(Q))
    return Machine(tau, (0, 0), sum(sizes), trajs, A, O, layers=sizes)


# --------------------------------------------------------------------------- FlexFringe
def flexfringe(trajs, A, O, ini=FF_INI, exe=FF_EXE, timeout=1800, extra=()):
    """Run the FlexFringe binary, read its automaton into a Machine. n_states counts every node
    of the final json, sinks included; info gives red and sinks. Symbols with no edge in the
    json count as lost."""
    if not os.path.exists(exe):
        raise FileNotFoundError(f'{exe} missing; see lib/flexfringe')
    E = symbols(trajs, A, O)
    end = (A + 1) * O                                     # termination symbol
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'traces.dat')
        with open(path, 'w') as f:
            f.write(f'{len(E)} {end + 1}\n')
            for e in E.tolist():
                f.write(f'1 {len(e) + 1} ' + ' '.join(map(str, e)) + f' {end}\n')
        run = subprocess.run([exe, '--ini', ini, '--output', 'json', '--outputfile', 'ff', *extra,
                              path], cwd=d, capture_output=True, text=True, timeout=timeout)
        out = os.path.join(d, 'ff.final.json')
        if run.returncode or not os.path.exists(out):
            raise RuntimeError(f'flexfringe failed ({run.returncode}): {run.stdout[-500:]} {run.stderr[-500:]}')
        with open(out) as f:
            g = json.load(f)
    delta = {(int(x['source']), int(x['name'])): int(x['target']) for x in g['edges']}
    nodes = g['nodes']
    return Machine(delta, nodes[0]['id'], len(nodes), trajs, A, O, red=sum(n['isred'] for n in nodes),
                   sinks=sum(n['issink'] for n in nodes))
