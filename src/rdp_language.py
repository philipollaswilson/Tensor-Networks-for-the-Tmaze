"""AdaCT-H with the language metric, the Table 1 'Language metric' column of Deb, Cipollone,
Jonsson, Ronca, Talebi, 'Offline RL in Regular Decision Processes: Sample Efficiency via Language
Metrics', ICLR 2025. Same trajectory format as rdp_baselines; true is never read. Returns an
rdp_baselines.Machine, so scoring is the same as for adact_h and flexfringe.

From the paper, as printed:
  Gamma = AO/R, a trace symbol is a o / r. Trace e_0 .. e_H, a_0 a dummy.
  G1 = {aO/R} u {AO/r} u {Ao/R}
  G2 = G1 u {ao/R} u {aO/r} u {Ao/r}
  G3 = G2 u {ao/r}
  C^l_k(G) = {{x_0 G_1 x_1 .. G_k x_k : |x_0 .. x_k| = l - k} : G_1 .. G_k in G}   (Definition 1)
  X_{i,j} = union over k in [j] of C^l_k(G_i), l = H - t
  L_X(p, p') = max over X in the set of |p(X) - p'(X)|                             (Definition 2)
  distinct iff L_X(Z1, Z2) >= sqrt(2 log(2|X| / delta) / min(|Z1|, |Z2|))          (Theorem 3)
  AdaCT-H as in Appendix A, with only TestDistinct replaced. Lemma 7 and 16 confirm the constant.

So X_{3,1} holds one language per basic pattern, 'the suffix contains at least one symbol
matching it', and |X_{3,1}| = |G3| = (1 + A)(1 + O)(1 + R) - 1, whatever H.

Choices the paper leaves open, or where its text needs reading:
  delta        0.05. The paper prints no value for Table 1. RegORL calls AdaCT-H with
               delta / (4 U A O) on half the data, but that is the planning pipeline; Table 1
               times 'automaton learning' and says nothing about it. Neither is done here.
  R, O         The learner needs the split of o into (code, r), since G1 has separate O and R
               patterns. rdp_envs packs o = code * R + index of r, so the caller passes
               R = len(env.rewards), an alphabet size like A and O: the reward is in the trace
               the paper's learner sees, so this is not a hint. |X| counts the declared alphabet,
               impossible pairs included, as the union bound does. a_0 is not counted: it never
               occurs in a suffix.
  key          The pseudocode's candidates are uao and line 4 strips 'a r o', and tau is on
               U x AO: the reward is NOT read by the automaton. key='ao' (default) does this, so
               a Machine edge covers every r of that (a, code). adact_h in rdp_baselines keys on
               the packed o, which does read r. key='aor' does that here too, for comparing like
               with like. It matters where r is the only sign of an event: in rdp_envs, the
               reset of Cheese and Mini-hall.
  j            j = 1 only: Table 1 uses X_{3,1}. i in 1, 2, 3 are all available.
  occurrence   'x G y' with |xy| = l - 1 is taken as 'at least one occurrence', as Example 2 says.
               A trace with three matches counts once. Example 4 writes |xy| = H - t + 1 for a
               language of X_{2,1}, which does not fit Definition 1 (l - 1 = H - t - 1); I follow
               Definition 1.
  G2 typo      The last set of G2 is printed {Ao/r | a in A, r in R}; read as o in O, r in R, which
               the printed count |G2| = |G1| + AO + AR + OR requires.
  t = H        l = 0, where C^l_k is undefined (k in [l] is empty). No languages, so L_X = 0 and
               every candidate merges: one final state U_{H+1}. Table 1's Corridor 11 needs this.
  order        Same as adact_h: candidates by decreasing count, ties by (state, label); 'element
               in Similar' is the first promoted. The paper says any, and states no loop order.
  merging      Z(u') grows at once (line 11), so later candidates are tested against the merged
               multiset, also for min(|Z1|, |Z2|).
  empty        Candidates with no data are skipped, as in adact_h: they would merge anywhere and
               change no count. Their edge stays undefined and Machine treats it as lost.
  Not stated   dataset size and behaviour policy of Table 1. Figure 2 alone gives K = 100, with
               the East-only policy of Example 4 presumably.
"""
import numpy as np

from rdp_baselines import Machine, symbols


def n_languages(A, O, R, i=3):
    """|G_i| = |X_{i,1}|, O and R the plain alphabet sizes (not the packed O)."""
    return [A + O + R, A + O + R + A * O + A * R + O * R, (1 + A) * (1 + O) * (1 + R) - 1][i - 1]


def threshold(n, n_x, delta=0.05):
    """Right side of TestDistinct, n = min(|Z1|, |Z2|)."""
    return np.sqrt(2 * np.log(2 * n_x / delta) / n)


def patterns(a, o, r, A, O, R, i=3):
    """Ids of the basic patterns of G_i matching symbols a o / r: array (..., m). Dense in
    [0, |G_i|), order a, o, r, ao, ar, or, aor."""
    p = [a, A + o, A + O + r]
    if i >= 2:
        s = A + O + R
        p += [s + a * O + o, s + A * O + a * R + r, s + A * O + A * R + o * R + r]
    if i >= 3:
        p.append(n_languages(A, O, R, 2) + (a * O + o) * R + r)
    return np.stack(p, -1)


def split(E, O, R):
    """Symbols a * O + o, packed o = code * R + r, as (a, code, r)."""
    return E // O, (E % O) // R, (E % O) % R


def language_table(E, A, O, R, i=3):
    """last[k, x] = last position of trace k with a symbol of pattern x, -1 if none. Suffix
    e_{t+1:H} is in the language of x iff last[k, x] > t. Position 0 (the dummy) is left out."""
    a, c, r = split(E, O, R)
    ids = patterns(a, c, r, A, O // R, R, i)
    last = np.full((len(E), n_languages(A, O // R, R, i)), -1, dtype=np.int16)
    rows = np.arange(len(E))[:, None]
    for p in range(1, E.shape[1]):
        last[rows, ids[:, p]] = p
    return last


def language_metric(z1, z2, A, O, R, i=3):
    """L_X of two lists of suffixes of packed symbols, X = X_{i,1}, by definition. Slow, used to
    check the fast one. O is packed."""
    out = 0.0
    for x in range(n_languages(A, O // R, R, i)):
        f = [sum(any(x in patterns(*(int(v) for v in split(e, O, R)), A, O // R, R, i) for e in z)
                 for z in zs) / len(zs) for zs in (z1, z2)]
        out = max(out, abs(f[0] - f[1]))
    return out


def adact_language(trajs, A, O, R, delta=0.05, i=3, j=1, key='ao'):
    """AdaCT-H with L_{X_{i,1}}. A, O as for adact_h (O packed), R = number of reward values. Returns
    a Machine, n_states = |U_0 u .. u U_{H+1}|, info['layers'] the layer sizes. key: see module
    docstring."""
    if j != 1:
        raise NotImplementedError('only X_{i,1}: Table 1 uses j = 1')
    if O % R or key not in ('ao', 'aor'):
        raise ValueError(f'O = {O} not a multiple of R = {R}, or key = {key}')
    E = symbols(trajs, A, O)
    N, H = len(E), E.shape[1] - 1
    last = language_table(E, A, O, R, i)
    n_x = last.shape[1]
    a, c, _ = split(E, O, R)
    lab = E if key == 'aor' else a * (O // R) + c         # what the automaton reads

    def edges(s):                                         # Machine symbols of a label
        if key == 'aor':
            return [s]
        return [(s // (O // R)) * O + (s % (O // R)) * R + r for r in range(R)]

    Z, tau, sizes = [np.arange(N)], {}, [1]
    for t in range(H + 1):
        groups = {}
        for q, idx in enumerate(Z):
            for s in np.unique(lab[idx, t]):
                groups[q, int(s)] = idx[lab[idx, t] == s]
        cand = sorted(groups, key=lambda g: (-len(groups[g]), g))
        P, n = np.zeros((len(cand), n_x)), np.zeros(len(cand))    # p-hat(X) and |Z| per promoted
        members, m = [], 0
        for c_ in cand:
            Zc, nc = groups[c_], len(groups[c_])
            pc = (last[Zc] > t).mean(0)
            home = None
            if m:                                         # TestDistinct against every promoted
                L = np.abs(P[:m] - pc).max(1)
                similar = np.flatnonzero(L < threshold(np.minimum(n[:m], nc), n_x, delta))
                if similar.size:
                    home = int(similar[0])
            if home is None:
                home, m = m, m + 1
                P[home], n[home] = pc, nc
                members.append([Zc])
            else:
                P[home] = (P[home] * n[home] + pc * nc) / (n[home] + nc)
                n[home] += nc
                members[home].append(Zc)
            for s in edges(c_[1]):
                tau[(t, c_[0]), s] = (t + 1, home)
        Z = [np.concatenate(z) for z in members]
        sizes.append(m)
    return Machine(tau, (0, 0), sum(sizes), trajs, A, O, layers=sizes, n_languages=n_x, key=key)
