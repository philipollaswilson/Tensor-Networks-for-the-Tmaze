"""Figures: from raw T-maze trajectories to hidden variables (full pymdp T-maze).

The agent's data are 3-step trajectories (o0, a1, o1, a2, o2) of 24 composite
observations (location x reward x cue bit), sampled from the true process under a
uniform policy. Nothing about locations, contexts or states is given to the learner.

Figure 1 (fig_hidden_states.png): the pipeline
  (a) the data: raw trajectories
  (b) the prediction table H[h, (a2, o2)] = p(o2 | h, a2), one row per distinct
      history h = (o0, a1, o1); rows that predict alike are one hidden state
  (c) singular values of H from data and from the trained MPS: rank 6
  (d) each history as a point (H projected on its principal axes); 24 histories land
      on 7 points; the centre sits halfway between the two cue states (a belief)
  (e) Bayesian model reduction: evidence of every merge, 24 histories down to 7 states

Figure 2 (fig_hidden_states_data.png): how much data, and what the states mean
  (a) the history cloud at growing data sizes
  (b) states found and agreement with the truth (ARI) against data size, 20 seeds
  (c) each learned state as a belief over the true hidden variables
      (location x context, pymdp's 8 states)
  (d) the learned state machine laid out on the maze

Projected on the label set position + reward (the cue bit is noise away from the cue).
BMR uses the Perks prior alpha = 1/12 (total prior mass 1 per action, as
structure_learner.py). Uses Saved_Models/FullTmaze.pt for the MPS rows.
Run from the repo root:  python src/make_fig_hidden_states.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bmr_states import bmr  # noqa: E402
from observable_embedding import LAB_FULL, load  # noqa: E402
import seed_protocol as SP  # noqa: E402
from make_extension_figures import FIGS  # noqa: E402

LAB = LAB_FULL['pos+rew']                   # 12 labels: 3 * position + reward
NL = 12
ALPHA = 1.0 / NL
N_TRAIN = int(0.8 * 4096)
LOC = ['Centre', 'Right', 'Left', 'Cue']
LOC_SHORT = ['C', 'R', 'L', 'Cue']
ACT = ['stay', 'right', 'left', 'cue']
RW = ['', 'cheese', 'shock']
COL = {0: '#898781', 1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a'}   # by location
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'

# the seven predictive states: (location, reward, cue bit or None), name, p(cheese on right)
STATES = [((0, 0, None), 'Centre', 0.5),
          ((3, 0, 0), 'Cue says right', 1.0),
          ((3, 0, 1), 'Cue says left', 0.0),
          ((1, 1, None), 'Right arm, cheese', 1.0),
          ((1, 2, None), 'Right arm, shock', 0.0),
          ((2, 1, None), 'Left arm, cheese', 0.0),
          ((2, 2, None), 'Left arm, shock', 1.0)]


def true_state(o1):
    p, r, c = o1 // 6, (o1 // 2) % 3, o1 % 2
    for k, ((lp, lr, lc), _, _) in enumerate(STATES):
        if p == lp and r == lr and (lc is None or lc == c):
            return k
    raise ValueError(o1)


# ------------------------------------------------------------ data and tables
LEAVES = SP.FULL.enumerate_weighted()
W = np.array([w for _, w in LEAVES], float)
W /= W.sum()
HIST = sorted({(o0[2], a1, SP.obs_index(*o1)) for (_, o0, a1, o1, _, _), _ in LEAVES})


def sample(n, seed):
    rng = np.random.default_rng(seed)
    return [LEAVES[i][0] for i in rng.choice(len(LEAVES), size=n, p=W)]


def counts_of(trajs):
    n = np.zeros((len(HIST), 4, NL))
    ix = {h: i for i, h in enumerate(HIST)}
    for (_, o0, a1, o1, a2, o2) in trajs:
        n[ix[(o0[2], a1, SP.obs_index(*o1))], a2, LAB[SP.obs_index(*o2)]] += 1
    return n


def rows(n):
    """Prediction rows p(label | h, a), flattened; unseen (h, a) get NaN."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return (n / n.sum(-1, keepdims=True)).reshape(len(n), -1)


def exact_rows():
    n = np.zeros((len(HIST), 4, NL))
    ix = {h: i for i, h in enumerate(HIST)}
    for (_, o0, a1, o1, a2, o2), w in LEAVES:
        n[ix[(o0[2], a1, SP.obs_index(*o1))], a2, LAB[SP.obs_index(*o2)]] += w
    return rows(n)


def mps_rows():
    T1, T2, T3 = load('FullTmaze')
    out = []
    for c0, a1, o1 in HIST:
        b = T1[0, 0, SP.obs_index(0, 0, c0), :] @ T2[:, a1, o1, :]
        amp = np.abs(np.einsum('j,jaq->aq', b, T3[:, :, :, 0])) ** 2
        p = np.zeros((4, NL))
        for o in range(24):
            p[:, LAB[o]] += amp[:, o]
        out.append((p / p.sum(1, keepdims=True)).ravel())
    return np.array(out)


def learn(n):
    """BMR over the histories seen; returns a state label per history (-1 unseen)."""
    seen = np.flatnonzero(n.sum((1, 2)) > 0)
    cl, acc, rej = bmr([n[i] for i in seen], ALPHA)
    lab = -np.ones(len(n), int)
    for k, c in enumerate(cl):
        lab[seen[c]] = k
    return lab, acc, rej


def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    M = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(M, (ai, bi), 1)
    c2 = lambda x: x * (x - 1) / 2  # noqa: E731
    s, sa, sb, n = c2(M).sum(), c2(M.sum(1)).sum(), c2(M.sum(0)).sum(), c2(len(a))
    e = sa * sb / n
    return float((s - e) / (0.5 * (sa + sb) - e)) if sa + sb > 2 * e else 1.0


TRUTH = np.array([true_state(o1) for _, _, o1 in HIST])


# ------------------------------------------------------------ embedding
def dist(x, Y):
    """L2 distance between prediction rows, over the actions both have seen (rescaled
    to all four actions); unseen (h, a) are NaN."""
    x, Y = x.reshape(4, NL), Y.reshape(len(Y), 4, NL)
    ok = ~np.isnan(x[:, 0])[None] & ~np.isnan(Y[:, :, 0])
    d2 = np.where(ok[..., None], (x[None] - Y) ** 2, 0).sum((1, 2))
    return np.sqrt(4 * d2 / np.maximum(ok.sum(1), 1))


def stress(z, Y, d):
    return ((np.linalg.norm(Y - z, axis=1) - d) ** 2).sum()


class Map2D:
    """Metric MDS (SMACOF) of the learned states' prediction rows; any other row is
    placed by its distances to them. Rotated so that the two cue states are vertical
    (display only: rotations keep every distance)."""

    def __init__(self, anchors, k_of_anchor, seed=0):
        A = np.array(anchors)
        D = np.array([dist(a, A) for a in A])
        rng, best = np.random.default_rng(seed), None
        for _ in range(30):
            Z = rng.normal(size=(len(A), 2))
            for _ in range(500):                  # Guttman transform
                E = np.linalg.norm(Z[:, None] - Z[None], axis=-1)
                B = -np.where(E > 1e-12, D / np.maximum(E, 1e-12), 0)
                B[np.diag_indices(len(A))] = -B.sum(1)
                Z = B @ Z / len(A)
            E = np.linalg.norm(Z[:, None] - Z[None], axis=-1)
            s = ((E - D) ** 2).sum() / 2
            if best is None or s < best[0]:
                best = (s, Z)
        Z = best[1]
        ix = {k: i for i, k in enumerate(k_of_anchor)}
        Z = Z - Z[ix[0]]                           # centre state at the origin
        v = Z[ix[1]] - Z[ix[2]]                    # cue says right, up
        th = np.arctan2(v[0], v[1])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Z = Z @ R.T
        if Z[ix[3], 0] < Z[ix[5], 0]:              # right arm on the right
            Z[:, 0] *= -1
        self.A, self.Z, self.D = A, Z, D
        self.raw_stress = best[0] / (D ** 2).sum() * 2
        self.anchor_of = ix

    def place(self, H):
        from scipy.optimize import minimize
        out = []
        for x in H:
            if np.isnan(x).all():
                out.append((np.nan, np.nan)); continue
            d = dist(x, self.A)
            z0 = self.Z[np.argmin(d)] + 0.01
            out.append(minimize(stress, z0, args=(self.Z, d)).x)
        return np.array(out)

    def state(self, k):
        return self.Z[self.anchor_of[k]]


# ------------------------------------------------------------ the learned model
def full_counts_of(trajs):
    """As counts_of, with the full 24-symbol next observation (for transitions)."""
    n = np.zeros((len(HIST), 4, 24))
    ix = {h: i for i, h in enumerate(HIST)}
    for (_, o0, a1, o1, a2, o2) in trajs:
        n[ix[(o0[2], a1, SP.obs_index(*o1))], a2, SP.obs_index(*o2)] += 1
    return n


def learned_model(trajs):
    """BMR states from the data, read out as a model:
       rows      pooled prediction row per state
       k_of      the true predictive state each learned state matches (for naming only)
       T         p(next state | state, a2): a next observation o2 leads to the state
                 whose histories ended in o2
       belief    p(location, context | state): the true process's posterior for each
                 member history, averaged over members by visits"""
    n, nf = counts_of(trajs), full_counts_of(trajs)
    lab, acc, rej = learn(n)
    K = lab.max() + 1
    vis = n.sum((1, 2))
    k_of = [np.bincount(TRUTH[lab == c], weights=vis[lab == c], minlength=len(STATES)).argmax()
            for c in range(K)]
    pooled = np.array([n[lab == c].sum(0) for c in range(K)])
    state_of_obs = {}
    for i, (_, _, o1) in enumerate(HIST):
        if lab[i] >= 0:
            state_of_obs.setdefault(o1, lab[i])
    T = np.zeros((K, 4, K))
    for c in range(K):
        m = nf[lab == c].sum(0)
        for o, s in state_of_obs.items():
            T[c, :, s] += m[:, o]
    T /= np.maximum(T.sum(-1, keepdims=True), 1e-12)
    belief = np.zeros((K, 8))
    for i in np.flatnonzero(lab >= 0):
        (loc, _, _), _, pr = STATES[TRUTH[i]]
        belief[lab[i], 2 * loc] += vis[i] * pr
        belief[lab[i], 2 * loc + 1] += vis[i] * (1 - pr)
    belief /= belief.sum(1, keepdims=True)
    return dict(n=n, lab=lab, acc=acc, rej=rej, K=K, k_of=k_of, rows=rows(pooled), T=T,
                belief=belief)


# ------------------------------------------------------------ style
def clean(ax, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(colors=MUTED, labelsize=7.5)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def title(ax, text, pad=6):
    ax.set_title(text, fontsize=9.5, color=INK, loc='left', pad=pad)


def marker(k, ax, x, y, size=150, alpha=1.0, z=3):
    """Hue = location; filled = cheese on the right, hollow = cheese on the left,
    diamond = undecided (the centre)."""
    (loc, _, _), _, pr = STATES[k]
    c = COL[loc]
    if pr == 0.5:
        ax.scatter(x, y, s=size, marker='D', color=c, edgecolor='white', lw=1.5, alpha=alpha, zorder=z)
    elif pr == 1.0:
        ax.scatter(x, y, s=size, marker='o', color=c, edgecolor='white', lw=1.5, alpha=alpha, zorder=z)
    else:
        ax.scatter(x, y, s=size, marker='o', facecolor='white', edgecolor=c, lw=2.2, alpha=alpha, zorder=z)


def legend_handles():
    h = []
    for (loc, _, _), name, pr in STATES:
        if pr == 0.5:
            h.append(Line2D([], [], marker='D', ls='', color=COL[loc], ms=7, label=name))
        elif pr == 1.0:
            h.append(Line2D([], [], marker='o', ls='', color=COL[loc], ms=8, label=name))
        else:
            h.append(Line2D([], [], marker='o', ls='', markerfacecolor='white',
                            markeredgecolor=COL[loc], mew=2, ms=8, label=name))
    return h


def obs_text(o):
    p, r, c = o // 6, (o // 2) % 3, o % 2
    return LOC_SHORT[p] + (f' {RW[r][:2]}' if r else '') + f' ·{c}'


def hist_text(h):
    c0, a1, o1 = h
    return f'C ·{c0}  {ACT[a1]:>5s} → {obs_text(o1):8s}'


# ------------------------------------------------------------ figure 1 panels
def panel_data(ax, trajs, n_show=28):
    ax.set_xlim(0, 5); ax.set_ylim(n_show + 0.6, -1.4)
    for j, t in enumerate(['o₀', 'a₁', 'o₁', 'a₂', 'o₂']):
        ax.text(j + 0.5, -0.8, t, ha='center', va='center', fontsize=9, color=INK, weight='bold')
    for i, (_, o0, a1, o1, a2, o2) in enumerate(trajs[:n_show]):
        for j, item in enumerate([o0, a1, o1, a2, o2]):
            if j % 2 == 0:
                o = SP.obs_index(*item)
                ax.add_patch(Rectangle((j + 0.03, i + 0.06), 0.94, 0.88, color=COL[o // 6],
                                       alpha=0.22, lw=0))
                ax.text(j + 0.5, i + 0.5, obs_text(o), ha='center', va='center', fontsize=6.3,
                        color=INK, family='monospace')
            else:
                ax.text(j + 0.5, i + 0.5, ACT[item], ha='center', va='center', fontsize=6.3,
                        color=MUTED, family='monospace')
    ax.text(2.5, n_show + 0.4, f'… {N_TRAIN} trajectories in all', ha='center', va='top',
            fontsize=7.5, color=MUTED)
    ax.axis('off')


def panel_table(ax, H, lab):
    order = np.lexsort((np.arange(len(HIST)), [m['k'] for m in lab]))
    cols = np.flatnonzero(np.nan_to_num(H).sum(0) > 0)
    M = np.nan_to_num(H[order][:, cols])
    ax.imshow(M, cmap='Greys', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    acts = cols // NL
    for b in np.flatnonzero(np.diff(acts)) + 0.5:
        ax.axvline(b, color=INK, lw=0.8)
    for a in range(4):
        idx = np.flatnonzero(acts == a)
        ax.text(idx.mean(), -0.9, f'a₂ = {ACT[a]}', ha='center', va='bottom', fontsize=7.5,
                color=INK, weight='bold')
    ks = np.array([lab[i]['k'] for i in order])
    for b in np.flatnonzero(np.diff(ks)) + 0.5:
        ax.axhline(b, color='white', lw=3)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([LOC_SHORT[(c % NL) // 3] + ('' if c % 3 == 0 else ' ' + RW[c % 3][:2])
                        for c in cols], rotation=90, fontsize=6.5, color=MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([hist_text(HIST[i]) for i in order], fontsize=6, family='monospace',
                       color=MUTED)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for r, i in enumerate(order):                 # learned state, as in (e)
        marker(lab[i]['k'], ax, -1.25, r, size=38)
    for k in np.unique(ks):
        mid = np.flatnonzero(ks == k).mean()
        ax.text(len(cols) - 0.2, mid, STATES[k][1], fontsize=7.5, color=INK, va='center')
    ax.set_xlim(-1.8, len(cols) - 0.5)


def panel_spectrum(ax, H_data, H_mps):
    for H, name, c, m in [(H_data, f'data ({N_TRAIN} trajectories)', '#2a78d6', 'o'),
                          (H_mps, 'trained MPS', '#eb6834', 's')]:
        s = np.linalg.svd(np.nan_to_num(H), compute_uv=False)
        s = s / s[0]
        ax.semilogy(np.arange(1, len(s) + 1), np.maximum(s, 1e-12), m + '-', color=c, ms=5,
                    lw=1.5, label=name)
    ax.axvline(6.5, color=MUTED, lw=0.8, ls=':')
    ax.text(6.3, 3e-6, 'rank 6', fontsize=8.5, color=INK, ha='right', weight='bold')
    ax.annotate('sampling noise', (7, 0.03), xytext=(10, 4), textcoords='offset points',
                fontsize=7, color=MUTED)
    ax.set_ylim(1e-7, 2.5); ax.set_xlim(0.5, 12.5)
    ax.set_xlabel('singular value index', fontsize=8, color=MUTED)
    ax.set_ylabel('singular value / largest', fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    clean(ax)


def panel_bmr(ax, acc, rej, n_seen):
    d = list(acc) + [rej]
    k_after = np.arange(n_seen - 1, n_seen - 1 - len(d), -1)
    ax.bar(k_after[:-1], acc, color='#2a78d6', width=0.7, label='merge accepted')
    ax.bar(k_after[-1:], [rej], color='#eb6834', width=0.7, label='best merge left: refused')
    ax.axhline(0, color=INK, lw=0.8)
    ax.invert_xaxis()
    ax.set_xticks(k_after[-1::-4])
    ax.set_xlabel('states if the merge is made', fontsize=8, color=MUTED)
    ax.set_ylabel('evidence change of the merge, ΔF (nats)', fontsize=8, color=MUTED)
    ax.text(k_after[-1] + 0.8, rej * 0.55, f'ΔF = {rej:.0f}\nstop at {k_after[-1] + 1}',
            fontsize=8, color=INK, ha='right', va='center')
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    clean(ax)


def panel_map(ax, emb, P, vis, Pm=None, names=True, pad=(0.55, 0.35)):
    c_r, c_l = emb.state(1), emb.state(2)
    ax.plot(*np.array([c_r, c_l]).T, color=MUTED, lw=0.8, ls=':', zorder=0)
    for k in range(len(STATES)):
        marker(k, ax, *emb.state(k), size=700, alpha=0.16, z=1)
    for i in np.argsort(-vis):
        if vis[i] > 0 and np.isfinite(P[i]).all():
            marker(TRUTH[i], ax, *P[i], size=14 + 2.0 * vis[i] ** 0.7, alpha=0.85)
    if Pm is not None:
        ax.scatter(Pm[:, 0], Pm[:, 1], marker='+', s=70, color=INK, lw=1.1, zorder=5)
    if names:
        for k, (_, name, _) in enumerate(STATES):
            x, y = emb.state(k)
            n_h = int((TRUTH == k).sum())
            right = x >= 0 if k > 2 else True
            ax.annotate(f'{name}\n{n_h} histories', (x, y), xytext=(16 if right else -16, 0),
                        textcoords='offset points', fontsize=7.3, color=INK, va='center',
                        ha='left' if right else 'right')
    lo, hi = emb.Z.min(0), emb.Z.max(0)
    span = hi - lo
    ax.set_xlim(lo[0] - pad[0] * span[0], hi[0] + pad[0] * span[0])
    ax.set_ylim(lo[1] - pad[1] * span[1], hi[1] + pad[1] * span[1])
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


# ------------------------------------------------------------ figure 2 panels
def panel_beliefs(ax, model):
    order = np.argsort(model['k_of'])
    M = model['belief'][order]
    ax.imshow(M, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for r in range(len(M)):
        for j in range(8):
            if M[r, j] > 0.005:
                ax.text(j, r, f'{M[r, j]:.2f}', ha='center', va='center', fontsize=6.8,
                        color='white' if M[r, j] > 0.6 else INK)
    for b in (1.5, 3.5, 5.5):
        ax.axvline(b, color='white', lw=3)
    ax.set_xticks(range(8))
    ax.set_xticklabels(['ch R', 'ch L'] * 4, fontsize=6.8, color=MUTED)
    for loc in range(4):
        ax.text(2 * loc + 0.5, -0.75, LOC[loc], ha='center', va='bottom', fontsize=7.5,
                color=INK, weight='bold')
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([STATES[model['k_of'][c]][1] for c in order], fontsize=7.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


MACHINE_POS = {0: (0, 0), 1: (0, 1.3), 2: (0, -1.3), 3: (2.6, 1.3), 4: (2.6, -1.3),
               5: (-2.6, -1.3), 6: (-2.6, 1.3)}


def panel_machine(ax, model):
    """Learned states placed by location (x) and context (y); arrows: learned
    transitions with p >= 0.05, self-loops left out (the arms absorb)."""
    T, k_of = model['T'], model['k_of']
    for y, txt in [(1.3, 'context:\ncheese right'), (0, 'context\nunknown'), (-1.3, 'context:\ncheese left')]:
        ax.text(-3.6, y, txt, fontsize=7.5, color=MUTED, ha='center', va='center')
    for x, txt in [(-2.6, 'left arm'), (0, 'centre / cue'), (2.6, 'right arm')]:
        ax.text(x, 2.25, txt, fontsize=8, color=INK, ha='center', weight='bold')
    drawn = {}
    for c in range(model['K']):
        for a in range(4):
            for c2 in np.flatnonzero(T[c, a] >= 0.05):
                if c2 == c:
                    continue
                drawn.setdefault((c, c2), []).append((ACT[a], T[c, a, c2]))
    for (c, c2), acts in drawn.items():
        p = max(q for _, q in acts)
        (x0, y0), (x1, y1) = MACHINE_POS[k_of[c]], MACHINE_POS[k_of[c2]]
        sure = p > 0.9
        rad = 0.5 if x0 == x1 else 0.15           # opposite pairs on one column bow apart
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=10,
                                     shrinkA=13, shrinkB=13, lw=1.6 if sure else 0.9,
                                     ls='-' if sure else (0, (3, 2)), color=INK if sure else MUTED,
                                     connectionstyle=f'arc3,rad={rad}', zorder=2))
        # midpoint of the arc3 curve: chord midpoint + rad/2 * (dy, -dx)
        mx = (x0 + x1) / 2 + rad / 2 * (y1 - y0)
        my = (y0 + y1) / 2 - rad / 2 * (x1 - x0)
        label = ', '.join(sorted({a for a, _ in acts})) + ('' if sure else f' {p:.2f}')
        ax.text(mx, my, label, fontsize=6.8, color=INK if sure else MUTED, ha='center',
                va='center', zorder=3, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))
    for c in range(model['K']):
        k = k_of[c]
        x, y = MACHINE_POS[k]
        marker(k, ax, x, y, size=380, z=4)
        if y > 0:
            ax.text(x, y + 0.3, STATES[k][1], fontsize=7.5, color=INK, ha='center', va='bottom', zorder=5)
        elif y < 0:
            ax.text(x, y - 0.3, STATES[k][1], fontsize=7.5, color=INK, ha='center', va='top', zorder=5)
        else:
            ax.text(x + 0.35, y, STATES[k][1], fontsize=7.5, color=INK, ha='left', va='center', zorder=5)
    ax.set_xlim(-4.2, 3.2); ax.set_ylim(-2.3, 2.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.6, label='certain (p > 0.9)'),
                       Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)), label='uncertain: action p')],
              fontsize=7, frameon=False, loc='lower center', bbox_to_anchor=(0.55, -0.08), ncol=2)


# ------------------------------------------------------------ figures
def figure1(trajs, H_mps, model, emb):
    n = model['n']
    H = rows(n)
    lab = [{'k': model['k_of'][c]} for c in model['lab']]
    vis = n.sum((1, 2))

    fig = plt.figure(figsize=(19.5, 10))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.72, 1.75, 1.0, 1.0], wspace=0.62, hspace=0.36)
    ax = fig.add_subplot(gs[:, 0]); panel_data(ax, trajs)
    title(ax, '(a) The data: raw trajectories\n(location, reward, cue bit)')
    ax = fig.add_subplot(gs[:, 1]); panel_table(ax, H, lab)
    title(ax, '(b) Prediction table: a row per history, p(o₂ | history, a₂).\n'
              '24 distinct histories; rows that predict alike are one hidden state', pad=20)
    ax = fig.add_subplot(gs[0, 2]); panel_spectrum(ax, H, H_mps)
    title(ax, '(c) Singular values of the table:\n6 independent predictions')
    ax = fig.add_subplot(gs[0, 3]); panel_bmr(ax, model['acc'], model['rej'], int((vis > 0).sum()))
    title(ax, f'(d) Bayesian model reduction: merge\nwhile the evidence rises → {model["K"]} states')
    ax = fig.add_subplot(gs[1, 2:])
    panel_map(ax, emb, emb.place(H), vis, emb.place(H_mps))
    title(ax, '(e) Every history as a point: a 2-D map (metric MDS) of 6-D prediction rows, so\n'
              f'distances are approximate (stress {emb.raw_stress:.2f}). 24 histories fall on 7 points')
    ax.legend(handles=legend_handles() + [Line2D([], [], marker='+', ls='', color=INK, ms=8,
                                                 label='MPS prediction')],
              fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.suptitle('From trajectories to hidden states, full T-maze (pymdp). The learner is given '
                 'only symbols: no locations, contexts or states', fontsize=12, color=INK,
                 x=0.1, ha='left', y=0.985)
    out = FIGS / 'fig_hidden_states.png'
    fig.savefig(out, bbox_inches='tight', dpi=170, facecolor='white')
    plt.close(fig)
    print(f'wrote {out}  (BMR: {model["K"]} states, ARI {ari(model["lab"], TRUTH):.3f}, '
          f'refused dF {model["rej"]:.1f}, MDS stress {emb.raw_stress:.3f})')


def figure2(model, emb, seeds=20):
    sizes = [16, 64, 256, N_TRAIN]
    fig = plt.figure(figsize=(19, 9.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], wspace=0.28, hspace=0.36)
    low = gs[1, :].subgridspec(1, 4, width_ratios=[0.7, 0.7, 0.85, 1.6], wspace=0.5)
    for j, N in enumerate(sizes):
        ax = fig.add_subplot(gs[0, j])
        n = counts_of(sample(N, 0))
        lab, _, _ = learn(n)
        seen = n.sum((1, 2)) > 0
        panel_map(ax, emb, emb.place(rows(n)), n.sum((1, 2)), names=False, pad=(0.12, 0.12))
        title(ax, f'{N} trajectories: {seen.sum()} of 24 histories seen\n'
                  f'BMR → {lab.max() + 1} states, ARI {ari(lab[seen], TRUTH[seen]):.2f}')
    fig.text(0.125, 0.965, '(a) The history cloud as data grow (seed 0). Discs: the learned '
             'states from all data; dots: histories, size = visits', fontsize=10, color=INK)

    Ns = np.unique(np.round(np.geomspace(8, N_TRAIN, 14)).astype(int))
    K, A = np.zeros((seeds, len(Ns))), np.zeros((seeds, len(Ns)))
    for s in range(seeds):
        for j, N in enumerate(Ns):
            n = counts_of(sample(N, 100 + s))
            lab, _, _ = learn(n)
            seen = n.sum((1, 2)) > 0
            K[s, j], A[s, j] = lab.max() + 1, ari(lab[seen], TRUTH[seen])
    for col, (Y, c, ylab, ylim, ttl) in enumerate([
            (A, '#2a78d6', 'agreement with the true states (ARI)', (-0.05, 1.05),
             '(b) Learned vs true states'),
            (K, '#2a78d6', 'states found', (0, 9), '(c) Number of states')]):
        ax = fig.add_subplot(low[col])
        ax.fill_between(Ns, np.percentile(Y, 25, 0), np.percentile(Y, 75, 0), color=c, alpha=0.15, lw=0)
        ax.plot(Ns, np.median(Y, 0), 'o-', color=c, lw=2, ms=4)
        if Y is K:
            ax.axhline(7, color=MUTED, ls=':', lw=1)
            ax.text(Ns[-1], 7.2, 'true: 7', fontsize=7.5, color=MUTED, ha='right')
        ax.set_xscale('log'); ax.set_ylim(*ylim)
        ax.set_xlabel('training trajectories', fontsize=8, color=MUTED)
        ax.set_ylabel(ylab + f'\nmedian and IQR, {seeds} seeds', fontsize=8, color=MUTED)
        clean(ax)
        title(ax, ttl)
    print('N      ', Ns)
    print('states ', np.median(K, 0))
    print('ARI    ', np.round(np.median(A, 0), 2))

    ax = fig.add_subplot(low[2]); panel_beliefs(ax, model)
    title(ax, '(d) What each learned state is:\np(location, context | learned state)', pad=26)
    ax = fig.add_subplot(low[3]); panel_machine(ax, model)
    title(ax, '(e) The learned transitions, states placed by location and context.\n'
              'Visiting the cue splits the centre into the two contexts')
    out = FIGS / 'fig_hidden_states_data.png'
    fig.savefig(out, bbox_inches='tight', dpi=170, facecolor='white')
    plt.close(fig)
    print(f'wrote {out}')


def main():
    trajs = sample(N_TRAIN, 0)
    model = learned_model(trajs)
    assert sorted(model['k_of']) == list(range(len(STATES))), model['k_of']
    emb = Map2D(model['rows'], model['k_of'])
    figure1(trajs, mps_rows(), model, emb)
    figure2(model, emb)


if __name__ == '__main__':
    main()

