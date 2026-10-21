"""Figure: observation-to-hidden-state mapping recovered by structure learning.

One row per hidden state that Bayesian model reduction selects on the learned MPS
(seed 0, alpha = 0.1). Left block: which action-observation histories the state is
inferred from (a filled cell assigns the history to the state). Middle block: what
the state predicts, p(next observation | state, next action), from the learned model.
Right column: the largest total-variation distance between a member history's learned
prediction and the analytic one, so every state can be checked against ground truth.
Top: minimal maze (4 states). Bottom: pymdp T-maze over position x reward (7 states).

Run from the repo root (with the seed-protocol models in Saved_Models/):
  python paper_mm/make_fig_states.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, 'src')]
from bmr_states import bmr, lnB, LAB_FULL, FULL, MIN, oi, load  # noqa: E402

SEED, ALPHA = int(os.environ.get('FIGSEED', 0)), 0.1
SURFACE, INK, MUTED, GRID = '#ffffff', '#1f1f1e', '#73726c', '#e4e3df'
SEQ = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
CMAP = plt.get_cmap('hot_r')
plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans', 'text.color': INK,
                     'xtick.color': INK, 'ytick.color': INK})


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(-1).max()


# ------------------------------------------------------------------ minimal maze
def minimal():
    T1, T2 = load(f'seedprot_minimal_s{SEED}')
    q = MIN.exact_joint()
    seqs = MIN.enumerate_sequences()
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(seqs), size=900)
    visits = {}
    for a1, o2, _, _ in (seqs[i] for i in draws[:720]):
        visits[(a1, o2)] = visits.get((a1, o2), 0) + 1
    keys = sorted(visits)
    P = np.abs(np.einsum('xaoi,ibpy->aobp', T1, T2)) ** 2; P /= P.sum()   # learned joint
    ph = [P[a1, o2].sum() for a1, o2 in keys]                            # learned p(history)
    hist, learned, analytic, counts = [], [], [], []
    for a1, o2 in keys:
        b = T1[:, a1, o2, :].sum(0)
        p = np.abs(np.einsum('i,ibp->bp', b, T2[:, :, :, 0])) ** 2
        p /= p.sum(1, keepdims=True)
        t = q[a1, o2] / q[a1, o2].sum(1, keepdims=True)
        hist.append(f'{MIN.ACT[a1]} → {MIN.OBS[o2]}')
        learned.append(p); analytic.append(t)
        counts.append(p * visits[(a1, o2)] / 3)
    cl, _, _ = bmr(counts, ALPHA)
    first_obs = lambda c: min(keys[i][1] for i in c)
    cl.sort(key=first_obs)
    names = {0: 'cheese trap', 1: 'shock trap', 2: 'cue: cheese right', 3: 'cue: cheese left'}
    states = [names[first_obs(c)] for c in cl]
    return dict(hist=hist, learned=learned, analytic=analytic, clusters=cl, states=states, counts=counts, ph=ph,
                acts=['Right', 'Left', 'Cue'], obs=['Cheese', 'Shock', 'See R', 'See L'])


# ------------------------------------------------------------------ full maze
def full():
    lab = LAB_FULL['pos+rew']
    T1, T2, T3 = load(f'seedprot_full_s{SEED}')
    leaves = FULL.enumerate_weighted()
    w = np.array([x[1] for x in leaves], float); w /= w.sum()
    q = FULL.exact_joint(leaves)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(len(leaves), size=4096, p=w)
    visits = {}
    for l in (leaves[i][0] for i in draws[:int(0.8 * 4096)]):
        k = (l[1][2], l[2], oi(*l[3]))
        visits[k] = visits.get(k, 0) + 1
    keys = sorted(visits)
    P = np.abs(np.einsum('xaoi,ibpj,jcqy->aobpcq', T1, T2, T3)) ** 2; P /= P.sum()   # learned joint
    ph = [P[0, oi(0, 0, c0), a1, o1].sum() for c0, a1, o1 in keys]                  # learned p(history)
    cols = [0, 4, 5, 7, 8, 9]                 # (centre,-) (R,cheese) (R,shock) (L,cheese) (L,shock) (cue,-)

    def to_lab(m):                             # (4 actions, 24 obs) -> (4, 12 pos x reward)
        out = np.zeros((4, 12))
        for o in range(24):
            out[:, lab[o]] += m[:, o]
        return out / out.sum(1, keepdims=True)

    pos, rew, act = ['centre', 'R', 'L', 'cue'], ['', ' cheese', ' shock'], ['centre', 'right', 'left', 'cue']
    hist, learned, analytic, counts = [], [], [], []
    for c0, a1, o1 in keys:
        b = np.einsum('i,ij->j', T1[0, 0, oi(0, 0, c0), :], T2[:, a1, o1, :])
        p = to_lab(np.abs(np.einsum('j,jaq->aq', b, T3[:, :, :, 0])) ** 2)
        t = to_lab(q[0, oi(0, 0, c0), a1, o1])
        assert np.delete(p, cols, 1).max() < 1e-3 and np.delete(t, cols, 1).max() < 1e-12
        if o1 // 6 == 3:                       # at the cue the context bit is the cue reading
            hist.append(f'cue → reads context {o1 % 2}')
        else:                                  # elsewhere the context bits are uninformative
            hist.append(f'{act[a1]} → {pos[o1 // 6]}{rew[(o1 // 2) % 3]}')
        learned.append(p[:, cols]); analytic.append(t[:, cols])
        counts.append(p * visits[(c0, a1, o1)] / 4)
    cl, _, _ = bmr(counts, ALPHA)
    order = lambda c: (min(keys[i][2] // 6 for i in c) % 3 if keys[c[0]][2] // 6 != 3 else 3,
                       (keys[c[0]][2] // 2) % 3, keys[c[0]][2] % 2)
    cl.sort(key=order)

    def name(c):
        o1 = keys[c[0]][2]
        p, r = o1 // 6, (o1 // 2) % 3
        return {0: 'centre (unresolved)', 3: f'cue: context {o1 % 2}'}.get(p, f'{pos[p]} arm,{rew[r]}')
    return dict(hist=hist, learned=learned, analytic=analytic, clusters=cl, counts=counts, ph=ph,
                states=[name(c) for c in cl], acts=act,
                obs=['centre', 'R cheese', 'R shock', 'L cheese', 'L shock', 'cue'])


# ------------------------------------------------------------------ drawing
def title_tag(d):
    return f'seed {SEED} {len(d["hist"])} histories'


def panel(fig, rect, d, title):
    cl, nS = d['clusters'], len(d['clusters'])
    order = [i for c in cl for i in sorted(c, key=lambda i: d['hist'][i])]  # by state, then label
    nH, nA, nO = len(order), len(d['acts']), len(d['obs'])
    # p(state | history): Dirichlet-categorical evidence of the history's outcome counts under each
    # state's pooled counts (the history's own counts left out of its state), uniform prior over states
    n = d['counts']
    a0 = np.full(n[0].shape[1], ALPHA)
    pooled = [sum(n[i] for i in c) for c in cl]
    assign = np.zeros((nS, nH))
    for col, i in enumerate(order):
        logev = np.array([(lnB(a0 + pooled[s] - (n[i] if i in cl[s] else 0) + n[i])
                           - lnB(a0 + pooled[s] - (n[i] if i in cl[s] else 0))).sum() for s in range(nS)])
        w = np.exp(logev - logev.max())
        assign[:, col] = w / w.sum()
    post = assign                                                     # p(state | history), reported
    # plotted: the hard mapping (each history's posterior on its own state exceeds 1 - 1e-22)
    assign = (post > 0.5).astype(float)                              # the mapping: history -> its state
    print(f'seed {SEED} {nH} histories: max p(other state | history) = {max(post[s, j] for s, c in enumerate(cl) for j in range(nH) if order[j] not in c):.2e}')
    pred = np.zeros((nS, nA * nO)); err = np.zeros(nS)
    for s, c in enumerate(cl):
        pred[s] = np.mean([d['learned'][i] for i in c], 0).ravel()
        err[s] = max(tv(d['learned'][i], d['analytic'][i]) for i in c)

    x0, y0, W, H = rect
    gap = 0.045
    wa = W * nH / (nH + nA * nO + 3)
    wp = W * nA * nO / (nH + nA * nO + 3)
    ax1 = fig.add_axes([x0, y0, wa, H]); ax2 = fig.add_axes([x0 + wa + gap, y0, wp, H])
    ax3 = fig.add_axes([x0 + wa + wp + gap + 0.03, y0, 0.066, H])

    ax1.imshow(assign, cmap=LinearSegmentedColormap.from_list('map', [SURFACE, INK]), vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    ax1.set_yticks(range(nS)); ax1.set_yticklabels(d['states'], fontsize=7)
    labels = [d['hist'][i] for i in order]                          # one tick per distinct history
    ticks, tlab, j = [], [], 0
    while j < nH:
        k = j
        while k + 1 < nH and labels[k + 1] == labels[j]:
            k += 1
        ticks.append((j + k) / 2); tlab.append(labels[j] + (f'  (×{k - j + 1})' if k > j else ''))
        j = k + 1
    ax1.set_xticks(ticks); ax1.set_xticklabels(tlab, rotation=90, fontsize=6.5)
    ax1.set_xlabel('history: action → observation', fontsize=7, color=MUTED)
    ax1.set_title('mapped histories', fontsize=7, color=MUTED)

    ax2.imshow(pred, cmap=CMAP, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    ax2.set_yticks([]); ax2.set_xticks(range(nA * nO))
    ax2.set_xticklabels(d['obs'] * nA, rotation=90, fontsize=6)
    for k, a in enumerate(d['acts']):
        ax2.text(k * nO + (nO - 1) / 2, -0.75, a, ha='center', va='bottom', fontsize=7, fontweight='bold')
        if k:
            ax2.axvline(k * nO - 0.5, color=MUTED, lw=0.8, zorder=5)
    ax2.set_title('predicted next observation\n', fontsize=7, color=MUTED, pad=8)
    ax2.set_xlabel('next observation, under each next action (bold)', fontsize=7, color=MUTED)

    ax3.set_xlim(-0.5, 0.5); ax3.set_ylim(nS - 0.5, -0.5)
    for s in range(nS):
        ax3.text(0, s, f'{err[s]:.3f}', ha='center', va='center', fontsize=6.5, color=INK)
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title('error vs\nanalytic', fontsize=7, color=MUTED)
    print(f'seed {SEED} {nH} histories max TV per state', np.round(err, 3).tolist())

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(SURFACE)
        for sp in ax.spines.values():
            sp.set_visible(True)                                     # frame every block
            sp.set_color(INK); sp.set_linewidth(1.0)
        ax.tick_params(length=0, pad=2)
        for s in range(1, nS):
            ax.axhline(s - 0.5, color=SURFACE, lw=3)
    col = -0.5
    for c in cl[:-1]:                                                # history groups
        col += len(c); ax1.axvline(col, color=SURFACE, lw=3)
    fig.text(x0 - 0.135, y0 + H + 0.055, title, fontsize=9, fontweight='bold', color=INK)
    return ax2


fig = plt.figure(figsize=(7.1, 7.4), facecolor=SURFACE)
ax_pred = panel(fig, (0.17, 0.745, 0.70, 0.16), minimal(), '(a)')
panel(fig, (0.17, 0.235, 0.70, 0.25), full(), '(b)')
bb = ax_pred.get_position()                                          # shared colorbar under (a)'s predictions
cax = fig.add_axes([bb.x0, 0.628, bb.width, 0.012])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1)), cax=cax,
                  orientation='horizontal')
cb.set_label('probability', fontsize=7, color=INK)
cb.outline.set_edgecolor(INK); cb.outline.set_linewidth(0.8)
cb.ax.tick_params(labelsize=6.5, length=2)
if os.environ.get('NOSAVE'):
    sys.exit()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fig6.png')
fig.savefig(out, dpi=300, facecolor=SURFACE)
fig.savefig(out.replace('.png', '.pdf'), facecolor=SURFACE)
print('wrote', out)
