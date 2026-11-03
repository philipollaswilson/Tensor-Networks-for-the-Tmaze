"""Figure: singular vectors as axes, learned hidden states as points.

Rows of the table H[h, (a, o)] = p(o | h, a) are the histories' predictions. Its SVD
gives the axes (right singular vectors, gauge invariant because H is built from
probabilities); each history is a point with coordinates U * s. Histories in one BMR
state land on the same point.

  (a) axis 2 vs axis 3     (b) axis 1 vs axis 4     (c) what each axis predicts (V^T)

Writes paper/figs/fig_svd_states.png. Run from the repo root:
  python src/make_fig_svd_states.py [seed]
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bmr_states import bmr, min_counts  # noqa: E402
from learned_pomdp import ACT, OBS  # noqa: E402
from make_extension_figures import FIGS  # noqa: E402

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']      # categorical slots 1-4
INK, MUTED = '#0b0b0b', '#52514e'


def state_name(members):
    o = members[0].split('/')[1]
    return {'Cheese': 'Cheese trap', 'Shock': 'Shock trap'}.get(o, f'Cue: {o[:-3]}')


def scatter(ax, P, groups, i, j, axis_names):
    lim = 1.6 * np.abs(P[:, [i, j]]).max()
    ax.axhline(0, color=MUTED, lw=0.5, zorder=0); ax.axvline(0, color=MUTED, lw=0.5, zorder=0)
    for k, (name, members, idx) in enumerate(groups):
        x, y = P[idx[0], i], P[idx[0], j]
        ax.scatter(x, y, s=140, color=SERIES[k], edgecolor='white', lw=2, zorder=3)
        if abs(x) > abs(y):                              # on the horizontal axis: label above
            kw = dict(xytext=(0, 12), ha='center', va='bottom')
        else:                                            # on the vertical axis: label right
            kw = dict(xytext=(12, 0), ha='left', va='center')
        ax.annotate(f'{name}\n' + '\n'.join(members), (x, y), textcoords='offset points',
                    fontsize=7, color=INK, **kw)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel(axis_names[i], color=INK); ax.set_ylabel(axis_names[j], color=INK)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def main(seed=0):
    names, counts = min_counts(seed, 'model')
    H = np.array([(c / c.sum(1, keepdims=True)).ravel() for c in counts])
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    r = int((s > 1e-6 * s[0]).sum())
    P = U[:, :r] * s[:r]                                    # each history as a point
    clusters, _, _ = bmr(counts, 0.1)
    groups = [(state_name([names[i] for i in c]), [names[i] for i in c], c) for c in clusters]

    axis_names = [f'axis {k + 1}  (s = {s[k]:.2f})' for k in range(r)]

    fig = plt.figure(figsize=(13.5, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.15], wspace=0.3)
    ax = fig.add_subplot(gs[0]); scatter(ax, P, groups, 1, 2, axis_names)
    ax.set_title('(a) Axes 2 and 3: cheese vs shock,\nleft cue vs right cue', fontsize=9.5)

    # axis 1 is nearly constant (the average prediction); axes 2-4 place the states
    # at the corners of a tetrahedron
    ax = fig.add_subplot(gs[1], projection='3d')
    lim = 1.4 * np.abs(P[:, 1:4]).max()
    for d in range(3):
        e = np.zeros((2, 3)); e[:, d] = (-lim, lim)
        ax.plot(*e.T, color=MUTED, lw=0.6)
    pts = [P[idx[0], 1:4] for _, _, idx in groups]
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            ax.plot(*np.array([pts[a], pts[b]]).T, color=MUTED, lw=0.6, ls=':')
    centre = np.mean(pts, 0)
    for k, ((name, _, _), p) in enumerate(zip(groups, pts)):
        ax.scatter(*p, s=120, color=SERIES[k], edgecolor='white', lw=1.5, depthshade=False)
        out = p + 0.45 * (p - centre) / np.linalg.norm(p - centre)
        ax.text(*out, name, fontsize=7.5, color=INK, ha='center', va='center')
    ax.set_xlabel('axis 2', fontsize=8); ax.set_ylabel('axis 3', fontsize=8); ax.set_zlabel('axis 4', fontsize=8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.tick_params(labelsize=6); ax.view_init(elev=40, azim=45)
    ax.set_title('(b) Axes 2-4: the four states are the\ncorners of a tetrahedron', fontsize=9.5)

    ax = fig.add_subplot(gs[2])
    m = np.abs(Vt[:r]).max()
    im = ax.imshow(Vt[:r], cmap='RdBu_r', vmin=-m, vmax=m, aspect='auto')
    ax.set_title('(c) What each axis predicts (rows of $V^T$)', fontsize=9.5, pad=22)
    ax.set_yticks(range(r)); ax.set_yticklabels([f'axis {k + 1}' for k in range(r)], fontsize=8)
    ax.set_xticks(range(H.shape[1]))
    ax.set_xticklabels([OBS[f % 4] for f in range(H.shape[1])], rotation=90, fontsize=7)
    for a in range(len(ACT)):
        if a:
            ax.axvline(4 * a - 0.5, color='k', lw=0.8)
        ax.text(4 * a + 1.5, -0.6, f'a = {ACT[a]}', ha='center', fontsize=8.5, weight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03, label='weight')

    fig.suptitle(f'Singular vectors are the axes; hidden states are the points '
                 f'(minimal maze, seed {seed}: 6 histories, rank {r}, {len(clusters)} states)',
                 fontsize=10.5, y=1.02)
    out = FIGS / 'fig_svd_states.png'
    fig.savefig(out); plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
