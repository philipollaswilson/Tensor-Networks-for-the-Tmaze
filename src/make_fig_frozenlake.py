"""Frozen-lake versions of fig_learned_pomdp and fig_svd_states.

fig_learned_pomdp_frozenlake.png
  (a) likelihood A over the learned states   (b) transition B, one block per action
  (c) the map with both episodes' paths      (d, e) beliefs q(s) through the agent's
  own 4-step episode (sophisticated EFE, remaining horizon) and a forced walk
  into the hole (Down, Right) followed by the agent's choices
fig_svd_states_frozenlake.png
  (a) axes 2 and 3 with the learned states as points   (b) singular values
  (c) what each axis predicts (rows of V^T)

Run from the repo root:  python src/make_fig_frozenlake.py [seed]
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from frozenlake_pomdp import (ACT, HORIZON, NS, OBS, bmr, counts, learned_model,  # noqa: E402
                              preferences, sophisticated)
from frozenlake_train import DESC  # noqa: E402
from make_extension_figures import FIGS, HEAT, annot  # noqa: E402
from make_fig_svd_states import INK, MUTED  # noqa: E402

TILE = {'S': '#dfe9f5', 'F': '#f4f7fb', 'H': '#3b4a5c', 'G': '#f2d27a'}
PATHS = ['#2a78d6', '#eb6834']                                # categorical slots 1-2
STATES9 = plt.get_cmap('tab10').colors[:NS]


def episode(A, B, C, D, forced):
    """Beliefs at each step; forced actions first, then sophisticated EFE with the
    remaining horizon. The observation is the one the model expects most."""
    q, cols, names, cells = D, [D], ['start'], [int(np.argmax(A @ D))]
    for t in range(HORIZON):
        a = forced[t] if t < len(forced) else int(np.argmax(sophisticated(A, B, C, q, HORIZON - t)[1]))
        q = B[:, :, a] @ q
        cols.append(q); names.append(f'do {ACT[a]}\n(predict)')
        o = int(np.argmax(A @ q))
        q = A[o] * q; q /= q.sum()
        cols.append(q); names.append(f'see\n{OBS[o]}')
        cells.append(o)
    return np.array(cols).T, names, cells


def draw_map(ax, paths):
    for r in range(3):
        for c in range(3):
            t = DESC[r][c]
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fc=TILE[t], ec='white', lw=2))
            ax.text(c - 0.38, r - 0.32, f'{t}{3 * r + c}', fontsize=8,
                    color='white' if t == 'H' else INK, va='center')
    for k, (cells, label) in enumerate(paths):
        off = (k - 0.5) * 0.12
        xy = np.array([(o % 3 + off, o // 3 + off) for o in cells])
        ax.plot(*xy.T, color=PATHS[k], lw=2.2, marker='o', ms=4, label=label, zorder=3)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(2.5, -0.5); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.03), frameon=False)


def fig_pomdp(seed):
    A, B, D, cells, _, _ = learned_model(seed)
    S = [OBS[o] for o in cells]
    C = preferences()
    q_a1 = sophisticated(A, B, C, D, HORIZON)[1]

    fig = plt.figure(figsize=(14, 8.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.55)
    gt = outer[0].subgridspec(1, 2, width_ratios=[1, 3.6], wspace=0.15)
    gb = outer[1].subgridspec(1, 3, width_ratios=[0.8, 1.6, 1.6], wspace=0.3)

    ax = fig.add_subplot(gt[0, 0])
    im = ax.imshow(A, cmap=HEAT, vmin=0, vmax=1)
    ax.set_title(r'Likelihood $A=p(o\,|\,s)$')
    ax.set_yticks(range(NS)); ax.set_yticklabels(OBS, fontsize=7)
    ax.set_xticks(range(len(S))); ax.set_xticklabels(S, rotation=90, fontsize=7)
    ax.set_xlabel('learned state', fontsize=8)

    ax = fig.add_subplot(gt[0, 1])
    Bw = np.concatenate([B[:, :, a] for a in range(len(ACT))], 1)
    ax.imshow(Bw, cmap=HEAT, vmin=0, vmax=1, aspect='auto')
    ax.set_title(r"Transition $B=p(s'\,|\,s,a)$, one block per action", pad=20)
    ax.set_yticks(range(len(S))); ax.set_yticklabels(S, fontsize=7)
    ax.set_ylabel("next state $s'$", fontsize=8)
    ax.set_xticks(range(Bw.shape[1])); ax.set_xticklabels(S * len(ACT), rotation=90, fontsize=6)
    for a in range(len(ACT)):
        if a:
            ax.axvline(a * len(S) - 0.5, color='k', lw=0.8)
        ax.text((a + 0.5) * len(S) - 0.5, -0.9, f'a = {ACT[a]}', ha='center', fontsize=8.5, weight='bold')
    annot(ax, Bw, small=5)

    runs = [([], f"Agent's choice ($q$ = {q_a1.max():.3f} for {ACT[int(np.argmax(q_a1))]} first)"),
            ([ACT.index('Down'), ACT.index('Right')], 'Forced: Down, Right, then the agent')]
    eps = [episode(A, B, C, D, forced) for forced, _ in runs]
    draw_map(fig.add_subplot(gb[0, 0]), [(e[2], lab.split(' (')[0]) for e, (_, lab) in zip(eps, runs)])
    for k, ((Q, names, _), (_, title)) in enumerate(zip(eps, runs)):
        ax = fig.add_subplot(gb[0, k + 1])
        ax.imshow(Q, cmap=HEAT, vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'Beliefs $q(s)$. {title}', fontsize=9, color=PATHS[k])
        ax.set_yticks(range(len(S))); ax.set_yticklabels(S, fontsize=7)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=6)
        annot(ax, Q, small=6)

    fig.colorbar(im, ax=fig.axes, shrink=0.6, pad=0.02, label='probability')
    fig.suptitle(f'Frozen lake ({"/".join(DESC)}, seed {seed}): generative model built from the '
                 f'{len(S)} learned states; EFE horizon {HORIZON}, goal +2, hole -4', y=0.97)
    out = FIGS / 'fig_learned_pomdp_frozenlake.png'
    fig.savefig(out); plt.close(fig)
    print(f'wrote {out}')


def fig_svd(seed):
    hists, cnts = counts(seed)
    H = np.array([(c / c.sum(1, keepdims=True)).ravel() for c in cnts])
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    r = int((s > 1e-2 * s[0]).sum())
    P = U[:, :r] * s[:r]
    clusters, _, _ = bmr(cnts, 0.1)
    clusters = sorted(clusters, key=lambda c: hists[c[0]][-1])

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.75, 1.5], wspace=0.55)

    ax = fig.add_subplot(gs[0])
    lim = 1.35 * np.abs(P[:, 1:3]).max()
    ax.axhline(0, color=MUTED, lw=0.5, zorder=0); ax.axvline(0, color=MUTED, lw=0.5, zorder=0)
    for k, c in enumerate(clusters):
        o = hists[c[0]][-1]
        ax.scatter(P[c, 1], P[c, 2], s=90, color=STATES9[o], edgecolor='white', lw=1.5, zorder=3,
                   label=f'{OBS[o]} ({len(c)} hist.)')
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False,
              handletextpad=0.2, labelcolor=INK)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel(f'axis 2  (s = {s[1]:.2f})'); ax.set_ylabel(f'axis 3  (s = {s[2]:.2f})')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title(f'(a) Axes 2 and 3: {len(hists)} histories\nland on {len(clusters)} points', fontsize=9.5)

    ax = fig.add_subplot(gs[1])
    k = np.arange(1, min(len(s), 16) + 1)
    ax.semilogy(k, s[:len(k)], 'o-', color=PATHS[0], ms=4, lw=1)
    ax.axvline(r + 0.5, color=MUTED, lw=0.8, ls=':')
    ax.text(r + 0.7, s[0], f'rank {r}', fontsize=8, color=MUTED, va='top')
    ax.set_xlabel('index'); ax.set_ylabel('singular value')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title('(b) Singular values: one axis per\nlearned state, then a gap', fontsize=9.5)

    ax = fig.add_subplot(gs[2])
    m = np.abs(Vt[:r]).max()
    im = ax.imshow(Vt[:r], cmap='RdBu_r', vmin=-m, vmax=m, aspect='auto')
    ax.set_title('(c) What each axis predicts (rows of $V^T$)', fontsize=9.5, pad=22)
    ax.set_yticks(range(r)); ax.set_yticklabels([f'axis {i + 1}' for i in range(r)], fontsize=8)
    ax.set_xticks(range(H.shape[1]))
    ax.set_xticklabels([OBS[f % NS] for f in range(H.shape[1])], rotation=90, fontsize=6)
    for a in range(len(ACT)):
        if a:
            ax.axvline(NS * a - 0.5, color='k', lw=0.8)
        ax.text(NS * a + (NS - 1) / 2, -0.8, f'a = {ACT[a]}', ha='center', fontsize=8.5, weight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label='weight')

    fig.suptitle(f'Singular vectors are the axes; hidden states are the points (frozen lake, seed {seed}: '
                 f'{len(hists)} histories, rank {r}, {len(clusters)} states)', fontsize=10.5, y=1.02)
    out = FIGS / 'fig_svd_states_frozenlake.png'
    fig.savefig(out); plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fig_pomdp(seed)
    fig_svd(seed)
