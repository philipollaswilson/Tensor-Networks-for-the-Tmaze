"""Figure: what our structure learner builds on MNIST (mnist_structure.py).
Run from the repo root:  python src/make_fig_mnist_structure.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rgm_lite as RG  # noqa: E402
import mnist_structure as MS  # noqa: E402
from make_fig_hidden_states import clean, title, INK, MUTED, GRID  # noqa: E402
from make_extension_figures import FIGS  # noqa: E402

BLUE, ORANGE = '#2a78d6', '#eb6834'


def reconstruct(x, basis, shape=(32, 32)):
    img = np.zeros(shape)
    for k, (r, c, mu, v, a) in enumerate(basis):
        img[r:r + MS.ND, c:c + MS.ND] = (mu + a[x[k]] * v).reshape(MS.ND, MS.ND)
    return img


def main():
    I, y = RG.mnist(12000)
    tr, te = slice(0, 10000), slice(10000, 12000)
    basis = MS.tiles_fit(I[tr])
    Xtr, Xte = MS.tiles_apply(I[tr], basis), MS.tiles_apply(I[te], basis)
    layers = MS.learn(Xtr, y[tr])
    Str, Ste = MS.run(layers, Xtr), MS.run(layers, Xte)
    P = Xtr.shape[1]
    K = np.array([L['K'] for L in layers])
    acc = np.array([(layers[t]['label'].argmax(1)[Ste[:, t]] == y[te]).mean() if t else 0.1
                    for t in range(P + 1)])
    mi = np.array([MS.mutual_info(layers[t]['label']) if t else 0.0 for t in range(P + 1)])
    print('test accuracy (final)', acc[-1], 'max states', K.max(), 'total states', K.sum())

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.8, 1.2, 1.2, 1.0], height_ratios=[1, 1.15],
                          hspace=0.45, wspace=0.35)

    # (a) input
    ax = fig.add_subplot(gs[0, 0])
    i = 0
    rec = reconstruct(Xtr[i], basis)
    ax.imshow(np.hstack([I[i], np.zeros((32, 3)), rec]), cmap='Greys', vmin=0, vmax=1)
    for g in range(0, 33, MS.ND):
        ax.plot([35 + g - 0.5] * 2, [-0.5, 31.5], color=ORANGE, lw=0.3)
        ax.plot([34.5, 66.5], [g - 0.5] * 2, color=ORANGE, lw=0.3)
    ax.text(15.5, 34, 'image', ha='center', va='top', fontsize=7.5, color=MUTED)
    ax.text(50.5, 34, 'as the learner sees it', ha='center', va='top', fontsize=7.5, color=MUTED)
    ax.axis('off')
    title(ax, '(a) An image and its 64 tile symbols\n(first SVD mode, 7 levels), read in\n'
              'raster order; the label comes last')

    # (b) the machine
    ax = fig.add_subplot(gs[0, 1:3])
    pos = {}
    for t, L in enumerate(layers):
        order = np.argsort(-L.get('visits', np.ones(1)))
        for rank, s in enumerate(order):
            pos[t, int(s)] = (t, rank - (L['K'] - 1) / 2)
    flows = {}
    for row in Str:
        for t in range(P):
            key = (t, int(row[t]), int(row[t + 1]))
            flows[key] = flows.get(key, 0) + 1
    segs, wid = [], []
    for (t, s, s2), n in flows.items():
        if n >= 5:
            segs.append([pos[t, s], pos[t + 1, s2]])
            wid.append(0.15 + 1.6 * (n / len(Str)) ** 0.5)
    ax.add_collection(LineCollection(segs, linewidths=wid, colors=MUTED, alpha=0.35, zorder=1))
    for t, L in enumerate(layers):
        if t == 0:
            ax.scatter([0], [0], s=12, color=INK, zorder=3)
            continue
        pl = L['label']
        purity = pl.max(1) / np.maximum(pl.sum(1), 1)
        xy = np.array([pos[t, s] for s in range(L['K'])])
        ax.scatter(xy[:, 0], xy[:, 1], s=4 + 40 * (L['visits'] / len(Str)) ** 0.5, c=purity,
                   cmap='Blues', vmin=0, vmax=1, edgecolor='none', zorder=3)
    for r in range(1, 8):
        ax.axvline(r * 8 + 0.5, color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(-1, P + 1); ax.set_ylim(-K.max() / 2 - 2, K.max() / 2 + 2)
    ax.set_xlabel('tile position (raster order; grey lines = image rows)', fontsize=8, color=MUTED)
    ax.set_yticks([])
    clean(ax, grid=None)
    title(ax, f'(b) The learned model: one hidden variable per position (its states are the '
              f'dots,\nsorted by visits), flows between them from 10,000 training images. '
              f'Colour = how sure\nthe state is of the digit. At most {K.max()} states per '
              f'position, {K.sum()} in all')

    # (c) states per position on the image grid
    ax = fig.add_subplot(gs[0, 3])
    grid = np.full((8, 8), np.nan)
    for k, (r, c, *_rest) in enumerate(basis):
        grid[r // MS.ND, c // MS.ND] = K[k + 1]
    im = ax.imshow(grid, cmap='Blues')
    for (r, c), v in np.ndenumerate(grid):
        if np.isfinite(v):
            ax.text(c, r, int(v), ha='center', va='center', fontsize=7,
                    color='white' if v > 0.6 * np.nanmax(grid) else INK)
    ax.set_xticks([]); ax.set_yticks([])
    title(ax, '(c) Hidden states after each tile, on\nthe image: memory peaks mid-stroke,\n'
              'drops where rows end in blank margin')

    # (d) what the digit evidence looks like along the scan
    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(range(P + 1), acc, '-', color=BLUE, lw=2, label='test accuracy from the state so far')
    ax2 = ax.twinx()
    ax2.plot(range(P + 1), mi / np.log(10), '--', color=ORANGE, lw=1.5,
             label='I(state ; digit) / ln 10 (train)')
    ax2.set_ylim(0, 1.05); ax.set_ylim(0, 1.05)
    ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    ax.set_xlabel('tiles read', fontsize=8, color=MUTED)
    ax.set_ylabel('fraction', fontsize=8, color=MUTED)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc='upper left')
    ax.text(P, acc[-1] + 0.03, f'{acc[-1]:.1%}', fontsize=8.5, color=INK, ha='right')
    clean(ax)
    title(ax, '(d) The state accumulates what the image says about the digit as the scan '
              'proceeds\n(both curves share the 0–1 scale; mutual information as a fraction '
              'of its maximum, ln 10)')

    # (e) what states mean: mean image of members at one position
    t = 36
    L = layers[t]
    top = np.argsort(-L['visits'])[:8]
    sub = gs[1, 2:].subgridspec(2, 8, height_ratios=[1, 0.55], hspace=0.15, wspace=0.12)
    for j, s in enumerate(top):
        mem = Str[:, t] == s
        ax = fig.add_subplot(sub[0, j])
        img = I[tr][mem].mean(0)
        ax.imshow(img, cmap='Greys', vmin=0, vmax=1)
        r0, c0 = basis[t - 1][0], basis[t - 1][1]
        ax.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), MS.ND, MS.ND, fill=False, ec=ORANGE, lw=1))
        ax.axis('off')
        ax.set_title(f'{int(L["visits"][s])} imgs', fontsize=7, color=MUTED, pad=2)
        ax = fig.add_subplot(sub[1, j])
        p = L['label'][s] / L['label'][s].sum()
        ax.bar(range(10), p, color=BLUE, width=0.7)
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_xticks(range(10)); ax.set_xticklabels(range(10), fontsize=5.5, color=MUTED)
        ax.tick_params(length=0)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
    fig.text(0.535, 0.475, f'(e) What a hidden state is: the 8 most visited states after tile '
             f'{t} (orange box = last tile read).\nTop: mean training image of the state\'s '
             'members. Bottom: p(digit | state)', fontsize=9.5, color=INK)

    fig.suptitle('Our structure learner on MNIST: images read tile by tile, histories merged '
                 'by Bayesian model reduction when they predict the next tiles and the label '
                 'alike', fontsize=12, color=INK, x=0.06, ha='left', y=0.995)
    out = FIGS / 'fig_mnist_structure.png'
    fig.savefig(out, bbox_inches='tight', dpi=170, facecolor='white')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
