"""Figure: a latent tree learned from pixels, with no order, tiles or neighbourhood given
(latent_tree.py), against fixed shapes (raster chain = MPS, k-d tree = locality prior).
Run from the repo root:  python src/make_fig_latent_tree.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import latent_tree as LT  # noqa: E402
from make_fig_hidden_states import clean, title, INK, MUTED  # noqa: E402
from make_extension_figures import FIGS  # noqa: E402

BLUE, ORANGE, GREY, GREEN = '#2a78d6', '#eb6834', '#c3c2b7', '#3a9a6a'
S = LT.SIDE


def spread(groups):
    d = []
    for g in groups:
        if len(g) > 1:
            rc = np.array([(p // S, p % S) for p in g])
            d.append(np.sqrt(((rc[:, None] - rc[None]) ** 2).sum(-1)).sum() / (len(g) * (len(g) - 1)))
    return float(np.mean(d))


def partition_map(ax, tree, k):
    groups = [tree.nodes[n]['pix'] for n in tree.log[-k]]
    lab = np.full(S * S, -1)
    for g, pix in enumerate(sorted(groups, key=len, reverse=True)):
        lab[pix] = g
    L = lab.reshape(S, S)
    cmap = plt.get_cmap('tab20')
    rgb = np.ones((S, S, 3)) * 0.97
    for (r, c), g in np.ndenumerate(L):
        if g >= 0:
            rgb[r, c] = cmap(g % 20)[:3]
    ax.imshow(rgb)
    for r in range(S):
        for c in range(S):
            if c + 1 < S and L[r, c] != L[r, c + 1]:
                ax.plot([c + 0.5] * 2, [r - 0.5, r + 0.5], color=INK, lw=0.8)
            if r + 1 < S and L[r, c] != L[r + 1, c]:
                ax.plot([c - 0.5, c + 0.5], [r + 0.5] * 2, color=INK, lw=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    return groups


def join_time(tree):
    """Step at which each pixel's leaf is first joined (blank = constant pixel)."""
    t = np.full(S * S, np.nan)
    for step, k in enumerate(range(len(tree.pix), len(tree.nodes))):
        for c in tree.nodes[k]['kids']:
            if c < len(tree.pix):
                t[tree.pix[c]] = step
    return t.reshape(S, S)


def main():
    Xtr, ytr, Xte, yte = LT.load()
    trees = {'learned (redundancy)': LT.Tree(Xtr, score='redundancy').learn(),
             'learned (evidence)': LT.Tree(Xtr).learn(),
             'raster chain (MPS)': LT.Tree(Xtr).fixed(LT.raster_chain()),
             'k-d tree (locality)': LT.Tree(Xtr).fixed(LT.kd_tree())}
    res, enc = {}, {}
    for name, tr in trees.items():
        res[name] = LT.evaluate(tr, Xtr, ytr, Xte, yte)
        enc[name] = (tr.encode(Xtr), tr.encode(Xte))
        print(name, {k: round(float(v), 3) for k, v in res[name].items()})
    pix = trees['k-d tree (locality)'].pix
    base = {'independent pixels': -LT.independent_ll(Xtr, Xte, pix).mean() / np.log(2),
            'Chow-Liu (visible tree)': -LT.chow_liu_ll(Xtr, Xte, pix).mean() / np.log(2)}
    print(base)
    sweep = {}
    for m in (1, 2, 4, 8, 16, None):
        for name, make in (('learned (redundancy)', lambda t: t.learn()),
                           ('raster chain (MPS)', lambda t: t.fixed(LT.raster_chain())),
                           ('k-d tree (locality)', lambda t: t.fixed(LT.kd_tree()))):
            sc = 'redundancy' if name.startswith('learned') else 'evidence'
            tr = make(LT.Tree(Xtr, m=m, score=sc))
            sweep.setdefault(name, []).append(-tr.loglik(Xte).mean() / np.log(2))
    print('sweep', sweep)

    fig = plt.figure(figsize=(19, 10.5))
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.42)

    # (a) learned partitions
    R = trees['learned (redundancy)']
    sub = gs[0, 0].subgridspec(1, 2, wspace=0.08)
    for c, k in enumerate((16, 4)):
        ax = fig.add_subplot(sub[0, c])
        g = partition_map(ax, R, k)
        ax.set_title(f'{k} subtrees', fontsize=8, color=MUTED, pad=3)
    groups = [R.nodes[n]['pix'] for n in R.log[-16]]
    allp, cuts = np.concatenate(groups), np.cumsum([len(g) for g in groups])[:-1]
    rng = np.random.default_rng(0)
    rnd = np.mean([spread(np.split(rng.permutation(allp), cuts)) for _ in range(20)])
    title(fig.add_subplot(gs[0, 0], frameon=False, xticks=[], yticks=[]),
          f'(a) The learned tree (redundancy rule), cut into\nits top 16 and 4 subtrees (white = '
          f'constant pixel).\nNobody said "neighbours": mean distance {spread(groups):.1f} px\n'
          f'within subtrees vs {rnd:.1f} for random groups', pad=8)

    # (b) evidence rule = learned order
    E = trees['learned (evidence)']
    sub = gs[0, 1].subgridspec(1, 2, wspace=0.08)
    for c, (tr, lab) in enumerate(((E, 'evidence rule'), (trees['raster chain (MPS)'], 'raster'))):
        ax = fig.add_subplot(sub[0, c])
        ax.imshow(join_time(tr), cmap='Blues_r')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lab, fontsize=8, color=MUTED, pad=3)
    depth = {}
    for name in ('learned (evidence)', 'learned (redundancy)'):
        tr, d = trees[name], {n: 0 for n in range(len(trees[name].pix))}
        for k in range(len(tr.pix), len(tr.nodes)):
            d[k] = max(d[c] for c in tr.nodes[k]['kids']) + 1
        depth[name] = d[len(tr.nodes) - 1]
    title(fig.add_subplot(gs[0, 1], frameon=False, xticks=[], yticks=[]),
          f'(b) When each pixel is joined (dark = early).\nThe plain evidence rule snowballs: '
          f'depth\n{depth["learned (evidence)"]} of {len(pix) - 1}, a chain (MPS) whose order the '
          f'data chose,\nstarting in the middle of the digit. Redundancy\nrule: depth '
          f'{depth["learned (redundancy)"]}', pad=8)

    # (c) likelihood
    ax = fig.add_subplot(gs[0, 2])
    names = list(base) + ['k-d tree (locality)', 'learned (evidence)', 'learned (redundancy)',
                          'raster chain (MPS)']
    vals = [base[n] if n in base else res[n]['nll_bits'] for n in names]
    cols = [GREY, GREY, GREY, BLUE, BLUE, ORANGE]
    for r, (n, v, c) in enumerate(zip(names, vals, cols)):
        ax.barh(r, v, color=c, height=0.5)
        ax.text(v + 0.6, r, f'{v:.1f}', va='center', fontsize=8, color=INK)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(40, 76)
    ax.set_xlabel('held-out bits per image (lower is better)', fontsize=8, color=MUTED)
    clean(ax, grid='x')
    title(ax, '(c) Density of 2,000 held-out images\n(146 non-constant pixels). Every latent\n'
              f'tree beats the best visible tree by '
              f'{base["Chow-Liu (visible tree)"] - max(vals[2:]):.0f}–'
              f'{base["Chow-Liu (visible tree)"] - min(vals[2:]):.0f} bits')

    # (d) outside budget
    ax = fig.add_subplot(gs[0, 3])
    xs = np.arange(6)
    for name, c in (('learned (redundancy)', BLUE), ('raster chain (MPS)', ORANGE),
                    ('k-d tree (locality)', GREY)):
        ax.plot(xs, sweep[name], 'o-', color=c, lw=2, ms=5, label=name)
    ax.set_xticks(xs); ax.set_xticklabels(['1', '2', '4', '8', '16', 'all'], fontsize=8)
    ax.set_xlabel('outside nodes each merge must predict (m)', fontsize=8, color=MUTED)
    ax.set_ylabel('held-out bits per image', fontsize=8, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False)
    clean(ax)
    title(ax, '(d) The one knob: how much of the outside\na merge looks at. "all" is the '
              'prior-free\nsetting; it costs at most 1 bit vs the best m')

    # (e) bond size vs subtree size
    ax = fig.add_subplot(gs[1, 0])
    for name, c, mk in (('k-d tree (locality)', GREY, 's'), ('raster chain (MPS)', ORANGE, '^'),
                        ('learned (redundancy)', BLUE, 'o')):
        tr = trees[name]
        sz = [len(N['pix']) for N in tr.nodes[len(tr.pix):]]
        K = [N['K'] for N in tr.nodes[len(tr.pix):]]
        ax.scatter(sz, K, s=10, color=c, marker=mk, alpha=0.7, label=name, edgecolor='none')
    ax.set_xscale('log')
    ax.set_xlabel('pixels in the subtree', fontsize=8, color=MUTED)
    ax.set_ylabel('bond (states of the node)', fontsize=8, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    clean(ax)
    title(ax, '(e) Bond size is set by the data: it grows\nwith the subtree, peaks at 50–80 pixels '
              '(of 146),\nand falls as the outside shrinks')

    # (f) digit information along the tree
    ax = fig.add_subplot(gs[1, 1])
    for name, c in (('k-d tree (locality)', GREY), ('raster chain (MPS)', ORANGE),
                    ('learned (redundancy)', BLUE)):
        tr = trees[name]
        info = LT.node_info(tr, enc[name][0], ytr)
        sz = [len(N['pix']) for N in tr.nodes[len(tr.pix):]]
        ax.scatter(sz, info, s=10, color=c, alpha=0.7, edgecolor='none', label=name)
    ax.set_xscale('log'); ax.set_ylim(0, 1)
    ax.set_xlabel('pixels in the subtree', fontsize=8, color=MUTED)
    ax.set_ylabel('I(node ; digit) / ln 10', fontsize=8, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    clean(ax)
    acc = {n: res[n]['best_acc'] for n in res}
    title(ax, '(f) No labels used. The digit is the information\nshared across a mid-sized cut; '
              'test accuracy\nfrom the best node: learned '
              f'{acc["learned (redundancy)"]:.0%} (redundancy),\n'
              f'{acc["learned (evidence)"]:.0%} (evidence), chain '
              f'{acc["raster chain (MPS)"]:.0%}, k-d {acc["k-d tree (locality)"]:.0%}')

    # (g) what the best node's states are
    name = 'learned (evidence)'
    tr = trees[name]
    Str = enc[name][0]
    info = LT.node_info(tr, Str, ytr)
    k = len(tr.pix) + int(info.argmax())
    N = tr.nodes[k]
    top = np.argsort(-N['nc'])[:10]
    sub = gs[1, 2:].subgridspec(2, 6, wspace=0.08, hspace=0.25)
    ax = fig.add_subplot(sub[:, 0])
    mask = np.zeros(S * S); mask[N['pix']] = 1
    ax.imshow(mask.reshape(S, S), cmap='Oranges', vmin=0, vmax=1.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'its {len(N["pix"])} pixels', fontsize=8, color=MUTED, pad=3)
    for q, s in enumerate(top):
        ax = fig.add_subplot(sub[q // 5, 1 + q % 5])
        mem = Str[k] == s
        ax.imshow(Xtr[mem].mean(0).reshape(S, S), cmap='Greys', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        J = np.bincount(ytr[mem], minlength=10)
        ax.set_title(f'n={mem.sum()}  "{J.argmax()}" {J.max() / mem.sum():.0%}', fontsize=7,
                     color=MUTED, pad=2)
    fig.text(0.52, 0.475, f'(g) What a bond carries: the most digit-informative node of the '
             f'evidence tree ({N["K"]} states). Its 10 most\nfrequent states, as the mean of all '
             'pixels of their member images, with the commonest digit: the\nstates are stroke '
             'shapes in its pixels, which predict the rest of the image', fontsize=9.5, color=INK)

    fig.suptitle('Structure learning without sequences, tiles or neighbourhoods: a latent tree '
                 'over the pixels of MNIST (14×14, binarised), built by joining the most dependent '
                 'nodes and merging states by BMR', fontsize=12, color=INK, x=0.06, ha='left',
                 y=0.995)
    out = FIGS / 'fig_latent_tree.png'
    fig.savefig(out, bbox_inches='tight', dpi=170, facecolor='white')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
