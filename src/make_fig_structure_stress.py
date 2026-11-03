"""Figure: the structure-learning pipeline on three T-maze variants (stress test).

Top row: the generator found from 3-step data, each state placed by the true location
(x) and context (y) of the histories it holds (naming only); edges are the learned
transitions. Bottom: what 2-step vs 3-step data give for each variant.
Run from the repo root:  python src/make_fig_structure_stress.py
"""
import pathlib
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import structure_pipeline as SPL  # noqa: E402
from make_extension_figures import FIGS  # noqa: E402

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'
COL = {0: '#898781', 1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a'}
X = {0: -1.0, 1: 3.0, 2: -3.0, 3: 1.0}
Y = {0: 1.0, 1: -1.0, 2: 0.0}
CTX = ['cheese R', 'cheese L', 'no cheese']
N_TRAJ = 20000

VARIANTS = [('pymdp', 'Original maze: every move goes straight\nto the named place'),
            ('centre_only', 'Location matters: the arms are\nreachable only from the centre'),
            ('three_ctx', 'A third context: no cheese anywhere')]


def placement(res):
    tot = defaultdict(lambda: defaultdict(int))
    for h, k in res['m']['of'].items():
        for st, c in res['truth'][h].items():
            tot[k][st] += c
    out = {}
    for i, k in enumerate(res['V']):
        d = tot[k]
        n = sum(d.values())
        locs = sorted({l for l, _ in d})
        ctx = defaultdict(float)
        for (_, c), v in d.items():
            ctx[c] += v / n
        x = np.mean([X[l] for l in locs])
        y = sum(p * Y[c] for c, p in ctx.items())
        name = '/'.join(SPL.LOC[l] for l in locs)
        cn = ' or '.join(CTX[c].replace('cheese ', '') if len(ctx) > 1 else CTX[c]
                         for c, p in sorted(ctx.items()) if p > 0.05)
        out[i] = (x, y, locs, f'{name}\n{cn}')
    return out


def panel(ax, res):
    pl = placement(res)
    G, D = res['gen']['G'], res['gen']['D']
    S = G.sum(1)
    for i in pl:
        for j in pl:
            if i != j and S[:, i, j].max() > 0.05:
                acts = [SPL.ACT[a] for a in range(4) if S[a, i, j] > 0.05]
                (x0, y0), (x1, y1) = pl[i][:2], pl[j][:2]
                ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9,
                                             shrinkA=14, shrinkB=14, lw=1.1, color=INK,
                                             connectionstyle='arc3,rad=0.12', zorder=2))
    sx, sy = 0.0, 2.15
    ax.scatter([sx], [sy], s=260, marker='s', color='white', edgecolor=INK, lw=1, zorder=4)
    ax.text(sx, sy, 'start', fontsize=6, ha='center', va='center', zorder=5)
    for i, w in enumerate(D):
        if w > 0.01:
            ax.add_patch(FancyArrowPatch((sx, sy), pl[i][:2], arrowstyle='-|>', mutation_scale=8,
                                         shrinkA=9, shrinkB=14, lw=0.8, ls=(0, (3, 2)), color=MUTED,
                                         zorder=1))
    for i, (x, y, locs, name) in pl.items():
        c = COL[locs[0]] if len(locs) == 1 else '#5c8f9e'
        ax.scatter([x], [y], s=380, color=c, edgecolor='white', lw=1.5, zorder=4)
        ax.text(x, y - 0.3, name, fontsize=6.6, ha='center', va='top', color=INK, zorder=5,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.8))
    ax.set_xlim(-4.0, 4.0); ax.set_ylim(-1.9, 2.5)
    ax.axis('off')


def verdict(res):
    n = len(res['V'])
    if res['prods']:
        p1, p2 = res['prods'][0]
        return f'{n} = {len(p1)} × {len(p2)}: a product of two hidden variables'
    return (f'{n} states, no product of two variables;\n{len(res["sps"])} closed groupings '
            '(hierarchical splits only)')


def main():
    rows = {}
    fig = plt.figure(figsize=(18, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.36], hspace=0.25, wspace=0.08)
    for c, (v, desc) in enumerate(VARIANTS):
        for T in (2, 3):
            env = SPL.MazeEnv(v)
            res = SPL.run(env.sample(N_TRAJ, T, 0), len(env.obs))
            rows[v, T] = res
        res = rows[v, 3]
        ax = fig.add_subplot(gs[0, c]); panel(ax, res)
        ax.set_title(f'({"abc"[c]}) {desc}\nfound from 3-step data: {verdict(res)}', fontsize=9.5,
                     color=INK, loc='left')
    ax = fig.add_subplot(gs[1, :]); ax.axis('off')
    head = ['', 'histories', 'states (BMR)', 'after unmixing', 'factorization']
    xs = [0.0, 0.25, 0.38, 0.52, 0.66]
    for x, h in zip(xs, head):
        ax.text(x, 0.95, h, fontsize=9, weight='bold', color=INK, transform=ax.transAxes, va='top')
    r = 0
    for v, desc in VARIANTS:
        for T in (2, 3):
            res = rows[v, T]
            y = 0.78 - 0.13 * r
            fac = (f'{len(res["prods"][0][0])} × {len(res["prods"][0][1])}' if res['prods']
                   else 'none')
            short = {'pymdp': 'Original maze', 'centre_only': 'Arms only from the centre',
                     'three_ctx': 'Third context (no cheese)'}[v]
            cells = [f'{short}, {T}-step data', len(res['N']),
                     res['m']['K'], len(res['V']), fac]
            for x, cell in zip(xs, cells):
                ax.text(x, y, str(cell), fontsize=8.5, color=INK if T == 3 else MUTED,
                        transform=ax.transAxes, va='top')
            r += 1
    ax.text(0.80, 0.78, '2-step data cannot show that the centre\nremembers the cue when the centre and the\n'
            'cue differ: the history needed is 2 steps\nlong and has no future in the data.\n'
            '3-step data recover pymdp\'s 2 × 4.', fontsize=8.3, color=MUTED, transform=ax.transAxes,
            va='top')
    fig.suptitle(f'Stress test: the same pipeline, three mazes ({N_TRAJ} random-action trajectories '
                 'each). Positions show the true location and context of each learned state; '
                 'colour = location', fontsize=11, color=INK, x=0.06, ha='left', y=0.995)
    out = FIGS / 'fig_structure_stress.png'
    fig.savefig(out, bbox_inches='tight', dpi=170, facecolor='white')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
