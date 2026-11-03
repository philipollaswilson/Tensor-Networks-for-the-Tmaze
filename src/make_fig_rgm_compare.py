"""Figure: RGM state generation vs. our factorized states (MNIST and a flapping bird).

RGM side: rgm_lite.py (SPM's spm_rgb2O + spm_MB_structure_learning, states only).
Our side: predictive states of the frame stream (2-frame contexts, BMR + refinement),
then the pair-closure factor search (factor_search.py).
Run from the repo root:  python src/make_fig_rgm_compare.py
"""
import math
import pathlib
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rgm_lite as RG  # noqa: E402
import factor_search as FS  # noqa: E402
from make_fig_hidden_states import ari, clean, title, INK, MUTED  # noqa: E402
from make_extension_figures import FIGS  # noqa: E402

BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#c3c2b7'


def bell(n):
    row = [1]
    for _ in range(n):
        nxt = [row[-1]]
        for x in row:
            nxt.append(nxt[-1] + x)
        row = nxt
    return row[0]


def bird_analysis():
    F, truth = RG.bird()
    O2, L2, _ = RG.discretise(F, nd=16, nb=16, mm=32, su=32, R=2)
    rgm = RG.rg_levels(O2, L2, dt=2)
    O1, _, _ = RG.discretise(F, nd=16, nb=16, mm=32, su=32, R=1)
    sym = [tuple(c) for c in O1.T]
    t0 = time.perf_counter()
    of, K, T = FS.stream_states(sym, k=2)
    succ = [[set(np.flatnonzero(T[i] > 0)) for i in range(K)]]
    sps = FS.sp_lattice(succ)
    prods = FS.products(sps, K)
    dt = time.perf_counter() - t0
    # state of each time step (from its 2-frame context), to compare with the truth
    ids = {s: i for i, s in enumerate(dict.fromkeys(sym))}
    x = [ids[s] for s in sym]
    st = np.array([of[(x[t - 1], x[t])] for t in range(1, len(x))])
    p, q = prods[0]
    f1, f2 = np.array(p)[st], np.array(q)[st]
    checks = dict(bob=ari(f1, truth[1:, 1]), wing=ari(f2, truth[1:, 0]),
                  joint=ari(st, truth[1:, 0] * 3 + truth[1:, 1]))
    B = [FS.factor_matrices([T], p)[0], FS.factor_matrices([T], q)[0]]
    return dict(F=F, truth=truth, rgm=rgm, K=K, sps=sps, prods=prods, B=B, checks=checks,
                distinct=len(set(sym)), secs=dt, T=T)


def mnist_analysis():
    I, y = RG.mnist(10000)
    j = np.concatenate([np.flatnonzero(y == d)[:13] for d in range(10)])
    O, L, _ = RG.discretise(I[j], nd=4, nb=7, mm=16, su=8)
    return dict(I=I[j], y=y[j], rgm=RG.rg_levels(O, L, dt=1), streams=len(O))


def strip(ax, frames, labels):
    ax.imshow(np.hstack([np.pad(f, 2, constant_values=0.35) for f in frames]), cmap='Greys',
              vmin=0, vmax=1)
    w = frames[0].shape[1] + 4
    for i, s in enumerate(labels):
        ax.text(i * w + w / 2, -4, s, ha='center', va='bottom', fontsize=7, color=MUTED)
    ax.axis('off')


def panel_levels(ax, rgm, unit):
    for n, lev in enumerate(rgm):
        s = np.array(lev['states'])
        ax.scatter(np.full(len(s), n + 1) + np.linspace(-0.15, 0.15, len(s)), s, s=22,
                   color=BLUE, zorder=3)
        ax.annotate(f'{len(s)} group{"s" if len(s) > 1 else ""}\nmedian {int(np.median(s))}',
                    (n + 1, s.max()), xytext=(0, 8), textcoords='offset points', ha='center',
                    va='bottom', fontsize=7.3, color=INK)
    ax.set_xticks(range(1, len(rgm) + 1))
    ax.set_xticklabels([f'level {n + 1}\n{unit(n)}' for n in range(len(rgm))], fontsize=7.5)
    ax.set_xlim(0.4, len(rgm) + 0.6)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.25 * (hi - lo))
    ax.set_ylabel('latent states per group', fontsize=8, color=MUTED)
    clean(ax)


def panel_cycle(ax, B, name):
    ax.imshow(B, cmap='Greys', vmin=0, vmax=1)
    ax.set_title(name, fontsize=8, color=INK, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GREY)


def main():
    b, m = bird_analysis(), mnist_analysis()
    K = b['K']
    print('bird: distinct frames', b['distinct'], 'states', K, 'SP sizes',
          sorted(len(set(p)) for p in b['sps']), 'products',
          [(len(set(p)), len(set(q))) for p, q in b['prods']], 'checks', b['checks'],
          f'search {b["secs"]:.3f}s')
    for n, lev in enumerate(b['rgm']):
        print(f'  RGM bird level {n + 1}: T={lev["T"]} states {lev["states"]} paths {lev["paths"]}')
    for n, lev in enumerate(m['rgm']):
        print(f'  RGM MNIST level {n + 1}: {len(lev["groups"])} groups, median states '
              f'{int(np.median(lev["states"]))}, max {max(lev["states"])}')

    fig = plt.figure(figsize=(19, 10.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 0.9, 1.0, 0.9], hspace=0.55, wspace=0.38)

    # bird row
    ax = fig.add_subplot(gs[0, 0])
    strip(ax, b['F'][:12, 8:56, 4:60], [f'w{w}\nb{s}' for w, s in b['truth'][:12]])
    title(ax, '(a) A flapping bird (synthetic): wing phase w cycles every 8 frames,\n'
              'the body bobs (b) every 3; the wing looks the same going up and down', pad=30)
    ax = fig.add_subplot(gs[0, 1])
    panel_levels(ax, b['rgm'], lambda n: f'{2 * 2 ** n} frames/step')
    title(ax, '(b) RGM states: unique patterns per\ntile group, then per 2×2 groups in\n'
              'space and pairs of steps in time')
    sub = gs[0, 2].subgridspec(2, 2, height_ratios=[1, 1.6], hspace=0.5, wspace=0.3)
    axs = fig.add_subplot(sub[0, :]); axs.axis('off')
    p, q = b['prods'][0]
    axs.text(0, 1, f'{b["distinct"]} distinct frames → {K} predictive states\n'
             f'(2-frame contexts: up- and downstroke differ)\n'
             f'{len(b["sps"])} closed groupings; 1 product: '
             f'{K} = {len(set(p))} × {len(set(q))}\n'
             f'vs. truth (ARI): bob {b["checks"]["bob"]:.2f}, wing {b["checks"]["wing"]:.2f}',
             fontsize=8.3, color=INK, va='top', transform=axs.transAxes)
    title(axs, '(c) Ours: states, then the factor search', pad=4)
    panel_cycle(fig.add_subplot(sub[1, 0]), b['B'][0], f'factor 1: {len(set(p))} values')
    panel_cycle(fig.add_subplot(sub[1, 1]), b['B'][1], f'factor 2: {len(set(q))} values')
    ax = fig.add_subplot(gs[0, 3])
    rows = [('RGM top level, one variable', b['rgm'][-1]['states'][0] ** 2, GREY),
            (f'ours, one variable ({K} states)', K * K, BLUE),
            (f'ours, factored ({len(set(p))} + {len(set(q))})', len(set(p)) ** 2 + len(set(q)) ** 2,
             ORANGE)]
    for r, (lab, v, c) in enumerate(rows):
        ax.barh(r, v, color=c, height=0.45)
        ax.text(v + 8, r, str(v), fontsize=8, va='center', color=INK)
        ax.text(0, r - 0.32, lab, fontsize=7.8, va='bottom', color=INK)
    ax.set_yticks([])
    ax.invert_yaxis(); ax.set_xlim(0, 700)
    ax.set_xlabel('transition-matrix entries', fontsize=8, color=MUTED)
    clean(ax, grid='x')
    title(ax, f'(d) Size. RGM\'s top step spans 4 frames,\nso it needs only '
              f'{b["rgm"][-1]["states"][0]} states but cannot say\nwhich frame within a step',
          pad=18)

    # MNIST row
    ax = fig.add_subplot(gs[1, 0])
    pick = [int(np.flatnonzero(m['y'] == d)[0]) for d in range(10)]
    strip(ax, m['I'][pick], [str(d) for d in range(10)])
    title(ax, f'(e) MNIST as in DEM_MNIST_RGM: 13 images per digit for structure\n'
              f'learning, 32×32, smoothed, equalised; 4×4 tiles → {m["streams"]} SVD streams',
          pad=16)
    ax = fig.add_subplot(gs[1, 1])
    panel_levels(ax, m['rgm'], lambda n: '')
    title(ax, '(f) RGM states on MNIST: from level 2\non, every image is its own state')
    ax = fig.add_subplot(gs[1, 2:]); ax.axis('off')
    ax.text(0, 1, 'What the comparison shows\n', fontsize=10, weight='bold', color=INK, va='top',
            transform=ax.transAxes)
    ax.text(0, 0.86,
            '• RGM state generation is "every unique pattern is a state", applied to 2×2 tiles in '
            'space\n  and pairs of steps in time. The grouping is fixed by locality; the top level '
            'is one\n  hidden variable whose states enumerate what was seen (130 images; the 24-frame '
            'cycle\n  in 4-frame steps).\n'
            '• Our states are predictive equivalence classes, so they carry memory where a frame '
            'is\n  ambiguous (up- vs. downstroke). The factor search then splits them by the '
            'dynamics:\n  wing × bob, without being told about locality or time scales.\n'
            f'• Search cost: closures of the {K * (K - 1) // 2} state pairs and their joins, '
            f'{b["secs"]:.2f} s,\n  instead of the Bell({K}) = {bell(K):.2e} groupings of {K} '
            'states.\n'
            '• MNIST has no dynamics: our factorization (by transitions) does not apply. The '
            'question\n  there is which tiles are independent given the digit; RGM assumes '
            'neighbouring tiles.',
            fontsize=8.6, color=INK, va='top', transform=ax.transAxes, linespacing=1.45)

    fig.suptitle('RGM state generation (rgm_lite, after SPM) vs. predictive states + factor '
                 'search', fontsize=12, color=INK, x=0.06, ha='left', y=0.995)
    out = FIGS / 'fig_rgm_compare.png'
    fig.savefig(out, bbox_inches='tight', dpi=170, facecolor='white')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
