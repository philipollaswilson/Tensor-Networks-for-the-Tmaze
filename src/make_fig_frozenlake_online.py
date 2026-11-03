"""Figure: online frozen lake, EFE agent against random exploration, per round.

Reads results/frozenlake_online/{efe,random}_s{seed}.json (frozenlake_online.py)
and writes paper/figs/fig_frozenlake_online.png. Small multiples, one measure each:
goal rate, hole rate, learned states, (cell, action) pairs tried, B error.

Run from the repo root:  python src/make_fig_frozenlake_online.py [seed]
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_extension_figures import FIGS, REPO  # noqa: E402
from make_fig_svd_states import INK, MUTED, SERIES  # noqa: E402

POLICIES = [('efe', 'EFE agent', '-'), ('random', 'Random', '--')]
PANELS = [('goal_rate', 'Goal reached (share of episodes)', None),
          ('hole_rate', 'Fell in the hole (share of episodes)', None),
          ('n_states', 'Learned states (BMR)', 9),
          ('cell_action_pairs_tried', '(cell, action) pairs tried', 28),
          ('B_TV_mean', 'Learned B vs truth (mean TV, tried pairs)', None)]


def main(seed=0):
    runs = {}
    for key, _, _ in POLICIES:
        p = REPO / 'results' / 'frozenlake_online' / f'{key}_s{seed}.json'
        if p.exists():
            runs[key] = json.loads(p.read_text())
    fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 3.4))
    for ax, (field, title, truth) in zip(axes, PANELS):
        for k, (key, label, ls) in enumerate(POLICIES):
            if key not in runs:
                continue
            xs = [r['episodes_total'] for r in runs[key]]
            ys = [r[field] if r[field] is not None else float('nan') for r in runs[key]]
            ax.plot(xs, ys, ls, color=SERIES[k], lw=2, marker='o', ms=4, label=label)
        if truth is not None:
            ax.axhline(truth, color=MUTED, lw=0.8, ls=':')
            ax.text(ax.get_xlim()[0], truth, ' truth', color=MUTED, fontsize=7, va='bottom')
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel('episodes so far', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis='y', color='#e6e6e6', lw=0.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f'Online frozen lake (seed {seed}): act, append sites, re-learn states, plan', y=1.03)
    fig.tight_layout()
    out = FIGS / 'fig_frozenlake_online.png'
    fig.savefig(out); plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
