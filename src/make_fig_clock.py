"""Figure: the clock (clock_born.py) against current methods (clock_compare.py).
Left: held-out gap above the true model vs training set size. Right: the hazard each model
implies at N = 2,000, against the truth.
Run from the repo root:  python src/make_fig_clock.py
"""
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_fig_hidden_states import clean, title, INK, MUTED  # noqa: E402
from make_extension_figures import FIGS  # noqa: E402

RES = HERE.parent / 'results' / 'clock'
NS = [100, 300, 1000, 2000]
SERIES = [('born', 'Born MPS, bond 2', '#2a78d6', 'o'),
          ('hmm', 'HMM, warm-started', '#eb6834', 's'),
          ('gru', 'GRU', '#7a4fd0', 'D'),
          ('count', 'Count learner', '#1f8a65', '^'),
          ('spectral', 'Spectral (SVD)', '#a88a10', 'v')]


def main():
    R = [json.loads((RES / ('compare_s%d.json' % s)).read_text()) for s in range(3)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1, 1.25]})

    for key, label, col, mk in SERIES:
        g = np.array([[r['runs'][str(n)][key]['gap'] for r in R] for n in NS])
        m = g.mean(1)
        a.errorbar(NS, m, yerr=[m - g.min(1), g.max(1) - m], color=col, marker=mk, ms=6,
                   lw=2, capsize=0, elinewidth=1, label=label, zorder=4 if key == 'born' else 3)
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xticks(NS)
    a.set_xticklabels([str(n) for n in NS])
    a.minorticks_off()
    a.set_xlabel('Training sequences (60 symbols each)', color=MUTED)
    a.set_ylabel('Held-out bits/symbol above truth', color=MUTED)
    a.grid(axis='y', color='#e8e7e1', lw=0.8)
    a.legend(frameon=False, fontsize=8.5, ncol=2, labelcolor=INK)
    clean(a)
    title(a, 'Born is closest at every data size')

    n = np.arange(R[0]['true_hazard'].__len__())
    h = lambda key: np.mean([r['runs']['2000'][key]['hazard'] for r in R], 0)
    b.plot(n, R[0]['true_hazard'], color=INK, lw=2.5, label='Truth', zorder=2)
    for key, label, col, mk in (SERIES[0], SERIES[2], SERIES[3]):
        b.plot(n, h(key), color=col, lw=1.5, marker=mk, ms=4, label=label, zorder=3)
    b.set_xlabel('Zeros since the last 1', color=MUTED)
    b.set_ylabel('p(next symbol is 1)', color=MUTED)
    b.grid(axis='y', color='#e8e7e1', lw=0.8)
    b.legend(frameon=False, fontsize=8.5, ncol=4, loc='upper center',
             bbox_to_anchor=(0.5, -0.17), labelcolor=INK)
    clean(b)
    title(b, 'Counting loses the rhythm where data run out; Born and the GRU keep it')

    fig.tight_layout()
    fig.savefig(FIGS / 'fig_clock.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
