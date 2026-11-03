"""Figure + table: position-only Mountain Car, learners vs RL baselines.

  (a) return per episode (20-episode mean)   (b) cumulative regret vs the near-optimal
  (c) evaluation return, exploration off     controller
  (d) memory used: learned states (classical) / belief nodes (Born)
Lines are means over seeds, shaded per-seed range. The table gives the IQM over seeds
with a stratified-bootstrap 95% CI (rliable protocol) of total regret and of the
last evaluation return.

Reads results/<cond>/ (mountaincar_sl.py, mountaincar_born.py, mountaincar_rl.py,
mountaincar_dreamer.py). Writes paper/figs/fig_mountaincar[_moving].png.
Run from the repo root:  python src/make_fig_mountaincar.py [--moving]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mountaincar_env as E  # noqa: E402
from make_extension_figures import FIGS  # noqa: E402
from make_fig_svd_states import INK, MUTED, SERIES  # noqa: E402

REPO = HERE.parent
AGENTS = [  # key, label, color, linestyle
    ('efe', 'Classical tensor, EFE + novelty (ours)', SERIES[0], '-'),
    ('born_D8', 'Born machine D=8, EFE + novelty (ours)', SERIES[1], '-'),
    ('dreamerv3', 'DreamerV3 (size1m)', SERIES[2], '-'),
    ('psrl_state', 'PSRL, told the state', SERIES[3], '--'),
    ('psrl_learned', 'Posterior sampling, learned states (ours)', SERIES[0], ':'),
    ('qlearning', 'Q-learning, told the state', MUTED, '--'),
    ('psrl_obs', 'PSRL on the observation (no memory)', MUTED, ':'),
]


def load(cond):
    out = {}
    d = REPO / 'results' / cond
    for key, *_ in AGENTS:
        runs = []
        if key == 'dreamerv3':
            for run in sorted(d.glob('dreamerv3_s*')):
                f = run / 'episodes.jsonl'
                if not f.exists():
                    continue
                eps = [json.loads(line) for line in open(f)]
                ev = [json.loads(line) for line in open(run / 'eval_episodes.jsonl')] \
                    if (run / 'eval_episodes.jsonl').exists() else []
                log = [{'return': r['return'], 'regret': r['optimal'] - r['return']} for r in eps]
                for k in range(E.EVAL_EVERY - 1, len(log), E.EVAL_EVERY):   # eval nearest in time
                    near = [r['return'] for r in ev if abs(r['train_episodes'] - (k + 1)) <= 5]
                    if near:
                        log[k]['eval_return'] = float(np.mean(near))
                runs.append(log)
        else:
            runs = [json.load(open(f)) for f in sorted(d.glob(f'{key}_s*.json'))]
        if runs:
            out[key] = runs
    return out


def band(ax, Y, x, color, ls, label):
    ax.plot(x, Y.mean(0), ls, color=color, lw=2, label=label)
    if len(Y) > 1:
        ax.fill_between(x, Y.min(0), Y.max(0), color=color, alpha=0.12, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--moving', action='store_true')
    args = ap.parse_args()
    cond = 'mountaincar_moving' if args.moving else 'mountaincar'
    data = load(cond)
    n = min(len(r) for runs in data.values() for r in runs)
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.9))
    rows = []
    for key, label, color, ls in AGENTS:
        if key not in data:
            continue
        runs = [r[:n] for r in data[key]]
        R = np.array([[e['return'] for e in r] for r in runs])
        w = 20
        Rs = np.array([np.convolve(x, np.ones(w) / w, mode='valid') for x in R])
        band(axes[0], Rs, np.arange(w, n + 1), color, ls, label)
        reg = np.cumsum([[e['regret'] for e in r] for r in runs], 1)
        band(axes[1], reg, np.arange(1, n + 1), color, ls, label)
        ev_idx = [k for k in range(n) if all('eval_return' in r[k] for r in runs)]
        if ev_idx:
            Ev = np.array([[r[k]['eval_return'] for k in ev_idx] for r in runs])
            band(axes[2], Ev, np.array(ev_idx) + 1, color, ls, label)
        mem = 'n_states' if 'n_states' in runs[0][0] else 'n_nodes' if 'n_nodes' in runs[0][0] else None
        if mem:
            band(axes[3], np.array([[e[mem] for e in r] for r in runs], float),
                 np.arange(1, n + 1), color, ls, label)
        tot = reg[:, -1]
        last_ev = Ev[:, -1] if ev_idx else np.full(len(runs), np.nan)
        first = [next((i for i, x in enumerate(r) if x > -200), None) for r in R]
        rows.append((label, len(runs), E.iqm_ci(tot), E.iqm_ci(last_ev) if ev_idx else None, first))
    titles = ['(a) Return per episode (20-episode mean)', '(b) Cumulative regret',
              '(c) Evaluation return, exploration off', '(d) Memory: states / belief nodes']
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=9, color=INK)
        ax.set_xlabel('episode (200 steps each)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis='y', color='#e6e6e6', lw=0.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    for ax in axes[[0, 2]]:
        ax.axhline(-200, color=MUTED, lw=0.6, ls=':')
        ax.set_ylim(-205, -60)
    axes[1].legend(fontsize=7, frameon=False, loc='upper left')
    flag = 'moving flag (switches side every %d episodes)' % E.SWITCH if args.moving else 'flag on the right'
    fig.suptitle(f'Position-only Mountain Car ({E.NB} position bins, velocity hidden), {flag}; '
                 f'{n} episodes', y=1.03)
    out = FIGS / f'fig_{cond}.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}\n')
    print(f"{'agent':45s} seeds  total regret IQM [95% CI]      last eval return IQM [CI]   first flag")
    for label, k, reg, ev, first in rows:
        evs = f'{ev[0]:7.1f} [{ev[1]:.1f}, {ev[2]:.1f}]' if ev else '   n/a'
        print(f'{label:45s} {k:5d}  {reg[0]:8.0f} [{reg[1]:.0f}, {reg[2]:.0f}]   {evs:28s} {first}')


if __name__ == '__main__':
    main()
