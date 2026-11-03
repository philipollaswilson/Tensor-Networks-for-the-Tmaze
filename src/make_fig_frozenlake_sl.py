"""Figure: online structure learning on the small frozen lake, against RL baselines.

  (a) goal reached (10-episode window)    (b) cumulative falls into the hole
  (c) states learned (bond dimension)     (d) the machine the EFE agent learned (seed 0)
Means over seeds with the per-seed range shaded. Agents: EFE structure learner
(frozenlake_sl.py; learns its states), PSRL (Dirichlet 0.1, as our agent) and Q-learning (frozenlake_rl.py; told
that the cell is the state), and random actions.

Writes paper/figs/fig_frozenlake_sl.png. Run from the repo root:
  python src/make_fig_frozenlake_sl.py [--seeds 5 --episodes 150]
"""
import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozenlake_rl as RL  # noqa: E402
import frozenlake_sl as SL  # noqa: E402
from make_extension_figures import FIGS  # noqa: E402
from make_fig_frozenlake import TILE  # noqa: E402
from make_fig_svd_states import INK, MUTED, SERIES  # noqa: E402

AGENTS = [('efe', 'EFE + novelty, learned states (ours)', '-', SL.run),
          ('psrl_learned', 'Posterior sampling, learned states (ours)', '-', SL.run),
          ('psrl_sparse', 'PSRL, given the state (best RL here)', '--', RL.run),
          ('qlearning', 'Q-learning, given the state', '-.', RL.run),
          ('random', 'Random', ':', SL.run)]
COLORS = SERIES + [MUTED]


def band(ax, Y, color, ls, label):
    x = np.arange(1, Y.shape[1] + 1)
    ax.plot(x, Y.mean(0), ls, color=color, lw=2, label=label)
    ax.fill_between(x, Y.min(0), Y.max(0), color=color, alpha=0.12, lw=0)


def draw_machine(ax, M):
    shown, _ = M.shown()
    for r in range(3):
        for c in range(3):
            t = SL.DESC[r][c]
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fc=TILE[t], ec='white', lw=2))
            ax.text(c - 0.4, r - 0.36, f'{t}{3 * r + c}', fontsize=7,
                    color='white' if t == 'H' else INK)
    pos = {}
    for s, o in enumerate(shown):                       # goal shows two states: +r and stuck
        c = o % SL.NC
        dx = 0.18 * (1 if o >= SL.NC else -1) if c == 8 else 0.0
        pos[s] = (c % 3 + dx, c // 3)
    p = M.p_obs()
    for s in range(M.K):
        for a in range(SL.NA):
            o = int(np.argmax(p[s, a]))
            t = M.f.get((s, a, o))
            if M.tried()[s, a] == 0 or p[s, a, o] < 0.5 or t is None or t == s:
                continue
            (x0, y0), (x1, y1) = pos[s], pos[t]
            if (x0, y0) == (x1, y1):
                continue
            ox, oy = 0.08 * np.sign(y1 - y0), -0.08 * np.sign(x1 - x0)
            ax.annotate('', (x0 + 0.68 * (x1 - x0) + ox, y0 + 0.68 * (y1 - y0) + oy),
                        (x0 + 0.22 * (x1 - x0) + ox, y0 + 0.22 * (y1 - y0) + oy),
                        arrowprops=dict(arrowstyle='->', color=SERIES[0], lw=1.2))
    for s, (x, y) in pos.items():
        ax.scatter(x, y, s=150, color=SERIES[0], edgecolor='white', lw=1.5, zorder=3)
        ax.text(x, y, f's{s}', color='white', fontsize=6, ha='center', va='center', zorder=4)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(2.5, -0.5); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--episodes', type=int, default=150)
    args = ap.parse_args()
    logs, machine = {}, None
    for key, _, _, fn in AGENTS:
        logs[key] = []
        for s in range(args.seeds):
            out = fn(key, s, args.episodes)
            log, M = out if isinstance(out, tuple) else (out, None)
            logs[key].append(log)
            if key == 'efe' and s == 0:
                machine = M

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8),
                             gridspec_kw={'width_ratios': [1, 1, 1, 0.8]})
    w = 10
    for k, (key, label, ls, _) in enumerate(AGENTS):
        goal = np.array([[r['result'] == 'goal' for r in log] for log in logs[key]], float)
        rate = np.array([np.convolve(g, np.ones(w) / w, mode='full')[:len(g)] for g in goal])
        holes = np.array([np.cumsum([r['result'] == 'hole' for r in log]) for log in logs[key]])
        band(axes[0], rate, COLORS[k], ls, label)
        band(axes[1], holes, COLORS[k], ls, label)
        if 'n_states' in logs[key][0][0]:
            states = np.array([[r['n_states'] for r in log] for log in logs[key]])
            band(axes[2], states, COLORS[k], ls, label)
    axes[2].axhline(10, color=MUTED, lw=0.8, ls=':')
    axes[2].text(args.episodes, 10, 'minimal machine: 10', color=MUTED, fontsize=7,
                 ha='right', va='bottom')
    titles = ['(a) Goal reached (10-episode window)', '(b) Falls into the hole (cumulative)',
              '(c) States learned (bond dimension)']
    for ax, title in zip(axes[:3], titles):
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel('episode (100 steps each)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis='y', color='#e6e6e6', lw=0.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[2].set_xscale('log')
    axes[1].legend(fontsize=7.5, frameon=False, loc='upper left')
    draw_machine(axes[3], machine)
    axes[3].set_title('(d) Machine learned by EFE (seed 0);\ngoal: entered (+r, right) / stuck (left)',
                      fontsize=9, color=INK)
    fig.suptitle(f'3x3 frozen lake, no task knowledge: observe (cell, reward), stuck after hole '
                 f'or goal, 100-step episodes ({args.seeds} seeds)', y=1.04)
    out = FIGS / 'fig_frozenlake_sl.png'
    fig.savefig(out); plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
