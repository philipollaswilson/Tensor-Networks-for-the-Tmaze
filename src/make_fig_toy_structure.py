"""Figure: the structure-learning rule on a toy you can see whole.

A hidden coin (L or R, 50/50). Two steps, each showing a hint of the coin (l or r) with
probability 0.4, else a blank (-). Then the coin is revealed. What must be remembered from
the two hint symbols to predict the reveal? Counted from samples, merged by BMR, unmixed.
Run from the repo root:  python src/make_fig_toy_structure.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gammaln

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_fig_hidden_states import clean, title, INK, MUTED  # noqa: E402
from make_extension_figures import FIGS  # noqa: E402

BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#c3c2b7'
HIST = ['ll', 'l-', '-l', '--', '-r', 'r-', 'rr']
COL = {'ll': BLUE, 'l-': BLUE, '-l': BLUE, '--': GREY, '-r': ORANGE, 'r-': ORANGE, 'rr': ORANGE}
ALPHA = np.full(2, 0.5)                           # Perks prior over the reveal {L, R}


def lnB(x):
    return gammaln(x).sum(-1) - gammaln(x.sum(-1))


def sample(n, p_hint=0.4, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = rng.choice(['L', 'R'])
        h = ''.join(c.lower() if rng.random() < p_hint else '-' for _ in range(2))
        out.append((h, c))
    return out


def dF(a, b):
    """ln evidence for one shared row minus two separate rows (> 0: merge)."""
    return lnB(ALPHA + a + b) + lnB(ALPHA) - lnB(ALPHA + a) - lnB(ALPHA + b)


def main():
    data = sample(400)
    N = {h: np.zeros(2) for h in HIST}
    for h, c in data:
        N[h][0 if c == 'L' else 1] += 1
    P = {h: N[h] / N[h].sum() for h in HIST}
    D = np.array([[dF(N[a], N[b]) if a != b else np.nan for b in HIST] for a in HIST])
    # greedy BMR
    groups = [[h] for h in HIST]
    while True:
        best, pair = 0, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                v = dF(sum(N[h] for h in groups[i]), sum(N[h] for h in groups[j]))
                if v > best:
                    best, pair = v, (i, j)
        if pair is None:
            break
        i, j = pair
        groups[i] += groups.pop(j)
    print('groups', groups)
    # unmix: is the blank-blank row a mixture of the other two states?
    sL = sum(N[h] for h in ('ll', 'l-', '-l'))
    sR = sum(N[h] for h in ('rr', 'r-', '-r'))
    pL, pR = sL / sL.sum(), sR / sR.sum()
    w = (P['--'][0] - pR[0]) / (pL[0] - pR[0])
    print('mixture weight on L', w, 'counts', {h: N[h].tolist() for h in HIST})

    fig = plt.figure(figsize=(19, 9.6))
    gs = fig.add_gridspec(2, 4, hspace=0.6, wspace=0.38, height_ratios=[1, 1])

    # (1) the process
    ax = fig.add_subplot(gs[0, 0]); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    box = dict(boxstyle='round,pad=0.4', fc='#f4f3ee', ec=GREY)
    ax.text(3.2, 8.4, 'hidden coin\nL or R (50/50)', fontsize=9, color=INK, bbox=box,
            va='center', ha='center')
    for k, x in enumerate((1.2, 4.6)):
        ax.text(x, 4.9, f'step {k + 1}\nhint (40%)\nor blank -', fontsize=8.3, color=INK,
                bbox=box, va='center', ha='center')
    ax.text(8.3, 4.9, 'reveal\nL or R', fontsize=8.3, color=INK, bbox=box, va='center',
            ha='center')
    for x in (1.2, 4.6, 8.3):
        ax.annotate('', (x, 6.1), (3.2, 7.6), arrowprops=dict(arrowstyle='->', color=MUTED, lw=1))
    ax.annotate('', (3.6, 4.9), (2.3, 4.9), arrowprops=dict(arrowstyle='->', color=INK, lw=1))
    ax.annotate('', (7.5, 4.9), (5.7, 4.9), arrowprops=dict(arrowstyle='->', color=INK, lw=1))
    ax.text(0.2, 1.6, 'What do you need to remember\nfrom the two hint symbols to predict\n'
            'the reveal? Nobody tells the learner\nabout the coin; it only sees\n'
            'sequences like  "-l L",  "-- R",  "rr R".', fontsize=8.8, color=INK, va='center')
    title(ax, '(1) A process you can see whole')

    # (2) counts
    ax = fig.add_subplot(gs[0, 1])
    y = np.arange(len(HIST))
    ax.barh(y, [N[h][0] for h in HIST], color='#8fb8e6', height=0.6, label='then L')
    ax.barh(y, [N[h][1] for h in HIST], left=[N[h][0] for h in HIST], color='#f3b08f',
            height=0.6, label='then R')
    for k, h in enumerate(HIST):
        ax.text(N[h].sum() + 2, k, f'{int(N[h][0])} L, {int(N[h][1])} R', va='center',
                fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([f'"{h}"' for h in HIST], fontsize=9, family='monospace')
    ax.invert_yaxis(); ax.set_xlim(0, 190)
    ax.set_xlabel('sequences (400 sampled)', fontsize=8, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, loc='lower right')
    clean(ax, grid='x')
    title(ax, '(2) COUNT. One row per history (what was\nseen so far): how often each reveal '
              'followed.\nThis table is all the learner ever uses')

    # (3) fingerprints
    ax = fig.add_subplot(gs[0, 2])
    for k, h in enumerate(HIST):
        ax.barh(k, P[h][0], color=COL[h], height=0.6)
        ax.text(P[h][0] + 0.02, k, f'{P[h][0]:.2f}', va='center', fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([f'"{h}"' for h in HIST], fontsize=9, family='monospace')
    ax.invert_yaxis(); ax.set_xlim(0, 1.15)
    ax.set_xlabel('p(reveal = L | history)', fontsize=8, color=MUTED)
    clean(ax, grid='x')
    title(ax, '(3) FINGERPRINT. Normalise each row: what\nthe history predicts. Three '
              'different values\nappear, though there are seven histories')

    # (4) BMR matrix
    ax = fig.add_subplot(gs[0, 3])
    lim = 40
    im = ax.imshow(np.clip(D, -lim, lim), cmap='RdBu', vmin=-lim, vmax=lim)
    for (i, j), v in np.ndenumerate(D):
        if np.isfinite(v):
            ax.text(j, i, f'{v:.0f}' if abs(v) >= 1 else f'{v:.1f}', ha='center', va='center',
                    fontsize=7, color='white' if abs(v) > 25 else INK)
    ax.set_xticks(y); ax.set_xticklabels(HIST, fontsize=8, family='monospace')
    ax.set_yticks(y); ax.set_yticklabels(HIST, fontsize=8, family='monospace')
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.set_label('ΔF (blue: merge)', fontsize=7.5, color=MUTED)
    title(ax, '(4) MERGE. For each pair: is one shared row\na better explanation of both '
              'count rows\nthan two? ΔF = ln p(one) − ln p(two)')

    # (5) merged states
    ax = fig.add_subplot(gs[1, 0]); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    names = {'ll': 'knows L', '-r': 'knows R', '--': 'no hint yet'}
    ys = [8, 5, 2]
    for g, yy in zip(sorted(groups, key=lambda g: HIST.index(g[0])), ys):
        c = COL[g[0]]
        ax.add_patch(plt.Circle((1.3, yy), 0.9, color=c, alpha=0.9))
        lab = 'knows L' if c == BLUE else 'knows R' if c == ORANGE else 'no hint'
        ax.text(1.3, yy, lab.replace(' ', '\n'), ha='center', va='center', fontsize=8,
                color='white' if c != GREY else INK)
        ax.text(2.7, yy, '  '.join(f'"{h}"' for h in g), va='center', fontsize=10,
                family='monospace', color=INK)
    title(ax, f'(5) Greedy merging while ΔF > 0 leaves\n{len(groups)} states. Order and '
              f'repetition of hints\nare forgotten: they predict nothing extra')

    # (6) unmix
    ax = fig.add_subplot(gs[1, 1])
    xs = np.linspace(0, 1, 2)
    ax.plot([0, 1], [0, 0], color=MUTED, lw=1)
    ax.scatter([pR[0], pL[0]], [0, 0], s=160, color=[ORANGE, BLUE], zorder=3)
    ax.scatter([P['--'][0]], [0], s=120, color=GREY, marker='D', zorder=3)
    ax.annotate('knows R', (pR[0], 0), xytext=(0, 14), textcoords='offset points', ha='center',
                fontsize=8.5, color=INK)
    ax.annotate('knows L', (pL[0], 0), xytext=(0, 14), textcoords='offset points', ha='center',
                fontsize=8.5, color=INK)
    ax.annotate(f'"--" = {w:.2f} × knows L\n     + {1 - w:.2f} × knows R', (P['--'][0], 0),
                xytext=(0, -38), textcoords='offset points', ha='center', fontsize=8.5, color=INK)
    ax.text(0.5, -0.75, 'Caveat: with one yes/no prediction every row lies\non this line. In the '
            'T-maze (12 outcomes, 4 actions)\nbeing a mixture is a real test, passed only by '
            'the centre', ha='center', fontsize=7.8, color=MUTED)
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(-1.05, 0.6); ax.set_yticks([])
    ax.set_xlabel('p(reveal = L)', fontsize=8, color=MUTED)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    title(ax, f'(6) UNMIX. "No hint" predicts between the two\nothers ({w:.2f}; 0.5 in truth): '
              'not a third situation,\nbut not knowing which of two. A belief')

    # (7) the model
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    for x, c, lab in ((2.5, BLUE, 'L'), (7.0, ORANGE, 'R')):
        ax.add_patch(plt.Circle((x, 6.2), 1.0, color=c))
        ax.text(x, 6.2, lab, ha='center', va='center', fontsize=13, color='white', weight='bold')
        ax.text(x, 4.4, f'hint {lab.lower()} or -,\nthen reveal {lab}', ha='center', va='top',
                fontsize=8.3, color=INK)
    ax.text(4.75, 9.0, f'start: D = ({w:.2f}, {1 - w:.2f})', ha='center', fontsize=9.5, color=INK)
    ax.text(0.3, 1.0, 'One hidden variable with 2 values, a start belief,\nand what each value emits.'
            ' Seeing a hint updates\nthe belief; it does not move the coin.', fontsize=8.8,
            color=INK)
    title(ax, '(7) The learned generative model:\nthe coin, recovered from counts')

    # (8) same rule, other variables
    ax = fig.add_subplot(gs[1, 3]); ax.axis('off')
    ax.text(0, 1, 'The rule never used time:\n', fontsize=10, weight='bold', color=INK,
            va='top', transform=ax.transAxes)
    ax.text(0, 0.86, 'states of A = groups of A-values with the\nsame p(B | A)\n\n'
            '• past → future: states are memory\n'
            '• image region → rest of image: tree bonds\n'
            '• image → label: classes\n'
            '• word context → next word: word classes\n\n'
            'Time adds a free cut (past | future) and\nreuse: one small machine serves every '
            'step.\nWithout it, the cuts must be searched for.', fontsize=8.8, color=INK,
            va='top', transform=ax.transAxes, linespacing=1.4)
    title(ax, '(8) Beyond sequences')

    fig.suptitle('Structure learning in one rule: two histories are the same state when they '
                 'predict the same future', fontsize=12.5, color=INK, x=0.06, ha='left', y=0.995)
    out = FIGS / 'fig_toy_structure.png'
    fig.savefig(out, bbox_inches='tight', dpi=170, facecolor='white')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
