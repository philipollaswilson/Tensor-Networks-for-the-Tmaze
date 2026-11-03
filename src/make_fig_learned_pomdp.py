"""Figure: the generative model built from the learned states, and the agent's beliefs.

  (a) likelihood A = p(o | s) over the learned states
  (b) transition B = p(s' | s, a), one block per action
  (c) beliefs q(s) through two episodes: the agent's own choice (Cue first) and a
      forced arm entry, with the action chosen by sophisticated EFE at each step

Style follows make_extension_figures.fig_ab. Writes paper/figs/fig_learned_pomdp.png.
Run from the repo root:  python src/make_fig_learned_pomdp.py [seed]
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from learned_pomdp import ACT, NULL, OBS, learned_model, preferences, risk, infogain, sophisticated  # noqa: E402
from make_extension_figures import FIGS, HEAT, annot  # noqa: E402

GAMMA = 16.0


def short(lab):
    """'{Left/Cheese, Right/Cheese}' -> 'Cheese trap'; '{Cue/LeftCue}' -> 'Cue: Left'."""
    if lab == 'Start':
        return 'Start'
    o = lab.strip('{}').split(', ')[0].split('/')[1]
    return {'Cheese': 'Cheese trap', 'Shock': 'Shock trap'}.get(o, f'Cue: {o[:-3]}')


def choose(A, B, C, q):
    """Horizon-1 EFE action from belief q (the last step has no further horizon)."""
    G = np.array([risk(A @ B[:, :, a] @ q, C) - infogain(A, B[:, :, a] @ q)
                  for a in range(B.shape[2])])
    return int(np.argmin(G))


def episode(A, B, C, first, o_first):
    """Beliefs at each step: start, predicted after a1, posterior after o1, ... ."""
    q = np.eye(B.shape[0])[0]
    cols, names = [q], ['start']
    a = first
    for step, o in enumerate([o_first, None]):
        q = B[:, :, a] @ q
        cols.append(q); names.append(f'do {ACT[a]}\n(predict)')
        if o is None:                                   # observe what the model expects most
            o = int(np.argmax(A @ q))
        q = A[o] * q; q /= q.sum()
        cols.append(q); names.append(f'see\n{OBS[o]}')
        if step == 0:
            a = choose(A, B, C, q)
    return np.array(cols).T, names


def main(seed=0):
    A, B, labels = learned_model(seed)
    S = [short(l) for l in labels]
    C = preferences(2, -4)
    _, q_a1 = sophisticated(A, B, C, GAMMA)
    agent_a1 = int(np.argmax(q_a1))

    fig = plt.figure(figsize=(11, 7))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.7)
    gs = outer[0].subgridspec(1, 2, width_ratios=[4, 9], wspace=0.35)
    gb = outer[1].subgridspec(1, 2, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(A, cmap=HEAT, vmin=0, vmax=1, aspect='auto')
    ax.set_title(r'Likelihood $A=p(o\,|\,s)$')
    ax.set_yticks(range(len(OBS))); ax.set_yticklabels(OBS, fontsize=8)
    ax.set_xticks(range(len(S))); ax.set_xticklabels(S, rotation=90, fontsize=7)
    annot(ax, A, small=7)

    ax = fig.add_subplot(gs[0, 1])
    Bw = np.concatenate([B[:, :, a] for a in range(len(ACT))], 1)
    ax.imshow(Bw, cmap=HEAT, vmin=0, vmax=1, aspect='auto')
    ax.set_title(r"Transition $B=p(s'\,|\,s,a)$, one block per action", pad=20)
    ax.set_yticks(range(len(S))); ax.set_yticklabels(S, fontsize=8)
    ax.set_ylabel("next state $s'$", fontsize=8)
    ax.set_xticks(range(Bw.shape[1])); ax.set_xticklabels(S * len(ACT), rotation=90, fontsize=6)
    for a in range(len(ACT)):
        if a:
            ax.axvline(a * len(S) - 0.5, color='k', lw=0.8)
        ax.text((a + 0.5) * len(S) - 0.5, -0.65, f'a = {ACT[a]}', ha='center', fontsize=8.5, weight='bold')
    annot(ax, Bw, small=6)

    for k, (first, o1, title) in enumerate([
            (agent_a1, OBS.index('LeftCue'),
             f'Agent\'s choice: {ACT[agent_a1]} first ($q$ = {q_a1[agent_a1]:.3f})'),
            (ACT.index('Left'), OBS.index('Shock'), 'Forced arm entry: Left first')]):
        ax = fig.add_subplot(gb[0, k])
        Q, names = episode(A, B, C, first, o1)
        ax.imshow(Q, cmap=HEAT, vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'Beliefs $q(s)$. {title}', fontsize=9)
        ax.set_yticks(range(len(S))); ax.set_yticklabels(S, fontsize=8)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7)
        annot(ax, Q, small=7)

    fig.colorbar(im, ax=fig.axes, shrink=0.6, pad=0.02, label='probability')
    out = FIGS / 'fig_learned_pomdp.png'
    fig.savefig(out); plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
