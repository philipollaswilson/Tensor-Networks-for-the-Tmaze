"""Generate the paper figure: ray-fidelity matrices of the learned models.

Two panels: minimal-maze MPS (6 histories) and full-maze MPS (7 histories),
pairwise fidelity |<b_i|b_j>|^2 between the bond rays induced by each
action--observation history. Blocks of high fidelity = one learned hidden state.

Style matches the paper's existing heatmaps (imshow + gist_heat_r + colorbar),
with values annotated so the figure reads in grayscale print.

Run from the repo root:  python src/make_paper_figures.py [outdir]
"""
import pathlib
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else REPO


def load_tensors(name):
    mps = torch.load(REPO / 'Saved_Models' / name, weights_only=False,
                     map_location='cpu')
    return [(m.num if hasattr(m, 'num') else m).detach().cpu().numpy()
            for m in mps.matrices]


def fidelity(rays):
    return np.abs(np.array([[np.vdot(x, y) for y in rays] for x in rays])) ** 2


def minimal_rays():
    T1, T2 = load_tensors('MinimalTmaze.pt')
    hists = [('R / Cheese', 0, 0), ('R / Shock', 0, 1), ('L / Cheese', 1, 0),
             ('L / Shock', 1, 1), ('Cue / R', 2, 2), ('Cue / L', 2, 3)]
    labels, rays = [], []
    for lbl, a1, o2 in hists:
        b = np.einsum('xaoi->i', T1[:, a1:a1 + 1, o2:o2 + 1, :])
        labels.append(lbl)
        rays.append(b / np.linalg.norm(b))
    return labels, fidelity(rays)


def full_rays():
    T1, T2, T3 = load_tensors('FullTmaze.pt')

    def oi(p, r, c):
        return 6 * p + 2 * r + c

    hists = [('Center', 0, (0, 0, 0)), ('R / Cheese', 1, (1, 1, 0)),
             ('R / Shock', 1, (1, 2, 0)), ('L / Cheese', 2, (2, 1, 0)),
             ('L / Shock', 2, (2, 2, 0)), ('Cue / ctx R', 3, (3, 0, 0)),
             ('Cue / ctx L', 3, (3, 0, 1))]
    labels, rays = [], []
    for lbl, a2, otup in hists:
        b = np.einsum('i,ij->j', T1[0, 0, 0, :], T2[:, a2, oi(*otup), :])
        labels.append(lbl)
        rays.append(b / np.linalg.norm(b))
    return labels, fidelity(rays)


def panel(ax, labels, F, title):
    im = ax.imshow(F, cmap='gist_heat_r', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = F[i, j]
            if v < 0.005:
                continue
            ax.text(j, i, f'{v:.2f}'.lstrip('0') if v < 1 else '1',
                    ha='center', va='center', fontsize=7,
                    color='white' if v > 0.6 else 'black')
    return im


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2),
                             gridspec_kw={'width_ratios': [6, 7]})
    lab_m, F_m = minimal_rays()
    lab_f, F_f = full_rays()
    panel(axes[0], lab_m, F_m, 'Minimal maze (6 histories)')
    im = panel(axes[1], lab_f, F_f, 'Full pymdp maze (7 histories)')
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r'ray fidelity $|\langle b_i | b_j \rangle|^2$', fontsize=9)
    out = OUT / 'fig8.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    print('wrote', out)
    print('minimal maze fidelity:\n', np.round(F_m, 2))
    print('full maze fidelity:\n', np.round(F_f, 2))


if __name__ == '__main__':
    main()
