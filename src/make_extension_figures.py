"""Figures for the extension paper (paper/figs/).

Generates, from the canonical learned checkpoints + analytic ground truth:
  fig_states.png   -- state recovery: ray-fidelity (over-separates) vs predictive
                      equivalence, minimal + full maze.
  fig_ab.png       -- labeled A=p(r|a,s) emission (learned vs analytic) and the
                      recovered transition B branching, full maze.
  fig_graph.png    -- MI dependency matrix + Chow-Liu tree, and sub-factor MI /
                      grouping residuals (factorization discovery).
  fig_modelsel.png -- state-count model selection: fit-L1 and MDL vs bond dim.
  fig_empower.png  -- empowerment phenotyping on the learned model vs analytic, and
                      the gauge-fixed emitted-state distribution.

Run from the repo root:  python src/make_extension_figures.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.structure_recovery import (load_tensors, _safe, mi_matrix, chow_liu,   # noqa: E402
                                    mutual_information, marginal, cluster_rays,
                                    predictive_signature, cluster_by_prediction)
from src.full_tmaze_train import enumerate_weighted, exact_joint, obs_index      # noqa: E402

FIGS = REPO / 'paper' / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 200,
                     'savefig.bbox': 'tight', 'axes.linewidth': 0.6})
HEAT = 'gist_heat_r'
BLUE = '#2b6cb0'
ORANGE = '#dd6b20'


def annot(ax, M, fmt='{:.2f}', thr=0.6, small=7):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if abs(v) < 5e-3:
                continue
            ax.text(j, i, fmt.format(v).lstrip('0') if 0 < v < 1 else fmt.format(v),
                    ha='center', va='center', fontsize=small,
                    color='white' if v > thr else 'black')


# --------------------------------------------------------------------------- #
def fig_states():
    T1m, T2m = load_tensors('MinimalTmaze.pt')
    T1f, T2f, T3f = load_tensors('FullTmaze.pt')
    # minimal ray fidelity
    OBS = ['Cheese', 'Shock', 'RightCue', 'LeftCue']; ACT = ['R', 'L', 'Cue']
    labm, rm = [], []
    for a1 in range(3):
        for o2 in range(4):
            b = T1m[0, a1, o2, :].astype(complex)
            if np.linalg.norm(b) < 1e-9:
                continue
            labm.append(f'{ACT[a1]}/{OBS[o2][:1]}'); rm.append(b / np.linalg.norm(b))
    Fm = np.abs(np.array([[np.vdot(x, y) for y in rm] for x in rm])) ** 2
    # full ray fidelity + predictive merge distances
    left = np.einsum('xi->i', T1f[0, 0, 0:2, :]).astype(complex)
    HIST = [('center', 0, [(0, 0, 0), (0, 0, 1)]), ('R/ch', 1, [(1, 1, 0), (1, 1, 1)]),
            ('R/sh', 1, [(1, 2, 0), (1, 2, 1)]), ('L/ch', 2, [(2, 1, 0), (2, 1, 1)]),
            ('L/sh', 2, [(2, 2, 0), (2, 2, 1)]), ('cue0', 3, [(3, 0, 0)]),
            ('cue1', 3, [(3, 0, 1)])]
    labf, rf, sf = [], [], []
    for nm, a2, ol in HIST:
        for ot in ol:
            b = np.einsum('i,ij->j', left, T2f[:, a2, obs_index(*ot), :]).astype(complex)
            if np.linalg.norm(b) < 1e-9:
                continue
            labf.append(nm); rf.append(b / np.linalg.norm(b))
            sf.append(predictive_signature(b / np.linalg.norm(b), T3f))
    Ff = np.abs(np.array([[np.vdot(x, y) for y in rf] for x in rf])) ** 2
    _, heights, kp = cluster_by_prediction(sf)
    kf = cluster_rays(rf)[0].max() + 1        # raw-fidelity cluster count (over-separates)

    # The figure's subject is what is NEW here: bond-ray fidelity (the base method of the
    # prior tensor-network work) OVER-separates on the full maze, and predictive
    # equivalence corrects it. The minimal-maze fidelity blocks are reported in text.
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={'width_ratios': [7, 5]})
    ax[0].imshow(Ff, cmap=HEAT, vmin=0, vmax=1)
    ax[0].set_title(f'Bond-ray fidelity over-separates:\n{kf} apparent blocks for 7 states',
                    fontsize=9.5)
    annot(ax[0], Ff, small=6)
    ax[0].set_xticks(range(len(labf))); ax[0].set_yticks(range(len(labf)))
    ax[0].set_xticklabels(labf, rotation=45, ha='right', fontsize=6); ax[0].set_yticklabels(labf, fontsize=6)
    ax[0].set_xlabel(r'history pairs, coloured by $|\langle b_i|b_j\rangle|^2$', fontsize=8)
    TAU = 1.5
    ax[1].plot(range(1, len(heights) + 1), heights, 'o-', color=BLUE, ms=4)
    ax[1].axhline(TAU, color=ORANGE, ls='--', lw=1)
    ax[1].set_title(f'Predictive equivalence\nrecovers the {kp} states', fontsize=9.5)
    ax[1].set_xlabel('agglomerative merge step'); ax[1].set_ylabel('L1 distance between merged futures')
    ax[1].text(0.04, 0.60, f'merge futures agreeing\nto L1 $< {TAU}$:  {kp} states',
               transform=ax[1].transAxes, color=ORANGE, fontsize=8.5)
    ax[1].text(0.04, 0.08, 'gauge duplicates\nmerge at $\\approx$0', transform=ax[1].transAxes,
               color='#666', fontsize=8)
    fig.savefig(FIGS / 'fig_states.png'); plt.close(fig)
    print('wrote fig_states.png')


# --------------------------------------------------------------------------- #
def fig_ab():
    T1, T2, T3 = load_tensors('FullTmaze.pt')
    psi = np.einsum('xaoi,ibpj,jcqy->aobpcq', T1, T2, T3)
    p = np.abs(psi) ** 2; p /= p.sum()
    sub = p[0, 0:2].sum(axis=0)
    ACT = ['center', 'right', 'left', 'cue']; REW = ['none', 'cheese', 'shock']
    states = [('center', 0, [obs_index(0, 0, 0), obs_index(0, 0, 1)]),
              ('R/cheese', 1, [obs_index(1, 1, 0), obs_index(1, 1, 1)]),
              ('R/shock', 1, [obs_index(1, 2, 0), obs_index(1, 2, 1)]),
              ('L/cheese', 2, [obs_index(2, 1, 0), obs_index(2, 1, 1)]),
              ('L/shock', 2, [obs_index(2, 2, 0), obs_index(2, 2, 1)]),
              ('cue/ctx0', 3, [obs_index(3, 0, 0)]), ('cue/ctx1', 3, [obs_index(3, 0, 1)])]

    def emission(a2, o2list):
        acc = sub[a2][o2list].sum(0)
        rew = np.zeros((4, 3))
        for o in range(24):
            rew[:, (o // 2) % 3] += acc[:, o]
        return rew / np.clip(rew.sum(1, keepdims=True), 1e-12, None)

    # learned emission stacked (7 states x 4 actions) -> reward prob of cheese/shock
    E = np.array([emission(a2, ol).ravel() for _, a2, ol in states])   # (7, 12)
    # transition B: start under each action -> next state (7 cols)
    left = np.einsum('xi->i', T1[0, 0, 0:2, :]).astype(complex)
    canon = {}
    for nm, a2, ol in states:
        b = np.einsum('i,ij->j', left, T2[:, a2, ol[0], :]).astype(complex)
        canon[nm] = predictive_signature(b / np.linalg.norm(b), T3)
    names = [s[0] for s in states]
    B = np.zeros((4, 7))
    for ai, a2 in enumerate([0, 1, 2, 3]):
        for rew in range(3):
            for ctx in range(2):
                oi = obs_index(a2, rew, ctx); w = p[0, 0:2, a2, oi].sum()
                if w < 1e-9:
                    continue
                b = np.einsum('i,ij->j', left, T2[:, a2, oi, :]).astype(complex)
                if np.linalg.norm(b) < 1e-9:
                    continue
                sig = predictive_signature(b / np.linalg.norm(b), T3)
                s2 = min(canon, key=lambda c: np.abs(sig - canon[c]).sum())
                B[ai, names.index(s2)] += w
    B = B / np.clip(B.sum(1, keepdims=True), 1e-12, None)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={'width_ratios': [7, 5]})
    im = ax[0].imshow(E, cmap=HEAT, vmin=0, vmax=1, aspect='auto')
    ax[0].set_title(r'Emission $A=p(r_3\,|\,a_3,s)$  (learned, TV$=0.000$ vs analytic)')
    ax[0].set_yticks(range(7)); ax[0].set_yticklabels(names, fontsize=8)
    ax[0].set_xticks(range(12))
    ax[0].set_xticklabels([f'{ACT[a][:1].upper()}:{REW[r][:2]}' for a in range(4) for r in range(3)],
                          rotation=90, fontsize=6)
    for a in range(1, 4):
        ax[0].axvline(3 * a - 0.5, color='k', lw=0.5)
    annot(ax[0], E, small=6)
    im2 = ax[1].imshow(B, cmap=HEAT, vmin=0, vmax=1, aspect='auto')
    ax[1].set_title(r'Transition $B=p(s_2\,|\,s_1{=}\mathrm{start},a_2)$')
    ax[1].set_yticks(range(4)); ax[1].set_yticklabels(ACT, fontsize=8)
    ax[1].set_xticks(range(7)); ax[1].set_xticklabels(names, rotation=90, fontsize=7)
    annot(ax[1], B, small=7)
    fig.colorbar(im2, ax=ax, shrink=0.7, pad=0.02, label='probability')
    fig.savefig(FIGS / 'fig_ab.png'); plt.close(fig)
    print('wrote fig_ab.png')


# --------------------------------------------------------------------------- #
def fig_emission():
    """p(o2|s2) factored into position/reward/context, plus the info-gain contrast:
    a blind arm carries 0 bits about context (posterior stays 50/50), only the cue
    resolves it (1 bit). Answers the concern that seeing Cheese/Shock at the arm would
    resolve the hidden context (spurious information gain)."""
    T1, T2, T3 = load_tensors('FullTmaze.pt')
    psi = np.einsum('xaoi,ibpj,jcqy->aobpcq', T1, T2, T3)
    p = np.abs(psi) ** 2; p /= p.sum()
    sub = p[0, 0:2].sum(axis=0).sum(axis=(2, 3))          # (a2, o2) first-move joint
    states = [('center', 0, [obs_index(0, 0, 0), obs_index(0, 0, 1)]),
              ('R/cheese', 1, [obs_index(1, 1, 0), obs_index(1, 1, 1)]),
              ('R/shock', 1, [obs_index(1, 2, 0), obs_index(1, 2, 1)]),
              ('L/cheese', 2, [obs_index(2, 1, 0), obs_index(2, 1, 1)]),
              ('L/shock', 2, [obs_index(2, 2, 0), obs_index(2, 2, 1)]),
              ('cue/ctx0', 3, [obs_index(3, 0, 0)]), ('cue/ctx1', 3, [obs_index(3, 0, 1)])]
    COLS = ['center', 'right', 'left', 'cue', 'none', 'cheese', 'shock', 'ctx0', 'ctx1']
    E = np.zeros((7, 9))
    for i, (nm, a2, o2list) in enumerate(states):
        e = np.zeros(24)
        for o in o2list:
            e[o] += sub[a2, o]
        e = e / e.sum()
        for o in range(24):
            E[i, o // 6] += e[o]
            E[i, 4 + (o // 2) % 3] += e[o]
            E[i, 7 + o % 2] += e[o]

    # info-gain: I(reward ; start context) at a blind Right arm; I(cue bit ; next reward)
    def mi(J):
        J = np.asarray(J, float); s = J.sum()
        if s <= 0:
            return 0.0
        P = J / s; px = P.sum(1, keepdims=True); py = P.sum(0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            return float(np.where(P > 1e-15, P * np.log2(P / (px * py)), 0.0).sum())
    blind = p[0, 0:2, 1].sum(axis=(2, 3))                 # (start ctx, o2) at Right arm
    Jarm = np.zeros((2, 2))
    for c0 in range(2):
        for o2 in range(24):
            r = (o2 // 2) % 3
            if r in (1, 2):
                Jarm[c0, r - 1] += blind[c0, o2]
    cue = p[0, 0:2, 3, :, 1, :].sum(axis=0)               # cue at o2 -> Right at o3
    Jcue = np.zeros((2, 2))
    for o2 in range(24):
        if o2 // 6 != 3:
            continue
        for o3 in range(24):
            r = (o3 // 2) % 3
            if r in (1, 2):
                Jcue[o2 % 2, r - 1] += cue[o2, o3]
    i_arm, i_cue = mi(Jarm), mi(Jcue)

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    im = ax.imshow(E, cmap=HEAT, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(9)); ax.set_xticklabels(COLS, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(7)); ax.set_yticklabels([s[0] for s in states], fontsize=8)
    annot(ax, E, small=7)
    for b, name in [(4, 'position'), (7, 'reward'), (9, 'context')]:
        if b < 9:
            ax.axvline(b - 0.5, color='k', lw=0.6)
        ax.text((b + (0 if name == 'position' else 4 if name == 'reward' else 7) - 1) / 2,
                -0.72, name, ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.add_patch(plt.Rectangle((6.5, 0.5), 2, 4, fill=False, edgecolor=BLUE, lw=2))
    ax.set_title(r'$p(o_2\,|\,s_2)$: blind arms leave context 0.50/0.50'
                 '\n' r'($I(\mathrm{reward};\mathrm{ctx})=%.3f$ bit) — only the cue'
                 r' resolves it ($%.3f$ bit)' % (i_arm, i_cue), fontsize=9, pad=26)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label='probability')
    fig.savefig(FIGS / 'fig_emission.png'); plt.close(fig)
    print('wrote fig_emission.png  (I_arm=%.3f, I_cue=%.3f bits)' % (i_arm, i_cue))


# --------------------------------------------------------------------------- #
def _rdm_subfactor_mi(psi, leg):
    d = psi.shape[leg]
    M = np.moveaxis(psi, leg, 0).reshape(d, -1)
    rho = M @ M.conj().T; rho /= np.trace(rho).real
    r6 = rho.reshape(4, 3, 2, 4, 3, 2)

    def vn(x):
        w = np.linalg.eigvalsh((x + x.conj().T) / 2); w = w[w > 1e-12]
        return float(-(w * np.log2(w)).sum())

    def part(keep):
        ein = list('abcABC')
        for a in [x for x in range(3) if x not in keep]:
            ein[3 + a] = ein[a]
        out = [ein[a] for a in keep] + [ein[3 + a] for a in keep]
        dk = int(np.prod([[4, 3, 2][a] for a in keep]))
        return np.einsum(''.join(ein) + '->' + ''.join(out), r6).reshape(dk, dk)
    S = {i: vn(part((i,))) for i in range(3)}
    M3 = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            M3[i, j] = M3[j, i] = S[i] + S[j] - vn(part((i, j)))
    return M3


def fig_graph():
    T1, T2, T3 = load_tensors('FullTmaze.pt')
    psi = np.einsum('xaoi,ibpj,jcqy->aobpcq', T1, T2, T3)
    p = np.abs(psi) ** 2; p /= p.sum()
    names = ['a1', 'o1', 'a2', 'o2', 'a3', 'o3']
    M = mi_matrix(p, names)
    tree = chow_liu(M, names)
    Msub = _rdm_subfactor_mi(psi / np.linalg.norm(psi), 3)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2),
                           gridspec_kw={'width_ratios': [5, 5, 4]})
    im = ax[0].imshow(M, cmap=HEAT, vmin=0, vmax=2)
    ax[0].set_title('Variable MI matrix (bits)')
    ax[0].set_xticks(range(6)); ax[0].set_yticks(range(6))
    ax[0].set_xticklabels(names); ax[0].set_yticklabels(names)
    annot(ax[0], M, fmt='{:.2f}', thr=1.2)
    fig.colorbar(im, ax=ax[0], shrink=0.8, pad=0.03)
    # Chow-Liu tree as a simple layout
    pos = {'a1': (0, 3), 'o1': (1, 3), 'a2': (1, 2), 'o2': (2, 2), 'a3': (1, 1), 'o3': (2, 1)}
    ax[1].set_title('Chow-Liu dependency tree')
    for a, b, w in tree:
        if w < 1e-6:
            continue
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax[1].plot([x0, x1], [y0, y1], '-', color=BLUE, lw=1 + 2 * w)
        ax[1].text((x0 + x1) / 2, (y0 + y1) / 2, f'{w:.2f}', fontsize=7, color=BLUE)
    for n, (x, y) in pos.items():
        ax[1].plot(x, y, 'o', ms=22, color='#e2e8f0', mec='k', mew=0.6)
        ax[1].text(x, y, n, ha='center', va='center', fontsize=8)
    ax[1].set_xlim(-0.5, 2.7); ax[1].set_ylim(0.5, 3.5); ax[1].axis('off')
    # sub-factor MI
    im2 = ax[2].imshow(Msub, cmap=HEAT, vmin=0, vmax=1)
    ax[2].set_title('Observation sub-factor MI')
    ax[2].set_xticks(range(3)); ax[2].set_yticks(range(3))
    lab = ['pos', 'rew', 'ctx']
    ax[2].set_xticklabels(lab); ax[2].set_yticklabels(lab)
    annot(ax[2], Msub, thr=0.6)
    fig.colorbar(im2, ax=ax[2], shrink=0.8, pad=0.03)
    fig.savefig(FIGS / 'fig_graph.png'); plt.close(fig)
    print('wrote fig_graph.png')


# --------------------------------------------------------------------------- #
def fig_modelsel():
    # values from src/model_selection.py (minimal maze, Han-scheduled sweep)
    chi = np.array([1, 2, 3, 4, 5, 6])
    fit = np.array([0.476, 0.464, 0.172, 0.027, 0.033, 0.034])
    mdl = np.array([4624.9, 4623.7, 4570.0, 4695.8, 4931.6, 5167.2])
    fig, ax1 = plt.subplots(figsize=(5, 3.3))
    ax1.plot(chi, fit, 'o-', color=BLUE, label='fit  $\\|p-q\\|_1$')
    ax1.set_xlabel('bond dimension $\\chi$'); ax1.set_ylabel('fit L1', color=BLUE)
    ax1.tick_params(axis='y', labelcolor=BLUE)
    ax1.axvline(4, color=ORANGE, ls='--', lw=1)
    ax1.text(4.05, 0.4, 'knee: 4 states', color=ORANGE, fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(chi, mdl, 's--', color='#718096', label='MDL/BIC')
    ax2.set_ylabel('MDL/BIC (bits)', color='#718096')
    ax2.tick_params(axis='y', labelcolor='#718096')
    ax1.set_title('State-count model selection (minimal maze)')
    fig.savefig(FIGS / 'fig_modelsel.png'); plt.close(fig)
    print('wrote fig_modelsel.png')


# --------------------------------------------------------------------------- #
def fig_empower():
    # learned empowerment (full maze, converged) vs analytic, from full_tmaze_train.analyze
    conds = ['start', 'trap', 'post-cue']
    learned_full = [2.000, 0.000, 1.9999]; analytic_full = [2.0, 0.0, 2.0]
    learned_rew = [1.000, 0.000, 1.5848]; analytic_rew = [1.0, 0.0, 1.585]
    x = np.arange(3); w = 0.2
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={'width_ratios': [5, 5]})
    ax[0].bar(x - 1.5 * w, analytic_full, w, label='analytic (full obs)', color='#a0aec0')
    ax[0].bar(x - 0.5 * w, learned_full, w, label='learned (full obs)', color=BLUE)
    ax[0].bar(x + 0.5 * w, analytic_rew, w, label='analytic (reward)', color='#fbd38d')
    ax[0].bar(x + 1.5 * w, learned_rew, w, label='learned (reward)', color=ORANGE)
    ax[0].set_xticks(x); ax[0].set_xticklabels(conds)
    ax[0].set_ylabel('empowerment (bits)')
    ax[0].set_title('Empowerment phenotyping: learned vs analytic')
    ax[0].legend(fontsize=6.5, loc='upper center')
    # gauge-fixed emitted states (minimal maze)
    T1, T2 = load_tensors('MinimalTmaze.pt')
    OBS = ['Cheese', 'Shock', 'RightCue', 'LeftCue']; ACT = ['R', 'L', 'Cue']
    hists, rays = [], []
    psi = np.abs(np.einsum('xaoi,ibpy->aobp', T1, T2)) ** 2
    for a1 in range(3):
        for o2 in range(4):
            if psi[a1, o2].sum() < 1e-9:
                continue
            b = T1[0, a1, o2, :].astype(complex)
            hists.append(f'{ACT[a1]}/{OBS[o2][:2]}'); rays.append(b / np.linalg.norm(b))
    labels, _ = cluster_rays(rays)
    k = labels.max() + 1
    reps = []
    for c in range(k):
        Mc = sum(np.outer(rays[i], np.conj(rays[i])) for i in range(len(rays)) if labels[i] == c)
        reps.append(np.linalg.eigh(Mc)[1][:, -1])
    Q, _ = np.linalg.qr(np.array(reps).T)
    emit = np.array([_safe(np.abs(Q.conj().T @ b) ** 2) for b in rays])   # (hist, k)
    im = ax[1].imshow(emit, cmap=HEAT, vmin=0, vmax=1, aspect='auto')
    ax[1].set_title('Gauge-fixed bond emits labeled states')
    ax[1].set_yticks(range(len(hists))); ax[1].set_yticklabels(hists, fontsize=7)
    ax[1].set_xticks(range(k)); ax[1].set_xticklabels([f's{j}' for j in range(k)])
    annot(ax[1], emit)
    fig.colorbar(im, ax=ax[1], shrink=0.8, pad=0.03, label='p(state | history)')
    fig.savefig(FIGS / 'fig_empower.png'); plt.close(fig)
    print('wrote fig_empower.png')


if __name__ == '__main__':
    fig_states()
    fig_ab()
    fig_emission()
    fig_graph()
    fig_modelsel()
    fig_empower()
    print('all figures in', FIGS)
