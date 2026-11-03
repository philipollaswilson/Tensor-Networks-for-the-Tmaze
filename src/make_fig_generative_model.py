"""Figures: from hidden states to a generative model (full pymdp T-maze).

Figure 3 (fig_generative_model.png): the T-maze, continued from fig_hidden_states*
  (a) mixture test: is a learned state's prediction row a mixture of the others?
  (b) the generator over the remaining states, laid out by its two factors
  (c) the factorization: transition matrices of each hidden variable
  (d) which hidden variable each observation modality depends on (pymdp's A)
  (e) model error vs data, inside the data's horizon and one step beyond it
  (f) hidden-state counts of the models

Figure 4 (fig_structure_learning_math.png): the pipeline as equations, one panel per
step, each with its T-maze instance.

Run from the repo root:  python src/make_fig_generative_model.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generative_model as GM  # noqa: E402
import make_fig_hidden_states as HS  # noqa: E402
import seed_protocol as SP  # noqa: E402
from make_fig_hidden_states import ACT, COL, INK, MUTED, GRID, STATES, clean, title, marker  # noqa: E402
from make_extension_figures import FIGS  # noqa: E402

BLUE, ORANGE = '#2a78d6', '#eb6834'
CTX_Y = {1.0: 1.1, 0.0: -1.1}                          # cheese right on top
PLACE_X = {0: 0.0, 1: 2.6, 2: -2.6}                    # by the location of the state


def name(res, k):
    return STATES[res['fl']['k_of'][k]][1]


def obs_label(os_):
    """Observations as text, the cue bit shown only when it is not a coin flip."""
    by = {}
    for o in os_:
        by.setdefault(o // 2, []).append(o % 2)
    out = []
    for pr, cs in sorted(by.items()):
        p, r = pr // 3, pr % 3
        s = HS.LOC_SHORT[p] + (f' {HS.RW[r]}' if r else '')
        out.append(s + ('' if len(cs) == 2 else f' ·{cs[0]}'))
    return ', '.join(out)


def factor_names(res):
    """Display names of the factor values, read off the member states (naming only)."""
    V = res['vertices']
    p1, p2 = res['prods'][0]
    ctx = ['cheese right' if STATES[res['fl']['k_of'][V[B[0]]]][2] == 1.0 else 'cheese left'
           for B in p1]
    place = []
    for B in p2:
        loc = STATES[res['fl']['k_of'][V[B[0]]]][0][0]
        place.append({1: 'in right arm', 2: 'in left arm', 3: 'not in an arm'}[loc])
    return ctx, place


def pos_of(res, i):
    k = res['fl']['k_of'][res['vertices'][i]]
    (loc, _, _), _, pr = STATES[k]
    return PLACE_X.get(loc if loc in (1, 2) else 0), CTX_Y[pr]


# ------------------------------------------------------------ figure 3 panels
def panel_mixture(ax, res):
    rep = res['report']
    order = sorted(rep, key=lambda k: HS.STATES.index(STATES[res['fl']['k_of'][k]]))
    bf = np.array([rep[k][0] for k in order])
    shown = np.where(np.isfinite(bf), bf, -1e4)
    cols = [BLUE if b > 0 else '#c3c2b7' for b in bf]
    ax.barh(range(len(order)), shown, color=cols, height=0.62)
    ax.set_xscale('symlog', linthresh=10)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([name(res, k) for k in order], fontsize=7.8, color=INK)
    ax.invert_yaxis()
    for r, (k, b) in enumerate(zip(order, bf)):
        txt = '−∞' if not np.isfinite(b) else f'{b:+.0f}'
        ax.text(-1.5 if b <= 0 else 1.5, r, txt, fontsize=7.5, color=INK, va='center',
                ha='right' if b <= 0 else 'left', zorder=4)
    k = next(k for k in order if rep[k][0] > 0)
    w = {name(res, j): v for j, v in rep[k][1].items() if v > 0}
    ax.text(0.99, 0.5, f'{name(res, k)} =\n' + '\n+ '.join(f'{v:.2f} · {n}' for n, v in w.items())
            + '\n\na belief over the\ntwo contexts, not\na new hidden state\n\n'
            '−∞: the row has an\noutcome that no\nmixture of others gives',
            transform=ax.transAxes, fontsize=7.4, color=INK, ha='right', va='center')
    ax.set_xlim(-2e4, 3e6)
    ax.set_xticks([-1e4, -1e2, 0, 1e2])
    ax.set_xlabel('ln Bayes factor: row = mixture of the other rows vs. a free row', fontsize=8,
                  color=MUTED)
    clean(ax, grid='x')
    ax.tick_params(axis='y', length=0)


def panel_generator(ax, res):
    G, D = res['gen']['G'], res['gen']['D']
    n = len(D)
    for y, txt in ((1.1, 'context:\ncheese right'), (-1.1, 'context:\ncheese left')):
        ax.text(-4.35, y, txt, fontsize=7.5, color=MUTED, ha='center', va='center')
    for x, txt in ((-2.6, 'place: left arm'), (0, 'place: not in an arm'), (2.6, 'place: right arm')):
        ax.text(x, 2.45, txt, fontsize=8, color=INK, ha='center', weight='bold')
    ax.scatter([0], [0], s=520, marker='s', color='white', edgecolor=INK, lw=1.2, zorder=4)
    ax.text(0, 0, 'start', fontsize=6.5, color=INK, ha='center', va='center', zorder=5)
    for i in range(n):
        if D[i] > 0.01:
            x, y = pos_of(res, i)
            ax.add_patch(FancyArrowPatch((0, 0), (x, y), arrowstyle='-|>', mutation_scale=9,
                                         shrinkA=13, shrinkB=14, lw=0.9, ls=(0, (3, 2)),
                                         color=MUTED, zorder=2))
            ax.text(x + 0.18, y * 0.5, f'D = {D[i]:.2f}', fontsize=6.8, color=MUTED, va='center')
    for i in range(n):
        x0, y0 = pos_of(res, i)
        selfs = []
        for j in range(n):
            lab = []
            for a in range(4):
                os_ = [o for o in range(GM.NO) if G[a, o, i, j] > 0.05]
                if os_:
                    lab.append((ACT[a], obs_label(os_)))
            if not lab:
                continue
            if j == i:
                selfs = lab
                continue
            x1, y1 = pos_of(res, j)
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=10,
                                         shrinkA=14, shrinkB=14, lw=1.5, color=INK, zorder=2))
            ax.text((x0 + x1) / 2, y0 + 0.17 * np.sign(y0), '\n'.join(f'{a} → {o}' for a, o in lab),
                    fontsize=6.6, color=INK, ha='center', va='bottom' if y0 > 0 else 'top',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none'), zorder=3)
        marker(res['fl']['k_of'][res['vertices'][i]], ax, x0, y0, size=380, z=4)
        if len(selfs) == 4 and len({o for _, o in selfs}) == 1:
            txt = f'any action → {selfs[0][1]}'
        else:
            txt = '\n'.join(f'{a} → {o}' for a, o in selfs)
        ax.text(x0, y0 + 0.32 * np.sign(y0), '↺ ' + txt, fontsize=6.6, color=MUTED, ha='center',
                va='bottom' if y0 > 0 else 'top', zorder=5)
    ax.set_xlim(-5.0, 3.6); ax.set_ylim(-2.6, 2.7)
    ax.set_aspect('equal'); ax.axis('off')


def panel_factors(fig, spec, res):
    Bs, err, _, _ = res['fac']
    ctx, place = factor_names(res)
    sub = spec.subgridspec(2, 4, height_ratios=[1, 1], hspace=0.9, wspace=0.35)
    for r, (B, names, lab) in enumerate(((Bs[0], ctx, 'context'), (Bs[1], place, 'place'))):
        for a in range(4):
            ax = fig.add_subplot(sub[r, a])
            ax.imshow(B[a], cmap='Greys', vmin=0, vmax=1)
            for i in range(len(B[a])):
                for j in range(len(B[a])):
                    if B[a, i, j] > 0.005:
                        ax.text(j, i, f'{B[a, i, j]:.0f}' if B[a, i, j] > 0.995 else f'{B[a, i, j]:.2f}',
                                ha='center', va='center', fontsize=6.5,
                                color='white' if B[a, i, j] > 0.6 else INK)
            short = [s.replace('cheese ', 'ch ').replace('in ', '').replace('not in an arm', 'no arm')
                     for s in names]
            ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
            ax.set_xticklabels(short, fontsize=6, rotation=40, ha='right', color=MUTED)
            ax.set_yticklabels(short if a == 0 else [], fontsize=6, color=MUTED)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f'{lab} · {ACT[a]}', fontsize=7.5, color=INK, pad=3)
    return err


def panel_dependence(ax, res):
    _, _, dep, _ = res['fac']
    ax.imshow(dep, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for m in range(3):
        for f in range(2):
            ax.text(f, m, f'{dep[m, f]:.2f}\n' + ('depends' if dep[m, f] > 0.05 else 'no effect'),
                    ha='center', va='center', fontsize=7.2,
                    color='white' if dep[m, f] > 0.6 else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['context', 'place'], fontsize=8, color=INK)
    ax.set_yticks(range(3)); ax.set_yticklabels(['location', 'reward', 'cue bit'], fontsize=8, color=INK)
    ax.set_xlabel('hidden variable', fontsize=8, color=MUTED)
    ax.set_ylabel('observation modality', fontsize=8, color=MUTED)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def sweep(seeds=10):
    Ns = np.unique(np.round(np.geomspace(30, 30000, 13)).astype(int))
    E = np.full((seeds, len(Ns), 4), np.nan)
    S = np.full((seeds, len(Ns), 3), np.nan)
    for s in range(seeds):
        for j, N in enumerate(Ns):
            try:
                res = GM.build(HS.sample(N, 200 + s))
            except AssertionError:
                continue
            ev = GM.evaluate(res)
            E[s, j] = [ev['flat', 2], ev['gen', 2], ev['flat', 3], ev['gen', 3]]
            S[s, j] = [res['fl']['K'], len(res['vertices']), len(res['prods']) > 0]
    return Ns, E, S


def panel_error(ax, Ns, E):
    for c, (lab, col, ls) in enumerate([('flat (7 states), within the data: step 2', BLUE, '-'),
                                        ('generator (6), within the data: step 2 (on the flat line)', ORANGE, '-'),
                                        ('flat, beyond the data: step 3', BLUE, '--'),
                                        ('generator, beyond the data: step 3', ORANGE, '--')]):
        Y = E[:, :, c]
        ax.plot(Ns, np.nanmedian(Y, 0), ls, color=col, lw=1.8, marker='o' if ls == '-' else 's',
                ms=3.5, label=lab)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('training trajectories', fontsize=8, color=MUTED)
    ax.set_ylabel('total variation of the next-observation\nprediction vs. the true maze (median)',
                  fontsize=8, color=MUTED)
    ax.axvline(HS.N_TRAIN, color=MUTED, lw=0.8, ls=':')
    ax.text(HS.N_TRAIN * 1.08, ax.get_ylim()[1] * 0.7, 'figs 1–2', fontsize=7, color=MUTED)
    ax.legend(fontsize=6.8, frameon=False, loc='lower left')
    clean(ax)


def panel_counts(ax, res, ev):
    rows = [('pymdp generative process', '4 locations × 2 contexts', 8, '#c3c2b7', None),
            ('flat: one variable, learned states', '7 predictive states', res['fl']['K'], BLUE,
             ev['flat', 3]),
            ('generator: mixtures removed', '6 = linear rank of the data', len(res['vertices']),
             ORANGE, ev['gen', 3]),
            ('factored generator', '2 contexts × 3 places', len(res['vertices']), ORANGE, ev['gen', 3])]
    for r, (lab, sub, k, c, tv) in enumerate(rows):
        ax.barh(r, k, color=c, height=0.6)
        ax.text(k + 0.15, r, sub, fontsize=7.4, color=INK, va='center')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 13.5); ax.set_xticks(range(0, 9, 2))
    ax.set_xlabel('hidden states', fontsize=8, color=MUTED)
    clean(ax, grid='x')
    ax.tick_params(axis='y', length=0)


def remember_example(res):
    """p(cheese | cue said right, back to the centre, then right) for each model."""
    out = {}
    o_cue, o_ctr = SP.obs_index(3, 0, 0), [SP.obs_index(0, 0, c) for c in range(2)]
    ch, sh = [SP.obs_index(1, 1, c) for c in range(2)], [SP.obs_index(1, 2, c) for c in range(2)]
    for key, T, D in (('flat', res['fl']['T'], res['fl']['D']), ('gen', res['gen']['G'], res['gen']['D'])):
        v = D @ T[3, o_cue]
        v = v @ T[0, o_ctr].sum(0)
        pc, ps = (v @ T[1, ch].sum(0)).sum(), (v @ T[1, sh].sum(0)).sum()
        out[key] = pc / (pc + ps)
    return out


def figure3(res, ev, Ns, E):
    fig = plt.figure(figsize=(19.5, 10.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 1.35, 1.05], height_ratios=[1.05, 1],
                          wspace=0.42, hspace=0.42)
    ax = fig.add_subplot(gs[0, 0]); panel_mixture(ax, res)
    title(ax, '(a) Is a state only a belief? Test each prediction row\nas a mixture of the other rows')
    ax = fig.add_subplot(gs[0, 1]); panel_generator(ax, res)
    title(ax, f'(b) The generator: {len(res["vertices"])} hidden states, the centre replaced by '
              'a 50/50 start.\nEdges: action → observation emitted. The layout is the factorization of (c)')
    err = panel_factors(fig, gs[0, 2], res)
    fig.text(0.682, 0.905, f'(c) {len(res["sps"])} partitions keep every transition inside one '
             f'block;\n{len(res["prods"])} pair of them is a product: {len(res["vertices"])} = '
             f'2 × 3, B = B_context ⊗ B_place ' + ('exactly' if err < 1e-9 else f'(max error {err:.1e})'), fontsize=9.5, color=INK)
    low = gs[1, :].subgridspec(1, 3, width_ratios=[0.55, 1.2, 1.1], wspace=0.55)
    ax = fig.add_subplot(low[0]); panel_dependence(ax, res)
    title(ax, "(d) What each observation depends on\n(pymdp's A): max change in p(o | s, a)")
    ax = fig.add_subplot(low[1]); panel_error(ax, Ns, E)
    ex = remember_example(res)
    title(ax, '(e) The data stop at step 2. At step 3 the flat model forgets the cue on returning\n'
              f'to the centre: p(cheese | cue said right, centre, right) = {ex["flat"]:.2f} flat, '
              f'{ex["gen"]:.2f} generator')
    ax = fig.add_subplot(low[2]); panel_counts(ax, res, ev)
    title(ax, '(f) How many hidden states? 7 is the smallest model whose\n'
              'state is read off the history; 6 is the smallest with hidden states')
    fig.suptitle('From hidden states to a generative model, full T-maze. Same data and learned '
                 f'states as figs 1–2 ({HS.N_TRAIN} trajectories)', fontsize=12, color=INK,
                 x=0.1, ha='left', y=0.99)
    out = FIGS / 'fig_generative_model.png'
    fig.savefig(out, bbox_inches='tight', dpi=170, facecolor='white')
    plt.close(fig)
    print(f'wrote {out}  remember example {ex}')


# ------------------------------------------------------------ figure 4
STEPS = [
    ('1  Count', r'$N(h,a,o)=\sum_n [\,h_n=h,\ a_n=a,\ o_n=o\,]$',
     'how often each history h was\nfollowed by action a, then outcome o',
     '24 histories × 4 actions × 12 outcomes'),
    ('2  Predict', r"$\hat p(o\,|\,h,a)=N(h,a,o)\,/\sum_{o'} N(h,a,o')$" + '\n'
     + r'$H[h,(a,o)]=\hat p(o\,|\,h,a)$',
     'one row per history: what\nit predicts for every action',
     'prediction table H, 24 × 48'),
    ('3  Rank', r'$H=U\Sigma V^{\top}$' + '\n' + r'$r=\#\{\sigma_i>\mathrm{noise}\}\ \leq\ \#\mathrm{states}$',
     'the number of independent\npredictions: a lower bound',
     'rank 6'),
    ('4  Merge (BMR)', r'$\Delta F=\sum_a \ln\dfrac{\mathrm{B}(\alpha+N_h+N_g)\,\mathrm{B}(\alpha)}'
     r'{\mathrm{B}(\alpha+N_h)\,\mathrm{B}(\alpha+N_g)}$' + '\n' + r'merge $h,g$ while $\Delta F>0$',
     'one state if one Dirichlet row\nexplains both counts better',
     '24 histories → 7 states'),
    ('5  Machine', r"$s=\sigma(h),\quad s'=f(s,a,o)=\sigma(h\,a\,o)$" + '\n'
     + r"$T^{(o|a)}_{s s'}=\hat p(o\,|\,s,a)\,[\,s'=f(s,a,o)\,]$",
     'states and where each outcome\nsends them: one hidden variable',
     '7 states, start = Centre'),
    ('6  Unmix', r'$\hat p(\cdot\,|\,s)=\sum_v w_v\,\hat p(\cdot\,|\,v),\ \ w\in\Delta$' + '\n'
     + r'$\ln\mathrm{BF}=\ln p(N_s|\mathrm{mix})-\ln p(N_s|\mathrm{Dir})$',
     'a row inside the hull of the others\nis a belief, not a hidden state',
     'Centre = ½ cue R + ½ cue L → 6'),
    ('7  Generator', r'$T^{(o|a)}_{ij}=p(o,j\,|\,i,a),\quad D=w_{\mathrm{start}}$' + '\n'
     + r'$p(o_{1:t}|a_{1:t})=D^{\top}T^{(o_1|a_1)}\cdots T^{(o_t|a_t)}\mathbf{1}$',
     'hidden states that persist unless\nthe data force a change',
     '6 hidden states, D = (½, ½)'),
    ('8  Factorize', r"blocks $\beta$ of $\pi$: $f_a(\beta)\subseteq\beta'\ \ \forall a,\beta$" + '\n'
     + r'$\pi_1\wedge\pi_2=0,\ |\pi_1||\pi_2|=|S|\ \Rightarrow\ S\cong S_1\times S_2$' + '\n'
     + r'$B_a=B^{1}_a\otimes B^{2}_a$',
     'split the state into variables\nthat the dynamics keep apart',
     '6 = context (2) × place (3)'),
]


def mini(ax, k, res, trajs):
    fl = res['fl']
    m = fl['m']
    n, lab = m['n'], m['lab']
    order = np.lexsort((np.arange(len(lab)), [fl['k_of'][c] for c in lab]))
    if k == 0:
        C = n[order].reshape(len(n), -1)
        ax.imshow(np.log1p(C), cmap='Greys', aspect='auto', interpolation='nearest')
    elif k == 1:
        ax.imshow(np.nan_to_num(HS.rows(n)[order]), cmap='Greys', vmin=0, vmax=1, aspect='auto',
                  interpolation='nearest')
        for b in np.flatnonzero(np.diff([fl['k_of'][c] for c in lab[order]])) + 0.5:
            ax.axhline(b, color='white', lw=1.5)
    elif k == 2:
        s = np.linalg.svd(np.nan_to_num(HS.rows(n)), compute_uv=False)
        ax.semilogy(range(1, len(s) + 1), s / s[0], 'o-', color=BLUE, ms=3, lw=1.2)
        ax.axvline(6.5, color=MUTED, ls=':', lw=0.8)
        ax.set_ylim(1e-3, 2)
        clean(ax)
        ax.set_xticks([1, 6, 12, 18, 24])
        return
    elif k == 3:
        acc, rej = m['acc'], m['rej']
        d = list(acc) + [rej]
        ax.bar(range(len(d)), d, color=[BLUE] * len(acc) + [ORANGE], width=0.7)
        ax.axhline(0, color=INK, lw=0.6)
        clean(ax)
        ax.set_xticks([])
        return
    elif k in (4, 6):
        T = fl['T'] if k == 4 else res['gen']['G']
        ks = range(fl['K']) if k == 4 else range(len(res['vertices']))
        P = {c: HS.MACHINE_POS[fl['k_of'][c]] for c in range(fl['K'])}
        if k == 6:
            P = {i: pos_of(res, i) for i in ks}
        S = T.sum(1)
        for i in ks:
            for j in ks:
                if i != j and S[:, i, j].max() > 0.05:
                    ax.add_patch(FancyArrowPatch(P[i], P[j], arrowstyle='-|>', mutation_scale=6,
                                                 shrinkA=6, shrinkB=6, lw=0.8, color=INK,
                                                 connectionstyle='arc3,rad=0.15'))
        for i in ks:
            kk = fl['k_of'][i] if k == 4 else fl['k_of'][res['vertices'][i]]
            marker(kk, ax, *P[i], size=55, z=4)
        if k == 6:
            for i in ks:
                if res['gen']['D'][i] > 0.01:
                    ax.add_patch(FancyArrowPatch((0, 0), P[i], arrowstyle='-|>', mutation_scale=6,
                                                 shrinkA=2, shrinkB=6, lw=0.7, ls=':', color=MUTED))
            ax.scatter([0], [0], s=18, marker='s', color='white', edgecolor=INK, lw=0.8, zorder=4)
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-1.9, 1.9)
        ax.set_aspect('equal'); ax.axis('off')
        return
    elif k == 5:
        rep = res['report']
        cen = next(c for c in rep if rep[c][0] > 0)
        w = rep[cen][1]
        (j1, w1), (j2, w2) = sorted(((j, v) for j, v in w.items() if v > 0), key=lambda t: -t[1])[:2]
        ax.plot([0, 1], [0, 0], color=MUTED, lw=1)
        for j, x in ((j1, 0), (j2, 1)):
            marker(fl['k_of'][j], ax, x, 0, size=90)
        marker(fl['k_of'][cen], ax, w2 / (w1 + w2), 0, size=90, z=5)
        ax.text(w2 / (w1 + w2), 0.25, f'{w1:.2f} : {w2:.2f}', ha='center', fontsize=7, color=INK)
        ax.text(0, -0.3, 'cue R', ha='center', fontsize=7, color=MUTED, va='top')
        ax.text(1, -0.3, 'cue L', ha='center', fontsize=7, color=MUTED, va='top')
        ax.set_xlim(-0.25, 1.25); ax.set_ylim(-0.7, 0.6); ax.axis('off')
        return
    elif k == 7:
        for i in range(len(res['vertices'])):
            marker(fl['k_of'][res['vertices'][i]], ax, *pos_of(res, i), size=70)
        for x in (-1.3, 1.3):
            ax.axvline(x, color=GRID, lw=1)
        ax.axhline(0, color=GRID, lw=1)
        ax.text(-3.4, 1.1, 'ctx R', fontsize=6.5, color=MUTED, ha='right', va='center')
        ax.text(-3.4, -1.1, 'ctx L', fontsize=6.5, color=MUTED, ha='right', va='center')
        for x, t in ((-2.6, 'L arm'), (0, 'no arm'), (2.6, 'R arm')):
            ax.text(x, 1.85, t, fontsize=6.5, color=MUTED, ha='center')
        ax.set_xlim(-4.6, 3.4); ax.set_ylim(-1.8, 2.2); ax.axis('off')
        return
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def figure4(res, trajs):
    fig = plt.figure(figsize=(20, 10.4))
    W, H = 0.215, 0.40
    xs = [0.03 + i * 0.245 for i in range(4)]
    ys = [0.52, 0.04]
    for k, (head, eq, what, tmaze) in enumerate(STEPS):
        r, c = divmod(k, 4)
        x, y = xs[c], ys[r]
        fig.patches.append(FancyBboxPatch((x, y), W, H, boxstyle='round,pad=0.004,rounding_size=0.01',
                                          transform=fig.transFigure, fc='#faf9f6', ec=GRID, lw=1,
                                          zorder=-1))
        fig.text(x + 0.012, y + H - 0.03, head, fontsize=12, weight='bold', color=INK)
        fig.text(x + 0.012, y + H - 0.065, what, fontsize=8.3, color=MUTED, va='top')
        lines = eq.split('\n')
        for i, line in enumerate(lines):
            step = 0.05 if len(lines) < 3 else 0.04
            fig.text(x + W / 2, y + H - 0.155 - step * (i - (len(lines) - 1) / 2), line,
                     fontsize=10.5, color=INK, ha='center', va='center')
        ax = fig.add_axes([x + 0.035, y + 0.05, W - 0.07, 0.125])
        mini(ax, k, res, trajs)
        fig.text(x + W / 2, y + 0.018, 'T-maze: ' + tmaze, fontsize=8.5, color=INK, ha='center',
                 weight='bold')
        if c < 3:
            fig.patches.append(FancyArrowPatch((x + W + 0.004, y + H / 2), (x + 0.245 - 0.004, y + H / 2),
                                               transform=fig.transFigure, arrowstyle='-|>',
                                               mutation_scale=14, color=MUTED, lw=1.4))
    fig.patches.append(FancyArrowPatch((xs[3] + W / 2, ys[0] - 0.004), (xs[0] + W / 2, ys[1] + H + 0.004),
                                       transform=fig.transFigure, arrowstyle='-|>', mutation_scale=14,
                                       color=MUTED, lw=1.4, connectionstyle='arc3,rad=0.03'))
    fig.text(0.03, 0.965, 'Structure learning, step by step: from trajectories to a factored '
             'generative model', fontsize=14, color=INK, weight='bold')
    fig.text(0.03, 0.935, 'Steps 1–5 find what the history must remember (predictive states); '
             '6–8 turn that memory into hidden variables. Every quantity is computed from counts; '
             'nothing about the maze is given.', fontsize=9.5, color=MUTED)
    out = FIGS / 'fig_structure_learning_math.png'
    fig.savefig(out, dpi=170, facecolor='white')
    plt.close(fig)
    print(f'wrote {out}')


def main():
    trajs = HS.sample(HS.N_TRAIN, 0)
    res = GM.build(trajs)
    assert res['prods'], 'no factorization found'
    ev = GM.evaluate(res)
    print({k: round(float(v), 4) for k, v in ev.items()})
    Ns, E, S = sweep()
    print('N', Ns)
    for c, lab in enumerate(['flat h2', 'gen h2', 'flat h3', 'gen h3']):
        print(f'{lab:8s}', np.round(np.nanmedian(E[:, :, c], 0), 4))
    print('vertices', np.nanmedian(S[:, :, 1], 0), 'product found', np.nanmean(S[:, :, 2], 0))
    figure3(res, ev, Ns, E)
    figure4(res, trajs)


if __name__ == '__main__':
    main()
