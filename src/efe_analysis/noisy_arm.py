"""Noisy-arm T-maze: when does the epistemic term of expected free energy select the Cue?

Environment (honest context model, as in efe_two_step.py, plus reliabilities):
  positions Center, LeftArm, RightArm, CuePos; context L or R, uniform.
  Entering LeftArm shows Cheese with probability r if the context is L and Shock with
  probability r if it is R (the opposite outcome otherwise); RightArm mirrored.
  The Cue shows SeeLeft with probability c if L, SeeRight with probability c if R.
  Arms are absorbing in position; every step in an arm draws a fresh outcome with the
  same reliability. From the Cue the agent may enter an arm or stay and re-read the cue.

Quantities per first action a at the choice point:
  risk(a)      = -E_{q(o|a)} ln softmax(C)_o
  epistemic(a) =  E_{q(o|a)} KL[q(s|o,a) || q(s|a)]   (state information gain, nats)
  G(a)         = risk(a) - epistemic(a);  at horizon 2 the sophisticated planner of
                 efe_two_step.py (second action conditioned on the first observation).
Selection with the epistemic term switched off ("preference only") uses G = risk.
Regions on the (r, c) grid, per profile and horizon:
  (a) epistemic alone selects the Cue: flat preferences, full EFE.
  (b) preference alone selects an arm, full EFE selects the Cue.
  (c) both preference alone and full EFE select the Cue.
  (d) the Cue is not selected by full EFE.

Empowerment (subjective potential, one step) after each first action, with the
information / interface decomposition of Proposition 1, and n-step empowerment for a
silent-arm control (arms show Null on entry and the outcome one step later).

Run: python noisy_arm.py   (numpy + matplotlib; pymdp cross-check appended if importable)
"""
from __future__ import annotations
import itertools
import os
import numpy as np

np.set_printoptions(precision=3, suppress=True)
LN2 = np.log(2.0)
POS = ["Center", "LeftArm", "RightArm", "CuePos"]
OBS = ["Cheese", "Shock", "SeeLeft", "SeeRight", "Null"]
ACT = ["Left", "Right", "Cue"]
GAMMA = 16.0

PROFILES = {
    "paper (+2/-4)":        np.array([2.0, -4.0, 0.0, 0.0, 0.0]),
    "mild (+1/-1)":         np.array([1.0, -1.0, 0.0, 0.0, 0.0]),
    "flat (0/0)":           np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "mild gambler (+2/-1)": np.array([2.0, -1.0, 0.0, 0.0, 0.0]),
    "gambler (+4/-2)":      np.array([4.0, -2.0, 0.0, 0.0, 0.0]),
}
R_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
C_GRID = [0.7, 0.9, 1.0]


def build_A(r, c, silent_arms=False):
    """A[o, pos, ctx]."""
    A = np.zeros((5, 4, 2))
    A[4, 0, :] = 1.0
    if silent_arms:
        A[4, 1, :] = 1.0
        A[4, 2, :] = 1.0
    else:
        A[0, 1, 0] = r; A[1, 1, 0] = 1 - r     # LeftArm, ctx L: Cheese w.p. r
        A[1, 1, 1] = r; A[0, 1, 1] = 1 - r     # LeftArm, ctx R: Shock w.p. r
        A[1, 2, 0] = r; A[0, 2, 0] = 1 - r     # RightArm, ctx L: Shock w.p. r
        A[0, 2, 1] = r; A[1, 2, 1] = 1 - r     # RightArm, ctx R: Cheese w.p. r
    A[2, 3, 0] = c; A[3, 3, 0] = 1 - c         # CuePos, ctx L: SeeLeft w.p. c
    A[3, 3, 1] = c; A[2, 3, 1] = 1 - c         # CuePos, ctx R: SeeRight w.p. c
    return A


B = np.zeros((4, 4, 3))                        # B[pos', pos, a]
for a, dest in enumerate([1, 2, 3]):
    B[dest, 0, a] = 1
    B[dest, 3, a] = 1
    B[1, 1, a] = 1; B[2, 2, a] = 1


def predict_obs(A, q_pos, q_ctx):
    return np.einsum("opc,p,c->o", A, q_pos, q_ctx)


def risk(q_o, C):
    lnp = C - np.log(np.sum(np.exp(C)))
    return float(-np.sum(q_o * lnp))


def infogain(A, q_pos, q_ctx):
    q_s = np.outer(q_pos, q_ctx)
    q_o = predict_obs(A, q_pos, q_ctx)
    ig = 0.0
    for o in range(5):
        if q_o[o] < 1e-15:
            continue
        post = A[o] * q_s / q_o[o]
        m = post > 1e-15
        ig += q_o[o] * float(np.sum(post[m] * (np.log(post[m]) - np.log(q_s[m]))))
    return ig


def posterior(A, q_pos, q_ctx, o):
    q_s = np.outer(q_pos, q_ctx)
    post = A[o] * q_s
    post /= post.sum()
    return post.sum(1), post.sum(0)


def step(q_pos, q_ctx, a):
    return B[:, :, a] @ q_pos, q_ctx


def efe_h1(A, C, use_epistemic=True):
    q_pos0 = np.array([1.0, 0, 0, 0]); q_ctx0 = np.array([0.5, 0.5])
    R = np.zeros(3); E = np.zeros(3)
    for a in range(3):
        qp, qc = step(q_pos0, q_ctx0, a)
        R[a] = risk(predict_obs(A, qp, qc), C)
        E[a] = infogain(A, qp, qc)
    G = R - (E if use_epistemic else 0.0)
    return R, E, G


def efe_h2(A, C, use_epistemic=True, gamma=GAMMA):
    """Sophisticated horizon 2: G(a1) = G1(a1) + E_{o1} E_{a2~sigma(-gamma G2)} G2(a2|o1)."""
    q_pos0 = np.array([1.0, 0, 0, 0]); q_ctx0 = np.array([0.5, 0.5])
    R1 = np.zeros(3); E1 = np.zeros(3); ER2 = np.zeros(3); EE2 = np.zeros(3); G = np.zeros(3)
    for a1 in range(3):
        qp1, qc1 = step(q_pos0, q_ctx0, a1)
        q_o1 = predict_obs(A, qp1, qc1)
        R1[a1] = risk(q_o1, C); E1[a1] = infogain(A, qp1, qc1)
        exp_G2 = 0.0
        for o1 in range(5):
            if q_o1[o1] < 1e-15:
                continue
            qp, qc = posterior(A, qp1, qc1, o1)
            R2 = np.zeros(3); E2 = np.zeros(3)
            for a2 in range(3):
                qp2, qc2 = step(qp, qc, a2)
                R2[a2] = risk(predict_obs(A, qp2, qc2), C); E2[a2] = infogain(A, qp2, qc2)
            G2 = R2 - (E2 if use_epistemic else 0.0)
            pi2 = np.exp(-gamma * G2); pi2 /= pi2.sum()
            ER2[a1] += q_o1[o1] * float(pi2 @ R2)
            EE2[a1] += q_o1[o1] * float(pi2 @ E2)
            exp_G2 += q_o1[o1] * float(pi2 @ G2)
        G[a1] = R1[a1] - (E1[a1] if use_epistemic else 0.0) + exp_G2
    return R1, E1, ER2, EE2, G


def selected(G, tol=1e-9):
    """Set of argmin actions (ties reported)."""
    m = G.min()
    return [ACT[i] for i in range(3) if G[i] <= m + tol]


def cue_wins(G, tol=1e-9):
    s = selected(G)
    return s == ["Cue"]


def arm_wins(G, tol=1e-9):
    s = selected(G)
    return "Cue" not in s


# ---------- empowerment ----------

def blahut_arimoto(W, iters=2000, tol=1e-12):
    """Capacity in bits of row-stochastic channel W (rows = inputs)."""
    W = np.asarray(W, float)
    n = W.shape[0]
    p = np.full(n, 1.0 / n)
    for _ in range(iters):
        q = p @ W
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where(W > 0, np.log(W / q), 0.0)
        c = np.exp(np.sum(W * lr, axis=1))
        p_new = p * c / np.sum(p * c)
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new
    q = p @ W
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(W > 0, np.log2(W / q), 0.0)
    return float(np.sum(p[:, None] * W * lr))


def choice_channel(A, q_ctx):
    """Rows Left/Right/Cue, columns OBS, at a three-way choice point under belief q_ctx."""
    W = np.zeros((3, 5))
    for a, pos in enumerate([1, 2, 3]):
        W[a] = np.einsum("oc,c->o", A[:, pos, :], q_ctx)
    return W


def arm_channel(A, pos, q_ctx):
    W = np.zeros((3, 5))
    for a in range(3):
        W[a] = np.einsum("oc,c->o", A[:, pos, :], q_ctx)
    return W


def emp_decomposition(A):
    """One-step subjective potential empowerment before and after each first action,
    with Proposition 1's information / interface split. Returns dict."""
    q0 = np.array([0.5, 0.5])
    E0 = blahut_arimoto(choice_channel(A, q0))
    out = {"E0": E0}
    q_pos0 = np.array([1.0, 0, 0, 0])
    for a in range(3):
        qp, qc = step(q_pos0, q0, a)
        q_o = predict_obs(A, qp, qc)
        info = 0.0; after = 0.0
        for o in range(5):
            if q_o[o] < 1e-15:
                continue
            _, qc_post = posterior(A, qp, qc, o)
            e_same = blahut_arimoto(choice_channel(A, qc_post))       # same interface, updated belief
            if a == 2:
                e_new = e_same                                          # Cue returns to the choice point
            else:
                e_new = blahut_arimoto(arm_channel(A, a + 1, qc_post)) # absorbing arm
            info += q_o[o] * e_same
            after += q_o[o] * e_new
        out[ACT[a]] = {"E_after": after, "info_term": info - E0, "interface_term": after - info,
                       "net": after - E0}
    return out


def nstep_capacity(A, q_pos, q_ctx, n):
    """Capacity of the channel from open-loop action sequences of length n to the
    observation sequence, starting from (q_pos, q_ctx) with context static."""
    seqs = list(itertools.product(range(3), repeat=n))
    obs_seqs = list(itertools.product(range(5), repeat=n))
    W = np.zeros((len(seqs), len(obs_seqs)))
    for i, seq in enumerate(seqs):
        # joint over (ctx, o_1..o_n): observations are conditionally independent given ctx and the path
        probs = np.zeros(len(obs_seqs))
        for ctx in range(2):
            pc = q_ctx[ctx]
            if pc < 1e-15:
                continue
            # position path is deterministic given start distribution and actions (start is a point mass here)
            qp = q_pos.copy()
            per_step = []
            for a in seq:
                qp = B[:, :, a] @ qp
                per_step.append(A[:, :, ctx] @ qp)          # p(o_t | ctx, path)
            for j, oseq in enumerate(obs_seqs):
                p = pc
                for t, o in enumerate(oseq):
                    p *= per_step[t][o]
                probs[j] += p
        W[i] = probs
    W = W[:, W.sum(0) > 0]
    return blahut_arimoto(W)


# ---------- main ----------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    P = lambda s="": out.append(s)
    P("# Noisy-arm T-maze: when the epistemic term does the work")
    P()
    P("Command: `python noisy_arm.py` (numpy, matplotlib; pymdp cross-check appended if importable). Nats unless stated.")
    P()
    P("Arms reveal the context with reliability r, the Cue with reliability c. Each step inside an arm draws a fresh outcome. "
      "Sophisticated horizon-2 planner as in efe_two_step.py. Selection = argmin G at precision gamma = 16 (ties listed).")
    P()

    # 1. region maps
    region_maps = {}   # (profile, horizon) -> dict[(r,c)] -> code
    P("## 1. Selection maps")
    P()
    P("Codes: a = epistemic term alone selects the Cue (flat preferences, full EFE); t = the preference term ties and the epistemic term "
      "breaks the tie toward the Cue; b = the preference term alone selects an arm and the epistemic term flips the choice to the Cue; "
      "c = the preference term alone and the full EFE both select the Cue; d = the full EFE does not select the Cue (tie or arm). "
      "The flat profile can only show a or d.")
    P()
    for prof, C in PROFILES.items():
        for horizon in (1, 2):
            codes = {}
            for r in R_GRID:
                for c in C_GRID:
                    A = build_A(r, c)
                    if horizon == 1:
                        _, _, G_full = efe_h1(A, C, True)
                        _, _, G_pref = efe_h1(A, C, False)
                    else:
                        *_, G_full = efe_h2(A, C, True)
                        *_, G_pref = efe_h2(A, C, False)
                    full_cue = cue_wins(G_full)
                    pref_cue = cue_wins(G_pref)
                    pref_arm = arm_wins(G_pref)
                    if prof.startswith("flat"):
                        code = "a" if full_cue else "d"
                    else:
                        if full_cue and pref_arm:
                            code = "b"
                        elif full_cue and not pref_cue:
                            code = "t"
                        elif full_cue and pref_cue:
                            code = "c"
                        else:
                            code = "d"
                    codes[(r, c)] = code
            region_maps[(prof, horizon)] = codes
            P(f"### {prof}, horizon {horizon}")
            P()
            P("| r \\ c | " + " | ".join(f"{c}" for c in C_GRID) + " |")
            P("|---|" + "---|" * len(C_GRID))
            for r in R_GRID:
                P(f"| {r} | " + " | ".join(codes[(r, c)] for c in C_GRID) + " |")
            P()

    # 2. detailed tables at a few grid points
    P("## 2. Per-action terms at selected grid points")
    P()
    for prof in ["flat (0/0)", "mild (+1/-1)", "paper (+2/-4)", "mild gambler (+2/-1)"]:
        C = PROFILES[prof]
        for (r, c) in [(0.5, 1.0), (0.7, 0.9), (0.8, 1.0), (0.9, 0.7), (1.0, 1.0)]:
            A = build_A(r, c)
            R, E, G = efe_h1(A, C)
            R1, E1, ER2, EE2, G2 = efe_h2(A, C)
            _, _, Gp1 = efe_h1(A, C, False)
            *_, Gp2 = efe_h2(A, C, False)
            P(f"### {prof}, r = {r}, c = {c}")
            P()
            P("| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |")
            P("|---|---|---|---|---|---|---|---|---|")
            for a in range(3):
                P(f"| {ACT[a]} | {R[a]:.3f} | {E[a]:.3f} | {G[a]:.3f} | {Gp1[a]:.3f} | {ER2[a]:.3f} | {EE2[a]:.3f} | {G2[a]:.3f} | {Gp2[a]:.3f} |")
            P()
            P(f"Selected h1 full: {selected(G)}; h1 pref only: {selected(Gp1)}; h2 full: {selected(G2)}; h2 pref only: {selected(Gp2)}.")
            P()

    # closed form for the horizon-1 epistemic term
    P("Closed form at horizon 1: epistemic(arm) = ln 2 - H(r), epistemic(Cue) = ln 2 - H(c), with H the binary entropy in nats, "
      "because under a uniform context prior each channel is binary symmetric. risk(arm) - risk(Cue) = "
      "(C_SeeLeft + C_SeeRight)/2 - (C_Cheese + C_Shock)/2 for every r and c, since the predicted outcome is a fair coin either way. "
      "So at horizon 1 the epistemic term favours the Cue exactly when c > r, by H(r) - H(c) nats, and the preference term is "
      "independent of the reliabilities.")
    P()

    # 3. empowerment decomposition
    P("## 3. Empowerment after each first action (bits), Proposition 1 decomposition")
    P()
    P("Subjective potential empowerment E0 at the choice point under the uniform context belief, then for each first action the "
      "expected empowerment after acting, split into the information term (belief updated, interface kept) and the interface term "
      "(move to the new interface at the updated belief). Cue returns to the choice point; arms are absorbing.")
    P()
    P("| r | c | E0 | action | E after | information term | interface term | net |")
    P("|---|---|---|---|---|---|---|---|")
    decomp_ok = True
    min_info = np.inf
    for r in [0.6, 0.8, 1.0]:
        for c in [0.9, 1.0]:
            A = build_A(r, c)
            d = emp_decomposition(A)
            for a in ACT:
                x = d[a]
                min_info = min(min_info, x["info_term"])
                if x["info_term"] < -1e-9:
                    decomp_ok = False
                if a == "Cue" and abs(x["interface_term"]) > 1e-9:
                    decomp_ok = False
                P(f"| {r} | {c} | {d['E0']:.3f} | {a} | {x['E_after']:.3f} | {x['info_term']:+.3f} | {x['interface_term']:+.3f} | {x['net']:+.3f} |")
    P()
    # full-grid check
    worst = 0.0
    for r in np.linspace(0.5, 1.0, 11):
        for c in np.linspace(0.5, 1.0, 11):
            d = emp_decomposition(build_A(r, c))
            for a in ACT:
                worst = min(worst, d[a]["info_term"])
                if a == "Cue" and abs(d[a]["interface_term"]) > 1e-9:
                    decomp_ok = False
    P(f"Decomposition check over the 11 x 11 grid r, c in [0.5, 1]: minimum information term = {worst:+.2e} "
      f"(non-negative required); Cue interface term identically zero: {'yes' if decomp_ok else 'NO'}. "
      f"Arms: interface term = -(empowerment at the choice point under the updated belief), net change negative whenever the arm is informative.")
    P()

    # 4. silent-arm control, n-step empowerment
    P("## 4. Silent-arm control: n-step empowerment")
    P()
    P("Arms show Null on entry and the outcome only on the following step (delayed information). Cue reliability c = 1, arm reliability r = 1. "
      "n-step empowerment = capacity from open-loop action sequences of length n to the observation sequence, in bits.")
    P()
    A_s = build_A(1.0, 1.0, silent_arms=False)
    # delayed variant: build a 5-position model where arm entry is a silent antechamber
    # Positions: Center, LeftAnte, RightAnte, CuePos, LeftArm, RightArm
    def nstep_silent(q_ctx, start, n):
        Bs = np.zeros((6, 6, 3))
        for a, dest in enumerate([1, 2, 3]):
            Bs[dest, 0, a] = 1; Bs[dest, 3, a] = 1
        for a in range(3):
            Bs[4, 1, a] = 1; Bs[5, 2, a] = 1      # antechamber -> arm on any action
            Bs[4, 4, a] = 1; Bs[5, 5, a] = 1      # arms absorbing
        As = np.zeros((5, 6, 2))
        As[4, 0, :] = 1; As[4, 1, :] = 1; As[4, 2, :] = 1
        As[2, 3, 0] = 1; As[3, 3, 1] = 1
        As[0, 4, 0] = 1; As[1, 4, 1] = 1
        As[1, 5, 0] = 1; As[0, 5, 1] = 1
        seqs = list(itertools.product(range(3), repeat=n))
        obs_seqs = list(itertools.product(range(5), repeat=n))
        W = np.zeros((len(seqs), len(obs_seqs)))
        for i, seq in enumerate(seqs):
            for ctx in range(2):
                pc = q_ctx[ctx]
                if pc < 1e-15:
                    continue
                qp = np.zeros(6); qp[start] = 1.0
                per_step = []
                for a in seq:
                    qp = Bs[:, :, a] @ qp
                    per_step.append(As[:, :, ctx] @ qp)
                for j, oseq in enumerate(obs_seqs):
                    p = pc
                    for t, o in enumerate(oseq):
                        p *= per_step[t][o]
                    W[i, j] += p
        W = W[:, W.sum(0) > 0]
        return blahut_arimoto(W)

    P("| situation | belief over context | 1-step | 2-step |")
    P("|---|---|---|---|")
    rows = [
        ("choice point, before acting", np.array([0.5, 0.5]), 0),
        ("after Cue (context resolved), at the choice point", np.array([1.0, 0.0]), 3),
        ("after Left (silent antechamber), context unresolved", np.array([0.5, 0.5]), 1),
        ("after Left (silent antechamber), context resolved", np.array([1.0, 0.0]), 1),
    ]
    for name, q, start in rows:
        e1 = nstep_silent(q, start, 1); e2 = nstep_silent(q, start, 2)
        P(f"| {name} | {q.tolist()} | {e1:.3f} | {e2:.3f} |")
    P()
    P("Standard (non-silent) maze for comparison, r = c = 1:")
    P()
    P("| situation | 1-step | 2-step |")
    P("|---|---|---|")
    for name, q, start in [("choice point, unresolved", np.array([0.5, 0.5]), 0), ("after Cue, resolved", np.array([1.0, 0.0]), 3), ("inside an arm", np.array([1.0, 0.0]), 1)]:
        P(f"| {name} | {nstep_capacity(A_s, np.eye(4)[start], q, 1):.3f} | {nstep_capacity(A_s, np.eye(4)[start], q, 2):.3f} |")
    P()

    # 5. figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        code_val = {"d": 0, "a": 1, "t": 1, "b": 2, "c": 3}
        cmap = ListedColormap(["#d9d9d9", "#4c72b0", "#dd8452", "#55a868"])
        panels = [("flat (0/0)", 1), ("mild (+1/-1)", 1), ("mild gambler (+2/-1)", 1),
                  ("flat (0/0)", 2), ("mild (+1/-1)", 2), ("mild gambler (+2/-1)", 2)]
        fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2))
        for ax, (prof, h) in zip(axes.ravel(), panels):
            codes = region_maps[(prof, h)]
            M = np.array([[code_val[codes[(r, c)]] for c in C_GRID] for r in R_GRID])
            ax.imshow(M, cmap=cmap, vmin=0, vmax=3, aspect="auto", origin="lower")
            ax.set_xticks(range(len(C_GRID))); ax.set_xticklabels(C_GRID)
            ax.set_yticks(range(len(R_GRID))); ax.set_yticklabels(R_GRID)
            ax.set_xlabel("cue reliability c"); ax.set_ylabel("arm reliability r")
            ax.set_title(f"{prof}, horizon {h}", fontsize=9)
            for i, r in enumerate(R_GRID):
                for j, c in enumerate(C_GRID):
                    ax.text(j, i, codes[(r, c)], ha="center", va="center", fontsize=8, color="black")
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color="#4c72b0", label="a / t: epistemic term alone selects the Cue (t: breaks a preference tie)"),
                            Patch(color="#dd8452", label="b: preferences alone pick an arm, epistemic term flips to the Cue"),
                            Patch(color="#55a868", label="c: preferences and epistemic term both select the Cue"),
                            Patch(color="#d9d9d9", label="d: Cue not selected (tie or arm)")],
                   loc="lower center", ncol=2, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        figdir = os.path.join(here, "figures"); os.makedirs(figdir, exist_ok=True)
        fpath = os.path.join(figdir, "fig_noisy_arm.png")
        fig.savefig(fpath, dpi=200)
        P(f"Figure written: {os.path.relpath(fpath, here)}")
        P()
    except Exception as e:  # pragma: no cover
        P(f"(figure skipped: {e!r})")
        P()

    # 6. pymdp cross-check at horizon 1
    try:
        try:
            from pymdp.legacy import utils
            from pymdp.legacy.agent import Agent
        except ImportError:
            from pymdp import utils
            from pymdp.agent import Agent
        P("## 5. pymdp cross-check, horizon 1 (negative EFE, larger is better)")
        P()
        P("| profile | r | c | Left | Right | Cue | q(pi) Cue | numpy G Cue - G Left |")
        P("|---|---|---|---|---|---|---|---|")
        for prof in ["flat (0/0)", "paper (+2/-4)", "mild gambler (+2/-1)"]:
            for (r, c) in [(0.5, 1.0), (0.7, 0.9), (1.0, 0.7)]:
                A = build_A(r, c)
                A_obj = utils.obj_array(1); A_obj[0] = A.copy()
                B_obj = utils.obj_array(2); B_obj[0] = B.copy()
                Bc = np.zeros((2, 2, 1)); Bc[:, :, 0] = np.eye(2); B_obj[1] = Bc
                C_obj = utils.obj_array(1); C_obj[0] = PROFILES[prof].copy()
                D_obj = utils.obj_array(2); D_obj[0] = np.array([1.0, 0, 0, 0]); D_obj[1] = np.array([0.5, 0.5])
                agent = Agent(A=A_obj, B=B_obj, C=C_obj, D=D_obj, policy_len=1, control_fac_idx=[0],
                              use_utility=True, use_states_info_gain=True, gamma=GAMMA)
                agent.reset(); agent.infer_states([4])
                q_pi, Gp = agent.infer_policies()
                _, _, Gn = efe_h1(A, PROFILES[prof])
                P(f"| {prof} | {r} | {c} | {Gp[0]:.3f} | {Gp[1]:.3f} | {Gp[2]:.3f} | {q_pi[2]:.3f} | {Gn[2]-Gn[0]:+.3f} (pymdp {-(Gp[2]-Gp[0]):+.3f}) |")
        P()
    except Exception as e:  # pragma: no cover
        P(f"(pymdp cross-check skipped: {e!r})")
        P()

    text = "\n".join(out)
    print(text)
    with open(os.path.join(here, "noisy_arm.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
