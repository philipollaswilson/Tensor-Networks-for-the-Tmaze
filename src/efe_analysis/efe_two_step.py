"""Horizon-2 expected free energy in the minimal T-maze, honest context model.

Question (internal review W4): at horizon 1 every action resolves the context,
so the epistemic term cancels and the Cue is selected by the risk term alone.
Does a two-step planner select the Cue through the epistemic term?

Answer computed here, two ways:

  * open-loop (pymdp-style): G(pi) = sum_t [ risk_t - infogain_t ] with beliefs
    propagated under the policy but NOT updated by the policy's own predicted
    observations. This is what pymdp's Agent(policy_len=2) computes.
  * sophisticated (closed-loop, Friston et al. 2021): the second action is chosen
    after conditioning on the first observation, G(a1) = G1(a1) +
    E_{o1} [ E_{a2 ~ sigma(-gamma G2(.|o1))} G2(a2 | o1) ].

Model.  Positions: Center, LeftArm, RightArm, CuePos. Context: cheese Left / Right,
uniform. From Center, Left/Right/Cue move to the arm / cue position. Arms are
absorbing: any action repeats the arm observation (Cheese or Shock, fixed by the
context). From the cue, Left/Right enter the arms (outcome fixed by context) and
Cue stays and re-observes the cue.

Observations: Cheese, Shock, SeeLeft, SeeRight (plus Null at Center for pymdp).
Risk_t = -E_{q(o_t|pi)} ln softmax(C)_{o_t};  infogain_t = E_{q(o_t|pi)}
KL[ q(s_t | o_t, pi) || q(s_t | pi) ]  (state information gain, nats).

Run:  python efe_two_step.py   (numpy only; pymdp cross-check at the end if importable)
"""
from __future__ import annotations
import itertools
import numpy as np

LN2 = np.log(2.0)
POS = ["Center", "LeftArm", "RightArm", "CuePos"]
CTX = ["L", "R"]
OBS = ["Cheese", "Shock", "SeeLeft", "SeeRight", "Null"]
ACT = ["Left", "Right", "Cue"]

# A[o, pos, ctx]
A = np.zeros((5, 4, 2))
A[4, 0, :] = 1                                   # Null at Center
A[0, 1, 0] = 1; A[1, 1, 1] = 1                   # LeftArm: Cheese if L, Shock if R
A[1, 2, 0] = 1; A[0, 2, 1] = 1                   # RightArm: Shock if L, Cheese if R
A[2, 3, 0] = 1; A[3, 3, 1] = 1                   # CuePos: SeeLeft if L, SeeRight if R

# B[pos', pos, a]
B = np.zeros((4, 4, 3))
for a, dest in enumerate([1, 2, 3]):
    B[dest, 0, a] = 1                            # from Center
    B[dest, 3, a] = 1                            # from Cue (Cue -> stay at cue = dest 3)
    B[1, 1, a] = 1; B[2, 2, a] = 1               # arms absorbing


def step(q_pos, q_ctx, a):
    """Propagate position under action a; context is static."""
    return B[:, :, a] @ q_pos, q_ctx


def predict_obs(q_pos, q_ctx):
    """q(o) = sum_{pos,ctx} A[o,pos,ctx] q(pos) q(ctx)  (factorised belief)."""
    return np.einsum("opc,p,c->o", A, q_pos, q_ctx)


def risk(q_o, C):
    lnp = C - np.log(np.sum(np.exp(C)))
    return float(-np.sum(q_o * lnp))


def infogain(q_pos, q_ctx, C=None):
    """State information gain about (pos, ctx) from the next observation, nats."""
    q_s = np.outer(q_pos, q_ctx)                 # (pos, ctx)
    q_o = predict_obs(q_pos, q_ctx)
    ig = 0.0
    for o in range(5):
        if q_o[o] < 1e-15:
            continue
        post = A[o] * q_s / q_o[o]
        m = post > 1e-15
        ig += q_o[o] * float(np.sum(post[m] * (np.log(post[m]) - np.log(q_s[m]))))
    return ig


def posterior(q_pos, q_ctx, o):
    q_s = np.outer(q_pos, q_ctx)
    post = A[o] * q_s
    post /= post.sum()
    return post.sum(1), post.sum(0)


def profile(name):
    return {
        "paper (+2/-4)": np.array([2.0, -4.0, 0.0, 0.0, 0.0]),
        "flat (0/0)":    np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "mild (+1/-1)":  np.array([1.0, -1.0, 0.0, 0.0, 0.0]),
    }[name]


def horizon1(C, gamma=16.0):
    rows = []
    q_pos0 = np.array([1.0, 0, 0, 0]); q_ctx0 = np.array([0.5, 0.5])
    for a in range(3):
        qp, qc = step(q_pos0, q_ctx0, a)
        r = risk(predict_obs(qp, qc), C); e = infogain(qp, qc)
        rows.append((ACT[a], r, e, r - e))
    G = np.array([x[3] for x in rows]); post = np.exp(-gamma * G); post /= post.sum()
    return rows, post


def horizon2_openloop(C, gamma=16.0):
    rows = []
    q_pos0 = np.array([1.0, 0, 0, 0]); q_ctx0 = np.array([0.5, 0.5])
    for a1, a2 in itertools.product(range(3), range(3)):
        qp1, qc1 = step(q_pos0, q_ctx0, a1)
        r1 = risk(predict_obs(qp1, qc1), C); e1 = infogain(qp1, qc1)
        qp2, qc2 = step(qp1, qc1, a2)          # belief NOT updated by o1
        r2 = risk(predict_obs(qp2, qc2), C); e2 = infogain(qp2, qc2)
        rows.append((f"{ACT[a1]}-{ACT[a2]}", r1, e1, r2, e2, (r1 - e1) + (r2 - e2)))
    G = np.array([x[-1] for x in rows]); post = np.exp(-gamma * G); post /= post.sum()
    return rows, post


def horizon2_sophisticated(C, gamma=16.0):
    """G(a1) = G1(a1) + E_{o1} E_{a2 ~ sigma(-gamma G2(.|o1))} G2(a2|o1)."""
    rows = []
    q_pos0 = np.array([1.0, 0, 0, 0]); q_ctx0 = np.array([0.5, 0.5])
    for a1 in range(3):
        qp1, qc1 = step(q_pos0, q_ctx0, a1)
        q_o1 = predict_obs(qp1, qc1)
        r1 = risk(q_o1, C); e1 = infogain(qp1, qc1)
        exp_r2 = exp_e2 = exp_G2 = 0.0
        for o1 in range(5):
            if q_o1[o1] < 1e-15:
                continue
            qp1_post, qc1_post = posterior(qp1, qc1, o1)
            G2 = np.zeros(3); R2 = np.zeros(3); E2 = np.zeros(3)
            for a2 in range(3):
                qp2, qc2 = step(qp1_post, qc1_post, a2)
                R2[a2] = risk(predict_obs(qp2, qc2), C); E2[a2] = infogain(qp2, qc2)
                G2[a2] = R2[a2] - E2[a2]
            pi2 = np.exp(-gamma * G2); pi2 /= pi2.sum()
            exp_r2 += q_o1[o1] * float(pi2 @ R2)
            exp_e2 += q_o1[o1] * float(pi2 @ E2)
            exp_G2 += q_o1[o1] * float(pi2 @ G2)
        rows.append((ACT[a1], r1, e1, exp_r2, exp_e2, (r1 - e1) + exp_G2))
    G = np.array([x[-1] for x in rows]); post = np.exp(-gamma * G); post /= post.sum()
    return rows, post


def main():
    out = []
    P = lambda s="": out.append(s)
    P("# Horizon-2 expected free energy, honest context model (nats)")
    P()
    P("Command: `python efe_two_step.py` (numpy; pymdp cross-check appended if importable).")
    P()
    for name in ["paper (+2/-4)", "flat (0/0)", "mild (+1/-1)"]:
        C = profile(name)
        P(f"## Preferences {name}: C over (Cheese, Shock, SeeLeft, SeeRight, Null) = {C.tolist()}")
        P()
        rows, post = horizon1(C)
        P("### Horizon 1")
        P()
        P("| action | risk | epistemic | G | q(pi), gamma=16 |")
        P("|---|---|---|---|---|")
        for (a, r, e, g), p in zip(rows, post):
            P(f"| {a} | {r:.3f} | {e:.3f} | {g:.3f} | {p:.3f} |")
        P()
        rows, post = horizon2_openloop(C)
        P("### Horizon 2, open-loop (pymdp policy_len=2 convention)")
        P()
        P("| policy | risk t1 | epist. t1 | risk t2 | epist. t2 | G | q(pi) |")
        P("|---|---|---|---|---|---|---|")
        for (pol, r1, e1, r2, e2, g), p in zip(rows, post):
            P(f"| {pol} | {r1:.3f} | {e1:.3f} | {r2:.3f} | {e2:.3f} | {g:.3f} | {p:.3f} |")
        P()
        rows, post = horizon2_sophisticated(C)
        P("### Horizon 2, sophisticated (second action conditioned on the first observation)")
        P()
        P("| first action | risk t1 | epist. t1 | E[risk t2] | E[epist. t2] | G | q(a1) |")
        P("|---|---|---|---|---|---|---|")
        for (a, r1, e1, r2, e2, g), p in zip(rows, post):
            P(f"| {a} | {r1:.3f} | {e1:.3f} | {r2:.3f} | {e2:.3f} | {g:.3f} | {p:.3f} |")
        P()
    P("## Reading")
    P()
    P("Every first action resolves the context, so the state information gain at t1 is ln 2 for all three,")
    P("and after any first observation nothing remains to be learned, so the sophisticated t2 epistemic term is 0")
    P("for every branch. The epistemic term never separates the actions at either horizon. What separates them at")
    P("horizon 2 is the decision value of the resolved belief: after the Cue the agent can still choose the cheese")
    P("arm (low t2 risk), after an arm it is absorbed (t2 risk equals t1 risk). Under the mild profile the three")
    P("actions tie at horizon 1 and the Cue wins at horizon 2 through this term alone. The open-loop convention")
    P("counts the same bit at both steps and cannot exploit it, which is why it prefers Cue-Cue.")
    text = "\n".join(out)

    # pymdp cross-check, open-loop policy_len=2
    try:
        try:
            from pymdp.legacy import utils
            from pymdp.legacy.agent import Agent
        except ImportError:
            from pymdp import utils
            from pymdp.agent import Agent
        A_obj = utils.obj_array(1); A_obj[0] = A.copy()
        B_obj = utils.obj_array(2); B_obj[0] = B.copy()
        Bc = np.zeros((2, 2, 1)); Bc[:, :, 0] = np.eye(2); B_obj[1] = Bc
        C_obj = utils.obj_array(1); C_obj[0] = profile("paper (+2/-4)")
        D_obj = utils.obj_array(2); D_obj[0] = np.array([1.0, 0, 0, 0]); D_obj[1] = np.array([0.5, 0.5])
        agent = Agent(A=A_obj, B=B_obj, C=C_obj, D=D_obj, policy_len=2, control_fac_idx=[0],
                      use_utility=True, use_states_info_gain=True, gamma=16.0)
        agent.reset(); agent.infer_states([4])
        q_pi, G = agent.infer_policies()
        text += "\n\n## pymdp cross-check (paper profile, policy_len=2, open-loop)\n\n| policy | -G (pymdp) | q(pi) |\n|---|---|---|\n"
        for pol, g, p in zip(agent.policies, G, q_pi):
            text += f"| {ACT[pol[0][0]]}-{ACT[pol[1][0]]} | {g:.3f} | {p:.3f} |\n"
        text += "\npymdp reports negative EFE (larger is better); compare with the open-loop table above.\n"
    except Exception as e:  # pragma: no cover
        text += f"\n\n(pymdp cross-check skipped: {e!r})\n"
    print(text)
    with open("efe_two_step.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
