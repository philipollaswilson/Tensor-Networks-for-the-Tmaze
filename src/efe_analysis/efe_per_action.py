"""
Per-action expected free energy and empowerment in the minimal T-maze,
under the honest context model (cheese location is a hidden variable that
every arm-entry observation also reveals).

Answers AI reviewer point 2 (no EFE decomposition shown) and grounds the
corrected treatment of BU2X point 3 / AI reviewer point 1 (the arms carry
1 bit of context information too, so the Cue is preferred through the risk
term of EFE, not through uniquely epistemic value).

Plain numpy, no dependencies beyond numpy. Cross-checked against pymdp's
own EFE computation at the bottom if pymdp is importable.

Conventions follow Friston et al. 2015 (Active inference and epistemic
value), eq. for a one-step policy pi = a:

    G(a) = -E_{q(o|a)}[ ln p(o | C) ]                       (risk / pragmatic)
           -E_{q(o|a)}[ KL( q(s|o,a) || q(s|a) ) ]          (epistemic value)

with p(o|C) = softmax(C). Lower G is better; the policy posterior is
softmax(-gamma * G). Both terms are reported in nats and in bits.
"""
from __future__ import annotations
import numpy as np

LN2 = np.log(2.0)

# ------------------------------------------------------------------
# Honest context model of the minimal T-maze
# ------------------------------------------------------------------
# Hidden state at t2 = (position, context)
POS = ["LeftArm", "RightArm", "CuePos"]
CTX = ["L", "R"]  # cheese on the Left / on the Right
OBS = ["Cheese", "Shock", "SeeLeft", "SeeRight"]
ACT = ["Left", "Right", "Cue"]

# Likelihood A[o, pos, ctx]
A = np.zeros((4, 3, 2))
A[OBS.index("Cheese"),   POS.index("LeftArm"),  CTX.index("L")] = 1
A[OBS.index("Shock"),    POS.index("LeftArm"),  CTX.index("R")] = 1
A[OBS.index("Shock"),    POS.index("RightArm"), CTX.index("L")] = 1
A[OBS.index("Cheese"),   POS.index("RightArm"), CTX.index("R")] = 1
A[OBS.index("SeeLeft"),  POS.index("CuePos"),   CTX.index("L")] = 1
A[OBS.index("SeeRight"), POS.index("CuePos"),   CTX.index("R")] = 1

# Action -> position is deterministic at the choice point.
def pos_after(a: str) -> int:
    return {"Left": 0, "Right": 1, "Cue": 2}[a]


def efe_one_step(a: str, q_ctx: np.ndarray, C: np.ndarray):
    """Return (G, risk, epistemic, IG_bits, q_o) for a single action from the
    choice point, with belief q_ctx over context and log-preferences C over
    observations."""
    p = pos_after(a)
    # q(s|a): position is known given a, context is q_ctx
    # q(o|a) = sum_ctx A[o, p, ctx] q(ctx)
    q_o = A[:, p, :] @ q_ctx
    # posterior over context given o and a
    eps = 1e-16
    epistemic = 0.0
    for o in range(4):
        if q_o[o] <= eps:
            continue
        q_ctx_post = A[o, p, :] * q_ctx / q_o[o]
        kl = np.sum(q_ctx_post * (np.log(q_ctx_post + eps) - np.log(q_ctx + eps)))
        epistemic += q_o[o] * kl
    lnp = C - np.log(np.sum(np.exp(C)))  # ln softmax(C)
    risk = -np.sum(q_o * lnp)
    G = risk - epistemic
    return G, risk, epistemic, epistemic / LN2, q_o


def channel_capacity(W: np.ndarray, iters: int = 2000, tol: float = 1e-12) -> float:
    """Blahut-Arimoto. W[a, o] = p(o|a). Returns capacity in bits."""
    n_a = W.shape[0]
    p = np.ones(n_a) / n_a
    eps = 1e-300
    for _ in range(iters):
        q_o = p @ W
        # c_a = exp( sum_o W[a,o] ln (W[a,o] / q_o) )
        with np.errstate(divide="ignore", invalid="ignore"):
            logratio = np.where(W > 0, np.log(W + eps) - np.log(q_o + eps), 0.0)
        c = np.exp(np.sum(W * logratio, axis=1))
        p_new = p * c / np.sum(p * c)
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new
    q_o = p @ W
    with np.errstate(divide="ignore", invalid="ignore"):
        logratio = np.where(W > 0, np.log(W + eps) - np.log(q_o + eps), 0.0)
    return float(np.sum(p[:, None] * W * logratio) / LN2)


def channel_at_choice_point(q_ctx: np.ndarray) -> np.ndarray:
    """W[a, o] = sum_ctx q(ctx) p(o | a, ctx) at the choice point."""
    return np.stack([A[:, pos_after(a), :] @ q_ctx for a in ACT])


def channel_in_trap(o_seen: str) -> np.ndarray:
    """Absorbing arm: every action repeats the observation just seen."""
    W = np.zeros((3, 4))
    W[:, OBS.index(o_seen)] = 1
    return W


def fmt(x):
    return f"{x:7.3f}"


def run(C_dict: dict, label: str, q_ctx=np.array([0.5, 0.5]), gamma: float = 16.0):
    C = np.array([C_dict[o] for o in OBS], dtype=float)
    print(f"\n### {label}")
    print("C (log-preferences over Cheese, Shock, SeeLeft, SeeRight):", C.tolist())
    print()
    print("| action | risk (nats) | epistemic (nats) | epistemic (bits) | G (nats) | q(o|a) Cheese/Shock/SeeL/SeeR |")
    print("|---|---|---|---|---|---|")
    Gs = []
    for a in ACT:
        G, risk, epi, ig_bits, q_o = efe_one_step(a, q_ctx, C)
        Gs.append(G)
        print(f"| {a:5s} | {fmt(risk)} | {fmt(epi)} | {fmt(ig_bits)} | {fmt(G)} | "
              + "/".join(f"{v:.2f}" for v in q_o) + " |")
    Gs = np.array(Gs)
    post = np.exp(-gamma * Gs)
    post /= post.sum()
    print()
    print(f"policy posterior softmax(-{gamma:g} G):",
          ", ".join(f"{a}={p:.3f}" for a, p in zip(ACT, post)))
    print("selected (argmin G):", ACT[int(np.argmin(Gs))])
    # actual empowerment under the stochastic policy posterior (BU2X 5)
    W = channel_at_choice_point(q_ctx)
    q_o = post @ W
    eps = 1e-300
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(W > 0, np.log(W + eps) - np.log(q_o + eps), 0.0)
    actual = float(np.sum(post[:, None] * W * lr) / LN2)
    print(f"actual empowerment under that policy posterior: {actual:.3f} bits "
          f"(potential is {channel_capacity(W):.3f} bits)")
    return Gs, post


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    q = np.array([0.5, 0.5])

    print("## Empowerment (subjective potential, one-step, bits)")
    W_mixed = channel_at_choice_point(q)
    E_mixed = channel_capacity(W_mixed)
    E_resolved = [channel_capacity(channel_at_choice_point(np.eye(2)[k])) for k in range(2)]
    E_trap = channel_capacity(channel_in_trap("Cheese"))
    print(f"choice point, context uniform (before Cue): {E_mixed:.3f}")
    print(f"choice point, context resolved (after Cue): {E_resolved[0]:.3f} (L), {E_resolved[1]:.3f} (R)")
    print(f"expected post-Cue empowerment: {0.5*sum(E_resolved):.3f}  >=  mixed {E_mixed:.3f}   [convexity lemma]")
    print(f"in an arm (absorbing): {E_trap:.3f}")
    print()
    print("Mixed channel W[a,o] at the choice point (rows Left, Right, Cue; cols Cheese, Shock, SeeLeft, SeeRight):")
    print(W_mixed)

    print("\n## Information gain about context, per action (bits)")
    for a in ACT:
        _, _, _, ig, _ = efe_one_step(a, q, np.zeros(4))
        print(f"{a:5s}: {ig:.3f}")
    print("(Under the honest model every action resolves the context. The paper's "
          "0/0/1 assignment came from collapsing both arms to a single hidden state.)")

    print("\n## One-step EFE per action at the choice point")
    # Setting 1: the paper's stated preferences, shock aversion > cheese preference
    run({"Cheese": 2.0, "Shock": -4.0, "SeeLeft": 0.0, "SeeRight": 0.0},
        "Setting 1: prefers cheese (+2), averse to shock (-4). This is the pymdp T-maze default profile.")
    # Setting 2: symmetric preferences: arms and Cue tie on risk, all tie on IG
    run({"Cheese": 2.0, "Shock": -2.0, "SeeLeft": 0.0, "SeeRight": 0.0},
        "Setting 2: symmetric (+2 / -2). Risk of an arm equals risk of the Cue, so all three actions tie.")
    # Setting 3: gambler: cheese preference exceeds shock aversion
    run({"Cheese": 4.0, "Shock": -2.0, "SeeLeft": 0.0, "SeeRight": 0.0},
        "Setting 3: gambler (+4 / -2). The arms are now preferred: same epistemic value, lower risk.")
    # Setting 4: no preferences at all: pure epistemic agent ties
    run({"Cheese": 0.0, "Shock": 0.0, "SeeLeft": 0.0, "SeeRight": 0.0},
        "Setting 4: flat preferences. Pure epistemic agent: three-way tie, the Cue is NOT uniquely epistemic.")

    print("\n## Closed form")
    print("With ln p(o) = C_o - lse(C): risk(arm) = lse(C) - (C_cheese + C_shock)/2, "
          "risk(Cue) = lse(C) - (C_seeL + C_seeR)/2, epistemic(all) = ln 2.")
    print("So under the honest model the Cue is selected iff "
          "(C_cheese + C_shock)/2 < (C_seeL + C_seeR)/2, i.e. iff shock aversion exceeds cheese preference "
          "relative to the neutral cue observations.")

    # ------------------------------------------------------------------
    # Cross-check against pymdp (optional)
    # ------------------------------------------------------------------
    try:
        try:  # pymdp >= 1.0 keeps the numpy API under pymdp.legacy
            from pymdp.legacy import utils
            from pymdp.legacy.agent import Agent
        except ImportError:
            from pymdp import utils
            from pymdp.agent import Agent

        # Two hidden-state factors: position (Center, LeftArm, RightArm, CuePos) and context (L, R)
        n_pos, n_ctx = 4, 2
        A_p = np.zeros((4, n_pos, n_ctx))   # one modality with 4 outcomes; add a 5th "Null" outcome at Center
        A_p = np.zeros((5, n_pos, n_ctx))
        A_p[4, 0, :] = 1  # Null at Center
        for ctx in range(2):
            A_p[0:4, 1:4, ctx] = A[:, :, ctx]  # LeftArm, RightArm, CuePos
        A_obj = utils.obj_array(1); A_obj[0] = A_p
        B_pos = np.zeros((n_pos, n_pos, 3))
        for a in range(3):
            B_pos[a + 1, :, a] = 1   # from anywhere, action a leads to position a+1 (choice-point interface)
        B_ctx = np.zeros((n_ctx, n_ctx, 1)); B_ctx[:, :, 0] = np.eye(2)
        B_obj = utils.obj_array(2); B_obj[0] = B_pos; B_obj[1] = B_ctx
        C_obj = utils.obj_array(1); C_obj[0] = np.array([2.0, -4.0, 0.0, 0.0, 0.0])
        D_obj = utils.obj_array(2); D_obj[0] = np.array([1.0, 0, 0, 0]); D_obj[1] = np.array([0.5, 0.5])
        agent = Agent(A=A_obj, B=B_obj, C=C_obj, D=D_obj, policy_len=1,
                      control_fac_idx=[0], use_utility=True, use_states_info_gain=True, gamma=16.0)
        agent.reset()
        agent.infer_states([4])           # observe Null at Center
        q_pi, G = agent.infer_policies()
        print("\n## pymdp cross-check (Setting 1, policy_len=1)")
        for a, g, p in zip(ACT, G, q_pi):
            print(f"{a:5s}: G = {g:7.3f}   q(pi) = {p:.3f}")
        print("(pymdp reports negative EFE with a sign convention where larger is better.)")
    except Exception as e:  # pragma: no cover
        print("\n(pymdp cross-check skipped:", repr(e), ")")
