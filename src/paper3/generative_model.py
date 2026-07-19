"""Generative model (A, B, C, D) for the persistent-latent T-maze (Paper III).

A transparent, factored generative model matching persistent_tmaze.py exactly,
built as plain numpy so a small active-inference agent (agents.py) can plan on it
and so workstream 2 (EFE-regret) can reuse the same EFE machinery. This is the
agent's model of the same environment the data generator samples; keeping them
identical is deliberate -- the agent is not misspecified, so any behavioural
difference between agents is due to preferences C and precision gamma, not to a
different world model.

Hidden state factors:
  location L in {0:center, 1:right, 2:left, 3:cue}   (arms 1,2 are absorbing)
  context  K in {0, 1}                               (persistent, hidden)

Observation modalities (the Paper II [4,3,2] encoding, kept factored here):
  position obs (4) = location, deterministic
  reward   obs (3) = {none, cheese, shock}; noisy readout of K at an arm
  cue      obs (2) = reads K at the cue, 0 elsewhere

Actions (control the location factor): go to {center, right, left, cue}.
"""
from __future__ import annotations

import numpy as np

N_LOC = 4
N_CTX = 2
N_ACT = 4
CENTER, RIGHT, LEFT, CUE = 0, 1, 2, 3
ARMS = (RIGHT, LEFT)


def preferred_reward(context: int, arm: int) -> int:
    """cheese (1) when context and arm agree in parity, else shock (2). Matches
    persistent_tmaze._preferred_reward."""
    return 1 if (context + arm) % 2 == 0 else 2


def build_A(reward_fidelity: float = 0.85):
    """Observation model, a list of three arrays [A_pos, A_rew, A_cue].

    A_pos[o, L]      = 1 if o == L
    A_rew[o, L, K]   = noisy readout of K at an arm; none elsewhere
    A_cue[o, L, K]   = 1 at cue when o == K; o == 0 otherwise
    """
    A_pos = np.eye(N_LOC)                                  # (4, 4)

    A_rew = np.zeros((3, N_LOC, N_CTX))                    # (3, 4, 2)
    for L in range(N_LOC):
        for K in range(N_CTX):
            if L in ARMS:
                pref = preferred_reward(K, L)
                other = 1 if pref == 2 else 2
                A_rew[pref, L, K] = reward_fidelity
                A_rew[other, L, K] = 1.0 - reward_fidelity
            else:
                A_rew[0, L, K] = 1.0                       # none

    A_cue = np.zeros((N_CTX, N_LOC, N_CTX))                # (2, 4, 2)
    for L in range(N_LOC):
        for K in range(N_CTX):
            A_cue[K if L == CUE else 0, L, K] = 1.0

    return [A_pos, A_rew, A_cue]


def build_B():
    """Transition model B[L', L, a]: go to location a; arms are absorbing. The
    context factor is persistent (identity), so it is not represented here -- the
    agent carries it as a static belief updated only by observation."""
    B = np.zeros((N_LOC, N_LOC, N_ACT))
    for L in range(N_LOC):
        for a in range(N_ACT):
            L_next = L if L in ARMS else a                 # absorbing arms
            B[L_next, L, a] = 1.0
    return B


def build_D():
    """Priors: start at center; context uniform (the hidden latent to resolve)."""
    d_loc = np.zeros(N_LOC); d_loc[CENTER] = 1.0
    d_ctx = np.full(N_CTX, 1.0 / N_CTX)
    return [d_loc, d_ctx]


def build_C(C_reward):
    """Log-preferences per observation modality. Preferences live on the reward
    modality (from the AgentSpec); position and cue are neutral."""
    C_pos = np.zeros(N_LOC)
    C_rew = np.asarray(C_reward, dtype=float)              # (3,)
    C_cue = np.zeros(N_CTX)
    return [C_pos, C_rew, C_cue]


if __name__ == "__main__":
    A = build_A(); B = build_B(); D = build_D()
    # A_rew columns are proper distributions
    assert np.allclose(A[1].sum(0), 1.0), "A_rew not normalised"
    assert np.allclose(A[0].sum(0), 1.0) and np.allclose(A[2].sum(0), 1.0)
    assert np.allclose(B.sum(0), 1.0), "B not normalised"
    # info geometry of the observation model, against the true (uniform) context
    def mi(joint):
        j = joint / joint.sum(); px = j.sum(1, keepdims=True); py = j.sum(0, keepdims=True)
        m = j > 0; return float(np.sum(j[m] * np.log2(j[m] / (px @ py)[m])))
    # reward vs context at an arm (average the two arms)
    igs = []
    for arm in ARMS:
        joint = np.zeros((3, N_CTX))
        for K in range(N_CTX):
            joint[:, K] = A[1][:, arm, K] * 0.5
        igs.append(mi(joint))
    print(f"I(reward; ctx | arm) from A_rew = {np.mean(igs):.3f} bit (expect 0.390)")
    joint = np.zeros((N_CTX, N_CTX))
    for K in range(N_CTX):
        joint[:, K] = A[2][:, CUE, K] * 0.5
    print(f"I(cue; ctx) from A_cue          = {mi(joint):.3f} bit (expect 1.000)")
