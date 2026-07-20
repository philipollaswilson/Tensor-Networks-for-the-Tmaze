"""Per-state recovery fidelity: what world-model can an observer build from
watching THIS agent? (Paper III, non-circular phenotype)

Two earlier attempts failed an adversarial audit and are kept as documented
negative results:

  1. blind_validation.py classified agents on their first-action marginal. That is
     the behaviour we hand-built into them, the trained MPS was never consulted,
     and a bare P(a1) histogram tied its accuracy 1.00.
  2. structure_phenotype.py thresholded which histories are "supported". The
     threshold is arbitrary and, measured, it does not bite: the reward-gambler
     still visits the cue ~2% of the time, so its cue histories clear any sane
     floor even though it has almost no data there. Binary coverage misses the
     real effect.

The real effect is continuous. An agent's MPS reconstructs the environment
accurately only where that agent's policy took it. So we measure, per latent
state, the L1 error between the predictive signature RECOVERED from the agent's
MPS and the ANALYTIC signature of the true generative model:

    error(state) = || p_MPS(o3 | a3, state) - p_true(o3 | a3, state) ||_1

This is a property of the learned tensor network (the signature comes from a bond
ray contracted against the third tensor); raw behaviour statistics give visit
counts, not reconstruction quality of latent structure. The resulting error map
is the phenotype: an info-seeker builds a model that resolves the hidden context;
a reward-greedy agent builds one accurate about arms but blind at the cue.

Run:  python -m src.paper3.recovery_fidelity
"""
from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from . import agents, persistent_tmaze as pt, phenotype as ph
from . import generative_model as gm
from .structure_phenotype import HISTORIES


def context_posterior(L1: int, o_rew: int, o_cue: int, A) -> np.ndarray:
    """q(K | o1) after one step -- the belief an ideal observer holds."""
    A_rew, A_cue = A[1], A[2]
    post = np.array([0.5 * A_rew[o_rew, L1, K] * A_cue[o_cue, L1, K]
                     for K in range(gm.N_CTX)])
    s = post.sum()
    return post / s if s > 1e-12 else np.full(gm.N_CTX, 0.5)


def true_signature(a1: int, otup, reward_fidelity: float = 0.85) -> np.ndarray:
    """Analytic p(o3 | a3, state) in the same (4 actions x 12 pos-reward) layout
    structure_recovery.predictive_signature produces.

    ⚠️ CONVENTION MISMATCH -- see DEVLOG 2026-07-19/20. This assumes the arm reward
    is RE-SAMPLED from the noisy channel at each step, matching agents.rollout.
    persistent_tmaze.enumerate_persistent instead makes the arm reward PERSIST
    (r2 = r1), following the original tmaze.py. The two generators therefore
    describe different environments at arm states.

    All agent-based results are internally consistent (rollout + this function
    agree). Anything trained on enumerate_persistent and scored here is NOT --
    that mismatch produces a flat ~2.04 error at all four arm states, which is how
    the bug was found. Resolve the convention before re-running the ceiling test.
    """
    A = gm.build_A(reward_fidelity)
    L1 = a1
    qK = context_posterior(L1, otup[1], otup[2], A)
    pr = np.zeros((gm.N_ACT, 12))
    for a3 in range(gm.N_ACT):
        L2 = L1 if L1 in gm.ARMS else a3          # absorbing arms
        for rew in range(3):
            pr[a3, 3 * L2 + rew] = float(np.sum(qK * A[1][rew, L2, :]))
        row = pr[a3].sum()
        if row > 1e-12:
            pr[a3] /= row
    return pr.ravel()


def recovered_signature(tensors, a1: int, otup):
    """p(o3 | a3, state) read off the agent's trained MPS via the bond ray."""
    sr = importlib.import_module("structure_recovery")
    T1, T2, T3 = tensors
    # T1 is (bond_left, action, observation, bond_right): the leading action is
    # the null one and the start observation is (0,0,0). Indexing with three
    # subscripts silently feeds the obs index into the action slot -- guard it.
    assert T1.ndim == 4, f"expected T1 (bond,action,obs,bond), got {T1.shape}"
    left = T1[0, 0, pt.obs_index(0, 0, 0), :].astype(complex)
    assert left.ndim == 1, f"left environment must be a vector, got {left.shape}"
    b = np.einsum("i,ij->j", left, T2[:, a1, pt.obs_index(*otup), :]).astype(complex)
    nb = np.linalg.norm(b)
    if nb < 1e-9:
        return None
    return sr.predictive_signature(b / nb, T3)


def fidelity_map(tensors, joint):
    """L1 recovery error per latent state, plus that state's Born weight."""
    out = {}
    for name, a1, otup in HISTORIES:
        rec = recovered_signature(tensors, a1, otup)
        w = float(joint[0, pt.obs_index(0, 0, 0), a1, pt.obs_index(*otup)].sum())
        if rec is None:
            out[name] = {"error": float("nan"), "weight": w}
            continue
        err = float(np.abs(rec - true_signature(a1, otup)).sum())
        out[name] = {"error": err, "weight": w}
    return out


def phenotype_agent(spec, n_episodes: int = 1200, epochs: int = 25, seed: int = 0):
    rollouts, stats = agents.rollout(spec, n_episodes, seed, verify=False)
    pool, _ = pt.rollouts_to_memory_pool(rollouts)
    mps, joint = ph.train_agent_mps(pool, epochs=epochs, seed=seed)
    tensors = [m.detach().cpu().numpy() for m in mps.matrices]
    return fidelity_map(tensors, joint), stats


if __name__ == "__main__":
    names = [h[0] for h in HISTORIES]
    print("Per-state recovery error (L1 of the recovered vs analytic predictive")
    print("signature). Lower = this agent's data supports recovering that state.\n")
    print(f"{'agent':16s}" + "".join(f"{n:>11s}" for n in names))
    rows = {}
    for spec in agents.ROSTER:
        fm, _ = phenotype_agent(spec)
        rows[spec.name] = fm
        print(f"{spec.name:16s}" + "".join(f"{fm[n]['error']:11.3f}" for n in names))
    print()
    print(f"{'(born weight)':16s}")
    for name, fm in rows.items():
        print(f"{name:16s}" + "".join(f"{fm[n]['weight']:11.4f}" for n in names))
