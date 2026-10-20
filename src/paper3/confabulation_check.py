"""Does the MPS CONFABULATE at conditionals the agent never exercised?

The identifiability law may not need a tensor network: you can detect "this
(state, action) was never sampled" by counting. If that is all there is, the
result is an identifiability note with an MPS attached.

The version that IS about generative models: counting tells you where you are
blind, but a FITTED model does not report a gap there -- it emits a
confident-looking distribution interpolated from bond geometry, with nothing
marking it as invented. If true, that is practically important: you cannot tell
from the model which parts were recovered and which were made up.

ADVERSARIAL DESIGN. Two ways this could be an artifact of my own metric, both
measured here rather than assumed away:

  1. NORMALISATION MANUFACTURES THE CONFIDENCE. predictive_signature normalises
     each action row. If the model assigns ~zero mass to an unexercised action,
     dividing by that tiny mass amplifies numerical noise into a spuriously
     peaked distribution. Then "confabulation" is my arithmetic, not the model's
     behaviour. So we report the RAW (pre-normalisation) row mass: if unexercised
     rows carry negligible mass, the model is effectively saying "this never
     happens" and the confidence is my artifact.

  2. CONFIDENT-AND-CORRECT IS GOOD GENERALISATION, NOT CONFABULATION. At an arm
     state the maze is absorbing, so every action leads to the same future -- a
     model that confidently predicts an unexercised action there is RIGHT, having
     inferred it from shared structure. Confabulation requires confident AND
     WRONG. So we cross-classify every row:

        exercised                      -> baseline
        unexercised, low error         -> correct generalisation (good!)
        unexercised, high error, low H -> CONFABULATION (confident and wrong)
        unexercised, high entropy      -> honest hedging (model signals ignorance)

Only the third cell supports the claim. We report all four.

Run:  python -m src.paper3.confabulation_check
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import agents, persistent_tmaze as pt, phenotype as ph
from . import recovery_fidelity as rf
from .gamma_sweep import _spec_at
from .structure_phenotype import HISTORIES

N_CLASS = 12                       # position(4) x reward(3) folded classes
MAX_H = float(np.log2(N_CLASS))    # 3.585 bits = maximum ignorance
HIGH_ERROR = 1.0                   # L1 over a single row (max 2.0)
LOW_ENTROPY = 0.5 * MAX_H          # "confident" = below half of maximum


def recovered_rows(tensors, a1, otup):
    """Per-action recovered future distribution AND its raw pre-normalisation
    mass. The raw mass is the control for artifact (1)."""
    T1, T2, T3 = tensors
    left = T1[0, 0, pt.obs_index(0, 0, 0), :].astype(complex)
    b = np.einsum("i,ij->j", left, T2[:, a1, pt.obs_index(*otup), :]).astype(complex)
    nb = np.linalg.norm(b)
    if nb < 1e-9:
        return None, None
    fut = np.abs(np.einsum("j,jcqy->cq", b / nb, T3)) ** 2      # (4 actions, 24 obs)
    pr = np.zeros((4, N_CLASS))
    for o in range(24):
        pr[:, 3 * (o // 6) + (o // 2) % 3] += fut[:, o]
    mass = pr.sum(1)                                            # raw, un-normalised
    safe = np.where(mass > 1e-12, mass, 1.0)
    return pr / safe[:, None], mass


def _entropy(p):
    p = np.clip(np.asarray(p, float), 0, None)
    s = p.sum()
    if s <= 1e-12:
        return float("nan")
    p = p / s
    m = p > 0
    return float(-(p[m] * np.log2(p[m])).sum())


def analyse_agent(spec, n_episodes: int = 800, epochs: int = 25, seed: int = 0):
    rollouts, _ = agents.rollout(spec, n_episodes, seed, verify=False)

    # which (state, a3) pairs were actually exercised, and how often
    counts = Counter()
    for a, o in rollouts:
        counts[(int(a[1]), tuple(int(x) for x in o[1]), int(a[2]))] += 1

    pool, _ = pt.rollouts_to_memory_pool(rollouts)
    mps, joint = ph.train_agent_mps(pool, epochs=epochs, seed=seed)
    tensors = [m.detach().cpu().numpy() for m in mps.matrices]

    rows = []
    for name, a1, otup in HISTORIES:
        rec, mass = recovered_rows(tensors, a1, otup)
        if rec is None:
            continue
        true = rf.true_signature(a1, otup).reshape(4, N_CLASS)
        total_mass = mass.sum()
        for a3 in range(4):
            n = counts[(a1, otup, a3)]
            rows.append({
                "state": name, "a3": a3, "count": n,
                "exercised": n > 0,
                "error": float(np.abs(rec[a3] - true[a3]).sum()),
                "entropy": _entropy(rec[a3]),
                "rel_mass": float(mass[a3] / total_mass) if total_mass > 0 else float("nan"),
            })
    return rows


def classify(r):
    if r["exercised"]:
        return "exercised"
    if r["error"] < HIGH_ERROR:
        return "generalised"          # unexercised but correct -- good
    if r["entropy"] < LOW_ENTROPY:
        return "CONFABULATED"         # confident AND wrong -- the claim
    return "hedged"                   # signals ignorance -- honest


def report(spec, rows):
    print(f"\n=== {spec.name} ===")
    buckets = {}
    for r in rows:
        buckets.setdefault(classify(r), []).append(r)
    print(f"{'category':>14s}{'n':>5s}{'mean err':>10s}{'mean H':>9s}{'mean rel-mass':>15s}")
    for k in ("exercised", "generalised", "CONFABULATED", "hedged"):
        b = buckets.get(k, [])
        if not b:
            continue
        print(f"{k:>14s}{len(b):5d}{np.nanmean([x['error'] for x in b]):10.3f}"
              f"{np.nanmean([x['entropy'] for x in b]):9.2f}"
              f"{np.nanmean([x['rel_mass'] for x in b]):15.4f}")
    return buckets


if __name__ == "__main__":
    print(f"max entropy (total ignorance) = {MAX_H:.2f} bits; "
          f"'confident' means H < {LOW_ENTROPY:.2f}")
    print("Artifact controls: rel-mass ~0 on unexercised rows would mean the\n"
          "normalisation manufactured the confidence, not the model.")

    all_buckets = {}
    for spec in (_spec_at(16.0), _spec_at(0.0)):     # deterministic vs random control
        rows = analyse_agent(spec)
        all_buckets[spec.name] = report(spec, rows)

    conf = all_buckets[_spec_at(16.0).name].get("CONFABULATED", [])
    print("\n--- verdict ---")
    if not conf:
        print("NO confabulation: every unexercised row is either correctly generalised\n"
              "or honestly hedged. The model does not invent confident structure, so the\n"
              "tensor-network-specific claim FAILS and the finding stays an\n"
              "identifiability result detectable by counting.")
    else:
        mm = float(np.nanmean([r["rel_mass"] for r in conf]))
        print(f"{len(conf)} confabulated rows (confident AND wrong), mean relative "
              f"mass {mm:.4f}.")
        if mm < 0.01:
            print("BUT their raw mass is negligible: the model assigns almost no\n"
                  "probability to these actions, so the apparent confidence is an\n"
                  "artifact of row normalisation. Claim NOT supported.")
        else:
            print("Their raw mass is substantial, so the model genuinely predicts these\n"
                  "actions occur and is confidently wrong about them. Claim SUPPORTED:\n"
                  "the fitted model invents structure where the data was silent, and\n"
                  "nothing in the model flags it.")
