"""Is the recovery floor our training budget, or something structural?

identifiability.py showed two regimes under a full-support policy:
    N=200 -> 3200 : clean statistical convergence (ratios 0.42, 0.52 vs 0.50)
    N > 3200      : a FLOOR at ~0.2-0.3, error stops falling

A floor means irreducible BIAS, and no amount of data removes bias. So the
positive identifiability claim (error -> 0 under full support) is NOT yet shown;
what is shown is convergence to a floor.

The floor is probably ours rather than fundamental, and Paper II is the evidence:
same bond settings, it reached TV <= 0.001 -- two orders of magnitude below our
plateau. It differed by using exact weighted enumeration instead of sampled
rollouts, and 200 epochs against our 40.

This module separates the candidates at FIXED N (so sampling noise is held
constant) by varying only budget and capacity:

    epochs 40 -> 100 -> 200 at bond 4   : is the floor TRAINING BUDGET?
    bond 4 -> 8 at the best epoch count : is the floor REPRESENTATIONAL CAPACITY?

Outcomes:
  error falls with epochs        -> floor was budget; identifiability holds, and
                                    we report the converged number.
  flat in epochs, falls with bond-> floor was capacity at chi=4; fixable, but the
                                    gamma sweep should be re-run at the larger bond.
  flat in both                   -> something structural binds in the sampled-
                                    rollout setting; the positive claim FAILS and
                                    that must be reported as such.

Run:  python -m src.paper3.floor_diagnosis
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from . import agents, persistent_tmaze as pt, phenotype as ph
from . import recovery_fidelity as rf
from .gamma_sweep import _spec_at
from .identifiability import support_check

CACHE = pathlib.Path(__file__).with_name("_floor_diagnosis.json")

N_FIXED = 3200          # where the floor appeared
CONFIGS = (             # (epochs, bond_dim)
    (40, 4),            # baseline, reproduces ~0.23
    (100, 4),
    (200, 4),
    (200, 8),
)


def measure(n_episodes: int, epochs: int, bond: int, seed: int = 0):
    rollouts, _ = agents.rollout(_spec_at(0.0), n_episodes, seed, verify=False)
    sup = support_check(rollouts)
    pool, _ = pt.rollouts_to_memory_pool(rollouts)
    mps, joint = ph.train_agent_mps(pool, epochs=epochs, bond_dim=bond, seed=seed)
    tensors = [m.detach().cpu().numpy() for m in mps.matrices]
    fmap = rf.fidelity_map(tensors, joint)
    errs = [v["error"] for v in fmap.values() if np.isfinite(v["error"])]
    return {"epochs": epochs, "bond": bond, "error": float(np.mean(errs)),
            "error_max": float(np.max(errs)),
            "bond_shapes": [list(t.shape) for t in tensors],
            **sup}


def run(configs=CONFIGS, n_episodes: int = N_FIXED, seed: int = 0):
    rows = []
    for epochs, bond in configs:
        r = measure(n_episodes, epochs, bond, seed)
        rows.append(r)
        print(f"  epochs={epochs:4d} bond={bond:2d}  mean err={r['error']:.4f}"
              f"  max={r['error_max']:.4f}  support={'full' if r['full_support'] else 'PARTIAL'}",
              flush=True)
    CACHE.write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    print(f"Fixed N={N_FIXED} (sampling noise held constant), full-support policy.\n"
          f"Varying only training budget and bond dimension.\n")
    rows = run()

    by_epoch = [r for r in rows if r["bond"] == 4]
    e0, e_last = by_epoch[0]["error"], by_epoch[-1]["error"]
    budget_gain = (e0 - e_last) / e0 if e0 > 0 else 0.0

    bond4_200 = next((r for r in rows if r["epochs"] == 200 and r["bond"] == 4), None)
    bond8_200 = next((r for r in rows if r["epochs"] == 200 and r["bond"] == 8), None)
    cap_gain = 0.0
    if bond4_200 and bond8_200 and bond4_200["error"] > 0:
        cap_gain = (bond4_200["error"] - bond8_200["error"]) / bond4_200["error"]

    print(f"\nbudget effect (40 -> 200 epochs at bond 4): {e0:.4f} -> {e_last:.4f}"
          f"  ({budget_gain:+.0%})")
    if bond4_200 and bond8_200:
        print(f"capacity effect (bond 4 -> 8 at 200 epochs): {bond4_200['error']:.4f}"
              f" -> {bond8_200['error']:.4f}  ({cap_gain:+.0%})")

    print("\n--- verdict ---")
    if budget_gain > 0.3:
        print("FLOOR WAS TRAINING BUDGET. Error falls substantially with more epochs at\n"
              "fixed data, so it is optimisation, not bias. The positive identifiability\n"
              "claim stands; report the converged error, and note the gamma sweep was run\n"
              "at a budget that inflates all its absolute numbers (the TREND is unaffected\n"
              "since every gamma got the same budget).")
    elif cap_gain > 0.3:
        print("FLOOR WAS CAPACITY at bond 4. More epochs did not help but more bond did,\n"
              "so it is representational bias. Fixable -- but the gamma sweep should be\n"
              "re-run at the larger bond before its numbers are quoted.")
    else:
        print("FLOOR IS STRUCTURAL: neither budget nor capacity removes it at fixed N.\n"
              "Full support is NOT sufficient for exact recovery in this sampled-rollout\n"
              "setting. The positive identifiability claim FAILS and must be reported as\n"
              "convergence-to-a-floor, with the floor's origin left open.")
