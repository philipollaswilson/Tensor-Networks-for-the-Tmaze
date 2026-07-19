"""Does the tensor network actually earn its place? (Paper III, adversarial check)

The blind-discrimination result was initially computed from RAW ROLLOUT ACTIONS,
never touching the trained MPS -- and a trivial baseline (the first-action
histogram, no criteria, no EFE, no tensor network) matched it at accuracy 1.00.
That makes the headline circular: agents built to act differently are told apart
by their actions.

This module runs the honest test the earlier pipeline skipped:

    At sample size N, does the phenotype read off the RECOVERED MPS estimate the
    agent's true policy marginal BETTER than the raw empirical histogram?

If the MPS never beats the histogram, the tensor network is not doing work in the
classification story and the paper must say so and justify itself differently
(e.g. by what only a full generative model provides: emission/transition
structure, empowerment, EFE on the recovered model). We report whichever way the
numbers fall.

Run:  python -m src.paper3.baseline_comparison
"""
from __future__ import annotations

import numpy as np

from . import agents, persistent_tmaze as pt, phenotype as ph


def true_marginal(spec, n_episodes: int = 20000, seed: int = 999) -> np.ndarray:
    """Ground-truth first-action marginal from a large sample."""
    rollouts, _ = agents.rollout(spec, n_episodes, seed, verify=False)
    return ph.recovered_action_marginal(ph.empirical_joint(rollouts))


def estimates_at_n(spec, n: int, seed: int, epochs: int = 30):
    """Return (raw_estimate, mps_estimate) of the first-action marginal from the
    SAME n episodes -- the only difference is whether we histogram them or fit a
    tensor network to them and read the marginal off the learned joint."""
    rollouts, _ = agents.rollout(spec, n, seed, verify=False)
    raw = ph.recovered_action_marginal(ph.empirical_joint(rollouts))
    pool, _ = pt.rollouts_to_memory_pool(rollouts)
    _mps, joint = ph.train_agent_mps(pool, epochs=epochs, seed=seed)
    mps = ph.recovered_action_marginal(joint)
    return raw, mps


def run(n_values=(10, 25, 50, 100), trials: int = 5, epochs: int = 30, roster=None):
    """Mean L1 error to the true marginal, raw histogram vs MPS-recovered."""
    roster = roster or agents.ROSTER
    truths = {s.name: true_marginal(s) for s in roster}
    print(f"{'N':>5s} {'raw L1':>9s} {'MPS L1':>9s} {'winner':>10s}")
    table = {}
    for n in n_values:
        raw_errs, mps_errs = [], []
        for s in roster:
            for t in range(trials):
                raw, mps = estimates_at_n(s, n, seed=1000 + 17 * t, epochs=epochs)
                raw_errs.append(np.abs(raw - truths[s.name]).sum())
                mps_errs.append(np.abs(mps - truths[s.name]).sum())
        r, m = float(np.mean(raw_errs)), float(np.mean(mps_errs))
        winner = "MPS" if m < r else ("raw" if r < m else "tie")
        table[n] = (r, m, winner)
        print(f"{n:5d} {r:9.4f} {m:9.4f} {winner:>10s}")
    return table


if __name__ == "__main__":
    print("Estimating the true first-action marginal, then comparing estimators\n"
          "built from the SAME data (histogram vs MPS-recovered marginal).\n")
    table = run()
    wins = sum(1 for _, _, w in table.values() if w == "MPS")
    print(f"\nMPS beat the raw histogram at {wins}/{len(table)} sample sizes.")
    if wins == 0:
        print("=> The tensor network adds NOTHING to this phenotype estimate. The\n"
              "   classification framing cannot justify the MPS; the paper must\n"
              "   justify it by what only a full generative model provides.")
