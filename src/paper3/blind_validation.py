"""Blind, pre-registered, predictive validation (Paper III, workstream 3).

Paper II validated against known ground truth, non-blind, with the state-count
threshold chosen after inspection. This is the single biggest credibility
upgrade and it is cheap: fix the classifier and the behavioural predictions
BEFORE fitting, build reference profiles on one set of seeds, then classify
held-out agents from DISJOINT seeds by their recovered agency profile alone --
never using the held-out label.

*** THIS CLASSIFIER IS CIRCULAR. Kept as the documented negative result. ***

It discriminates the roster by nearest reference centroid over
(rationality, cue_visit, arm_first). It reports accuracy 1.00 -- and that number
is worthless as evidence for the paper's thesis, for three reasons found by
adversarial audit:

  1. cue_visit / arm_first are computed straight off `actions[1]` in the rollouts.
     They ARE the agents' defining behaviour (the info-seeker is the one built to
     visit the cue), so this recovers the agents by the signature we hand-built
     into them. An earlier docstring here claimed they were "marginals of the
     RECOVERED model" -- that was false; nothing in this module touches an MPS.
  2. A trivial baseline -- the raw first-action histogram P(a1), with no criteria,
     no EFE, no inverse-C and no tensor network -- also scores 1.00 (12/12). So
     none of the machinery contributes to the number.
  3. "Held-out" is weak: the same three specs re-run under different RNG seeds.
     The agents are near-deterministic, so held-out points land on the reference
     centroids and 1.00 is guaranteed by construction, not earned.

Paper II earned its tensor network because its claims REQUIRED the latent model
(hidden-state partition from bond rays, A/B, subsystem MI, empowerment). The
error here was choosing a phenotype that raw behaviour supplies for free.

See structure_phenotype.py for the non-circular replacement (phenotype by the
latent structure recoverable from each agent's MPS) and structure_vs_visitrate.py
for the check that it is not the visit rate restated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import agency_criteria, agents


@dataclass(frozen=True)
class PreRegistration:
    """Frozen BEFORE fitting -- committed to git, then never edited. Immutability
    is the whole point: nothing here may be chosen after seeing the held-out
    results."""

    features: tuple = ("rationality", "cue_visit", "arm_first")
    rule: str = "nearest reference centroid, Euclidean distance"
    n_episodes: int = 2000
    reference_seeds: tuple = (0, 1, 2)          # build centroids on these
    holdout_seeds: tuple = (100, 101, 102, 103) # classify these (disjoint)
    behavioural_predictions: tuple = (
        "info-seeker visits the cue first (cue_first > 0.8)",
        "reward-gambler goes straight to an arm (arm_first > 0.8)",
    )


def _profile_vector(spec, seed, n_episodes) -> np.ndarray:
    """(rationality, cue_visit, arm_first) for one agent run. cue_visit and
    arm_first are RAW behavioural frequencies, not recovered-model quantities --
    which is why this classifier is circular (see module docstring)."""
    rollouts, _ = agents.rollout(spec, n_episodes, seed, verify=False)
    prof = agency_criteria.profile_agent(spec.name, rollouts)
    vis = agency_criteria.visitation_signature(rollouts)
    return np.array([prof.rationality, vis["cue_visit"], vis["arm_first"]])


def fit_references(prereg: PreRegistration, roster=None):
    """Reference centroid per agent, averaged over the reference seeds."""
    roster = roster or agents.ROSTER
    centroids = {}
    for spec in roster:
        pts = [_profile_vector(spec, s, prereg.n_episodes) for s in prereg.reference_seeds]
        centroids[spec.name] = np.mean(pts, axis=0)
    return centroids


def classify(vector: np.ndarray, centroids: dict) -> str:
    """Nearest-centroid label for a profile vector (the frozen decision rule)."""
    names = list(centroids)
    d = [np.linalg.norm(vector - centroids[n]) for n in names]
    return names[int(np.argmin(d))]


def run_blind_discrimination(prereg: PreRegistration | None = None, roster=None):
    """Fit references on the reference seeds, then blind-classify each agent on
    the disjoint held-out seeds. Returns (accuracy, confusion, centroids)."""
    prereg = prereg or PreRegistration()
    roster = roster or agents.ROSTER
    centroids = fit_references(prereg, roster)

    names = [s.name for s in roster]
    confusion = {t: {p: 0 for p in names} for t in names}
    correct = total = 0
    for spec in roster:
        for s in prereg.holdout_seeds:
            v = _profile_vector(spec, s, prereg.n_episodes)
            pred = classify(v, centroids)
            confusion[spec.name][pred] += 1
            correct += int(pred == spec.name)
            total += 1
    return correct / total, confusion, centroids


if __name__ == "__main__":
    prereg = PreRegistration()
    acc, confusion, centroids = run_blind_discrimination(prereg)
    print("Reference centroids (rationality, cue_visit, arm_first):")
    for name, c in centroids.items():
        print(f"  {name:16s} ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")
    print(f"\nBlind held-out classification (seeds {prereg.holdout_seeds}):")
    header = "true \\ pred".ljust(16) + "".join(f"{n[:12]:>14s}" for n in confusion)
    print("  " + header)
    for t, row in confusion.items():
        print("  " + t.ljust(16) + "".join(f"{row[p]:>14d}" for p in row))
    print(f"\nblind accuracy = {acc:.2f}")
