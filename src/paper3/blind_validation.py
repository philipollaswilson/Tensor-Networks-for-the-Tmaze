"""Blind, pre-registered, predictive validation (Paper III, workstream 3).

Paper II validated against known ground truth, non-blind, with the state-count
threshold chosen after inspection. This is the single biggest credibility
upgrade and it is cheap: fix the classifier and the behavioural predictions
BEFORE fitting, build reference profiles on one set of seeds, then classify
held-out agents from DISJOINT seeds by their recovered agency profile alone --
never using the held-out label.

The killer result: a profile that tells a curious agent from a gambler, without
being told which is which, is genuine phenotyping. We discriminate the roster
(info-seeker / reward-gambler / habitual) by nearest reference centroid in a
recovered-profile space of three stable coordinates:

  rationality  -- agency criterion (EFE-consistency), separates the habitual agent
  cue_visit    -- recovered epistemic-seeking (state-visitation coverage of the
                  recovered policy-weighted model), separates the info-seeker
  arm_first    -- recovered reward-commitment, separates the reward-gambler

We deliberately do NOT use intentionality as a discrimination axis: cue-seeking
is degenerate between reward-preference and pure curiosity (see agency_criteria),
so the inverse-inferred goal depth is an unstable classifier. cue_visit /
arm_first are marginals of the RECOVERED model (reproduced by the fitted MPS to
<0.04 L1), so this is discrimination by recovered phenotype, not by raw label.
All three coordinates are behaviour/recovery-derived, so the blind test needs no
per-agent retraining.
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
    """The recovered-profile coordinate (rationality, cue_visit, arm_first) for
    one agent run. No labels or held-out data used in the computation."""
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
