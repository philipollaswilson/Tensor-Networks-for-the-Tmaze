"""The killer experiment: agent discrimination by recovered phenotype.

Ties the spine together (workstreams 1-4):

  1. generate rollouts from each agent in ROSTER          (agents.py)
     on the persistent-latent T-maze                      (persistent_tmaze.py)
  2. fit one MPS per agent on that agent's OWN rollouts    (Paper II trainer)
  3. recover each agent's structure                        (structure_recovery.py)
     and full agency profile                               (agency_criteria.py)
  4. show the profiles SEPARATE by character, then
     BLIND-classify a held-out agent from its phenotype    (blind_validation.py)

A profile that tells a curious agent from a gambler, without being told which is
which, is the genuine phenotyping result Paper II could only gesture at.

Run from the repo root:  python src/paper3/phenotype.py
"""
from __future__ import annotations

from . import agency_criteria, agents, blind_validation, persistent_tmaze


def train_agent_mps(agent_rollouts, spec):
    """Fit one MPS on a single agent's rollouts.

    TODO(paper3): reuse the Han-scheduler recipe from src/full_tmaze_train.py,
    swapping the exhaustive uniform-action pool for this agent's rollout pool.
    """
    raise NotImplementedError("train per-agent MPS (full_tmaze_train.py recipe)")


def run_experiment(n_episodes: int = 5000, seed: int = 0):
    """End-to-end discrimination experiment.

    TODO(paper3): for each spec in agents.ROSTER -> rollout -> train_agent_mps ->
    structure_recovery -> agency_criteria.profile_agent; collect profiles, show
    separation, then blind-classify a held-out agent.
    """
    raise NotImplementedError("wire steps 1-4 into the discrimination experiment")


if __name__ == "__main__":
    run_experiment()
