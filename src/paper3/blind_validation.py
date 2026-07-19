"""Blind, pre-registered, predictive validation (Paper III, workstream 3).

Paper II validated against known ground truth, non-blind, with the state-count
threshold chosen after inspection. This is the single biggest credibility
upgrade and it is cheap: fix the hypotheses and thresholds BEFORE fitting, hold
out data the exact model cannot memorise, and make one falsifiable behavioural
prediction confirmed out-of-sample.

  pre_registration — the frozen thresholds and structure hypotheses, written
      down before any fitting touches the held-out set.
  held-out test    — split whole sub-branches (not random rows) so the recovered
      structure must PREDICT unseen conditionals rather than memorise them.
  behavioural prediction — e.g. "the info-seeker will visit the cue first";
      confirmed on trajectories never used in fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreRegistration:
    """Thresholds and hypotheses frozen before fitting. Instantiate once, commit
    it, then never edit — that immutability is the whole point."""

    state_count_threshold: float          # e.g. predictive-future L1 tau
    structure_hypotheses: tuple           # dependency edges predicted a priori
    behavioural_predictions: tuple        # falsifiable per-agent claims


def split_holdout(episodes, held_out_branches):
    """Partition episodes into (train, test) by whole sub-branch, so the exact
    model cannot memorise the test conditionals.

    TODO(paper3): remove entire (context, first-action) branches from train and
    reserve them for the predictive test.
    """
    raise NotImplementedError("split by sub-branch, not by random row")


def predictive_score(recovered_model, test_episodes):
    """Score how well the recovered structure predicts held-out conditionals.

    TODO(paper3): compare recovered p(o_t | history) against the held-out
    empirical conditionals; report a single out-of-sample fidelity.
    """
    raise NotImplementedError("evaluate recovered conditionals on held-out data")


def check_prediction(prereg: PreRegistration, agent_name, rollouts) -> bool:
    """Confirm/deny a pre-registered behavioural prediction out-of-sample.

    TODO(paper3): test the frozen claim (e.g. cue-first for the info-seeker)
    against rollouts that were never used in fitting.
    """
    raise NotImplementedError("test a frozen behavioural prediction out-of-sample")
