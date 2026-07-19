"""The full three agency criteria on a recovered model (Paper III, workstream 2).

Paper II reported only empowerment (controllability). This is the direct answer
to "empowerment isn't agency": operationalise all three Paper I criteria on the
MPS-recovered generative model, with empowerment kept as the controllability
facet of a fuller profile.

  intentionality — inverse-infer the preference vector C from the recovered
      A, B and the observed behaviour; goal-directedness = whether the recovered
      C renders the observed policy an expected-free-energy (EFE) minimiser.
  rationality    — the EFE-optimality gap (regret) of the behaviour under the
      recovered model: how far the agent is from the EFE-optimal policy.
  explainability — fidelity and description length of the recovered model.

The recovered A, B come from Paper II's src/structure_recovery.py; empowerment
from the existing Blahut-Arimoto code. The new numerics are inverse-C inference
and the EFE-regret computation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgencyProfile:
    """The full phenotype for one agent, read off its recovered model."""

    agent: str
    empowerment: float          # controllability (Paper II carry-over)
    intentionality: float       # goal-directedness score in [0, 1]
    rationality: float          # 1 - normalised EFE regret, in [0, 1]
    explainability: float       # model fidelity / description-length score
    recovered_C: tuple          # inverse-inferred preference vector


def infer_preferences(A, B, behaviour):
    """Inverse-infer the preference vector C that best rationalises `behaviour`.

    TODO(paper3): solve for C such that the observed policy is (near-)EFE-optimal
    under the recovered A, B. Inverse active inference / max-likelihood over C.
    """
    raise NotImplementedError("inverse-infer C from recovered A, B and behaviour")


def efe_regret(A, B, C, behaviour):
    """EFE-optimality gap of `behaviour` under the recovered model and C.

    TODO(paper3): EFE(observed policy) - EFE(optimal policy), normalised.
    """
    raise NotImplementedError("compute EFE regret against the optimal policy")


def profile_agent(agent_name, recovered_model, behaviour) -> AgencyProfile:
    """Assemble the full AgencyProfile for one recovered agent model.

    TODO(paper3): combine empowerment (existing) + intentionality + rationality
    + explainability into one profile.
    """
    raise NotImplementedError("assemble empowerment + 3 criteria into a profile")
