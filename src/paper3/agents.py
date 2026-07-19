"""Per-agent rollout generators (Paper III, workstream 1).

Paper II trained on exhaustive uniform-action rollouts, so it recovered the
ENVIRONMENT's generative model. To phenotype an AGENT we instead train each MPS
on that agent's OWN behaviour. This module produces those rollouts for the three
characters in the killer experiment:

  * info-seeker   — high preference precision; visits the cue before an arm.
  * reward-gambler— low precision / high risk; goes straight to an arm.
  * habitual/random — flat policy over actions.

Each agent is a pymdp active-inference agent differing only in its preference
vector C and policy precision (gamma). State-space coverage under a real policy
is no longer guaranteed (an info-seeker rarely enters an arm blind); that
partiality is realistic and is itself part of the recovered phenotype.

Reuses the pymdp TMaze wiring already present in src/full_tmaze_train.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentSpec:
    """A named active-inference agent, defined by its preferences and precision.

    name:      label carried through to the recovered phenotype.
    C_reward:  preference over the reward modality (none, cheese, shock).
    gamma:     policy precision. High -> sharply goal-directed; low -> diffuse.
    """

    name: str
    C_reward: tuple[float, float, float]
    gamma: float


INFO_SEEKER = AgentSpec("info-seeker", C_reward=(0.0, 4.0, -4.0), gamma=16.0)
REWARD_GAMBLER = AgentSpec("reward-gambler", C_reward=(0.0, 2.0, -0.5), gamma=1.0)
HABITUAL = AgentSpec("habitual", C_reward=(0.0, 0.0, 0.0), gamma=0.0)

ROSTER = [INFO_SEEKER, REWARD_GAMBLER, HABITUAL]


def build_agent(spec: AgentSpec):
    """Instantiate a pymdp agent from an AgentSpec.

    TODO(paper3): construct the pymdp Agent with A/B/D from the TMaze env and
    C/gamma from the spec (see the pymdp TMazeEnv usage patterns imported in
    src/full_tmaze_train.py).
    """
    raise NotImplementedError("build pymdp Agent from spec (A/B/D/C/gamma)")


def rollout(spec: AgentSpec, n_episodes: int, seed: int, persistent_latent: bool = True):
    """Generate n_episodes of this agent acting in the (persistent-latent) T-maze.

    Returns rollouts in the same (actions, observations) shape as
    persistent_tmaze.enumerate_persistent so they feed the identical MPS trainer.

    TODO(paper3): drive build_agent(spec) through the env for n_episodes; when
    persistent_latent, sample context once per episode (see persistent_tmaze).
    """
    raise NotImplementedError("run the agent through the env and collect rollouts")
