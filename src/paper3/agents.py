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

Architecture (settled against the installed pymdp API):
  * pymdp.envs.TMaze exposes generate_A / generate_B / generate_D and
    reset / step, with cue_validity and reward_probability as the noisy channels.
  * We drive a real pymdp.agent.Agent (genuine active-inference planning) against
    OUR persistent-latent environment (persistent_tmaze.py) -- the one that
    answers Philip and matches the Paper II [4,3,2] pipeline -- so the science
    stays in one environment.
  * Each agent shares the same recognition model (A, B, D); they differ ONLY in
    the preference vector C and policy precision gamma, so any phenotype
    difference is attributable to character, not to a different world model.
  * Rollouts are mapped into our (position, reward, ctx_read) encoding and fed to
    persistent_tmaze.to_memory_pool, reusing the whole training path.

VERIFICATION TARGET for the implementation: the info-seeker must actually visit
the cue before an arm, and the reward-gambler must go straight to an arm -- check
this on sampled rollouts before trusting any downstream phenotype.
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
    """Instantiate a pymdp Agent from an AgentSpec.

    TODO(paper3):
      A, B, D = pymdp.envs.TMaze(cue_validity=..., reward_probability=0.85,
                                 punishment_probability=0.85).generate_{A,B,D}()
      C = preference vector over the reward modality from spec.C_reward (uniform
          on the other modalities).
      return pymdp.agent.Agent(A=A, B=B, C=C, D=D, gamma=spec.gamma,
                               policy_len=2)   # two decision steps
    All specs share A/B/D; only C and gamma vary.
    """
    raise NotImplementedError("build pymdp Agent from spec (shared A/B/D, per-spec C/gamma)")


def rollout(spec: AgentSpec, n_episodes: int, seed: int):
    """Generate n_episodes of this agent acting in the persistent-latent T-maze.

    Returns rollouts in the same (actions, observations) int-array shape as
    persistent_tmaze.enumerate_persistent, so they feed the identical trainer via
    persistent_tmaze.to_memory_pool.

    TODO(paper3): loop per episode --
        env.reset() with context sampled once (persistent);
        for each of the 2 steps: agent.infer_states(obs) ->
        agent.infer_policies() -> a = agent.sample_action() -> obs = env.step(a);
        record (action, obs) mapped into our (position, reward, ctx_read) triple.
      Then assert the VERIFICATION TARGET (info-seeker visits cue, gambler does
      not) before returning.
    """
    raise NotImplementedError("run the pymdp agent through our env and collect rollouts")
