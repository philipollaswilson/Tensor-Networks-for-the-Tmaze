"""Per-agent rollout generators (Paper III, workstream 1).

Paper II trained on exhaustive uniform-action rollouts, so it recovered the
ENVIRONMENT's generative model. To phenotype an AGENT we instead train each MPS
on that agent's OWN behaviour. This module produces those rollouts for the three
characters in the killer experiment:

  * info-seeker    -- plans ahead; visits the cue, then the correct arm.
  * reward-gambler -- myopic; goes straight to an arm for immediate reward.
  * habitual       -- flat policy over actions (zero precision).

Architecture. The installed pymdp is the JAX build: its Agent is a functional
JAX object and its TMaze has a different 5-location state space. Rather than
drive that heavy API in a foreign environment, we implement a small, transparent
active-inference agent (expected-free-energy minimisation) directly on OUR
generative model (generative_model.py), which matches persistent_tmaze.py
exactly. This keeps everything in one verified environment and gives us the EFE
machinery workstream 2 (rationality / EFE-regret) needs anyway.

The agent is a genuine active-inference agent: it selects policies by minimising
expected free energy G = -(epistemic value) - (pragmatic value), i.e. it trades
off resolving the hidden context (salience) against reaching preferred rewards
(utility), with policy precision gamma. Agents differ ONLY in the preference
vector C, precision gamma, and planning horizon -- the recognition model (A, B,
D) is shared and correctly specified, so behavioural differences are attributable
to character, not to a different or wrong world model.

  Why horizon separates info-seeker from gambler: cue-then-correct-arm is
  genuinely reward-optimal, so ANY 2-step rational agent finds it. The gambler
  goes straight to an arm because it is myopic (horizon 1) -- an interpretable
  trait (impulsivity), not a rigged preference.

VERIFICATION TARGET: the info-seeker must visit the cue before an arm, and the
reward-gambler must go straight to an arm. rollout() asserts this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import generative_model as gm

_EPS = 1e-12


@dataclass
class AgentSpec:
    """A named active-inference agent.

    name:      label carried through to the recovered phenotype.
    C_reward:  log-preference over the reward modality (none, cheese, shock).
    gamma:     policy precision. High -> sharply goal-directed; 0 -> uniform.
    horizon:   planning depth in steps (2 = plans cue-then-arm; 1 = myopic).
    """

    name: str
    C_reward: tuple[float, float, float]
    gamma: float
    horizon: int


INFO_SEEKER = AgentSpec("info-seeker", C_reward=(0.0, 4.0, -4.0), gamma=16.0, horizon=2)
REWARD_GAMBLER = AgentSpec("reward-gambler", C_reward=(0.0, 2.0, -0.5), gamma=8.0, horizon=1)
HABITUAL = AgentSpec("habitual", C_reward=(0.0, 0.0, 0.0), gamma=0.0, horizon=1)

ROSTER = [INFO_SEEKER, REWARD_GAMBLER, HABITUAL]


# --- small math helpers ----------------------------------------------------

def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1.0); q = np.clip(q, _EPS, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


# --- the agent -------------------------------------------------------------

class ActiveInferenceAgent:
    """Expected-free-energy agent on the persistent-latent T-maze model.

    Location is directly observed (deterministic A_pos), so the only belief the
    agent maintains is the posterior over the hidden context, q(K). Planning
    enumerates action sequences up to `horizon`, scoring each by expected free
    energy with one-step-ahead belief branching over predicted observations.
    """

    def __init__(self, spec: AgentSpec, reward_fidelity: float = 0.85):
        self.spec = spec
        self.A = gm.build_A(reward_fidelity)          # [A_pos, A_rew, A_cue]
        self.C = gm.build_C(spec.C_reward)            # [C_pos, C_rew, C_cue]
        self.qK = gm.build_D()[1].copy()              # context prior [0.5, 0.5]

    def reset(self):
        self.qK = gm.build_D()[1].copy()

    def update_context(self, L: int, o_rew: int, o_cue: int):
        """Bayesian posterior over the hidden context given an observation."""
        A_rew, A_cue = self.A[1], self.A[2]
        post = np.array([self.qK[K] * A_rew[o_rew, L, K] * A_cue[o_cue, L, K]
                         for K in range(gm.N_CTX)])
        s = post.sum()
        if s > _EPS:
            self.qK = post / s

    def _efe(self, qK: np.ndarray, L: int, actions: tuple[int, ...]) -> float:
        """Expected free energy of an action sequence from (qK, L).

        G = sum over steps of E_o[ -info_gain(K) - pragmatic(o) + future ].
        Lower is better: epistemic (resolving K) and pragmatic (preferred reward)
        value both reduce G. Branches over predicted observations so the epistemic
        term is a genuine expected information gain.
        """
        if not actions:
            return 0.0
        a, rest = actions[0], actions[1:]
        L2 = L if L in gm.ARMS else a                 # absorbing arms
        A_rew, A_cue, C_rew = self.A[1], self.A[2], self.C[1]
        G = 0.0
        for o_rew in range(3):
            for o_cue in range(gm.N_CTX):
                po = float(np.sum([qK[K] * A_rew[o_rew, L2, K] * A_cue[o_cue, L2, K]
                                   for K in range(gm.N_CTX)]))
                if po < _EPS:
                    continue
                post = np.array([qK[K] * A_rew[o_rew, L2, K] * A_cue[o_cue, L2, K]
                                 for K in range(gm.N_CTX)])
                post /= post.sum()
                info_gain = _kl(post, qK)             # epistemic value (salience)
                pragmatic = C_rew[o_rew]              # utility (pos/cue neutral)
                future = self._efe(post, L2, rest)
                G += po * (-info_gain - pragmatic + future)
        return G

    def act(self, L: int, steps_left: int, rng: np.random.Generator) -> int:
        """Choose an action from the current location by EFE-softmax policy
        selection over a receding horizon."""
        h = min(self.spec.horizon, steps_left)
        if h <= 0:
            return int(rng.integers(gm.N_ACT))
        policies = _enumerate_policies(h)
        G = np.array([self._efe(self.qK, L, pi) for pi in policies])
        qpi = _softmax(-self.spec.gamma * G)
        # marginal over the first action, then sample
        first = np.zeros(gm.N_ACT)
        for pi, w in zip(policies, qpi):
            first[pi[0]] += w
        first /= first.sum()
        return int(rng.choice(gm.N_ACT, p=first))


def _enumerate_policies(horizon: int):
    from itertools import product
    return list(product(range(gm.N_ACT), repeat=horizon))


# --- rollouts --------------------------------------------------------------

def _sample_observation(L: int, K: int, A, rng: np.random.Generator):
    """Sample (position, reward, ctx_read) from the true model at state (L, K)."""
    A_pos, A_rew, A_cue = A
    pos = int(rng.choice(gm.N_LOC, p=A_pos[:, L]))
    rew = int(rng.choice(3, p=A_rew[:, L, K]))
    cue = int(rng.choice(gm.N_CTX, p=A_cue[:, L, K]))
    return pos, rew, cue


def rollout(spec: AgentSpec, n_episodes: int, seed: int, reward_fidelity: float = 0.85,
            context_prior: float = 0.5, verify: bool = True):
    """Generate n_episodes of this agent acting in the persistent-latent T-maze.

    Returns a list of ``(actions, observations)`` where actions is a length-3 int
    array (leading null a1) and observations is a 3x3 int array of
    (position, reward, ctx_read) rows -- the exact shape enumerate_persistent /
    to_memory_pool consume, so rollouts feed the identical MPS trainer.

    Episode structure mirrors persistent_tmaze: a null start observation at
    center, then two decision steps. Context K is drawn ONCE per episode (hidden).
    """
    rng = np.random.default_rng(seed)
    agent = ActiveInferenceAgent(spec, reward_fidelity)
    A = agent.A
    episodes = []
    cue_first = 0  # visited cue before entering any arm
    arm_first = 0  # entered an arm at the first decision step

    for _ in range(n_episodes):
        K = int(rng.random() < context_prior)         # persistent hidden context
        agent.reset()
        L = gm.CENTER
        actions = [0]                                  # leading null action
        obs_rows = [(gm.CENTER, 0, 0)]                 # null start at center
        visited_cue = False
        first_arm_step = None

        for step in range(2):                          # two decision steps
            steps_left = 2 - step
            a = agent.act(L, steps_left, rng)
            L = L if L in gm.ARMS else a               # absorbing arms
            pos, rew, cue = _sample_observation(L, K, A, rng)
            agent.update_context(L, rew, cue)
            actions.append(a)
            obs_rows.append((pos, rew, cue))
            if L == gm.CUE:
                visited_cue = True
            if L in gm.ARMS and first_arm_step is None:
                first_arm_step = step

        episodes.append((np.array(actions, dtype=int), np.array(obs_rows, dtype=int)))
        if visited_cue and (first_arm_step is None or first_arm_step > 0):
            cue_first += 1
        if first_arm_step == 0:
            arm_first += 1

    stats = {"cue_first_frac": cue_first / n_episodes,
             "arm_first_frac": arm_first / n_episodes}
    if verify:
        _verify_behaviour(spec, stats)
    return episodes, stats


def _verify_behaviour(spec: AgentSpec, stats: dict):
    """Fail loudly if an agent does not behave as its character requires."""
    if spec.name == "info-seeker" and stats["cue_first_frac"] < 0.8:
        raise AssertionError(
            f"info-seeker should visit the cue first; got "
            f"cue_first_frac={stats['cue_first_frac']:.2f}")
    if spec.name == "reward-gambler" and stats["arm_first_frac"] < 0.8:
        raise AssertionError(
            f"reward-gambler should go straight to an arm; got "
            f"arm_first_frac={stats['arm_first_frac']:.2f}")


if __name__ == "__main__":
    for spec in ROSTER:
        eps, stats = rollout(spec, n_episodes=2000, seed=0, verify=False)
        print(f"{spec.name:16s} cue_first={stats['cue_first_frac']:.2f}  "
              f"arm_first={stats['arm_first_frac']:.2f}  ({len(eps)} episodes)")
