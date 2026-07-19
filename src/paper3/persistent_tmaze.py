"""Persistent-latent T-maze data generator (Paper III, workstream 4).

Answers Philip's point directly, and does it without cheating.

In the Paper II dataset (src/tmaze.py) the third observation modality IS the
context bit, emitted every step, and it is redrawn i.i.d. each step. So context
was both (a) directly observed and (b) not persistent: a blind arm's reward told
you nothing about it, which is why "only the cue is informative" held. That made
Philip's info-gain worry vacuous rather than answered.

Here context is a genuine PERSISTENT HIDDEN latent:

  * c is drawn ONCE per episode (context_prior) and never changes.
  * c is emitted only INDIRECTLY:
      - at the CUE, the third modality reads c exactly    -> safe, 1-bit channel
      - at an ARM, the reward is a NOISY readout of c      -> risky channel,
        I(reward; c | arm) = 1 - H(fidelity) < 1 bit
      - everywhere else the third modality is 0 (context hidden).
  * so both the cue and an arm are informative about the hidden state; the cue
    is the safe one. That is the canonical Friston T-maze, and it is the honest
    (non-deterministic) version: an arm leaks context but does not resolve it.

Observation encoding is UNCHANGED from Paper II, so the whole Paper II pipeline
(structure_recovery, factorization_test, gauge_fix_states, model_selection)
transfers as-is:
  position p in {0:center, 1:right, 2:left, 3:cue}
  reward   r in {0:none, 1:cheese, 2:shock}   (noisy readout of c at arms)
  ctx-read k in {0, 1}                          (= c only at the cue, else 0)
  composite obs index = 6*p + 2*r + k

The ONLY change from Paper II is the data-generating process, not the shape.

Run from the repo root:  python src/paper3/persistent_tmaze.py
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PersistentTMazeConfig:
    """Configuration for the persistent-latent enumeration.

    reward_fidelity: P(the context-preferred reward | you entered that arm). The
        arm is a binary-symmetric channel from the hidden context to the reward,
        with crossover (1 - fidelity). 1.0 would make the arm resolve context
        perfectly (cheating); < 1.0 leaks but does not resolve it. Default 0.85.
    context_prior:  P(context = 1). 0.5 keeps the latent maximally uncertain.
    """

    reward_fidelity: float = 0.85
    context_prior: float = 0.5


def _preferred_reward(context: int, arm: int) -> int:
    """The reward this arm 'should' give under the context: cheese (1) when
    context and arm agree in parity, shock (2) otherwise. The noisy channel
    flips this with probability (1 - fidelity)."""
    return 1 if (context + arm) % 2 == 0 else 2


def _arm_reward_dist(cfg: PersistentTMazeConfig, context: int, arm: int):
    """Distribution over reward {none, cheese, shock} at an arm, given the hidden
    context. Noisy: the preferred reward with prob fidelity, the other with the
    remainder. Returns a length-3 numpy array indexed by reward."""
    pref = _preferred_reward(context, arm)
    other = 1 if pref == 2 else 2
    dist = np.zeros(3)
    dist[pref] = cfg.reward_fidelity
    dist[other] = 1.0 - cfg.reward_fidelity
    return dist


def _ctx_read(position: int, context: int) -> int:
    """Third observation modality: reads the hidden context ONLY at the cue
    (position 3); 0 everywhere else, so context stays hidden off-cue."""
    return context if position == 3 else 0


# --- enumeration -----------------------------------------------------------

def _enumerate_with_context(cfg: PersistentTMazeConfig):
    """Internal enumeration that also yields the ground-truth hidden context c.

    Yields ``(actions, observations, weight, c)``. The public
    ``enumerate_persistent`` strips c (it is hidden and must never reach the
    model); the diagnostics keep it to measure information against the true
    latent rather than against an inferred one.
    """
    rows: list[tuple[np.ndarray, np.ndarray, float, int]] = []
    for c in range(2):
        p_ctx = cfg.context_prior if c == 1 else (1.0 - cfg.context_prior)
        # step 0: always the null start at center, context hidden
        o0 = (0, 0, 0)
        for a1 in range(4):
            p1 = a1
            r1_dist = _arm_reward_dist(cfg, c, p1) if p1 in (1, 2) else np.array([1.0, 0, 0])
            for r1 in np.nonzero(r1_dist)[0]:
                pr1 = r1_dist[r1]
                o1 = (p1, int(r1), _ctx_read(p1, c))
                for a2 in range(4):
                    p2 = a1 if a1 in (1, 2) else a2  # once in an arm, you stay
                    if p1 in (1, 2) and p2 in (1, 2):
                        # already in the arm: the reward you saw persists
                        r2_dist = np.zeros(3); r2_dist[int(r1)] = 1.0
                    elif p2 in (1, 2):
                        # newly entering an arm (from center or after the cue)
                        r2_dist = _arm_reward_dist(cfg, c, p2)
                    else:
                        r2_dist = np.array([1.0, 0, 0])
                    for r2 in np.nonzero(r2_dist)[0]:
                        pr2 = r2_dist[r2]
                        o2 = (p2, int(r2), _ctx_read(p2, c))
                        actions = np.array([0, a1, a2], dtype=int)
                        obs = np.array([o0, o1, o2], dtype=int)
                        weight = p_ctx * (1.0 / 4.0) * (1.0 / 4.0) * pr1 * pr2
                        rows.append((actions, obs, float(weight), c))

    total = sum(r[2] for r in rows)
    return [(a, o, w / total, c) for a, o, w, c in rows]


def enumerate_persistent(cfg: PersistentTMazeConfig | None = None):
    """Exhaustively enumerate 3-step episodes with a persistent HIDDEN context.

    Returns a list of ``(actions, observations, weight)`` where ``actions`` is a
    length-3 int array (first is the null a1), ``observations`` is a 3x3 int
    array of (position, reward, ctx_read) rows, and ``weight`` is the episode
    probability under a uniform action prior. Weights sum to 1. The hidden
    context c is deliberately NOT returned -- it never reaches the model.

    Mirrors src/tmaze.py's exhaustive branch, changed so that (i) context is one
    persistent hidden latent c, (ii) the third modality reads c only at the cue,
    and (iii) an arm's reward is a noisy readout of c.
    """
    cfg = cfg or PersistentTMazeConfig()
    return [(a, o, w) for a, o, w, _c in _enumerate_with_context(cfg)]


def obs_index(p: int, r: int, k: int) -> int:
    """Composite observation index, matching src/tmaze.py."""
    return 6 * p + 2 * r + k


# --- sanity diagnostics ----------------------------------------------------

def _mutual_information(joint: np.ndarray) -> float:
    j = joint / joint.sum()
    px = j.sum(axis=1, keepdims=True)
    py = j.sum(axis=0, keepdims=True)
    mask = j > 0
    return float(np.sum(j[mask] * np.log2(j[mask] / (px @ py)[mask])))


def info_gain_at_arm(cfg: PersistentTMazeConfig | None = None) -> float:
    """I(reward; context | arm) for a first-step arm entry (a1 in {1,2}).

    Measured against the TRUE hidden context (not one inferred from the reward,
    which would be circular). Conditioned on WHICH arm, since Right/Left have
    opposite reward parity. Under the noisy readout this is 1 - H(fidelity): with
    fidelity 0.85, ~0.390 bits. The arm LEAKS context but does not RESOLVE it --
    the honest, non-cheating number.
    """
    cfg = cfg or PersistentTMazeConfig()
    joints = {1: np.zeros((3, 2)), 2: np.zeros((3, 2))}  # arm -> reward x context
    for actions, obs, w, c in _enumerate_with_context(cfg):
        arm = int(actions[1])
        if arm in (1, 2):
            joints[arm][int(obs[1, 1]), c] += w
    total = sum(j.sum() for j in joints.values())
    return float(sum((j.sum() / total) * _mutual_information(j) for j in joints.values()))


def info_gain_at_cue(cfg: PersistentTMazeConfig | None = None) -> float:
    """I(cue reading; context) at the cue (a1 == 3). Exactly 1 bit -- the cue
    resolves the hidden state, risk-free. Measured against the true context."""
    cfg = cfg or PersistentTMazeConfig()
    joint = np.zeros((2, 2))  # ctx_read x context
    for actions, obs, w, c in _enumerate_with_context(cfg):
        if int(actions[1]) == 3:
            joint[int(obs[1, 2]), c] += w
    return _mutual_information(joint)


# --- training bridge -------------------------------------------------------

def _install_import_shims():
    """Stub the optional gym/pymdp bits mpstwo imports at load time but does not
    need here (same shim as src/full_tmaze_train.py)."""
    import importlib
    import sys
    import types
    for mod, attrs in [('gym_bandits', []), ('gym_bandits.bandit', ['BanditEnv']),
                       ('pymdp', []), ('pymdp.envs', ['TMazeEnv']),
                       ('gym', []), ('gym.spaces', ['MultiDiscrete', 'Discrete', 'Box'])]:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            m = types.ModuleType(mod)
            sys.modules[mod] = m
        for a in attrs:
            if not hasattr(m, a):
                setattr(m, a, type(a, (), {}))


def to_memory_pool(episodes, replicas: int = 10000, dtype="torch.complex128"):
    """Convert enumerated episodes into an mpstwo MemoryPool for MPS training.

    Episode weights are real-valued (noisy rewards are not dyadic), so each
    episode is pushed round(weight * replicas) times. Larger `replicas` reduces
    the rounding error; 10000 reproduces fidelity 0.85 to <1e-4.

    Mirrors src/full_tmaze_train.py main(): obs map [4,3,2], act map [4,1].
    """
    import torch
    _install_import_shims()
    from mpstwo.data.datasets.memory_pool import MemoryPool
    from mpstwo.data.datastructs import TensorDict
    from mpstwo.utils.mapping import MultiOneHotMap

    torch_dtype = eval(dtype)
    obs_map = MultiOneHotMap([4, 3, 2])
    act_map = MultiOneHotMap([4, 1])

    pool = MemoryPool()
    pushed = 0
    for actions, obs, w in episodes:
        count = int(round(w * replicas))
        if count == 0:
            continue
        sample = TensorDict({
            "action": act_map(torch.tensor([[int(a), 0] for a in actions])).type(torch_dtype),
            "observation": obs_map(torch.tensor([list(map(int, row)) for row in obs])).type(torch_dtype),
        })
        for _ in range(count):
            pool.push_no_update(sample)
        pushed += count
    pool._update_table()
    return pool, pushed


def rollouts_to_memory_pool(rollouts, dtype="torch.complex128"):
    """Build an mpstwo MemoryPool from raw agent rollouts (agents.rollout output).

    Each rollout is a ``(actions, observations)`` pair and is an independent
    sample, so each is pushed exactly once (no weights, unlike the exhaustive
    enumeration). Mirrors to_memory_pool's obs [4,3,2] / act [4,1] maps.
    """
    import torch
    _install_import_shims()
    from mpstwo.data.datasets.memory_pool import MemoryPool
    from mpstwo.data.datastructs import TensorDict
    from mpstwo.utils.mapping import MultiOneHotMap

    torch_dtype = eval(dtype)
    obs_map = MultiOneHotMap([4, 3, 2])
    act_map = MultiOneHotMap([4, 1])

    pool = MemoryPool()
    for actions, obs in rollouts:
        sample = TensorDict({
            "action": act_map(torch.tensor([[int(a), 0] for a in actions])).type(torch_dtype),
            "observation": obs_map(torch.tensor([list(map(int, row)) for row in obs])).type(torch_dtype),
        })
        pool.push_no_update(sample)
    pool._update_table()
    return pool, len(pool)


if __name__ == "__main__":
    eps = enumerate_persistent()
    print(f"enumerated {len(eps)} episodes, weights sum to {sum(w for *_, w in eps):.6f}")
    print(f"I(reward; context | arm) at a blind arm = {info_gain_at_arm():.3f} bits "
          f"(noisy: 1 - H(0.85) ~= 0.390; the arm LEAKS but does not RESOLVE context)")
    print(f"I(cue reading; context) at the cue       = {info_gain_at_cue():.3f} bits "
          f"(the cue RESOLVES context, risk-free)")
