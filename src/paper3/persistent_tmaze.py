"""Persistent-latent T-maze data generator (Paper III, workstream 4).

Answers Philip's point directly. In the Paper II dataset (src/tmaze.py,
make_dataset) the context bits c0, c1, c2 are looped INDEPENDENTLY, so context
is redrawn every step and is not a persistent episode latent: only the bit read
AT the cue governs the next arm, and a blind arm's reward is a true 50/50 coin.
That is why "only the cue is informative" held there.

Here context is drawn ONCE per episode: c0 = c1 = c2 = c. The reward at an arm
is governed by that single latent, so an arm outcome now carries information
about the hidden state. This is the canonical Friston T-maze geometry:
  * the CUE resolves the context risk-free,
  * an ARM resolves it too, but at the risk of shock.
Both channels are informative; the cue is the safe one. Contrast this with the
Paper II claim "only the cue is informative", which was an artifact of the
i.i.d.-per-step context, not of the extraction.

Observation encoding (matches src/tmaze.py):
  position p in {0:center, 1:right, 2:left, 3:cue}
  reward   r in {0:none, 1:cheese, 2:shock}
  context  c in {0, 1}
  composite obs index = 6*p + 2*r + c

Run from the repo root:  python src/paper3/persistent_tmaze.py
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def _default_arm_reward(context: int, arm: int) -> int:
    """Deterministic reward at an arm under a persistent context.

    cheese (1) when context and arm agree in parity, shock (2) otherwise. This
    is the persistent-latent analogue of tmaze.py's post-cue rule
    ``2 - (c1 + p2) % 2``, now applied to EVERY arm entry (blind or post-cue)
    because the context is stable across the episode.
    """
    return 1 if (context + arm) % 2 == 0 else 2


@dataclass
class PersistentTMazeConfig:
    """Configuration for the persistent-latent enumeration.

    arm_reward: (context, arm) -> reward. Swap for a stochastic rule to make the
        arm a noisy (rather than perfect) readout of context; the default is
        deterministic so I(reward; context) at an arm is exactly 1 bit.
    context_prior: P(context = 1). 0.5 keeps the latent maximally uncertain.
    """

    arm_reward: Callable[[int, int], int] = _default_arm_reward
    context_prior: float = 0.5


# --- enumeration -----------------------------------------------------------

def enumerate_persistent(cfg: PersistentTMazeConfig | None = None):
    """Exhaustively enumerate 3-step episodes with a persistent context.

    Returns a list of ``(actions, observations, weight)`` tuples where
    ``actions`` is a length-3 array of ints (the first action is the null a1),
    ``observations`` is a 3x3 int array of (position, reward, context) rows, and
    ``weight`` is the episode probability. Weights sum to 1.

    The control flow mirrors src/tmaze.py make_dataset's exhaustive branch, with
    the single change that c0 = c1 = c2 = c and the arm reward is governed by c.
    """
    cfg = cfg or PersistentTMazeConfig()
    episodes: list[tuple[np.ndarray, np.ndarray, float]] = []

    for c in range(2):
        p_ctx = cfg.context_prior if c == 1 else (1.0 - cfg.context_prior)
        for a1 in range(4):
            p1 = a1
            # reward at step 1
            if p1 in (1, 2):
                r1 = cfg.arm_reward(c, p1)
                r1_opts = [r1]
            else:
                r1_opts = [0]
            for r1 in r1_opts:
                for a2 in range(4):
                    p2 = a2
                    if a1 in (1, 2):  # already committed to an arm, stay
                        p2 = a1
                    if p2 in (1, 2):
                        r2 = cfg.arm_reward(c, p2)
                        r2_opts = [r2]
                    else:
                        r2_opts = [0]
                    for r2 in r2_opts:
                        actions = np.array([0, a1, a2], dtype=int)
                        obs = np.array(
                            [[0, 0, c], [p1, r1, c], [p2, r2, c]], dtype=int
                        )
                        # every remaining branch is deterministic given (c, a1, a2),
                        # so the weight is just the context prior times the uniform
                        # action prior over the two free actions.
                        weight = p_ctx * (1.0 / 4.0) * (1.0 / 4.0)
                        episodes.append((actions, obs, weight))

    total = sum(w for _, _, w in episodes)
    episodes = [(a, o, w / total) for a, o, w in episodes]
    return episodes


def obs_index(p: int, r: int, c: int) -> int:
    """Composite observation index, matching src/tmaze.py."""
    return 6 * p + 2 * r + c


# --- sanity diagnostics ----------------------------------------------------

def info_gain_at_arm(episodes) -> float:
    """I(reward; context | arm) for a blind arm entered at step 1 (a1 in {1,2}).

    Conditioned on WHICH arm, because Right and Left have opposite reward parity
    under the deterministic rule; marginalizing over both arms washes the signal
    out (P(reward) is uniform per context) and hides the info-gain. The canonical
    question Philip raised is: given you entered a specific arm and saw the
    reward, is the context resolved?

    In the i.i.d. dataset this is ~0 bits (Paper II): a blind arm is a coin. Under
    a persistent latent with the deterministic reward rule it is ~1 bit: the arm
    outcome reveals the hidden state. That is the whole point of this environment.
    """
    # per-arm joint over (reward, context); average the per-arm MI weighted by
    # the arm's probability -> conditional MI I(reward; context | arm).
    joints = {1: np.zeros((3, 2)), 2: np.zeros((3, 2))}  # arm -> reward x context
    for actions, obs, w in episodes:
        arm = actions[1]
        if arm in (1, 2):  # a1 is a blind arm
            r, c = obs[1, 1], obs[1, 2]
            joints[arm][r, c] += w
    total = sum(j.sum() for j in joints.values())
    return float(sum(
        (j.sum() / total) * _mutual_information(j) for j in joints.values()
    ))


def _mutual_information(joint: np.ndarray) -> float:
    j = joint / joint.sum()
    px = j.sum(axis=1, keepdims=True)
    py = j.sum(axis=0, keepdims=True)
    mask = j > 0
    return float(np.sum(j[mask] * np.log2(j[mask] / (px @ py)[mask])))


def to_memory_pool(episodes, dtype="torch.float32"):
    """Convert enumerated episodes into an mpstwo MemoryPool for training.

    TODO(paper3): mirror full_tmaze_train.py's MultiOneHotMap wiring so the
    persistent-latent data drops straight into the existing MPS trainer. Left as
    a stub so this module imports without the mpstwo stack present.
    """
    raise NotImplementedError(
        "wire MultiOneHotMap + MemoryPool as in src/full_tmaze_train.py"
    )


if __name__ == "__main__":
    eps = enumerate_persistent()
    print(f"enumerated {len(eps)} episodes, weights sum to {sum(w for *_, w in eps):.6f}")
    ig = info_gain_at_arm(eps)
    print(f"I(reward; context | arm) at a blind arm = {ig:.3f} bits "
          f"(i.i.d. dataset gives ~0.000; persistent latent gives ~1.000)")
