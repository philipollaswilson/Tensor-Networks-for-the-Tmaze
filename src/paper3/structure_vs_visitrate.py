"""Adversarial check: is the recovered structure more than the visit rate restated?

The obvious objection to structure_phenotype.py is that `n_supported` is just the
first-action histogram thresholded -- an agent that never visits the cue has no
cue histories, which is the visit rate wearing a hat. If `n_states` were also a
deterministic function of the visit rate, the tensor network would again be
decorative and we should say so.

The decisive test holds the visit rate FIXED and varies something only the latent
model can see. Two variants of the SAME agent:

    greedy  -- normal policy at both decision steps
    explore -- identical first-step policy, but the SECOND action is uniform

Their first-action marginals are statistically identical, so any behaviour-level
phenotype (cue_visit, arm_first, P(a1)) cannot tell them apart. But Paper II's
predictive signature is action-conditioned, p(o3 | a3, state): an agent that only
ever takes one third action leaves the other rows of that conditional unvisited,
so less of each state's future is recoverable. If the recovered structure differs
between these two, it carries information the visit rate does not.

We report whichever way it falls, including "no difference".

Run:  python -m src.paper3.structure_vs_visitrate
"""
from __future__ import annotations

import numpy as np

from . import agents, persistent_tmaze as pt, phenotype as ph
from . import generative_model as gm
from . import structure_phenotype as sp


def rollout_variant(spec, n_episodes: int, seed: int, explore_second: bool):
    """Rollouts whose FIRST-step policy is the agent's own, but whose SECOND
    action is either the agent's own (greedy) or uniform-random (explore).

    Mirrors agents.rollout exactly apart from the second-action override, so the
    first-action marginal is unchanged by construction.
    """
    rng = np.random.default_rng(seed)
    agent = agents.ActiveInferenceAgent(spec)
    A = agent.A
    episodes = []
    for _ in range(n_episodes):
        K = int(rng.random() < 0.5)
        agent.reset()
        L = gm.CENTER
        actions = [0]
        obs_rows = [(gm.CENTER, 0, 0)]
        for step in range(2):
            if step == 1 and explore_second:
                a = int(rng.integers(gm.N_ACT))     # uniform second action
            else:
                a = agent.act(L, 2 - step, rng)
            L = L if L in gm.ARMS else a
            pos, rew, cue = agents._sample_observation(L, K, A, rng)
            agent.update_context(L, rew, cue)
            actions.append(a)
            obs_rows.append((pos, rew, cue))
        episodes.append((np.array(actions, int), np.array(obs_rows, int)))
    return episodes


def first_action_marginal(rollouts):
    h = np.zeros(gm.N_ACT)
    for a, _o in rollouts:
        h[int(a[1])] += 1
    return h / h.sum()


def run(spec=None, n_episodes: int = 3000, epochs: int = 60, seed: int = 0):
    spec = spec or agents.INFO_SEEKER
    out = {}
    for variant, explore in (("greedy", False), ("explore", True)):
        rollouts = rollout_variant(spec, n_episodes, seed, explore)
        pool, _ = pt.rollouts_to_memory_pool(rollouts)
        mps, joint = ph.train_agent_mps(pool, epochs=epochs, seed=seed)
        tensors = [m.detach().cpu().numpy() for m in mps.matrices]
        rec = sp.recover_structure(tensors, joint)
        rec["first_action"] = first_action_marginal(rollouts)
        out[variant] = rec
    return out


if __name__ == "__main__":
    print(f"Same agent ({agents.INFO_SEEKER.name}), same first-step policy,\n"
          f"second action greedy vs uniform. Behaviour-level phenotypes cannot\n"
          f"distinguish these; can the recovered latent structure?\n")
    out = run()
    for variant, rec in out.items():
        print(f"{variant:8s} P(a1)={np.round(rec['first_action'], 3)}  "
              f"states={rec['n_states']}  supported={rec['n_supported']}/7")
        print(f"{'':8s} recoverable: {', '.join(rec['supported'])}")
    d_beh = np.abs(out["greedy"]["first_action"] - out["explore"]["first_action"]).sum()
    same_states = out["greedy"]["n_states"] == out["explore"]["n_states"]
    print(f"\nfirst-action L1 difference = {d_beh:.3f}  (≈0 => behaviourally identical)")
    if same_states:
        print("Recovered state count IDENTICAL => on this test the structure adds\n"
              "nothing beyond the visit rate. Report honestly.")
    else:
        print("Recovered state count DIFFERS at matched behaviour => the recovered\n"
              "latent structure carries information the visit rate does not.")
