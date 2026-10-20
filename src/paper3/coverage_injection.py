"""Applying the known fix -- and measuring what it costs.

The gamma sweep showed competence destroys identifiability. The fix is not new:
randomise some actions to restore coverage. It is DAgger's move in imitation
learning, randomisation restoring positivity in causal inference, and exploratory
data collection in offline RL (see RELATED_WORK.md).

What is NOT in those literatures, as far as we can find, is the EXCHANGE RATE.
Injecting exploration restores the observer's ability to identify the world
model, but it makes the agent act against its own preferences -- it walks into
shocks it would otherwise avoid. So identifiability is bought with the agent's
performance, and the price is measurable:

    epsilon = P(override the agent's action with a uniform random one)

    as epsilon rises:  action entropy UP, recovery error DOWN  (observer gains)
                       agent reward DOWN                        (agent pays)

That trade is the practical question anyone auditing a deployed agent faces: how
much performance must I cost this system to learn how its world works? Reporting
the curve turns a known fix into a design guideline.

Run:  python -m src.paper3.coverage_injection
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from . import agents, persistent_tmaze as pt, phenotype as ph
from . import recovery_fidelity as rf
from . import generative_model as gm
from .gamma_sweep import _spec_at
from .structure_phenotype import HISTORIES

CACHE = pathlib.Path(__file__).with_name("_coverage_injection.json")

EPSILONS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)


def rollout_with_injection(spec, n_episodes: int, seed: int, epsilon: float):
    """The agent's own policy, except each action is replaced by a uniform random
    one with probability epsilon. epsilon=0 is pure agent, epsilon=1 is uniform.

    This is deliberately an INTERVENTION on data collection, not a change to the
    agent's objective: we are not making the agent legible, we are overriding it.
    """
    rng = np.random.default_rng(seed)
    agent = agents.ActiveInferenceAgent(spec)
    A = agent.A
    episodes, cheese, shock = [], 0, 0
    for _ in range(n_episodes):
        K = int(rng.random() < 0.5)
        agent.reset()
        L = gm.CENTER
        actions, obs_rows = [0], [(gm.CENTER, 0, 0)]
        for step in range(2):
            a = agent.act(L, 2 - step, rng)
            if rng.random() < epsilon:
                a = int(rng.integers(gm.N_ACT))       # override
            L = L if L in gm.ARMS else a
            pos, rew, cue = agents._sample_observation(L, K, A, rng)
            agent.update_context(L, rew, cue)
            actions.append(a)
            obs_rows.append((pos, rew, cue))
        if obs_rows[-1][1] == 1:
            cheese += 1
        elif obs_rows[-1][1] == 2:
            shock += 1
        episodes.append((np.array(actions, int), np.array(obs_rows, int)))
    return episodes, {"cheese_rate": cheese / n_episodes, "shock_rate": shock / n_episodes}


def action_entropy_weighted(rollouts):
    """Visitation-weighted H(a3|state) -- the identifiability gate."""
    from collections import Counter
    counts = Counter()
    for a, o in rollouts:
        counts[(int(a[1]), tuple(int(x) for x in o[1]), int(a[2]))] += 1
    num = den = 0.0
    for name, a1, otup in HISTORIES:
        c = np.array([counts[(a1, otup, a3)] for a3 in range(4)], float)
        if c.sum() == 0:
            continue
        p = c / c.sum()
        H = float(max(0.0, -(p[p > 0] * np.log2(p[p > 0])).sum()))
        num += H * c.sum(); den += c.sum()
    return num / den if den else float("nan")


def measure(epsilon: float, spec=None, n_episodes: int = 1200, epochs: int = 25, seed: int = 0):
    spec = spec or _spec_at(16.0)      # the most degenerate agent
    rollouts, perf = rollout_with_injection(spec, n_episodes, seed, epsilon)
    H = action_entropy_weighted(rollouts)
    pool, _ = pt.rollouts_to_memory_pool(rollouts)
    mps, joint = ph.train_agent_mps(pool, epochs=epochs, seed=seed)
    tensors = [m.detach().cpu().numpy() for m in mps.matrices]
    fmap = rf.fidelity_map(tensors, joint)
    errs = [v["error"] for v in fmap.values() if np.isfinite(v["error"])]
    return {"epsilon": epsilon, "H": H, "error": float(np.mean(errs)), **perf}


def run(epsilons=EPSILONS, **kw):
    rows = []
    for e in epsilons:
        r = measure(e, **kw)
        rows.append(r)
        print(f"  eps={e:4.2f}  H={r['H']:.2f}  err={r['error']:.3f}"
              f"  cheese={r['cheese_rate']:.2f}  shock={r['shock_rate']:.2f}", flush=True)
    CACHE.write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    print("Injecting coverage into a deterministic agent (gamma=16).\n"
          "Observer gains identifiability; the agent pays in reward.\n")
    rows = run()

    print(f"\n{'eps':>6s}{'H(a3)':>8s}{'recovery err':>14s}{'cheese':>9s}{'shock':>8s}")
    for r in rows:
        print(f"{r['epsilon']:6.2f}{r['H']:8.2f}{r['error']:14.3f}"
              f"{r['cheese_rate']:9.2f}{r['shock_rate']:8.2f}")

    e0, e1 = rows[0], rows[-1]
    d_err = e0["error"] - e1["error"]
    d_cheese = e0["cheese_rate"] - e1["cheese_rate"]
    print(f"\nfull randomisation buys {d_err:+.3f} recovery error"
          f" and costs {d_cheese:+.3f} cheese rate")

    # the useful number: recovery gained per unit of performance sacrificed
    print(f"\n{'eps':>6s}{'err reduction':>15s}{'cheese lost':>13s}{'gain/cost':>11s}")
    for r in rows[1:]:
        de = e0["error"] - r["error"]
        dc = e0["cheese_rate"] - r["cheese_rate"]
        ratio = de / dc if dc > 1e-9 else float("inf")
        print(f"{r['epsilon']:6.2f}{de:15.3f}{dc:13.3f}{ratio:11.2f}")
    print("\nThe knee of that ratio is the design guideline: the cheapest epsilon\n"
          "that buys most of the identifiability. If small epsilon recovers most of\n"
          "the error, auditing a deployed agent is cheap; if it needs eps~1, you\n"
          "cannot audit without effectively replacing the agent with a random one.")
