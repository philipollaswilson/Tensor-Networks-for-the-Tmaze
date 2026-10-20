"""The determinism/identifiability tradeoff curve (Paper III headline figure).

fidelity_analysis established the law on three hand-picked agents: recovery error
is gated by action entropy, not visit frequency. Three points is an anecdote.
This sweeps ONE agent family along its policy precision gamma -- the single knob
that turns a random explorer (gamma=0) into a deterministic expert (gamma=16) --
and measures how much identifiability each increment of competence costs.

Everything else is held fixed: same preferences C, same horizon, same generative
model, same training budget. So the x-axis is competence and the y-axis is how
well an observer can reconstruct the world from watching.

Expected shape (the claim under test): as gamma rises, measured action entropy
H(a3) falls and recovery error rises. If instead error is flat in gamma, the
identifiability law does not actually bind and we say so.

Outputs a table and caches results to _gamma_sweep.json.

Run:  python -m src.paper3.gamma_sweep
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from . import agents, persistent_tmaze as pt, phenotype as ph
from . import recovery_fidelity as rf
from .structure_phenotype import HISTORIES

CACHE = pathlib.Path(__file__).with_name("_gamma_sweep.json")

#: policy precisions spanning random -> deterministic
GAMMAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def _spec_at(gamma: float) -> agents.AgentSpec:
    """The info-seeker family with precision dialled to gamma. Preferences and
    horizon are held at the info-seeker's values so gamma is the only variable."""
    base = agents.INFO_SEEKER
    return agents.AgentSpec(f"gamma={gamma:g}", C_reward=base.C_reward,
                            gamma=gamma, horizon=base.horizon)


def measure(gamma: float, n_episodes: int = 1200, epochs: int = 25, seed: int = 0):
    """Behaviour + recovery quality for one precision setting."""
    spec = _spec_at(gamma)
    rollouts, stats = agents.rollout(spec, n_episodes, seed, verify=False)

    # measured action entropy per state (the identifiability gate)
    ent, weights = {}, {}
    for name, a1, otup in HISTORIES:
        acts = [int(a[2]) for a, o in rollouts
                if int(a[1]) == a1 and tuple(int(x) for x in o[1]) == otup]
        if not acts:
            continue
        c = np.bincount(acts, minlength=4).astype(float)
        p = c / c.sum()
        ent[name] = float(max(0.0, -(p[p > 0] * np.log2(p[p > 0])).sum()))
        weights[name] = len(acts) / len(rollouts)

    pool, _ = pt.rollouts_to_memory_pool(rollouts)
    mps, joint = ph.train_agent_mps(pool, epochs=epochs, seed=seed)
    tensors = [m.detach().cpu().numpy() for m in mps.matrices]
    fmap = rf.fidelity_map(tensors, joint)

    errs = [v["error"] for v in fmap.values() if np.isfinite(v["error"])]
    # visitation-weighted entropy: how varied this agent is where it actually goes
    wsum = sum(weights.values()) or 1.0
    H_w = sum(ent[k] * weights[k] for k in ent) / wsum
    return {
        "gamma": gamma,
        "cue_visit": stats["cue_first_frac"],
        "H_mean": float(np.mean(list(ent.values()))) if ent else float("nan"),
        "H_weighted": float(H_w),
        "error_mean": float(np.mean(errs)) if errs else float("nan"),
        "n_states": len(errs),
        "per_state": {k: v["error"] for k, v in fmap.items()},
        "entropy_per_state": ent,
    }


def run(gammas=GAMMAS, n_episodes: int = 1200, epochs: int = 25, seed: int = 0,
        use_cache: bool = True):
    if use_cache and CACHE.exists():
        return json.loads(CACHE.read_text())
    rows = [measure(g, n_episodes, epochs, seed) for g in gammas]
    CACHE.write_text(json.dumps(rows, indent=2))
    return rows


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 2 else float("nan")


if __name__ == "__main__":
    rows = run(use_cache=False)
    print("Determinism vs identifiability: one agent family, gamma is the only knob.\n")
    print(f"{'gamma':>7s}{'cue_visit':>11s}{'H(a3) mean':>12s}{'H weighted':>12s}"
          f"{'recovery err':>14s}")
    for r in rows:
        print(f"{r['gamma']:7.1f}{r['cue_visit']:11.2f}{r['H_mean']:12.2f}"
              f"{r['H_weighted']:12.2f}{r['error_mean']:14.3f}")

    g = [r["gamma"] for r in rows]
    H = [r["H_weighted"] for r in rows]
    E = [r["error_mean"] for r in rows]
    print(f"\ncorr(gamma, H_weighted)  = {_corr(g, H):+.3f}   (expect negative:"
          " precision removes action variation)")
    print(f"corr(gamma, error)       = {_corr(g, E):+.3f}   (expect positive:"
          " competence costs identifiability)")
    print(f"corr(H_weighted, error)  = {_corr(H, E):+.3f}   (the law itself)")
    print("\nIf corr(gamma, error) is ~0, the tradeoff does not bind in this range"
          " and the\nthree-agent result was an artifact of the particular specs. Report"
          " either way.")
