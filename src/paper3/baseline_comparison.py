"""Can the RECOVERED model identify an agent where raw behaviour cannot?

Discrimination was dropped as a headline because of a hard information-theoretic
wall: the MPS is trained on the rollouts, so any feature computed from it is a
deterministic function of that same data. By the data-processing inequality,

    I(agent ; MPS features)  <=  I(agent ; rollouts)

Raw behaviour is a sufficient statistic. No MPS-derived quantity can carry MORE
information about which agent produced the data. Asymptotically, discrimination
can never favour the tensor network -- which is exactly why the original accuracy
1.00 was unfalsifiable as evidence (the trivial baseline was already at 1.00).

BUT the DPI bounds information, not ESTIMATOR QUALITY. With finite samples a
structured model can beat a raw frequency table by trading variance for bias: a
low-bond MPS shares statistical strength across states, where an empirical joint
at N=10 is mostly zeros. So there is one legitimate, falsifiable version of the
discrimination claim:

    At small N, does the MPS-recovered joint identify the agent more reliably
    than the empirical joint estimated from the same N episodes?

Both methods see identical data and use the identical decision rule (nearest
reference agent by L1). The ONLY difference is whether the distribution is
estimated by counting or by fitting a tensor network. That isolates the
regularisation effect and makes the comparison a fair fight.

We report whichever way it falls, including "the histogram wins".

Run:  python -m src.paper3.baseline_comparison
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from . import agents, persistent_tmaze as pt, phenotype as ph

CACHE = pathlib.Path(__file__).with_name("_baseline_cache.json")


def hard_roster(gammas=(3.0, 4.0, 5.0)):
    """A DELIBERATELY HARD roster: one agent family at adjacent precisions.

    The default roster (info-seeker / gambler / habitual) is categorically
    different, so both methods hit 1.00 by N=10 and the comparison saturates --
    a test at ceiling cannot tell two estimators apart. These agents differ only
    subtly, which is where a better estimator could actually show an advantage.
    """
    from .gamma_sweep import _spec_at
    return [_spec_at(g) for g in gammas]


def reference_joints(n_episodes: int = 20000, seed: int = 999, roster=None):
    """Well-characterised reference distribution per agent, from a large sample.
    Shared by both methods, so neither is advantaged by the references."""
    refs = {}
    for spec in (roster or agents.ROSTER):
        rollouts, _ = agents.rollout(spec, n_episodes, seed, verify=False)
        refs[spec.name] = ph.empirical_joint(rollouts)
    return refs


def _classify(joint, refs):
    names = list(refs)
    d = [np.abs(joint - refs[n]).sum() for n in names]
    return names[int(np.argmin(d))]


def trial(spec, n: int, seed: int, refs, epochs: int = 30):
    """One head-to-head at sample size n. Returns (raw_correct, mps_correct)."""
    rollouts, _ = agents.rollout(spec, n, seed, verify=False)

    raw_joint = ph.empirical_joint(rollouts)                 # counting
    pool, _ = pt.rollouts_to_memory_pool(rollouts)
    _mps, mps_joint = ph.train_agent_mps(pool, epochs=epochs, seed=seed)  # fitting

    return (_classify(raw_joint, refs) == spec.name,
            _classify(mps_joint, refs) == spec.name)


def run(n_values=(5, 10, 20, 40, 80), trials: int = 6, epochs: int = 30,
        use_cache: bool = True, roster=None, cache=None):
    cache = cache or CACHE
    roster = roster or agents.ROSTER
    if use_cache and cache.exists():
        return json.loads(cache.read_text())
    refs = reference_joints(roster=roster)
    out = []
    for n in n_values:
        raw_hits = mps_hits = total = 0
        for spec in roster:
            for t in range(trials):
                r_ok, m_ok = trial(spec, n, seed=2000 + 31 * t, refs=refs, epochs=epochs)
                raw_hits += int(r_ok); mps_hits += int(m_ok); total += 1
        out.append({"n": n, "raw": raw_hits / total, "mps": mps_hits / total,
                    "trials": total})
        print(f"  n={n:4d}  raw={raw_hits/total:.2f}  mps={mps_hits/total:.2f}"
              f"  ({total} trials)", flush=True)
    cache.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print("Agent identification from N episodes: empirical joint (counting) vs\n"
          "MPS-recovered joint (fitting). Same data, same decision rule.\n")
    rows = run(use_cache=False)
    print(f"\n{'N':>6s}{'raw hist':>10s}{'MPS':>8s}{'delta':>8s}")
    for r in rows:
        print(f"{r['n']:6d}{r['raw']:10.2f}{r['mps']:8.2f}{r['mps']-r['raw']:+8.2f}")
    wins = sum(1 for r in rows if r["mps"] > r["raw"] + 0.02)
    losses = sum(1 for r in rows if r["raw"] > r["mps"] + 0.02)
    print()
    if wins and wins >= losses:
        print(f"MPS beats the histogram at {wins}/{len(rows)} sample sizes. The recovered\n"
              "model identifies the agent where counting cannot -- the finite-sample\n"
              "version of the discrimination claim survives. Note this is an ESTIMATOR\n"
              "result, not an information result: asymptotically the histogram catches up.")
    else:
        print(f"MPS does NOT beat the histogram (wins {wins}, losses {losses}). The\n"
              "finite-sample discrimination claim fails too; report that plainly and\n"
              "rest the paper on the identifiability law instead.")
