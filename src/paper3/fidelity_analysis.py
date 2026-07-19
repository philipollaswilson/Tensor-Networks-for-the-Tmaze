"""Is per-state recovery fidelity anything more than the visit rate? (audit)

recovery_fidelity.py measures, per latent state, how badly each agent's MPS
reconstructs that state's predictive signature. The obvious objection is that
this is data volume wearing a hat: a state the agent rarely visits is recovered
badly, so error(state) is just -log(visits) and the tensor network again adds
nothing a visit count could not.

This module tests that directly. Pool every (agent, state) pair, regress the
recovery error on log10 of the state's Born weight, and report R^2:

  * R^2 near 1  -> recovery error IS the visit rate restated. Say so plainly;
                   the fidelity map would then be a prettier histogram.
  * R^2 lower   -> the recovered model carries information visit counts do not,
                   and the residuals say where (a low-data state recovered well
                   means the MPS generalised across the bond; a high-data state
                   recovered badly means the model could not represent it).

Either way we print the residual table so the claim is inspectable rather than
asserted.

Run:  python -m src.paper3.fidelity_analysis
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from . import agents, recovery_fidelity as rf
from .structure_phenotype import HISTORIES

CACHE = pathlib.Path(__file__).with_name("_fidelity_cache.json")


def gather(n_episodes: int = 1200, epochs: int = 25, seed: int = 0, use_cache: bool = True):
    """Per-agent fidelity maps, cached so the audit does not retrain."""
    if use_cache and CACHE.exists():
        return json.loads(CACHE.read_text())
    data = {}
    for spec in agents.ROSTER:
        fm, stats = rf.phenotype_agent(spec, n_episodes, epochs, seed)
        data[spec.name] = {"map": fm, "cue_visit": stats["cue_first_frac"]}
    CACHE.write_text(json.dumps(data, indent=2))
    return data


def analyse(data):
    names = [h[0] for h in HISTORIES]
    xs, ys, tags = [], [], []
    for agent, d in data.items():
        for state in names:
            e = d["map"][state]["error"]
            w = d["map"][state]["weight"]
            if not np.isfinite(e) or w <= 0:
                continue
            xs.append(np.log10(w)); ys.append(e); tags.append((agent, state))
    x, y = np.asarray(xs), np.asarray(ys)
    # least squares fit  error ~ a*log10(weight) + b
    Aa = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(Aa, y, rcond=None)
    pred = Aa @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    resid = y - pred
    return {"r2": r2, "coef": coef.tolist(), "tags": tags,
            "x": x.tolist(), "y": y.tolist(), "resid": resid.tolist()}


if __name__ == "__main__":
    data = gather()
    res = analyse(data)
    print("Regression:  recovery_error ~ a*log10(born weight) + b")
    print(f"  a = {res['coef'][0]:.3f}   b = {res['coef'][1]:.3f}   R^2 = {res['r2']:.3f}\n")

    order = np.argsort(-np.abs(res["resid"]))
    print(f"{'agent':16s}{'state':11s}{'weight':>9s}{'error':>9s}{'resid':>9s}")
    for i in order:
        a, s = res["tags"][i]
        print(f"{a:16s}{s:11s}{10 ** res['x'][i]:9.4f}{res['y'][i]:9.3f}{res['resid'][i]:9.3f}")

    print()
    if res["r2"] > 0.9:
        print(f"VERDICT: R^2={res['r2']:.3f} -- recovery error is essentially the visit\n"
              "rate restated. The fidelity map does NOT justify the tensor network.")
    else:
        print(f"VERDICT: R^2={res['r2']:.3f} -- visit rate leaves substantial variance\n"
              "unexplained. Inspect the residual table above: those are states whose\n"
              "recoverability is not predicted by how often the agent went there.")
