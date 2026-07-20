"""Is the gamma/error trend identifiability, or just uneven under-training?

The gamma sweep fixed epochs=25, not CONVERGENCE. Agents at different precision
produce data of differing diversity, which may fit at different rates -- so part
of the error trend could be that high-gamma models are simply less converged,
not that their conditionals are unidentifiable.

The decisive test is a budget comparison on IDENTICAL data:

    if the error is UNDER-TRAINING, a 4x longer budget collapses it.
    if the error is UNIDENTIFIABILITY, no budget helps -- the conditional was
    never sampled, so there is nothing for gradient descent to fit.

For each gamma we train the same rollout pool at 25 and at 100 epochs and compare
the recovery error, and we record the final training loss as corroboration that
the longer run really did descend further.

A pass looks like: error essentially unchanged at 4x budget, while training loss
does improve (so the extra budget was real, it just could not buy identifiability).
A fail looks like: error drops materially at 100 epochs, meaning the sweep was
measuring convergence speed and must be re-run matched on convergence.

Run:  python -m src.paper3.convergence_check
"""
from __future__ import annotations

import json
import logging
import pathlib

import numpy as np

from . import persistent_tmaze as pt, phenotype as ph
from . import recovery_fidelity as rf
from .gamma_sweep import _spec_at
from . import agents

CACHE = pathlib.Path(__file__).with_name("_convergence_check.json")

GAMMAS = (0.0, 4.0, 16.0)      # low / mid / high competence
BUDGETS = (25, 100)            # 4x budget difference


class _LossCatcher(logging.Handler):
    """Capture the trainer's reported train loss so we can show the longer run
    actually descended further (otherwise 'no change' proves nothing)."""

    def __init__(self):
        super().__init__()
        self.last = None

    def emit(self, record):
        msg = record.getMessage()
        if "Train loss" in msg:
            try:
                self.last = float(msg.split(":")[-1])
            except ValueError:
                pass


def _train_and_score(pool, epochs, seed):
    catcher = _LossCatcher()
    log = logging.getLogger("mpstwo.model.mpstwo_trainer")
    log.addHandler(catcher)
    try:
        mps, joint = ph.train_agent_mps(pool, epochs=epochs, seed=seed)
    finally:
        log.removeHandler(catcher)
    tensors = [m.detach().cpu().numpy() for m in mps.matrices]
    fmap = rf.fidelity_map(tensors, joint)
    errs = [v["error"] for v in fmap.values() if np.isfinite(v["error"])]
    return float(np.mean(errs)), catcher.last, fmap


def run(gammas=GAMMAS, budgets=BUDGETS, n_episodes: int = 800, seed: int = 0):
    rows = []
    for g in gammas:
        # SAME rollouts for both budgets, so only the training budget differs
        rollouts, stats = agents.rollout(_spec_at(g), n_episodes, seed, verify=False)
        pool, _ = pt.rollouts_to_memory_pool(rollouts)
        rec = {"gamma": g, "cue_visit": stats["cue_first_frac"]}
        for e in budgets:
            err, loss, _fmap = _train_and_score(pool, e, seed)
            rec[f"err@{e}"] = err
            rec[f"loss@{e}"] = loss
        rec["err_delta"] = rec[f"err@{budgets[1]}"] - rec[f"err@{budgets[0]}"]
        rows.append(rec)
        print(f"  gamma={g:5.1f}  err@{budgets[0]}={rec[f'err@{budgets[0]}']:.3f}"
              f"  err@{budgets[1]}={rec[f'err@{budgets[1]}']:.3f}"
              f"  delta={rec['err_delta']:+.3f}"
              f"  loss {rec[f'loss@{budgets[0]}']} -> {rec[f'loss@{budgets[1]}']}",
              flush=True)
    CACHE.write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    print("Same data, 25 vs 100 epochs. Under-training would collapse the error;\n"
          "unidentifiability will not budge.\n")
    rows = run()
    lo, hi = BUDGETS
    print(f"\n{'gamma':>7s}{f'err@{lo}':>10s}{f'err@{hi}':>10s}{'delta':>9s}"
          f"{'trend held?':>13s}")
    for r in rows:
        held = "yes" if abs(r["err_delta"]) < 0.25 * max(r[f"err@{lo}"], 1e-9) else "NO"
        print(f"{r['gamma']:7.1f}{r[f'err@{lo}']:10.3f}{r[f'err@{hi}']:10.3f}"
              f"{r['err_delta']:+9.3f}{held:>13s}")

    spread_lo = rows[-1][f"err@{lo}"] - rows[0][f"err@{lo}"]
    spread_hi = rows[-1][f"err@{hi}"] - rows[0][f"err@{hi}"]
    print(f"\nerror spread (gamma {rows[-1]['gamma']:g} minus {rows[0]['gamma']:g}):"
          f"  at {lo} epochs = {spread_lo:+.3f}   at {hi} epochs = {spread_hi:+.3f}")
    if spread_hi > 0.75 * spread_lo:
        print("\nPASS: the competence/identifiability gap SURVIVES a 4x training budget,\n"
              "so the gamma sweep was not measuring convergence speed.")
    else:
        print("\nFAIL: the gap shrinks materially with more training. The sweep was\n"
              "partly measuring convergence, and must be re-run matched on convergence.")
