"""A data-only identifiability certificate for a recovered model.

The practitioner's actual problem: you fit the Paper II pipeline to some agent's
behaviour, you get a recovered generative model, and you have NO GROUND TRUTH to
check it against. Recovery error is unmeasurable in the wild. So which parts of
the model do you trust?

Worse, the obvious proxy actively misleads: training loss IMPROVES as recovery
DEGRADES (3.689 -> 1.661 while error 0.46 -> 3.8), because a concentrated policy
is easy to model. A practitioner watching the loss curve feels reassured while
being wrong.

This module computes a certificate from the DATA AND MODEL ALONE -- no ground
truth -- flagging, per conditional, whether it is identified:

    support(state, action)   how many times the agent exercised it
    mass(state, action)      probability the recovered model puts there
    verdict                  IDENTIFIED / WEAK / UNIDENTIFIED

and validates it: the certificate should predict the (normally unmeasurable)
recovery error. We can check that here because this environment HAS ground truth,
which is the point of validating a diagnostic in a setting where you can.

Grounding: this is a positivity/overlap check (causal inference), equivalently a
coverage/concentrability check (offline RL). See RELATED_WORK.md. The contribution
is not the principle but the per-conditional, data-only instrument and its
validation.

Run:  python -m src.paper3.certificate
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

import numpy as np

from . import agents, persistent_tmaze as pt, phenotype as ph
from . import recovery_fidelity as rf
from .confabulation_check import recovered_rows
from .gamma_sweep import _spec_at
from .structure_phenotype import HISTORIES

CACHE = pathlib.Path(__file__).with_name("_certificate.json")

#: exercise counts below this are treated as unreliable rather than absent
WEAK_SUPPORT = 10


def certify(tensors, rollouts):
    """Per-conditional identifiability certificate. Uses ONLY the data and the
    fitted model -- never ground truth -- so it is computable in the wild."""
    counts = Counter()
    for a, o in rollouts:
        counts[(int(a[1]), tuple(int(x) for x in o[1]), int(a[2]))] += 1

    out = []
    for name, a1, otup in HISTORIES:
        rec, mass = recovered_rows(tensors, a1, otup)
        if rec is None:
            continue
        total = mass.sum()
        for a3 in range(4):
            n = counts[(a1, otup, a3)]
            rel = float(mass[a3] / total) if total > 0 else 0.0
            if n == 0:
                verdict = "UNIDENTIFIED"
            elif n < WEAK_SUPPORT:
                verdict = "WEAK"
            else:
                verdict = "IDENTIFIED"
            out.append({"state": name, "a3": a3, "support": int(n),
                        "rel_mass": rel, "verdict": verdict})
    return out


def validate(tensors, rollouts):
    """Check the certificate against the recovery error it is meant to predict.
    Only possible because this environment has analytic ground truth."""
    cert = certify(tensors, rollouts)
    by_key = {(c["state"], c["a3"]): c for c in cert}
    for name, a1, otup in HISTORIES:
        rec, _ = recovered_rows(tensors, a1, otup)
        if rec is None:
            continue
        true = rf.true_signature(a1, otup).reshape(4, 12)
        for a3 in range(4):
            k = (name, a3)
            if k in by_key:
                by_key[k]["error"] = float(np.abs(rec[a3] - true[a3]).sum())
    return [c for c in cert if "error" in c]


def summarise(rows, label):
    print(f"\n=== {label} ===")
    print(f"{'verdict':>14s}{'n':>5s}{'mean support':>14s}{'mean error':>12s}")
    order = ["IDENTIFIED", "WEAK", "UNIDENTIFIED"]
    stats = {}
    for v in order:
        b = [r for r in rows if r["verdict"] == v]
        if not b:
            continue
        me = float(np.mean([r["error"] for r in b]))
        stats[v] = me
        print(f"{v:>14s}{len(b):5d}{np.mean([r['support'] for r in b]):14.1f}{me:12.3f}")
    return stats


if __name__ == "__main__":
    print("Certificate computed from data + model only (no ground truth), then\n"
          "validated against the recovery error it claims to predict.")

    all_rows = []
    for g in (0.0, 4.0, 16.0):
        spec = _spec_at(g)
        rollouts, _ = agents.rollout(spec, 1200, 0, verify=False)
        pool, _ = pt.rollouts_to_memory_pool(rollouts)
        mps, _joint = ph.train_agent_mps(pool, epochs=25, seed=0)
        tensors = [m.detach().cpu().numpy() for m in mps.matrices]
        rows = validate(tensors, rollouts)
        summarise(rows, f"gamma={g:g}")
        for r in rows:
            r["gamma"] = g
        all_rows.extend(rows)

    CACHE.write_text(json.dumps(all_rows, indent=2))

    print("\n=== pooled: does the certificate predict recovery failure? ===")
    stats = summarise(all_rows, "all agents pooled")

    ident = [r["error"] for r in all_rows if r["verdict"] == "IDENTIFIED"]
    unid = [r["error"] for r in all_rows if r["verdict"] == "UNIDENTIFIED"]
    if ident and unid:
        sep = float(np.mean(unid) - np.mean(ident))
        # AUC: probability a random UNIDENTIFIED row has higher error than a
        # random IDENTIFIED one. 1.0 = perfect separation, 0.5 = useless.
        wins = sum(1 for u in unid for i in ident if u > i)
        ties = sum(1 for u in unid for i in ident if u == i)
        auc = (wins + 0.5 * ties) / (len(unid) * len(ident))
        print(f"\nmean error  IDENTIFIED {np.mean(ident):.3f}  vs  "
              f"UNIDENTIFIED {np.mean(unid):.3f}   (gap {sep:+.3f})")
        print(f"AUC = {auc:.3f}  (1.0 = the certificate perfectly ranks bad"
              " conditionals; 0.5 = useless)")
        if auc > 0.8:
            print("\nThe certificate WORKS: flags computed with no ground truth rank the\n"
                  "untrustworthy conditionals. This is the practical deliverable -- it\n"
                  "tells a user which parts of a recovered model to believe, in a setting\n"
                  "where recovery error itself cannot be measured.")
        else:
            print("\nThe certificate does NOT reliably rank bad conditionals. Report that;\n"
                  "the diagnostic is not usable as-is.")
