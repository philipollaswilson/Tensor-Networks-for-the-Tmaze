"""Phenotype an agent by the LATENT STRUCTURE recoverable from its behaviour.

This replaces the earlier, circular discrimination. That version classified
agents on their first-action marginal -- a statistic raw rollouts hand you for
free, so the tensor network was decorative and a histogram baseline tied it at
accuracy 1.00.

Paper II earned its tensor network because its claims REQUIRED the latent model:
a hidden-state partition recovered by clustering bond rays, emission/transition
matrices, subsystem mutual information, empowerment on the learned model. None of
those exist without an MPS. So we phenotype agents the same way, per agent:

    Train one MPS on an agent's own behaviour, then ask what generative structure
    an observer can RECOVER from it -- which histories carry support in the bond,
    and how many predictively-distinct hidden states survive.

This is the honest reading of "state-space coverage is no longer guaranteed"
(PAPER3 workstream 1): an info-seeker's data can resolve the cue-context states;
a gambler that never visits the cue leaves them unrecoverable, so the model an
observer can build from watching it is genuinely smaller. The phenotype is a
property of the recovered latent model, not a behaviour frequency.

Encoding note: the histories differ from Paper II's because our third modality is
the cue-reading (0 off-cue), not a context bit emitted everywhere -- so each arm
outcome has a single observation code instead of two context copies.

Run:  python -m src.paper3.structure_phenotype
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from . import agents, persistent_tmaze as pt, phenotype as ph
from . import generative_model as gm

#: (label, first-real-action, observation triple) for every reachable history
#: after one decision step in the persistent-latent maze.
HISTORIES = [
    ("center",    gm.CENTER, (0, 0, 0)),
    ("R/cheese",  gm.RIGHT,  (1, 1, 0)),
    ("R/shock",   gm.RIGHT,  (1, 2, 0)),
    ("L/cheese",  gm.LEFT,   (2, 1, 0)),
    ("L/shock",   gm.LEFT,   (2, 2, 0)),
    ("cue/ctx0",  gm.CUE,    (3, 0, 0)),
    ("cue/ctx1",  gm.CUE,    (3, 0, 1)),
]

#: Born probability below which a history is treated as unsupported -- an
#: observer watching this agent essentially never sees it, so nothing about the
#: corresponding latent state is recoverable.
SUPPORT_FLOOR = 1e-3


def _structure_recovery():
    """Import Paper II's recovery helpers (predictive signature + predictive-
    equivalence clustering) so Paper III uses the SAME validated criterion."""
    import importlib
    return importlib.import_module("structure_recovery")


def recover_structure(tensors, joint, tau: float = 1.5, support_floor: float = SUPPORT_FLOOR):
    """Recover the latent structure an observer can build from this agent's MPS.

    Returns a dict with the supported histories, their Born weights, and the
    number of predictively-distinct hidden states among them (Paper II's
    predictive-equivalence criterion at threshold tau).
    """
    sr = _structure_recovery()
    T1, T2, T3 = tensors
    # left environment into the second bond: null action, start observation (0,0,0)
    left = T1[0, 0, pt.obs_index(0, 0, 0), :].astype(complex)

    supported, weights, sigs = [], {}, []
    for name, a2, otup in HISTORIES:
        oi = pt.obs_index(*otup)
        w = float(joint[0, pt.obs_index(0, 0, 0), a2, oi].sum())
        weights[name] = w
        b = np.einsum("i,ij->j", left, T2[:, a2, oi, :]).astype(complex)
        nb = np.linalg.norm(b)
        if nb < 1e-9 or w < support_floor:
            continue                      # unrecoverable from this agent's data
        supported.append(name)
        sigs.append(sr.predictive_signature(b / nb, T3))

    if len(sigs) >= 2:
        _labels, _h, n_states = sr.cluster_by_prediction(sigs, tau=tau)
    else:
        n_states = len(sigs)
    return {"supported": supported, "weights": weights,
            "n_supported": len(supported), "n_states": int(n_states)}


def phenotype_agent(spec, n_episodes: int = 3000, epochs: int = 60, seed: int = 0):
    """Train this agent's MPS, then recover its latent structure."""
    rollouts, stats = agents.rollout(spec, n_episodes, seed, verify=False)
    pool, _ = pt.rollouts_to_memory_pool(rollouts)
    mps, joint = ph.train_agent_mps(pool, epochs=epochs, seed=seed)
    tensors = [m.detach().cpu().numpy() for m in mps.matrices]
    rec = recover_structure(tensors, joint)
    rec["stats"] = stats
    rec["cue_visit"] = stats["cue_first_frac"]
    return rec


if __name__ == "__main__":
    print("Recovering latent structure per agent (Paper II criterion, per-agent MPS)\n")
    rows = {}
    for spec in agents.ROSTER:
        rec = phenotype_agent(spec)
        rows[spec.name] = rec
        print(f"{spec.name:16s} states={rec['n_states']}  "
              f"supported={rec['n_supported']}/7  cue_visit={rec['cue_visit']:.2f}")
        print(f"{'':16s} recoverable: {', '.join(rec['supported'])}")
        drop = [n for n in rec['weights'] if n not in rec['supported']]
        if drop:
            print(f"{'':16s} unrecoverable: {', '.join(drop)}")
        print()
