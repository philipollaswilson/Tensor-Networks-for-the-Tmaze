"""The killer experiment: agent discrimination by recovered phenotype.

Ties the spine together (workstreams 1-4):

  1. generate rollouts from each agent in ROSTER          (agents.py)
     on the persistent-latent T-maze                      (persistent_tmaze.py)
  2. fit one MPS per agent on that agent's OWN rollouts    (Paper II trainer)
  3. recover each agent's structure                        (structure_recovery.py)
     and full agency profile                               (agency_criteria.py)
  4. show the profiles SEPARATE by character, then
     BLIND-classify a held-out agent from its phenotype    (blind_validation.py)

A profile that tells a curious agent from a gambler, without being told which is
which, is the genuine phenotyping result Paper II could only gesture at.

Run from the repo root:  python -m src.paper3.phenotype        (full run)
                         python -m src.paper3.phenotype smoke  (quick smoke test)
"""
from __future__ import annotations

import numpy as np

from . import agents, persistent_tmaze


def train_agent_mps(pool, epochs: int = 200, bond_dim: int = 4, seed: int = 0):
    """Fit one MPS on a single agent's rollout pool.

    Reuses the converged recipe from src/full_tmaze_train.py (random init, SGD
    lr=5e-3, batch 32, Han scheduler) on the [4,3,2]->24 observation space. The
    only change from Paper II is the training set: this agent's own behaviour
    instead of exhaustive uniform-action rollouts. Returns (mps, joint) where
    joint is the contracted Born distribution p over (a1,o1,a2,o2,a3,o3).
    """
    import torch
    persistent_tmaze._install_import_shims()
    from mpstwo.model.mpstwo import MPSTwo
    from mpstwo.model.mpstwo_trainer import MPSTrainer
    from mpstwo.model.optimizers import SGD
    from mpstwo.model.schedulers import Han

    torch.manual_seed(seed)
    dtype = torch.complex128
    mps = MPSTwo(3, feature_dim_obs=24, feature_dim_act=4, bond_dim=bond_dim,
                 init_mode='random', max_bond=16, cutoff=0.01, dtype=dtype)
    optim = SGD(mps, lr=5e-3)
    trainer = MPSTrainer(mps, pool, optimizer=optim, batch_size=32,
                         scheduler=Han(optim, safe_loss_threshold=5e-3, lr_shrink_rate=0.8),
                         log_dir_suffix='paper3_agent', device='cpu')
    with torch.inference_mode():
        trainer.train(epochs)
    try:
        trainer.close()
    except (NameError, Exception):
        pass

    T1, T2, T3 = [m.detach().cpu().numpy() for m in mps.matrices]
    psi = np.einsum('xaoi,ibpj,jcqy->aobpcq', T1, T2, T3)
    p = np.abs(psi) ** 2
    p /= p.sum()
    return mps, p


def empirical_joint(rollouts):
    """Empirical Born-target joint p(a1,o1,a2,o2,a3,o3) from raw rollouts, the
    distribution the per-agent MPS is fit to. Shape (4,24,4,24,4,24)."""
    q = np.zeros((4, 24, 4, 24, 4, 24))
    oi = persistent_tmaze.obs_index
    for actions, obs in rollouts:
        i0, i1, i2 = (oi(*map(int, obs[t])) for t in range(3))
        q[int(actions[0]), i0, int(actions[1]), i1, int(actions[2]), i2] += 1.0
    return q / q.sum()


def run_experiment(n_episodes: int = 5000, epochs: int = 200, seed: int = 0):
    """Steps 1-3: per-agent rollout -> MPS fit, reporting how well each agent's
    MPS reproduces that agent's own behaviour (L1 to the empirical joint).

    Steps 4-5 (agency_criteria.profile_agent on each recovered model, then blind
    discrimination) are the next build stage; this driver produces the trained
    per-agent models they consume.
    """
    results = {}
    for i, spec in enumerate(agents.ROSTER):
        rollouts, stats = agents.rollout(spec, n_episodes, seed + i, verify=True)
        pool, n = persistent_tmaze.rollouts_to_memory_pool(rollouts)
        mps, p = train_agent_mps(pool, epochs=epochs, seed=seed)
        q = empirical_joint(rollouts)
        l1 = float(np.abs(p - q).sum())
        results[spec.name] = {"stats": stats, "fit_L1": l1, "n": n}
        print(f"{spec.name:16s} fit L1={l1:.3f}  "
              f"cue_first={stats['cue_first_frac']:.2f}  n={n}")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # quick end-to-end check: tiny data, few epochs -- confirms the pipe runs
        run_experiment(n_episodes=300, epochs=15, seed=0)
    else:
        run_experiment()
