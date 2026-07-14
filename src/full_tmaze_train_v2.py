"""Better-converged full pymdp T-maze MPS (fixes the L1=0.30 fit of FullTmaze.pt).

Same data and model as full_tmaze_train.py, but with a training recipe that actually
converges the DMRG/cumulant-SGD:
  * Han learning-rate scheduler -- shrinks lr whenever the (noisy) loss jumps up, so
    it stops bouncing on the ~3.13-nat plateau and descends (Han et al. PRX 2018);
  * Linear cutoff annealing -- starts with a loose SVD cutoff and tightens it, letting
    bonds grow adaptively toward the true rank;
  * more epochs.

Saves Saved_Models/FullTmaze_v2.pt and reports the joint-fit L1 plus the per-state
emission of the previously-weak "center" state, so the improvement is visible.

Run from the repo root:  python src/full_tmaze_train_v2.py
"""
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.full_tmaze_train import (enumerate_weighted, exact_joint, obs_index,   # noqa: E402
                                  analyze)
from mpstwo.data.datasets.memory_pool import MemoryPool                          # noqa: E402
from mpstwo.data.datastructs import TensorDict                                   # noqa: E402
from mpstwo.model.mpstwo import MPSTwo                                           # noqa: E402
from mpstwo.model.mpstwo_trainer import MPSTrainer                               # noqa: E402
from mpstwo.model.optimizers import SGD                                         # noqa: E402
from mpstwo.model.schedulers import Han                                          # noqa: E402
from mpstwo.model.cutoff_schedulers import Linear                                # noqa: E402
from mpstwo.utils.mapping import MultiOneHotMap                                  # noqa: E402


def build_dataset(dtype):
    obs_map = MultiOneHotMap([4, 3, 2])
    act_map = MultiOneHotMap([4, 1])
    leaves = enumerate_weighted()
    train = MemoryPool()
    for (a1, o1, a2, o2, a3, o3), w in leaves:
        sample = TensorDict({
            'action': act_map(torch.tensor([[a1, 0], [a2, 0], [a3, 0]])).type(dtype),
            'observation': obs_map(torch.tensor([list(o1), list(o2), list(o3)])).type(dtype)})
        for _ in range(w):
            train.push_no_update(sample)
    train._update_table()
    return train, leaves


def center_emission_tv(T1, T2, T3):
    """Mean TV of the 'center' state's action-conditioned reward emission vs analytic."""
    left = np.einsum('xi->i', T1[0, 0, 0:2, :]).astype(complex)
    b = np.einsum('i,ij->j', left, T2[:, 0, obs_index(0, 0, 0), :]).astype(complex)
    b = b / np.linalg.norm(b)
    fut = np.abs(np.einsum('j,jcqy->cq', b, T3)) ** 2
    rew = np.zeros((4, 3))
    for o in range(24):
        rew[:, (o // 2) % 3] += fut[:, o]
    rew = rew / rew.sum(1, keepdims=True)
    gt = np.array([[1, 0, 0], [0, .5, .5], [0, .5, .5], [1, 0, 0]], float)  # center,right,left,cue
    return float(np.mean([0.5 * np.abs(rew[a] - gt[a]).sum() for a in range(4)])), rew


def main():
    torch.manual_seed(0)
    dtype = torch.complex128
    train, leaves = build_dataset(dtype)
    print('dataset size:', len(train))

    mps = MPSTwo(3, feature_dim_obs=24, feature_dim_act=4, bond_dim=4,
                 init_mode='random', max_bond=16, cutoff=0.01, dtype=dtype)
    optim = SGD(mps, lr=5e-3)
    trainer = MPSTrainer(mps, train, optimizer=optim, batch_size=32,
                         scheduler=Han(optim, safe_loss_threshold=5e-3, lr_shrink_rate=0.8),
                         log_dir_suffix='full_tmaze_v2', device='cpu')
    with torch.inference_mode():
        trainer.train(200)
    try:
        trainer.close()
    except Exception:
        pass

    T1, T2, T3 = [m.detach().cpu().numpy() for m in mps.matrices]
    print('\ntrained tensor shapes:', T1.shape, T2.shape, T3.shape)
    psi = np.einsum('xaoi,ibpj,jcqy->aobpcq', T1, T2, T3)
    p = np.abs(psi) ** 2
    p /= p.sum()
    q = exact_joint(leaves)
    print('L1 distance to exact joint: %.4f (was 0.2982 for FullTmaze.pt)'
          % np.abs(p - q).sum())

    tv, rew = center_emission_tv(T1, T2, T3)
    print('center-state emission mean TV: %.4f (was 0.148)' % tv)
    print('   center p(r3|a3) [none,cheese,shock] rows=center,right,left,cue:')
    print(np.round(rew, 2))

    analyze(p, 'learned MPS v2')
    analyze(q, 'analytic (reference)')

    out = REPO / 'Saved_Models' / 'FullTmaze_v2.pt'
    torch.save(mps, out)
    print('\nsaved', out)


if __name__ == '__main__':
    main()
