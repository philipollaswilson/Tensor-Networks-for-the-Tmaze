"""Train an MPS on the minimal 4-observation T-maze and extract its structure.

This finishes the program sketched in 5_observation_Tmaze.ipynb (dev/benny_v2):
train a tensor network unsupervised on the minimal maze, then check whether the
learned bond states match the stipulated hidden states.

Minimal maze (trap scenario, as in the paper):
  observations: 0=Cheese, 1=Shock, 2=Right(cue obs), 3=Left(cue obs)
  actions:      0=Right,  1=Left,  2=Cue
  two timesteps: (a1 -> o2), (a2 -> o3)
  context ctx in {0: cheese right, 1: cheese left}, uniform.
  arms are traps: o3 = o2 for any a2 after entering an arm.
  after the cue: a2 = correct arm -> Cheese, wrong arm -> Shock, stay -> same cue obs.

Analysis after training:
  * joint fidelity vs the exact distribution
  * empowerment (Blahut-Arimoto) at t1, in the trap, and post-cue
  * gauge-invariant hidden-state extraction: cluster the bond rays induced by
    each history (a1, o2) by pairwise fidelity, and report each cluster's
    future conditional p(o3 | a2).

Run from the repo root:  python src/minimal_tmaze_train.py
"""
import itertools
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# the package __init__ chain pulls in env deps (gym_bandits, pymdp, ...) that
# training does not need; stub any that are missing so the imports resolve
import importlib
import types

for _mod, _attrs in [('gym_bandits', []), ('gym_bandits.bandit', ['BanditEnv']),
                     ('pymdp', []), ('pymdp.envs', ['TMazeEnv']),
                     ('gym', []), ('gym.spaces', ['MultiDiscrete', 'Discrete', 'Box'])]:
    try:
        _m = importlib.import_module(_mod)
    except ImportError:
        _m = types.ModuleType(_mod)
        sys.modules[_mod] = _m
    for _a in _attrs:
        if not hasattr(_m, _a):
            setattr(_m, _a, type(_a, (), {}))

from mpstwo.data.datasets.memory_pool import MemoryPool
from mpstwo.data.datastructs import TensorDict
from mpstwo.model.mpstwo import MPSTwo
from mpstwo.model.mpstwo_trainer import MPSTrainer
from mpstwo.model.optimizers import SGD
from mpstwo.utils.mapping import MultiOneHotMap

OBS = ['Cheese', 'Shock', 'RightCue', 'LeftCue']
ACT = ['Right', 'Left', 'Cue']

def o2_of(ctx, a1):
    if a1 == 0:
        return 0 if ctx == 0 else 1
    if a1 == 1:
        return 0 if ctx == 1 else 1
    return 2 if ctx == 0 else 3

def o3_of(ctx, a1, o2, a2):
    if a1 in (0, 1):
        return o2                       # trap
    if a2 == 2:
        return o2                       # stay at cue
    return 0 if a2 == ctx else 1        # correct arm -> cheese

def enumerate_sequences():
    seqs = []
    for ctx, a1, a2 in itertools.product(range(2), range(3), range(3)):
        o2 = o2_of(ctx, a1)
        o3 = o3_of(ctx, a1, o2, a2)
        seqs.append((a1, o2, a2, o3))
    return seqs                         # 18 sequences, each with prob 1/18

def exact_joint():
    p = np.zeros((3, 4, 3, 4))
    for a1, o2, a2, o3 in enumerate_sequences():
        p[a1, o2, a2, o3] += 1 / 18
    return p

def main():
    torch.manual_seed(0)
    dtype = torch.complex128
    obs_map = MultiOneHotMap([4])
    act_map = MultiOneHotMap([3])

    train = MemoryPool()
    for _ in range(50):                 # replicate the exact support for batching
        for a1, o2, a2, o3 in enumerate_sequences():
            sample = TensorDict({
                'action': act_map(torch.tensor([[a1], [a2]])).type(dtype),
                'observation': obs_map(torch.tensor([[o2], [o3]])).type(dtype),
            })
            train.push_no_update(sample)
    train._update_table()

    mps = MPSTwo(2, feature_dim_obs=4, feature_dim_act=3, bond_dim=2,
                 init_mode='positive', max_bond=8, cutoff=0.01, dtype=dtype)
    trainer = MPSTrainer(mps, train, optimizer=SGD(mps, lr=1e-2),
                         batch_size=10, log_dir_suffix='minimal_tmaze',
                         device='cpu')
    with torch.inference_mode():
        trainer.train(40)
    try:
        trainer.close()
    except NameError:
        pass  # trainer.close() calls wandb.finish() even when wandb was never set up

    T1, T2 = [m.detach().cpu().numpy() for m in mps.matrices]
    print('\ntrained tensor shapes:', T1.shape, T2.shape)

    psi = np.einsum('xaoi,ibpy->aobp', T1, T2)
    p = np.abs(psi) ** 2
    p /= p.sum()
    q = exact_joint()
    print('L1 distance to exact joint: %.4f  (max entry err %.4f)'
          % (np.abs(p - q).sum(), np.abs(p - q).max()))

    def capacity(W, iters=5000):
        W = np.asarray(W, float)
        keep = W.sum(1) > 1e-12
        W = W[keep] / W[keep].sum(1, keepdims=True)
        qd = np.full(W.shape[0], 1 / W.shape[0])
        for _ in range(iters):
            out = (qd[:, None] * W).sum(0)
            with np.errstate(divide='ignore', invalid='ignore'):
                D = np.where(W > 0, W * np.log2(W / out), 0).sum(1)
            qn = qd * np.exp2(D); qn /= qn.sum()
            if np.abs(qn - qd).max() < 1e-12:
                qd = qn; break
            qd = qn
        out = (qd[:, None] * W).sum(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            return float(np.where(qd[:, None] * W > 0,
                                  qd[:, None] * W * np.log2(W / out), 0).sum())

    # empowerment at t1
    W1 = p.sum(axis=(2, 3))
    W1 = W1 / W1.sum(1, keepdims=True)
    print('\nempowerment t1 (learned): %.4f bits  (analytic 1.0)' % capacity(W1))

    # trap and post-cue
    for name, a1, o2, target in [('trap (Right,Cheese)', 0, 0, 0.0),
                                 ('post-cue (Cue,RightCue)', 2, 2, np.log2(3))]:
        W = p[a1, o2] / p[a1, o2].sum()
        W = W / W.sum(1, keepdims=True)
        print('empowerment %s: %.4f bits  (analytic %.4f)'
              % (name, capacity(W), target))

    # gauge-invariant hidden states: cluster history-induced bond rays
    print('\n---- bond rays by history (a1, o2) ----')
    hists, rays = [], []
    for a1 in range(3):
        for o2 in range(4):
            if p[a1, o2].sum() < 1e-6:
                continue
            b = np.einsum('xaoi->i', T1[:, a1:a1+1, o2:o2+1, :])
            b = b / np.linalg.norm(b)
            hists.append(f'{ACT[a1]:5s}/{OBS[o2]}')
            rays.append(b)
    F = np.abs(np.array([[np.vdot(x, y) for y in rays] for x in rays])) ** 2
    print('histories:', hists)
    print('pairwise ray fidelity |<b_i|b_j>|^2:')
    print(np.round(F, 2))

    # future conditional of each ray
    print('\np(o3|a2) per history ray:')
    for h, b in zip(hists, rays):
        fut = np.abs(np.einsum('i,ibpy->bp', b, T2)) ** 2
        fut = fut / fut.sum(1, keepdims=True)
        print(f'  {h:16s}: ' + '  '.join(
            f'{ACT[a]}->' + np.array2string(np.round(fut[a], 2)) for a in range(3)))

    out = pathlib.Path(__file__).resolve().parents[1] / 'Saved_Models' / 'MinimalTmaze.pt'
    torch.save(mps, out)
    print('\nsaved trained model to', out)

if __name__ == '__main__':
    main()
