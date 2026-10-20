"""Train a converged MPS on the FULL pymdp T-maze and extract its structure.

Same program as minimal_tmaze_train.py, but on the full three-step maze with
24 composite observations (position 4 x reward 3 x context 2) and 4 actions.
Differences from the original tmaze.py setup:
  * sequences are weighted by their true probability (the exhaustive enumeration
    in tmaze.py gives every leaf equal weight, which mis-weights branches whose
    outcome count differs; probabilities here are dyadic so exact integer
    replication reproduces the true distribution),
  * a smaller truncation cutoff and larger max bond, because the true amplitude
    rank across the second bond is ~7 (center-unresolved, 4 arm/outcome traps,
    2 cue contexts) while SamuelModel.pt was truncated to bond 4.

Analysis: empowerment (Blahut-Arimoto) at start / trap / post-cue over the full
observation space and the reward modality, plus gauge-invariant hidden-state
extraction by history-ray fidelity clustering across the second bond.

Run from the repo root:  python src/full_tmaze_train.py
"""
import itertools
import pathlib
import sys
import types
import importlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

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
from mpstwo.model.schedulers import Han
from mpstwo.utils.mapping import MultiOneHotMap

ACT = ['center', 'right', 'left', 'cue']


def enumerate_weighted():
    """All (a1,o1,a2,o2,a3,o3) leaves with true probability numerators.

    Random draws (prob 1/2 each): c0; r1 if arm entered blind; c1 (unless at
    cue, where it reveals ctx -- we enumerate ctx via the cue leaf itself);
    r2 / c2 analogously. Actions uniform (handled by equal action fan-out).
    Weight = 2**(n_max - n_draws) so all weights are integers.
    Follows the branch logic of mpstwo/tmaze.py exactly.
    """
    leaves = []
    for c0 in range(2):
        for a1 in range(4):
            p1 = a1
            r1_opts = [1, 2] if a1 in (1, 2) else [0]
            for r1 in r1_opts:
                for c1 in range(2):
                    for a2 in range(4):
                        p2 = a1 if a1 in (1, 2) else a2
                        if p1 in (1, 2) and p2 in (1, 2):
                            r2_opts = [r1]
                        elif p1 == 3 and p2 in (1, 2):
                            r2_opts = [2 - (c1 + p2) % 2]
                        elif p2 in (1, 2):
                            r2_opts = [1, 2]
                        else:
                            r2_opts = [0]
                        for r2 in r2_opts:
                            c2_opts = [c1] if (a1 == 3 and a2 == 3) else [0, 1]
                            for c2 in c2_opts:
                                n_draws = (1 + len(r1_opts).bit_length() - 1
                                           + 1 + (len(r2_opts) > 1) + (len(c2_opts) > 1))
                                # draws: c0, r1?(1 if 2 opts), c1, r2?, c2?
                                n = 1 + (len(r1_opts) > 1) + 1 + (len(r2_opts) > 1) + (len(c2_opts) > 1)
                                weight = 2 ** (5 - n)
                                leaves.append(((0, (0, 0, c0), a1, (p1, r1, c1),
                                                a2, (p2, r2, c2)), weight))
    return leaves


def obs_index(p, r, c):
    return 6 * p + 2 * r + c


def exact_joint(leaves):
    q = np.zeros((4, 24, 4, 24, 4, 24))
    for (a1, o1, a2, o2, a3, o3), w in leaves:
        q[a1, obs_index(*o1), a2, obs_index(*o2), a3, obs_index(*o3)] += w
    return q / q.sum()


def capacity(W, iters=6000):
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


def reward_marginal(W24):
    W = np.zeros((W24.shape[0], 3))
    for o in range(24):
        W[:, (o // 2) % 3] += W24[:, o]
    return W


def analyze(p, label):
    print(f'\n==== {label} ====')
    results = {}
    # start: condition a1=0, o1 = (center,none,*)
    sub = p[0, 0:2].sum(axis=0)
    W1 = sub.sum(axis=(2, 3))
    W1 = W1 / W1.sum(1, keepdims=True)
    results['start'] = (capacity(W1), capacity(reward_marginal(W1)))
    # trap: a2=right, o2=(right,cheese,*)
    W2 = sub[1, [obs_index(1, 1, 0), obs_index(1, 1, 1)]].sum(axis=0)
    W2n = W2 / W2.sum(1, keepdims=True)
    results['trap'] = (capacity(W2n), capacity(reward_marginal(W2n)))
    # post-cue: a2=cue, o2=(cue,none,c) averaged over c
    caps = []
    for c in (0, 1):
        W3 = sub[3, obs_index(3, 0, c)]
        W3 = W3 / W3.sum(1, keepdims=True)
        caps.append((capacity(W3), capacity(reward_marginal(W3))))
    results['postcue'] = tuple(np.mean(caps, axis=0))
    for k, (cf, cr) in results.items():
        print(f'  {k:8s}: full = {cf:.4f} bits | reward-only = {cr:.4f} bits')
    return results


def main():
    torch.manual_seed(0)
    dtype = torch.complex128
    obs_map = MultiOneHotMap([4, 3, 2])
    act_map = MultiOneHotMap([4, 1])

    leaves = enumerate_weighted()
    total = sum(w for _, w in leaves)
    print(f'{len(leaves)} leaves, total weight {total}')

    train = MemoryPool()
    for (a1, o1, a2, o2, a3, o3), w in leaves:
        sample = TensorDict({
            'action': act_map(torch.tensor([[a1, 0], [a2, 0], [a3, 0]])).type(dtype),
            'observation': obs_map(torch.tensor([list(o1), list(o2), list(o3)])).type(dtype),
        })
        for _ in range(w):
            train.push_no_update(sample)
    train._update_table()
    print('dataset size:', len(train))

    # converged recipe: a Han learning-rate scheduler shrinks lr whenever the noisy
    # DMRG/cumulant loss jumps up, so it descends past the plateau that a fixed lr
    # bounces on (fit L1 ~0.005 vs ~0.30 for a fixed-lr run of the same budget).
    mps = MPSTwo(3, feature_dim_obs=24, feature_dim_act=4, bond_dim=4,
                 init_mode='random', max_bond=16, cutoff=0.01, dtype=dtype)
    optim = SGD(mps, lr=5e-3)
    trainer = MPSTrainer(mps, train, optimizer=optim, batch_size=32,
                         scheduler=Han(optim, safe_loss_threshold=5e-3, lr_shrink_rate=0.8),
                         log_dir_suffix='full_tmaze', device='cpu')
    with torch.inference_mode():
        trainer.train(200)
    try:
        trainer.close()
    except NameError:
        pass

    T1, T2, T3 = [m.detach().cpu().numpy() for m in mps.matrices]
    print('\ntrained tensor shapes:', T1.shape, T2.shape, T3.shape)

    psi = np.einsum('xaoi,ibpj,jcqy->aobpcq', T1, T2, T3)
    p = np.abs(psi) ** 2
    p /= p.sum()
    q = exact_joint(leaves)
    print('L1 distance to exact joint: %.4f (max entry err %.5f)'
          % (np.abs(p - q).sum(), np.abs(p - q).max()))

    analyze(q, 'analytic (true distribution)')
    analyze(p, 'learned MPS (converged run)')

    # hidden states across the second bond, gauge-invariant
    print('\n---- history rays across second bond ----')
    hists, rays = [], []
    HIST = [('center', 0, [(0, 0, 0), (0, 0, 1)]),
            ('trap R/cheese', 1, [(1, 1, 0), (1, 1, 1)]),
            ('trap R/shock', 1, [(1, 2, 0), (1, 2, 1)]),
            ('trap L/cheese', 2, [(2, 1, 0), (2, 1, 1)]),
            ('trap L/shock', 2, [(2, 2, 0), (2, 2, 1)]),
            ('cue ctx0', 3, [(3, 0, 0)]),
            ('cue ctx1', 3, [(3, 0, 1)])]
    for name, a2, obs_list in HIST:
        for o1 in (0, 1):
            pass
        for otup in obs_list:
            b = np.einsum('xi,ij->j',
                          T1[:, 0, 0:2, :].sum(axis=1),
                          T2[:, a2, obs_index(*otup), :])
            n = np.linalg.norm(b)
            if n < 1e-9:
                continue
            hists.append(f'{name} c={otup[2]}')
            rays.append(b / n)
    F = np.abs(np.array([[np.vdot(x, y) for y in rays] for x in rays])) ** 2
    print('histories:', hists)
    np.set_printoptions(linewidth=200)
    print('pairwise ray fidelity |<b_i|b_j>|^2:')
    print(np.round(F, 2))

    out = pathlib.Path(__file__).resolve().parents[1] / 'Saved_Models' / 'FullTmaze.pt'
    torch.save(mps, out)
    print('\nsaved trained model to', out)


if __name__ == '__main__':
    main()
