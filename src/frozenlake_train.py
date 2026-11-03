"""Train an MPS on frozen lake under the seed protocol used for the minimal maze.

Map (the repo's default, mpstwo/frozenlake.py config):
    S F F        cells 0 1 2
    F H F              3 4 5
    F F G              6 7 8
Observation = the cell; actions 0=Left 1=Down 2=Right 3=Up (gymnasium order).
Hole and goal are absorbing. Episodes start at S and run T = 5 steps under a
uniform policy, one MPS site per step: site t carries (a_t, o_t).

As in seed_protocol.run_minimal: sample N trajectories from the true process,
split 80/20, train on the 80 percent, report L1 to the analytic joint and the
held-out NLL. The seed-0 model is saved as Saved_Models/frozenlake_s0.pt.

Run from the repo root:  python src/frozenlake_train.py [--seeds 1] [--epochs 30] [--slippery]
"""
import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
import minimal_tmaze_train  # noqa: E402,F401  (installs the import stubs)
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv  # noqa: E402

from mpstwo.data.datasets.memory_pool import MemoryPool   # noqa: E402
from mpstwo.data.datastructs import TensorDict           # noqa: E402
from mpstwo.model.mpstwo import MPSTwo                   # noqa: E402
from mpstwo.model.mpstwo_trainer import MPSTrainer       # noqa: E402
from mpstwo.model.optimizers import SGD                  # noqa: E402
from mpstwo.utils.mapping import MultiOneHotMap          # noqa: E402
from seed_protocol import empirical_joint, nll, _l1      # noqa: E402

DESC = ['SFF', 'FHF', 'FFG']
T = 5
ACT = ['Left', 'Down', 'Right', 'Up']
NS = 9
OBS = [f'{DESC[c // 3][c % 3]}{c}' for c in range(NS)]      # 'S0', 'F1', ..., 'H4', 'G8'
HOLE, GOAL = 4, 8
BATCH = 128


def true_B(slippery=False):
    """B[s', s, a] from the gymnasium transition table."""
    P = FrozenLakeEnv(desc=DESC, is_slippery=slippery).P
    B = np.zeros((NS, NS, len(ACT)))
    for s in range(NS):
        for a in range(len(ACT)):
            for prob, s2, _, _ in P[s][a]:
                B[s2, s, a] += prob
    return B


def enumerate_weighted(B):
    """Every (a1, o1, ..., aT, oT) with its probability under the uniform policy."""
    leaves = []
    for acts in itertools.product(range(len(ACT)), repeat=T):
        paths = [((), 1.0, 0)]
        for a in acts:
            paths = [(obs + (s2,), w * B[s2, s, a], s2)
                     for obs, w, s in paths for s2 in np.flatnonzero(B[:, s, a] > 0)]
        for obs, w, _ in paths:
            leaves.append((tuple(x for pair in zip(acts, obs) for x in pair), w / len(ACT) ** T))
    return leaves


def run(seed, n_traj, epochs, save, slippery=False):
    rng = np.random.default_rng(seed)
    dtype = torch.complex128
    leaves = enumerate_weighted(true_B(slippery))
    w = np.array([x[1] for x in leaves]); w /= w.sum()
    shape = (len(ACT), NS) * T
    q = np.zeros(shape)
    for idx, wt in leaves:
        q[idx] += wt
    draws = rng.choice(len(leaves), size=n_traj, p=w)
    n_tr = int(0.8 * n_traj)
    train_idx = [leaves[i][0] for i in draws[:n_tr]]
    test_idx = [leaves[i][0] for i in draws[n_tr:]]

    obs_map = MultiOneHotMap([NS]); act_map = MultiOneHotMap([len(ACT)])
    pool = MemoryPool()
    for idx in train_idx:
        acts, obs = idx[0::2], idx[1::2]
        pool.push_no_update(TensorDict({
            'action': act_map(torch.tensor([[a] for a in acts])).type(dtype),
            'observation': obs_map(torch.tensor([[o] for o in obs])).type(dtype)}))
    pool._update_table()

    t0 = time.time()
    for attempt in range(8):                            # as in run_full: retry a zero-psi init
        torch.manual_seed(seed + 1000 * attempt)
        mps = MPSTwo(T, feature_dim_obs=NS, feature_dim_act=len(ACT), bond_dim=4,
                     init_mode='random', max_bond=16, cutoff=0.01, dtype=dtype)
        trainer = MPSTrainer(mps, pool, optimizer=SGD(mps, lr=1e-2), batch_size=BATCH,
                             log_dir_suffix=f'frozenlake_s{seed}', device='cpu')
        try:
            with torch.inference_mode():
                trainer.train(epochs)
            break
        except ZeroDivisionError as err:
            print(f'[seed {seed}] init hit zero psi ({err}); retrying')
    else:
        raise RuntimeError(f'seed {seed}: training failed for 8 inits')
    try:
        trainer.close()
    except NameError:
        pass

    Ts = [m.detach().cpu().numpy() for m in mps.matrices]
    psi = Ts[0].sum(0)
    for Tk in Ts[1:]:
        psi = np.tensordot(psi, Tk, axes=(-1, 0))
    p = np.abs(psi[..., 0]) ** 2; p /= p.sum()
    emp_test = empirical_joint(shape, test_idx)
    res = {'seed': seed, 'slippery': slippery, 'n_train': n_tr, 'n_test': n_traj - n_tr,
           'epochs': epochs, 'train_seconds': time.time() - t0,
           'bond_dims': [int(Tk.shape[-1]) for Tk in Ts[:-1]],
           'L1_model_vs_analytic': _l1(p, q),
           'L1_model_vs_heldout': _l1(p, emp_test),
           'L1_analytic_vs_heldout': _l1(q, emp_test),
           'NLL_heldout_model': nll(p, test_idx),
           'NLL_heldout_analytic': nll(q, test_idx)}
    if save:
        torch.save(mps, REPO / 'Saved_Models' / f'frozenlake{"_slip" if slippery else ""}_s{seed}.pt')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--n-traj', type=int, default=10000)
    ap.add_argument('--slippery', action='store_true')
    args = ap.parse_args()
    out = REPO / 'results' / 'frozenlake'
    out.mkdir(parents=True, exist_ok=True)
    for s in range(args.seeds):
        r = run(s, args.n_traj, args.epochs, save=(s == 0), slippery=args.slippery)
        print(json.dumps(r, indent=1))
        with open(out / f'seed{s}{"_slip" if args.slippery else ""}.json', 'w') as f:
            json.dump(r, f, indent=1)


if __name__ == '__main__':
    main()
