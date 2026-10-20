"""Seeded, held-out protocol for the learned T-maze models (Paper I, Sec. learned).

For each seed: sample N trajectories from the true generative process under a
uniform-action policy, split 80/20 by trajectory, train an MPS on the 80 percent,
and evaluate on the held-out 20 percent and against the analytic environment:

  * fit: L1 between the learned joint and the analytic joint; L1 between the
    learned joint and the held-out empirical joint (with the analytic-vs-held-out
    L1 as the sampling-noise floor);
  * generalisation: held-out negative log-likelihood per trajectory, against the
    analytic NLL (the entropy of the true process) as the floor;
  * empowerment (Blahut-Arimoto) at start, in the trap, and post-cue, read off
    the learned model, against the analytic values (full maze: full observation
    space and reward modality);
  * structure: number of hidden states recovered by greedy clustering of the
    gauge-invariant history-ray fidelities (merge when |<b_i|b_j>|^2 > 0.5) and
    the maximum total-variation distance between each history's learned next-
    observation conditional and the analytic one.

Seed 0 models are saved as Saved_Models/seedprot_{maze}_s0.pt and the canonical
checkpoints are never touched.

Run from the repo root:
  python src/seed_protocol.py --maze minimal --seeds 5 --epochs 40
  python src/seed_protocol.py --maze full --seeds 5 --epochs 200
"""
import argparse
import itertools
import json
import pathlib
import sys
import time

import os

import numpy as np
import torch

if os.environ.get('SEEDPROT_THREADS'):
    torch.set_num_threads(int(os.environ['SEEDPROT_THREADS']))

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))

import minimal_tmaze_train as MIN          # noqa: E402  (installs the import stubs)
import full_tmaze_train as FULL            # noqa: E402

from mpstwo.data.datasets.memory_pool import MemoryPool   # noqa: E402
from mpstwo.data.datastructs import TensorDict           # noqa: E402
from mpstwo.model.mpstwo import MPSTwo                   # noqa: E402
from mpstwo.model.mpstwo_trainer import MPSTrainer       # noqa: E402
from mpstwo.model.optimizers import SGD                  # noqa: E402
from mpstwo.model.schedulers import Han                  # noqa: E402
from mpstwo.utils.mapping import MultiOneHotMap          # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() and '--cpu' not in sys.argv else 'cpu'
capacity = FULL.capacity
reward_marginal = FULL.reward_marginal
obs_index = FULL.obs_index


def cluster_rays(rays, thresh=0.5):
    """Greedy fidelity clustering (as in structure_recovery.cluster_rays)."""
    F = np.abs(np.array([[np.vdot(x, y) for y in rays] for x in rays])) ** 2
    labels = -np.ones(len(rays), int)
    k = 0
    for i in range(len(rays)):
        if labels[i] >= 0:
            continue
        labels[i] = k
        for j in range(i + 1, len(rays)):
            if labels[j] < 0 and F[i, j] > thresh:
                labels[j] = k
        k += 1
    return labels, F


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def _l1(p, q):
    return float(np.abs(p - q).sum())


def nll(joint, idx_list):
    """Mean negative log-likelihood (nats) of trajectories given as index tuples."""
    vals = np.array([joint[idx] for idx in idx_list])
    vals = np.clip(vals, 1e-300, None)
    return float(-np.log(vals).mean())


def empirical_joint(shape, idx_list):
    e = np.zeros(shape)
    for idx in idx_list:
        e[idx] += 1
    return e / e.sum()


# --------------------------------------------------------------------------
# minimal maze
# --------------------------------------------------------------------------
def run_minimal(seed, n_traj, epochs, save):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dtype = torch.complex128
    seqs = MIN.enumerate_sequences()                   # 18, uniform
    q = MIN.exact_joint()
    draws = rng.integers(0, len(seqs), size=n_traj)
    n_tr = int(0.8 * n_traj)
    train_idx = [seqs[i] for i in draws[:n_tr]]
    test_idx = [seqs[i] for i in draws[n_tr:]]

    obs_map = MultiOneHotMap([4]); act_map = MultiOneHotMap([3])
    pool = MemoryPool()
    for a1, o2, a2, o3 in train_idx:
        pool.push_no_update(TensorDict({
            'action': act_map(torch.tensor([[a1], [a2]])).type(dtype),
            'observation': obs_map(torch.tensor([[o2], [o3]])).type(dtype)}))
    pool._update_table()

    mps = MPSTwo(2, feature_dim_obs=4, feature_dim_act=3, bond_dim=2,
                 init_mode='positive', max_bond=8, cutoff=0.01, dtype=dtype)
    trainer = MPSTrainer(mps, pool, optimizer=SGD(mps, lr=1e-2), batch_size=10,
                         log_dir_suffix=f'seedprot_minimal_s{seed}', device='cpu')
    t0 = time.time()
    with torch.inference_mode():
        trainer.train(epochs)
    try:
        trainer.close()
    except NameError:
        pass
    T1, T2 = [m.detach().cpu().numpy() for m in mps.matrices]
    psi = np.einsum('xaoi,ibpy->aobp', T1, T2)
    p = np.abs(psi) ** 2; p /= p.sum()
    emp_test = empirical_joint(p.shape, test_idx)

    res = {'seed': seed, 'n_train': n_tr, 'n_test': n_traj - n_tr, 'epochs': epochs,
           'train_seconds': time.time() - t0,
           'L1_model_vs_analytic': _l1(p, q),
           'L1_model_vs_heldout': _l1(p, emp_test),
           'L1_analytic_vs_heldout': _l1(q, emp_test),
           'NLL_heldout_model': nll(p, test_idx),
           'NLL_heldout_analytic': nll(q, test_idx)}
    # empowerment
    W1 = p.sum(axis=(2, 3)); W1 = W1 / W1.sum(1, keepdims=True)
    res['emp_start'] = capacity(W1)
    W = p[0, 0] / p[0, 0].sum(); W = W / W.sum(1, keepdims=True)
    res['emp_trap'] = capacity(W)
    caps = []
    for a1, o2 in [(2, 2), (2, 3)]:
        W = p[a1, o2] / p[a1, o2].sum(); W = W / W.sum(1, keepdims=True)
        caps.append(capacity(W))
    res['emp_postcue'] = float(np.mean(caps))
    # structure: history rays at the bond
    hists, rays, tvs = [], [], []
    for a1 in range(3):
        for o2 in range(4):
            if q[a1, o2].sum() < 1e-9:
                continue
            b = np.einsum('xaoi->i', T1[:, a1:a1 + 1, o2:o2 + 1, :]); b = b / np.linalg.norm(b)
            hists.append((a1, o2)); rays.append(b)
            fut = np.abs(np.einsum('i,ibpy->bp', b, T2)) ** 2
            fut = fut / fut.sum(1, keepdims=True)
            truth = q[a1, o2] / q[a1, o2].sum(); truth = truth / truth.sum(1, keepdims=True)
            tvs.append(max(tv(fut[a], truth[a]) for a in range(3)))
    labels, F = cluster_rays(rays, 0.5)
    res['n_states_fidelity'] = int(labels.max() + 1)
    res['n_states_truth'] = 4
    res['max_TV_history_conditional'] = float(max(tvs))
    res['fidelity_matrix'] = np.round(F, 3).tolist()
    res['histories'] = [f'{MIN.ACT[a]}/{MIN.OBS[o]}' for a, o in hists]
    if save:
        torch.save(mps, REPO / 'Saved_Models' / f'seedprot_minimal_s{seed}.pt')
    return res


# --------------------------------------------------------------------------
# full maze
# --------------------------------------------------------------------------
def run_full(seed, n_traj, epochs, save):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dtype = torch.complex128
    leaves = FULL.enumerate_weighted()
    w = np.array([wt for _, wt in leaves], float); w /= w.sum()
    q = FULL.exact_joint(leaves)

    def idx_of(leaf):
        (a1, o1, a2, o2, a3, o3) = leaf
        return (a1, obs_index(*o1), a2, obs_index(*o2), a3, obs_index(*o3))

    draws = rng.choice(len(leaves), size=n_traj, p=w)
    n_tr = int(0.8 * n_traj)
    train_leaves = [leaves[i][0] for i in draws[:n_tr]]
    test_idx = [idx_of(leaves[i][0]) for i in draws[n_tr:]]

    obs_map = MultiOneHotMap([4, 3, 2]); act_map = MultiOneHotMap([4, 1])
    pool = MemoryPool()
    for (a1, o1, a2, o2, a3, o3) in train_leaves:
        pool.push_no_update(TensorDict({
            'action': act_map(torch.tensor([[a1, 0], [a2, 0], [a3, 0]])).type(dtype),
            'observation': obs_map(torch.tensor([list(o1), list(o2), list(o3)])).type(dtype)}))
    pool._update_table()

    # the cumulant SGD raises when a training sequence gets zero amplitude after
    # truncation; a fresh random init clears it, so retry with an offset init seed
    # (the data split is fixed by `seed`; only the init changes) and record it.
    t0 = time.time()
    for attempt in range(8):
        init_seed = seed + 1000 * attempt
        torch.manual_seed(init_seed)
        mps = MPSTwo(3, feature_dim_obs=24, feature_dim_act=4, bond_dim=4,
                     init_mode='random', max_bond=16, cutoff=0.01, dtype=dtype)
        optim = SGD(mps, lr=5e-3)
        trainer = MPSTrainer(mps, pool, optimizer=optim, batch_size=32,
                             scheduler=Han(optim, safe_loss_threshold=5e-3, lr_shrink_rate=0.8),
                             log_dir_suffix=f'seedprot_full_s{seed}', device=DEVICE)
        try:
            with torch.inference_mode():
                trainer.train(epochs)
            break
        except ZeroDivisionError as err:
            print(f'[seed {seed}] init {init_seed} hit zero psi ({err}); retrying with a new init')
            continue
    else:
        raise RuntimeError(f'seed {seed}: training failed for 8 inits')
    try:
        trainer.close()
    except NameError:
        pass
    T1, T2, T3 = [m.detach().cpu().numpy() for m in mps.matrices]
    psi = np.einsum('xaoi,ibpj,jcqy->aobpcq', T1, T2, T3)
    p = np.abs(psi) ** 2; p /= p.sum()
    emp_test = empirical_joint(p.shape, test_idx)

    res = {'seed': seed, 'init_seed': init_seed, 'n_train': n_tr, 'n_test': n_traj - n_tr,
           'epochs': epochs, 'train_seconds': time.time() - t0,
           'L1_model_vs_analytic': _l1(p, q),
           'L1_model_vs_heldout': _l1(p, emp_test),
           'L1_analytic_vs_heldout': _l1(q, emp_test),
           'NLL_heldout_model': nll(p, test_idx),
           'NLL_heldout_analytic': nll(q, test_idx)}
    a = FULL.analyze(p, f'seed {seed}')
    for k, (cf, cr) in a.items():
        res[f'emp_{k}_full'] = cf; res[f'emp_{k}_rew'] = cr
    # structure: seven canonical histories across the second bond
    HIST = [('Center', 0, (0, 0, 0)), ('R/Cheese', 1, (1, 1, 0)), ('R/Shock', 1, (1, 2, 0)),
            ('L/Cheese', 2, (2, 1, 0)), ('L/Shock', 2, (2, 2, 0)),
            ('Cue/ctxR', 3, (3, 0, 0)), ('Cue/ctxL', 3, (3, 0, 1))]
    rays, tvs, names = [], [], []
    sub_p = p[0, 0:2].sum(axis=0); sub_q = q[0, 0:2].sum(axis=0)
    for name, a2, otup in HIST:
        b = np.einsum('i,ij->j', T1[0, 0, 0, :], T2[:, a2, obs_index(*otup), :])
        n = np.linalg.norm(b)
        if n < 1e-9:
            continue
        rays.append(b / n); names.append(name)
        o2 = obs_index(*otup)
        Wp = sub_p[a2, o2]; Wq = sub_q[a2, o2]
        if Wq.sum() < 1e-12:
            continue
        Wp = Wp / Wp.sum(1, keepdims=True); Wq = Wq / Wq.sum(1, keepdims=True)
        Rp, Rq = reward_marginal(Wp), reward_marginal(Wq)
        tvs.append(max(tv(Rp[a3], Rq[a3]) for a3 in range(4)))
    labels, F = cluster_rays(rays, 0.5)
    res['n_states_fidelity'] = int(labels.max() + 1)
    res['n_states_truth'] = 7
    res['max_TV_history_conditional_reward'] = float(max(tvs))
    res['fidelity_matrix'] = np.round(F, 3).tolist()
    res['histories'] = names
    if save:
        torch.save(mps, REPO / 'Saved_Models' / f'seedprot_full_s{seed}.pt')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maze', choices=['minimal', 'full'], required=True)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed-start', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--n-traj', type=int, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    epochs = args.epochs or (40 if args.maze == 'minimal' else 200)
    n_traj = args.n_traj or (900 if args.maze == 'minimal' else 4096)
    out = pathlib.Path(args.out) if args.out else REPO / 'results' / 'seed_protocol'
    out.mkdir(parents=True, exist_ok=True)
    runner = run_minimal if args.maze == 'minimal' else run_full
    results = []
    for s in range(args.seed_start, args.seed_start + args.seeds):
        r = runner(s, n_traj, epochs, save=(s == 0))
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ('fidelity_matrix',)}, indent=1))
        with open(out / f'{args.maze}_seed{s}.json', 'w') as f:
            json.dump(r, f, indent=1)
    keys = [k for k, v in results[0].items() if isinstance(v, (int, float)) and k != 'seed']
    summary = {k: (float(np.mean([r[k] for r in results])), float(np.std([r[k] for r in results], ddof=1) if len(results) > 1 else 0.0))
               for k in keys}
    with open(out / f'{args.maze}_summary.json', 'w') as f:
        json.dump({'n_seeds': len(results), 'mean_sd': summary}, f, indent=1)
    print('\nSUMMARY (mean, sd over %d seeds)' % len(results))
    for k, (m, sd) in summary.items():
        print(f'  {k:34s} {m:9.4f} +- {sd:.4f}')


if __name__ == '__main__':
    main()
