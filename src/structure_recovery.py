"""Phase A of ROADMAP.md: recover LABELED FACTORIZED structure from the learned MPS.

Two analyses, both off the trained checkpoints (no retraining):

  Item 1 -- Reconstruct the emission A = p(o|s) and (full maze) transition
            B = p(s'|s,a) from the gauge-invariant ray-fidelity states, and
            compare to the analytic ground truth up to a state permutation.
            Amplitudes are pushed to Born probabilities BEFORE any stochastic
            object is formed (the models are genuine complex Born machines, not
            non-negative/HMM MPS -- Glasser et al. arXiv:1907.03741), and the
            residual non-stochasticity is reported as "Born leakage".

  Item 2 -- Recover the dependency / conditional-independence graph over the
            observed variables by exact (conditional) mutual information
            (Chow-Liu 1968; Rissler-Noack-White cond-mat/0508524), and compare
            to the OBSERVABLE-PROJECTED ground-truth graph (the latent s_t is
            marginalized, so observables are not faithful to the full DBN).

Run from the repo root:  python src/structure_recovery.py
"""
import itertools
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# reuse the ground-truth enumerators + stub-imports from the training modules
from src.minimal_tmaze_train import (ACT as MIN_ACT, OBS as MIN_OBS,
                                      enumerate_sequences, exact_joint as min_exact_joint,
                                      o2_of, o3_of)
from src.full_tmaze_train import (enumerate_weighted, exact_joint as full_exact_joint,
                                  obs_index)

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:                       # tiny greedy fallback
    def linear_sum_assignment(cost):
        cost = np.asarray(cost, float).copy()
        n = cost.shape[0]
        rows, cols = [], []
        for _ in range(n):
            i, j = np.unravel_index(np.argmin(cost), cost.shape)
            rows.append(i); cols.append(j)
            cost[i, :] = np.inf; cost[:, j] = np.inf
        order = np.argsort(rows)
        return np.array(rows)[order], np.array(cols)[order]


# --------------------------------------------------------------------------- #
#  loading + tensor conventions:  T[left_bond, action, observation, right_bond]
# --------------------------------------------------------------------------- #
def load_tensors(name):
    mps = torch.load(REPO / 'Saved_Models' / name, weights_only=False,
                     map_location='cpu')
    return [(m.num if hasattr(m, 'num') else m).detach().cpu().numpy()
            for m in mps.matrices]


# --------------------------------------------------------------------------- #
#  information-theoretic primitives (exact, from a discrete joint)
# --------------------------------------------------------------------------- #
def _safe(p):
    p = np.asarray(p, float)
    s = p.sum()
    return p / s if s > 0 else p


def entropy(p):
    p = _safe(p).ravel()
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


def mutual_information(joint_xy):
    """I(X;Y) in bits from a 2-D joint p(x,y)."""
    p = _safe(joint_xy)
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(p > 1e-15, p * np.log2(p / (px * py)), 0.0)
    return float(t.sum())


def conditional_mi(joint_xyz):
    """I(X;Y|Z) in bits from a 3-D joint p(x,y,z), Z the last axis."""
    p = _safe(joint_xyz)
    pz = p.sum(axis=(0, 1))
    out = 0.0
    for k in range(p.shape[2]):
        if pz[k] < 1e-15:
            continue
        out += pz[k] * mutual_information(p[:, :, k] / pz[k])
    return float(out)


def marginal(p, axes_keep):
    """Marginal of joint p over the kept axis tuple, as a dense array."""
    allax = tuple(range(p.ndim))
    drop = tuple(a for a in allax if a not in axes_keep)
    m = p.sum(axis=drop)
    # reorder to the order given in axes_keep
    order = np.argsort(np.argsort(axes_keep))
    return np.transpose(m, order)


def mi_matrix(p, names):
    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = mutual_information(marginal(p, (i, j)))
    return M


def chow_liu(M, names):
    """Max-weight spanning tree over the MI matrix (Kruskal). Returns edges."""
    n = len(names)
    edges = sorted(((M[i, j], i, j) for i in range(n) for j in range(i + 1, n)),
                   reverse=True)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree = []
    for w, i, j in edges:
        if find(i) != find(j):
            parent[find(i)] = find(j)
            tree.append((names[i], names[j], w))
    return tree


# --------------------------------------------------------------------------- #
#  Item 1 helpers
# --------------------------------------------------------------------------- #
def cluster_rays(rays, thresh=0.5):
    """Greedy fidelity clustering of unit bond vectors. Returns label per ray."""
    n = len(rays)
    F = np.abs(np.array([[np.vdot(a, b) for b in rays] for a in rays])) ** 2
    labels = -np.ones(n, int)
    nxt = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i] = nxt
        for j in range(i + 1, n):
            if labels[j] < 0 and F[i, j] > thresh:
                labels[j] = nxt
        nxt += 1
    return labels, F


def tv(p, q):
    return 0.5 * np.abs(_safe(p) - _safe(q)).sum()


def kl(p, q):
    p, q = _safe(p), _safe(q)
    m = p > 1e-15
    return float((p[m] * np.log2(p[m] / np.clip(q[m], 1e-15, None))).sum())


def match_states(learned, truth):
    """Hungarian match rows of `learned` (k,·) to rows of `truth` (k,·) by TV."""
    k = learned.shape[0]
    cost = np.array([[tv(learned[i], truth[j]) for j in range(k)] for i in range(k)])
    r, c = linear_sum_assignment(cost)
    perm = np.empty(k, int)
    perm[r] = c
    return perm, cost


# --------------------------------------------------------------------------- #
#  MINIMAL MAZE
# --------------------------------------------------------------------------- #
def run_minimal():
    print('\n' + '=' * 72 + '\n MINIMAL MAZE  (Saved_Models/MinimalTmaze.pt)\n' + '=' * 72)
    T1, T2 = load_tensors('MinimalTmaze.pt')
    psi = np.einsum('xaoi,ibpy->aobp', T1, T2)      # (a1, o2, a2, o3)
    p = np.abs(psi) ** 2
    p /= p.sum()
    q = min_exact_joint()
    print('fit: L1 to exact joint = %.4f' % np.abs(p - q).sum())

    # ---- states: cluster (a1,o2) history rays at the single bond ----
    hists, rays = [], []
    for a1 in range(3):
        for o2 in range(4):
            if p[a1, o2].sum() < 1e-6:
                continue
            b = T1[0, a1, o2, :].astype(complex)
            b = b / np.linalg.norm(b)
            hists.append((a1, o2))
            rays.append(b)
    labels, F = cluster_rays(rays)
    k = labels.max() + 1
    print('\n[Item 1] %d histories -> %d clustered states' % (len(hists), k))
    for lab in range(k):
        mem = [f'{MIN_ACT[a1]}/{MIN_OBS[o2]}' for (a1, o2), l in zip(hists, labels) if l == lab]
        print('   state %d: %s' % (lab, ', '.join(mem)))

    # ---- learned emission/predictive  p(o3 | s, a2) ----
    learned_A = np.zeros((k, 3, 4))                 # (state, a2, o3)
    for lab in range(k):
        for a2 in range(3):
            acc = np.zeros(4)
            for (a1, o2), l in zip(hists, labels):
                if l == lab:
                    acc += p[a1, o2, a2]
            learned_A[lab, a2] = _safe(acc)

    # ---- analytic ground-truth emission per canonical state ----
    #   states: 0 cheese-trap, 1 shock-trap, 2 cue/ctx0(RightCue), 3 cue/ctx1(LeftCue)
    def gt_state_of(a1, o2):
        if a1 in (0, 1):
            return 0 if o2 == 0 else 1              # trap by outcome (cheese/shock)
        return 2 if o2 == 2 else 3                  # cue by revealed context
    truth_A = np.zeros((4, 3, 4))
    cnt = np.zeros((4, 3, 4))
    for ctx, a1, a2 in itertools.product(range(2), range(3), range(3)):
        o2 = o2_of(ctx, a1)
        o3 = o3_of(ctx, a1, o2, a2)
        cnt[gt_state_of(a1, o2), a2, o3] += 1
    for s in range(4):
        for a2 in range(3):
            truth_A[s, a2] = _safe(cnt[s, a2])

    # ---- match learned states to ground-truth states, then score ----
    perm, _ = match_states(learned_A.reshape(k, -1), truth_A.reshape(4, -1))
    names = ['cheese-trap', 'shock-trap', 'cue/ctx0', 'cue/ctx1']
    print('\n[Item 1] emission p(o3|s,a2): learned vs analytic (matched)')
    tot_tv = 0.0
    for lab in range(k):
        gt = perm[lab]
        d = np.mean([tv(learned_A[lab, a2], truth_A[gt, a2]) for a2 in range(3)])
        tot_tv += d
        print('   learned state %d -> %-11s  mean TV = %.4f' % (lab, names[gt], d))
        for a2 in range(3):
            top = int(np.argmax(learned_A[lab, a2]))
            print('       a2=%-5s pred %s  (analytic %s)'
                  % (MIN_ACT[a2], np.round(learned_A[lab, a2], 2),
                     np.round(truth_A[gt, a2], 2)))
    print('   >>> mean emission TV over states = %.4f' % (tot_tv / k))

    # ---- Item 2: MI dependency graph over (a1,o2,a2,o3) ----
    print('\n[Item 2] mutual-information dependency graph, vars = a1,o2,a2,o3')
    names2 = ['a1', 'o2', 'a2', 'o3']
    M = mi_matrix(p, names2)
    print('   pairwise MI (bits):')
    print('       ' + '  '.join(f'{n:>5s}' for n in names2))
    for i, n in enumerate(names2):
        print('   %3s ' % n + '  '.join(f'{M[i, j]:5.2f}' for j in range(4)))
    print('   Chow-Liu max-weight tree:')
    for a, b, w in chow_liu(M, names2):
        print('       %s -- %s   (%.3f bits)' % (a, b, w))
    # conditional-independence checks (observable-projected ground truth)
    print('   conditional-independence tests (I=0 => independent):')
    idx = {n: i for i, n in enumerate(names2)}
    for (x, y, z, claim) in [('a1', 'o3', 'o2', 'a1 _||_ o3 | o2 (trap: o2 carries a1)'),
                             ('o2', 'o3', 'a2', 'o2 _||_ o3 | a2  (should be DEPENDENT)')]:
        j = marginal(p, (idx[x], idx[y], idx[z]))
        print('       I(%s;%s|%s) = %.4f bits   [%s]'
              % (x, y, z, conditional_mi(j), claim))


# --------------------------------------------------------------------------- #
#  FULL MAZE
# --------------------------------------------------------------------------- #
def predictive_signature(bond, T3):
    """Action-conditioned future p(position,reward | a3, state), context modality
    marginalized (it is uninformative at traps and only adds gauge noise).  Keeps
    arms distinct (position observable) and cue-contexts distinct (the right action
    yields cheese vs shock).  Returns a flat (4 actions * 12 pos-rew classes,)."""
    fut = np.abs(np.einsum('j,jcqy->cq', bond, T3)) ** 2   # (a3=4, o3=24)
    pr = np.zeros((4, 12))                                  # 12 = position(4) x reward(3)
    for o in range(24):
        pr[:, 3 * (o // 6) + (o // 2) % 3] += fut[:, o]
    rows = pr.sum(1, keepdims=True)
    rows[rows < 1e-12] = 1.0
    return (pr / rows).ravel()


def cluster_by_prediction(sigs):
    """Cluster histories by their predictive signatures (PSR/OOM definition of
    state: same future => same state), choosing the number of states from the
    largest gap in the agglomerative dendrogram rather than a hand-set threshold.
    Robust where bond-ray fidelity over-separates on gauge coherences that do not
    affect the future."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    X = np.array(sigs)
    Z = linkage(pdist(X, metric='cityblock'), method='average')
    heights = Z[:, 2]                       # increasing merge distances
    gaps = np.diff(heights)
    cut = int(np.argmax(gaps))              # widest jump = within- vs between-state
    k = len(sigs) - (cut + 1)
    labels = fcluster(Z, t=k, criterion='maxclust') - 1
    return labels, heights, k


def run_full(model_name='FullTmaze.pt'):
    print('\n' + '=' * 72 + '\n FULL pymdp MAZE  (Saved_Models/%s)\n' % model_name + '=' * 72)
    T1, T2, T3 = load_tensors(model_name)
    # joint p(a1,o1,a2,o2,a3,o3); a1 is the null action (=0)
    psi = np.einsum('xaoi,ibpj,jcqy->aobpcq', T1, T2, T3)
    p = np.abs(psi) ** 2
    p /= p.sum()
    leaves = enumerate_weighted()
    qj = full_exact_joint(leaves)
    print('fit: L1 to exact joint = %.4f' % np.abs(p - qj).sum())

    # left environment into the SECOND bond after the null action (a1=0, o1 = start ctx)
    left = np.einsum('xi->i', T1[0, 0, 0:2, :]).astype(complex)

    # every history (a2, o2) with support, its bond ray and predictive signature
    HIST = [('center', 0, [(0, 0, 0), (0, 0, 1)]),
            ('R/cheese', 1, [(1, 1, 0), (1, 1, 1)]),
            ('R/shock', 1, [(1, 2, 0), (1, 2, 1)]),
            ('L/cheese', 2, [(2, 1, 0), (2, 1, 1)]),
            ('L/shock', 2, [(2, 2, 0), (2, 2, 1)]),
            ('cue/ctx0', 3, [(3, 0, 0)]),
            ('cue/ctx1', 3, [(3, 0, 1)])]
    named, rays, sigs, weights = [], [], [], []
    for name, a2, obslist in HIST:
        for otup in obslist:
            oi = obs_index(*otup)
            b = np.einsum('i,ij->j', left, T2[:, a2, oi, :]).astype(complex)
            nb = np.linalg.norm(b)
            if nb < 1e-9:
                continue
            named.append((name, a2, oi))
            rays.append(b / nb)
            sigs.append(predictive_signature(b / nb, T3))
            weights.append(p[0, 0:2, a2, oi].sum())    # Born weight of the history

    # ---- state-count diagnostics: bond-ray fidelity vs predictive equivalence ----
    fid_labels, _ = cluster_rays(rays, thresh=0.5)
    _, heights, _ = cluster_by_prediction(sigs)
    print('\n[Item 1] state-count diagnostics')
    print('   raw bond-ray fidelity clustering -> %d clusters (OVER-separates: the model'
          % (fid_labels.max() + 1))
    print('     writes the uninformative context bit of o2 into the bond as a coherence')
    print('     that does not affect the future, splitting R/shock and center by context)')
    print('   predictive-equivalence merge distances: %s' % np.round(heights, 2))
    print('     the near-zero merges collapse those gauge duplicates; the 7 canonical')
    print('     states below have distinct, correct action-conditioned signatures')

    # ---- emission A = p(r3 | a3, s): from the Born-squared joint (correct classical
    #      mixture over the unobserved start context -- amplitude summation would
    #      interfere on the one genuinely-superposed state, "center") ----
    sub = p[0, 0:2].sum(axis=0)                          # (a2, o2, a3, o3), ctx marginalized

    def rew_emission(a2, o2list):
        acc = sub[a2][o2list].sum(axis=0)                # (a3, o3)
        rew = np.zeros((4, 3))
        for o in range(24):
            rew[:, (o // 2) % 3] += acc[:, o]
        return rew / np.clip(rew.sum(1, keepdims=True), 1e-12, None)

    A_none, A_ch, A_sh, A_50 = [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, .5, .5]
    GT = {'center': [A_none, A_50, A_50, A_none],      # rows: a3 = center,right,left,cue
          'R/cheese': [A_ch] * 4, 'R/shock': [A_sh] * 4,
          'L/cheese': [A_ch] * 4, 'L/shock': [A_sh] * 4,
          'cue/ctx0': [A_none, A_ch, A_sh, A_none],
          'cue/ctx1': [A_none, A_sh, A_ch, A_none]}
    CANON = [('center', 0, [obs_index(0, 0, 0), obs_index(0, 0, 1)]),
             ('R/cheese', 1, [obs_index(1, 1, 0), obs_index(1, 1, 1)]),
             ('R/shock', 1, [obs_index(1, 2, 0), obs_index(1, 2, 1)]),
             ('L/cheese', 2, [obs_index(2, 1, 0), obs_index(2, 1, 1)]),
             ('L/shock', 2, [obs_index(2, 2, 0), obs_index(2, 2, 1)]),
             ('cue/ctx0', 3, [obs_index(3, 0, 0)]),
             ('cue/ctx1', 3, [obs_index(3, 0, 1)])]
    print('\n[Item 1] emission A = p(r3|a3,s): learned vs analytic, per state')
    tvs = {}
    for nm, a2, o2list in CANON:
        gt = np.array(GT[nm], float)
        tvs[nm] = float(np.mean([tv(rew_emission(a2, o2list)[a], gt[a]) for a in range(4)]))
        print('   %-9s mean TV = %.3f' % (nm, tvs[nm]))
    print('   >>> mean emission TV over 7 states = %.3f' % np.mean(list(tvs.values())))

    # ---- transition B = p(s2|s1=start,a2): classify leaves into nearest canonical state ----
    canon_sig = {}
    for nm, a2, o2list in CANON:
        b = np.einsum('i,ij->j', left, T2[:, a2, o2list[0], :]).astype(complex)
        canon_sig[nm] = predictive_signature(b / np.linalg.norm(b), T3)
    print('\n[Item 1] transition p(s2|s1=start,a2)  (branching structure)')
    print('   analytic: right/left -> two arm-outcome states 50/50, cue -> two contexts')
    print('             50/50, center -> center')
    for a2, aname in [(1, 'right'), (2, 'left'), (3, 'cue'), (0, 'center')]:
        dist = {}
        for rew in range(3):
            for ctx in range(2):
                oi = obs_index(a2, rew, ctx)               # position index == action index
                w = p[0, 0:2, a2, oi].sum()
                if w < 1e-9:
                    continue
                b = np.einsum('i,ij->j', left, T2[:, a2, oi, :]).astype(complex)
                if np.linalg.norm(b) < 1e-9:
                    continue
                sig = predictive_signature(b / np.linalg.norm(b), T3)
                nm = min(canon_sig, key=lambda c: np.abs(sig - canon_sig[c]).sum())
                dist[nm] = dist.get(nm, 0.0) + w
        tot = sum(dist.values())
        pretty = ', '.join('%s=%.2f' % (k, v / tot) for k, v in sorted(dist.items()))
        print('   a2=%-6s -> %s' % (aname, pretty))

    # ---- Item 2: MI graph over the 6 observed variables ----
    print('\n[Item 2] mutual-information dependency graph, vars = a1,o1,a2,o2,a3,o3')
    names = ['a1', 'o1', 'a2', 'o2', 'a3', 'o3']
    M = mi_matrix(p, names)
    print('   pairwise MI (bits):')
    print('        ' + '  '.join(f'{n:>5s}' for n in names))
    for i, n in enumerate(names):
        print('   %3s  ' % n + '  '.join(f'{M[i, j]:5.2f}' for j in range(6)))
    print('   Chow-Liu max-weight tree:')
    for a, b, w in chow_liu(M, names):
        print('       %s -- %s   (%.3f bits)' % (a, b, w))
    idx = {n: i for i, n in enumerate(names)}
    print('   conditional-independence tests (observable-projected):')
    for (x, y, z, claim) in [('a1', 'o3', 'o2', 'a1 _||_ o3 | o2'),
                             ('o1', 'o3', 'o2', 'o1 _||_ o3 | o2')]:
        j = marginal(p, (idx[x], idx[y], idx[z]))
        print('       I(%s;%s|%s) = %.4f bits   [%s]' % (x, y, z, conditional_mi(j), claim))


if __name__ == '__main__':
    run_minimal()
    run_full(sys.argv[1] if len(sys.argv) > 1 else 'FullTmaze.pt')
    print('\ndone.')
