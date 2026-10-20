"""Item 5 of ROADMAP.md: gauge-fix the MPS so the bond index EMITS labeled states.

The bond index of an MPS is only defined up to a gauge transformation
A^(n) -> A^(n) G,  A^(n+1) -> G^-1 A^(n+1)  (Orus arXiv:1306.2164, Schollwoeck
arXiv:1008.3477): it carries no intrinsic labeling, which is why states had to be
recovered post-hoc by ray-fidelity clustering.  Here we choose the residual gauge
that rotates the bond into the cluster basis, so the bond index itself becomes a
labeled hidden state and the model emits p(state | history) directly.

Recipe (Aizpurua et al. arXiv:2401.00867; gauge brief):
  1. per-history bond vectors + their fidelity clusters (k states);
  2. a cluster representative = dominant eigenvector of the within-cluster bond RDM
     (phase-free, robust to noisy reps);
  3. orthonormalize the k representatives (QR) and complete to a UNITARY U;
  4. apply U as the gauge -- unitary, so the represented distribution is preserved
     exactly; the bond then emits p_i(h) = |<q_i | psi_h>|^2 over labeled states.

Validation: distribution preserved to machine precision; emitted per-history
distribution near one-hot (monosemantic); agreement (ARI) with the original
clustering; off-diagonal "leakage" mass into the unlabeled complement.

Run from the repo root:  python src/gauge_fix_states.py
"""
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.structure_recovery import load_tensors, cluster_rays, _safe   # noqa: E402


def adjusted_rand(a, b):
    """ARI between two label vectors (no sklearn dependency)."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    ua, ub = np.unique(a), np.unique(b)
    C = np.array([[np.sum((a == x) & (b == y)) for y in ub] for x in ua], float)
    sa = C.sum(1); sb = C.sum(0)
    comb = lambda x: x * (x - 1) / 2
    idx = comb(C).sum()
    exp = comb(sa).sum() * comb(sb).sum() / comb(n)
    mx = 0.5 * (comb(sa).sum() + comb(sb).sum())
    return (idx - exp) / (mx - exp) if mx != exp else 1.0


def dominant_evec(vectors):
    """Dominant eigenvector of sum_i |v_i><v_i| (phase-free cluster representative)."""
    M = sum(np.outer(v, v.conj()) for v in vectors)
    w, V = np.linalg.eigh(M)
    return V[:, -1]


def unitary_completion(Q):
    """Extend the orthonormal columns of Q (D x k) to a full unitary (D x D)."""
    D, k = Q.shape
    A = np.zeros((D, D), complex)
    A[:, :k] = Q
    # fill the rest with vectors orthogonal to Q via QR of a random-free basis
    B = np.eye(D, dtype=complex)
    for col in range(D):
        v = B[:, col].copy()
        v -= A[:, :max(k, 1)] @ (A[:, :max(k, 1)].conj().T @ v) if k else v
        # re-orthogonalize against already-placed extra columns
        for c in range(k, D):
            if np.linalg.norm(A[:, c]) > 1e-12:
                v -= A[:, c] * (A[:, c].conj() @ v)
        nv = np.linalg.norm(v)
        if nv > 1e-9:
            # place into first empty slot >= k
            for c in range(k, D):
                if np.linalg.norm(A[:, c]) < 1e-12:
                    A[:, c] = v / nv
                    break
    return A


def main():
    print('=' * 72 + '\n GAUGE-FIX TO SYMBOLIC STATES  (Saved_Models/MinimalTmaze.pt)\n' + '=' * 72)
    T1, T2 = load_tensors('MinimalTmaze.pt')
    psi = np.einsum('xaoi,ibpy->aobp', T1, T2)
    p = np.abs(psi) ** 2
    Z = p.sum()

    OBS = ['Cheese', 'Shock', 'RightCue', 'LeftCue']
    ACT = ['Right', 'Left', 'Cue']
    hists, rays, weights = [], [], []
    for a1 in range(3):
        for o2 in range(4):
            w = p[a1, o2].sum()
            if w < 1e-9:
                continue
            b = T1[0, a1, o2, :].astype(complex)
            hists.append(f'{ACT[a1]}/{OBS[o2]}')
            rays.append(b / np.linalg.norm(b))
            weights.append(w)
    labels, _ = cluster_rays(rays)
    k = labels.max() + 1
    D = T1.shape[3]
    print('\n%d histories, bond dim %d, %d fidelity clusters' % (len(rays), D, k))

    # cluster representatives -> orthonormal frame -> unitary gauge
    reps = [dominant_evec([rays[i] for i in range(len(rays)) if labels[i] == c])
            for c in range(k)]
    Q, _ = np.linalg.qr(np.array(reps).T)          # (D, k) orthonormal
    U = unitary_completion(Q[:, :k])
    assert np.allclose(U.conj().T @ U, np.eye(D), atol=1e-8), 'U not unitary'

    # apply the gauge and confirm the distribution is preserved exactly
    T1g = np.einsum('xaoi,ij->xaoj', T1, U)
    T2g = np.einsum('ij,jbpy->ibpy', U.conj().T, T2)
    psi_g = np.einsum('xaoi,ibpy->aobp', T1g, T2g)
    print('gauge unitary; distribution preserved: max|p-p_g| = %.2e'
          % np.abs(np.abs(psi_g) ** 2 / np.abs(psi_g).__pow__(2).sum() - p / Z).max())

    # the bond now EMITS a labeled-state distribution per history
    print('\nemitted state distribution p(state | history) = |<q_i|psi_h>|^2')
    print('   (first %d coords = labeled states, remainder = unlabeled leakage)' % k)
    emit_labels = []
    leak_tot = 0.0
    for h, b, lab in zip(hists, rays, labels):
        amp = U.conj().T @ b
        pr = np.abs(amp) ** 2
        pr = pr / pr.sum()
        emit_labels.append(int(np.argmax(pr[:k])))
        leak = pr[k:].sum()
        leak_tot += leak
        print('   %-16s -> %s  leak=%.3f  (cluster %d)'
              % (h, np.array2string(np.round(pr[:k], 2)), leak, lab))
    print('\nmonosemanticity: mean max-prob = %.3f, mean leakage = %.3f'
          % (np.mean([np.abs(U.conj().T @ b)[:k].max() ** 2 / (np.abs(U.conj().T @ b) ** 2).sum()
                      for b in rays]), leak_tot / len(rays)))
    print('agreement with fidelity clustering: ARI = %.3f'
          % adjusted_rand(labels, emit_labels))
    print('\n=> the bond index is now a labeled hidden state: the model emits states')
    print('   directly instead of us clustering them post-hoc.')


if __name__ == '__main__':
    main()
