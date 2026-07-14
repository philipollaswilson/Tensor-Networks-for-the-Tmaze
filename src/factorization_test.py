"""Item 4 of ROADMAP.md: does the learned MPS DISCOVER the observation factorization?

The full-maze MPS is fed a FLAT 24-dim observation leg (MultiOneHotMap([4,3,2])
flattens the outer product); the 4x3x2 = position x reward x context structure is
assumed only in our *analysis*.  Here we test whether the learned tensors respect
that structure, from the single-site reduced density matrix rho = Tr_rest |psi><psi|
(no retraining):

  * quantum mutual information I(A:B) = S(rho_A)+S(rho_B)-S(rho_AB) between the three
    candidate sub-factors (I=0  <=>  rho_A (x) rho_B, i.e. independent);
  * the trace-distance residual ||rho - rho_pos (x) rho_rew (x) rho_ctx||_1 / 2;
  * the same residual for ALTERNATIVE groupings of 24, to check that position/reward/
    context is the grouping the model most nearly factorizes over.

Nonzero position-reward and reward-context coupling is EXPECTED (it is causally real
in the maze): the target is a sparse, structured coupling matching the DBN, not zero.

Refs: Convy et al. arXiv:2103.00105 (MI for TN structure); quantum mutual information.
Run from the repo root:  python src/factorization_test.py
"""
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.structure_recovery import load_tensors, _safe   # noqa: E402

MOD = ['position', 'reward', 'context']
DIMS = (4, 3, 2)                          # obs index = 6*pos + 2*rew + ctx


def von_neumann(rho):
    """S(rho) in bits from a density matrix (Hermitian, trace 1)."""
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    w = w[w > 1e-12]
    return float(-(w * np.log2(w)).sum())


def single_site_rdm(psi, leg):
    """Reduced density matrix at observation `leg` of a pure state psi (all legs)."""
    d = psi.shape[leg]
    M = np.moveaxis(psi, leg, 0).reshape(d, -1)
    rho = M @ M.conj().T
    return rho / np.trace(rho).real


def partial_over(rho6, keep):
    """From rho reshaped (4,3,2,4,3,2) keep the tuple of sub-factor axes, trace rest."""
    keep = tuple(keep)
    tr = tuple(a for a in range(3) if a not in keep)
    # sum matched bra/ket indices on the traced axes
    ein_in = list('abcABC')
    for a in tr:
        ein_in[3 + a] = ein_in[a]                 # tie ket index to bra index -> trace
    out = [ein_in[a] for a in keep] + [ein_in[3 + a] for a in keep]
    r = np.einsum(''.join(ein_in) + '->' + ''.join(out), rho6)
    dk = int(np.prod([DIMS[a] for a in keep]))
    return r.reshape(dk, dk)


def analyse_leg(psi, leg, label):
    print('\n---- observation leg %s ----' % label)
    rho = single_site_rdm(psi, leg)
    rho6 = rho.reshape(*DIMS, *DIMS)
    print('   rank(rho) = %d, S(rho) = %.3f bits, marginal p(o) purity = %.3f'
          % (np.linalg.matrix_rank(rho, tol=1e-9), von_neumann(rho),
             float((np.diag(rho).real ** 2).sum())))

    S = {i: von_neumann(partial_over(rho6, (i,))) for i in range(3)}
    print('   quantum MI between sub-factors (bits):')
    for i in range(3):
        for j in range(i + 1, 3):
            Sij = von_neumann(partial_over(rho6, (i, j)))
            I = S[i] + S[j] - Sij
            print('       I(%-8s : %-8s) = %+.3f' % (MOD[i], MOD[j], I))

    # trace-distance residual vs the fully-factorized product
    rho_prod = np.ones((1, 1), complex)
    for i in range(3):
        rho_prod = np.kron(rho_prod, partial_over(rho6, (i,)))
    # reorder product legs back to (pos,rew,ctx) order (kron already in that order)
    resid = 0.5 * np.abs(np.linalg.eigvalsh(rho - rho_prod)).sum()
    print('   trace-distance ||rho - rho_pos(x)rho_rew(x)rho_ctx||/2 = %.3f' % resid)
    return rho6


def grouping_residual(rho6, groups):
    """Trace distance between rho and the product over `groups` (list of axis tuples)."""
    prod = np.ones((1, 1), complex)
    perm = []
    for g in groups:
        prod = np.kron(prod, partial_over(rho6, g))
        perm += list(g)
    # rho reordered so its axes follow `perm`
    axes = perm + [a + 3 for a in perm]
    rho_perm = np.transpose(rho6, axes).reshape(24, 24)
    return 0.5 * np.abs(np.linalg.eigvalsh(rho_perm - prod)).sum()


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'FullTmaze.pt'
    T1, T2, T3 = load_tensors(model_name)
    psi = np.einsum('xaoi,ibpj,jcqy->aobpcq', T1, T2, T3)   # (a1,o1,a2,o2,a3,o3)
    psi = psi / np.linalg.norm(psi)
    print('=' * 72 + '\n FACTORIZATION DISCOVERY  (Saved_Models/%s)\n' % model_name + '=' * 72)

    rho6 = analyse_leg(psi, 3, 'o2 (mid step)')
    analyse_leg(psi, 5, 'o3 (final step)')

    print('\n---- candidate groupings of the 24-dim leg (o2), by trace-distance residual ----')
    cands = {'position | reward | context (4x3x2, true)': [(0,), (1,), (2,)],
             '(position,reward) | context   (12x2)':      [(0, 1), (2,)],
             '(position,context) | reward   (8x3)':       [(0, 2), (1,)],
             '(reward,context) | position   (6x4)':       [(1, 2), (0,)],
             'fully joined                  (24)':        [(0, 1, 2)]}
    rows = sorted(((grouping_residual(rho6, g), name) for name, g in cands.items()))
    for r, name in rows:
        print('   %-44s residual = %.3f' % (name, r))
    print('   (lower = the model factorizes more cleanly over that grouping; the fully')
    print('    joined grouping is always 0 and is the trivial reference)')


if __name__ == '__main__':
    main()
