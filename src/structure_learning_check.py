"""Does the learned MPS structure-learn interpretable hidden states?

The bond between timestep-2 and timestep-3 tensors is the model's latent state
after the first real action. We Schmidt-decompose the (history | future) cut,
then ask two questions:
  1. How many states did the model discover (effective rank / entanglement spectrum)?
  2. Are they interpretable? For each canonical history (center / trap-cheese /
     trap-shock / cue-ctx0 / cue-ctx1) we compute the bond state it induces, and
     for each Schmidt state we compute the future reward profile p(r3 | a3, state).
Ground-truth expectation if structure learning works:
  trap histories -> states that predict their own reward under every action;
  cue histories  -> states that predict win/loss depending on arm choice;
  center history -> a state with 50/50 arm outcomes (context unresolved).
"""
import sys
import numpy as np
import torch

REPO = r'C:\Users\mahau\OneDrive\Desktop\projects\Tensor-Networks-for-the-Tmaze'
sys.path.insert(0, REPO)

mps = torch.load(REPO + r'\Saved_Models\SamuelModel.pt', weights_only=False, map_location='cpu')
T1, T2, T3 = [ (m.num if hasattr(m, 'num') else m).detach().cpu().numpy() for m in mps.matrices ]

# ---- entanglement spectrum across both cuts ----
# cut A: between t1-site and t2-site (bond dim 2)
psi12 = np.einsum('xaoi->aoi', T1)                    # (4,24,2)
M = psi12.reshape(-1, psi12.shape[-1])
# weight by the future to get the true Schmidt spectrum of the full state:
psi_full = np.einsum('xaoi,ibpj,jcqy->aobpcq', T1, T2, T3)
sh = psi_full.shape
for cut, mat in [('t1|t2', psi_full.reshape(sh[0]*sh[1], -1)),
                 ('t2|t3', psi_full.reshape(sh[0]*sh[1]*sh[2]*sh[3], -1))]:
    s = np.linalg.svd(mat, compute_uv=False)
    s2 = (s**2) / (s**2).sum()
    print(f'cut {cut}: schmidt weights = {np.round(s2[:6], 4)}')

# ---- Schmidt basis across the t2|t3 cut ----
U, S, Vh = np.linalg.svd(psi_full.reshape(-1, sh[4]*sh[5]), full_matrices=False)
k = int((S**2 / (S**2).sum() > 1e-4).sum())
print(f'\neffective number of hidden states after first action: {k}')
V = Vh[:k]                                            # (k, 4*24) future amplitudes

def reward_profile(vec):
    """p(r3 | a3) from a future amplitude vector over (a3,o3)."""
    p = np.abs(vec.reshape(4, 24))**2
    pr = np.zeros((4, 3))
    for o in range(24):
        pr[:, (o // 2) % 3] += p[:, o]
    rows = pr.sum(1, keepdims=True); rows[rows == 0] = 1
    return pr / rows

ACT = ['center', 'right', 'left', 'cue']
print('\n---- Schmidt states: future reward profile p(r3|a3) [none, win, loss] ----')
for j in range(k):
    print(f'state {j} (weight {S[j]**2/(S**2).sum():.3f}):')
    rp = reward_profile(V[j])
    for a in range(4):
        print(f'   a3={ACT[a]:6s}: {np.round(rp[a], 2)}')

# ---- which state does each canonical history land on? ----
def bond_state(a2, o2_indices):
    """Contract history (a1=0, o1 in {0,1}, a2, o2 in o2_indices) -> bond vector."""
    v = np.zeros(T3.shape[0], dtype=complex)
    amps = []
    for o1 in (0, 1):
        for o2 in o2_indices:
            amps.append(np.einsum('xi,ij->j',
                                  T1[:, 0, o1, :], T2[:, a2, o2, :]))
    # histories are alternatives (classical mixture); compare each separately
    return amps

HIST = {
    'center (o2=center/none)': (0, [0, 1]),
    'trap right/cheese':       (1, [8, 9]),
    'trap right/shock':        (1, [10, 11]),
    'trap left/cheese':        (2, [14, 15]),
    'cue, ctx obs = 0':        (3, [18]),
    'cue, ctx obs = 1':        (3, [19]),
}
print('\n---- history -> distribution over Schmidt states ----')
for name, (a2, idxs) in HIST.items():
    amps = bond_state(a2, idxs)
    # average the induced state distribution over the alternative o-realizations
    dists = []
    for amp in amps:
        coeff = amp @ np.conj(U.reshape(-1, len(S))[..., :k]).T if False else None
        # project the bond vector onto the Schmidt basis of the future:
        # future state given bond vector b is sum_j b_j T3[j] ; expand in V basis
        fut = np.einsum('j,jcqy->cq', amp, T3).reshape(-1)
        n = np.linalg.norm(fut)
        if n < 1e-12:
            continue
        fut /= n
        w = np.abs(V.conj() @ fut)**2
        dists.append(w / w.sum())
    d = np.mean(dists, axis=0)
    print(f'{name:26s}: {np.round(d, 3)}')

# ---- reward profile of each history directly (sanity) ----
print('\n---- history -> p(r3|a3) directly ----')
for name, (a2, idxs) in HIST.items():
    amps = bond_state(a2, idxs)
    ps = []
    for amp in amps:
        fut = np.einsum('j,jcqy->cq', amp, T3)
        ps.append(np.abs(fut)**2)
    p = np.mean([x / x.sum() for x in ps], axis=0)
    pr = np.zeros((4, 3))
    for o in range(24):
        pr[:, (o // 2) % 3] += p[:, o]
    rows = pr.sum(1, keepdims=True); rows[rows == 0] = 1
    pr = pr / rows
    print(f'{name:26s}: right->{np.round(pr[1], 2)}  left->{np.round(pr[2], 2)}')
