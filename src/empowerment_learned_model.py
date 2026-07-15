"""Empowerment analysis of the learned T-maze tensor network (SamuelModel.pt).

Loads the trained MPS, contracts the exact joint p(a1,o1,a2,o2,a3,o3) = |psi|^2 / Z,
verifies it against the environment's generative rules, and computes channel-capacity
empowerment (Blahut-Arimoto) in three conditions:
  (1) from the start position,
  (2) after entering an arm (trap),
  (3) after visiting the cue,
each over the full 24-dim observation space and over the reward modality alone.

Obs index = 6*position + 2*reward + context  (position in 0..3, reward 0..2, context 0..1)
Actions: 0=center, 1=right, 2=left, 3=cue.
Timestep 0 is a null action (0) with obs (center, none, random context).
"""
import sys
import numpy as np
import torch

REPO = r'C:\Users\mahau\OneDrive\Desktop\projects\Tensor-Networks-for-the-Tmaze'
sys.path.insert(0, REPO)

mps = torch.load(REPO + r'\Saved_Models\SamuelModel.pt', weights_only=False,
                 map_location='cpu')
mats = mps.matrices
print('type(matrices[0]):', type(mats[0]))
Ts = []
for m in mats:
    arr = m.num if hasattr(m, 'num') else m
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    Ts.append(np.asarray(arr))
T1, T2, T3 = Ts
print('shapes:', T1.shape, T2.shape, T3.shape)

# joint amplitude psi[a1,o1,a2,o2,a3,o3]
psi = np.einsum('xaoi,ibpj,jcqy->aobpcq', T1, T2, T3)
p = np.abs(psi) ** 2
p /= p.sum()
print('joint shape:', p.shape, '| sum:', p.sum())

# ---- sanity checks against the environment's rules ----
# t0: null action (a1=0); obs must be center/none, context random
p_o1 = p[0].sum(axis=(1, 2, 3, 4))
p_o1 /= p_o1.sum()
top = np.argsort(p_o1)[::-1][:4]
print('p(o1 | a1=0) top indices:', [(int(i), round(float(p_o1[i]), 4)) for i in top])
# expect mass ~0.5/0.5 on indices 0 and 1

def cond_o_given_a(joint_ao):
    """joint_ao: (4,24) unnormalized -> row-normalized p(o|a); zero rows dropped later."""
    out = joint_ao.copy().astype(float)
    rows = out.sum(axis=1)
    ok = rows > 1e-12
    out[ok] = out[ok] / rows[ok][:, None]
    return out, ok

def blahut_arimoto(W, iters=8000, tol=1e-12):
    """Channel capacity in bits of W[o|a] (rows = inputs)."""
    W = np.asarray(W, float)
    keep = W.sum(axis=1) > 1e-12
    W = W[keep]
    W = W / W.sum(axis=1, keepdims=True)
    n = W.shape[0]
    q = np.full(n, 1.0 / n)
    for _ in range(iters):
        Q = q[:, None] * W
        out = Q.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            D = np.where(W > 0, W * np.log2(W / out[None, :]), 0.0).sum(axis=1)
        qn = q * np.exp2(D)
        qn /= qn.sum()
        if np.max(np.abs(qn - q)) < tol:
            q = qn
            break
        q = qn
    Q = q[:, None] * W
    out = Q.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        C = np.where(Q > 0, Q * np.log2(W / out[None, :]), 0.0).sum()
    return C, q

def reward_marginal(W24):
    """Collapse p(o|a) over 24 obs to p(r|a) over 3 rewards."""
    W = np.zeros((W24.shape[0], 3))
    for o in range(24):
        W[:, (o // 2) % 3] += W24[:, o]
    return W

REPORT = {}

# ---- condition 1: at start (condition on a1=0, o1 in {0,1}) ----
sub = p[0, 0:2].sum(axis=0)          # (a2,o2,a3,o3)
joint_a2o2 = sub.sum(axis=(2, 3))     # (4,24)
W1, _ = cond_o_given_a(joint_a2o2)
C_full, q1 = blahut_arimoto(W1)
C_rew, _ = blahut_arimoto(reward_marginal(W1))
REPORT['start'] = (C_full, C_rew)
print('\n[start] capacity full-obs = %.4f bits | reward-only = %.4f bits' % (C_full, C_rew))
print('        p(o2|a2) row supports:')
for a, name in enumerate(['center', 'right', 'left', 'cue']):
    nz = [(int(o), round(float(W1[a, o]), 3)) for o in range(24) if W1[a, o] > 5e-3]
    print(f'        a2={name}: {nz}')

# ---- condition 2: trap (a2=right, o2 = right/cheese, either context obs) ----
for r1, lbl in [(1, 'cheese'), (2, 'shock')]:
    idx = [6 * 1 + 2 * r1 + 0, 6 * 1 + 2 * r1 + 1]
    sub2 = p[0, 0:2].sum(axis=0)[1, idx].sum(axis=0)   # (a3,o3)
    W2, _ = cond_o_given_a(sub2)
    Cf, _ = blahut_arimoto(W2)
    Cr, _ = blahut_arimoto(reward_marginal(W2))
    if lbl == 'cheese':
        REPORT['trap'] = (Cf, Cr)
    print('[trap-%s] capacity full-obs = %.4f | reward-only = %.4f' % (lbl, Cf, Cr))

# ---- condition 3: post-cue (a2=cue, o2 = cue/none/context c) ----
caps_f, caps_r = [], []
for c in (0, 1):
    idx = 6 * 3 + 2 * 0 + c
    sub3 = p[0, 0:2].sum(axis=0)[3, idx]               # (a3,o3)
    W3, _ = cond_o_given_a(sub3)
    Cf, _ = blahut_arimoto(W3)
    Cr, _ = blahut_arimoto(reward_marginal(W3))
    caps_f.append(Cf); caps_r.append(Cr)
    print('[post-cue ctx=%d] capacity full-obs = %.4f | reward-only = %.4f' % (c, Cf, Cr))
    for a, name in enumerate(['center', 'right', 'left', 'cue']):
        Wr = reward_marginal(W3)
        print(f'        a3={name}: p(r|a) = {np.round(Wr[a], 3)}')
REPORT['postcue'] = (float(np.mean(caps_f)), float(np.mean(caps_r)))

# ---- analytic ground truth from the environment rules ----
print('\n---- ground truth (analytic, same conditions) ----')
def gt_start():
    W = np.zeros((4, 24))
    W[0, [0, 1]] = 0.5                       # center: none, ctx random
    for a, pos in [(1, 1), (2, 2)]:          # arms: win/loss 50/50, ctx random
        for r in (1, 2):
            for c in (0, 1):
                W[a, 6 * pos + 2 * r + c] = 0.25
    W[3, [18, 19]] = 0.5                     # cue: none, ctx revealed (uniform prior)
    return W
def gt_trap():
    W = np.zeros((4, 24))
    for a in range(4):                        # absorbing: same obs forever
        for c in (0, 1):
            W[a, 6 * 1 + 2 * 1 + c] = 0.5
    return W
def gt_postcue(c):
    W = np.zeros((4, 24))
    W[0, [0, 1]] = 0.5
    win, lose = (1, 2) if c == 1 else (2, 1)  # r2 = 2-(c1+p2)%2: p2 with (c+p2) even -> win
    for cc in (0, 1):
        W[win if False else 0, 0] += 0        # placeholder no-op
    for cc in (0, 1):
        W[1, 6 * 1 + 2 * (2 - (c + 1) % 2) + cc] = 0.5
        W[2, 6 * 2 + 2 * (2 - (c + 2) % 2) + cc] = 0.5
    W[3, 6 * 3 + 2 * 0 + c] = 1.0             # staying at cue keeps ctx obs
    return W
for name, W in [('start', gt_start()), ('trap', gt_trap()),
                ('post-cue c=0', gt_postcue(0)), ('post-cue c=1', gt_postcue(1))]:
    Cf, _ = blahut_arimoto(W)
    Cr, _ = blahut_arimoto(reward_marginal(W))
    print('[GT %s] full-obs = %.4f | reward-only = %.4f' % (name, Cf, Cr))

print('\nSUMMARY (learned model): start=%.3f/%.3f trap=%.3f/%.3f postcue=%.3f/%.3f (full/reward bits)'
      % (REPORT['start'][0], REPORT['start'][1], REPORT['trap'][0], REPORT['trap'][1],
         REPORT['postcue'][0], REPORT['postcue'][1]))
