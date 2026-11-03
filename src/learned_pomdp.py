"""Active inference on a generative model built from the learned hidden states.

Structure learning (bmr_states.py) groups the histories (a1, o2) of the minimal
maze into predictive states. This script turns those states into a POMDP and runs
the same expected-free-energy planner as efe_analysis/efe_two_step.py on it:

  states  s = Start + one per BMR cluster
  A[o, s] = which observation each state shows (Start shows Null)
  B[s', s, a]:
     from Start:  sum_o p(o2 = o | a1 = a) [state(o) = s']   (learned marginal)
     from s:      sum_o p(o3 = o | s, a2 = a) [state(o) = s']  (pooled BMR counts)
  D = Start,  C = log-preferences over (Cheese, Shock, cues, Null)

state(o) maps an observation to the learned state that shows it. That is valid
when every cluster shows a single observation and every observation belongs to
a single cluster; the script checks this. No context variable is assumed.

Run from the repo root:  python src/learned_pomdp.py [seed]
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'efe_analysis')]
from bmr_states import bmr, min_counts  # noqa: E402
from observable_embedding import MIN, load  # noqa: E402
import efe_two_step as HAND  # noqa: E402

OBS = MIN.OBS + ['Null']
ACT = MIN.ACT
NULL = len(OBS) - 1


# ------------------------------------------------------------ build the model
def learned_model(seed, alpha=0.1):
    names, counts = min_counts(seed, 'model')
    clusters, _, _ = bmr(counts, alpha)
    hist = [tuple(n.split('/')) for n in names]                 # (action, obs)

    # observation -> state; each cluster must show one observation
    state_of_obs, labels = {}, ['Start']
    for k, c in enumerate(clusters, start=1):
        shown = {OBS.index(hist[i][1]) for i in c}
        assert len(shown) == 1, f'cluster {k} shows several observations: {shown}'
        o = shown.pop()
        assert o not in state_of_obs, f'observation {OBS[o]} shown by two states'
        state_of_obs[o] = k
        labels.append('{' + ', '.join(sorted(names[i] for i in c)) + '}')
    nS, nO, nA = len(labels), len(OBS), len(ACT)

    A = np.zeros((nO, nS))
    A[NULL, 0] = 1
    for o, s in state_of_obs.items():
        A[o, s] = 1

    # learned joint p(a1, o2, a2, o3) from the Born rule
    T1, T2 = load(f'seedprot_minimal_s{seed}')
    p = np.abs(np.einsum('xaoi,ibpy->aobp', T1, T2)) ** 2
    p /= p.sum()
    p_o2 = p.sum((2, 3))
    p_o2 /= p_o2.sum(1, keepdims=True)                          # p(o2 | a1)

    B = np.zeros((nS, nS, nA))
    for a in range(nA):
        for o, s in state_of_obs.items():
            B[s, 0, a] += p_o2[a, o]
    for k, c in enumerate(clusters, start=1):
        n = sum(counts[i] for i in c)                           # pooled (a2, o3) counts
        q = n / n.sum(1, keepdims=True)
        for a in range(nA):
            for o, s in state_of_obs.items():
                B[s, k, a] += q[a, o]
    B /= B.sum(0, keepdims=True)                                # drop o3 never shown at t2
    return A, B, labels


# ------------------------------------------------------------ EFE (one factor)
def risk(q_o, C):
    return float(-q_o @ (C - np.log(np.exp(C).sum())))


def infogain(A, q_s):
    q_o = A @ q_s
    ig = 0.0
    for o in np.flatnonzero(q_o > 1e-15):
        post = A[o] * q_s / q_o[o]
        m = post > 1e-15
        ig += q_o[o] * float(post[m] @ (np.log(post[m]) - np.log(q_s[m])))
    return ig


def sophisticated(A, B, C, gamma=16.0):
    """G(a1) = G1(a1) + E_{o1} E_{a2 ~ sigma(-gamma G2)} G2(a2 | o1), as in efe_two_step."""
    q0 = np.eye(B.shape[0])[0]
    rows = []
    for a1 in range(B.shape[2]):
        q1 = B[:, :, a1] @ q0
        q_o1 = A @ q1
        r1, e1 = risk(q_o1, C), infogain(A, q1)
        exp_G2 = 0.0
        for o1 in np.flatnonzero(q_o1 > 1e-15):
            post = A[o1] * q1 / q_o1[o1]
            R2 = np.array([risk(A @ B[:, :, a] @ post, C) for a in range(B.shape[2])])
            E2 = np.array([infogain(A, B[:, :, a] @ post) for a in range(B.shape[2])])
            pi2 = np.exp(-gamma * (R2 - E2)); pi2 /= pi2.sum()
            exp_G2 += q_o1[o1] * float(pi2 @ (R2 - E2))
        rows.append((r1, e1, (r1 - e1) + exp_G2))
    G = np.array([x[2] for x in rows])
    post = np.exp(-gamma * G); post /= post.sum()
    return rows, post


def preferences(cheese, shock):
    C = np.zeros(len(OBS))
    C[OBS.index('Cheese')], C[OBS.index('Shock')] = cheese, shock
    return C


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    A, B, labels = learned_model(seed)
    print(f'Learned generative model, minimal maze, seed {seed}\n')
    for s, lab in enumerate(labels):
        print(f'  state {s}: {lab:28s} shows {OBS[int(np.argmax(A[:, s]))]}')
    print('\n  B[next state | state, action]  (rows: state, then action)')
    for s in range(len(labels)):
        for a in range(len(ACT)):
            nz = {f's{t}': round(float(B[t, s, a]), 3) for t in np.flatnonzero(B[:, s, a] > 1e-3)}
            print(f'    s{s} --{ACT[a]:5s}--> {nz}')

    for name, (ch, sh) in [('paper (+2/-4)', (2, -4)), ('mild (+1/-1)', (1, -1)), ('flat (0/0)', (0, 0))]:
        rows, post = sophisticated(A, B, preferences(ch, sh))
        hrows, hpost = HAND.horizon2_sophisticated(HAND.profile(name))
        hand = {r[0]: (r[2], r[5], p) for r, p in zip(hrows, hpost)}
        print(f'\nPreferences {name}, horizon 2 sophisticated, gamma=16')
        print('  action | info gain (learned / hand) | G (learned / hand) | q(a) (learned / hand)')
        for a, ((r1, e1, g), p) in enumerate(zip(rows, post)):
            he, hg, hp = hand[ACT[a]]
            print(f'  {ACT[a]:6s} | {e1:6.3f} / {he:6.3f}           | {g:7.3f} / {hg:7.3f}  '
                  f'| {p:.3f} / {hp:.3f}')
