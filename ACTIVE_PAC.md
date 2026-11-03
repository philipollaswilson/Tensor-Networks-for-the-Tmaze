# PAC-style guarantees for active structure learning

Follows [MERGE_PAC.md](MERGE_PAC.md), whose Lemmas, Theorems and notation are used as they stand (dF, rho, G, JS_w, kappa, B_M,
BISIM, GROW, Prop 8). Checks: [src/active_pac_check.py](src/active_pac_check.py), **44 of 44 pass, in 77 to 107 s by machine load**, on two threads. Code read:
`Episode`, `future_windows`, `split_round`, `bmr_rows` in [src/kmaze.py](src/kmaze.py), and [src/active_k2.py](src/active_k2.py) (Section 7). From `kmaze.py` the checks call only
`future_windows`, read-only. **The learner itself was not run.** Every number below comes from a small synthetic case.

Status labels as in MERGE_PAC.md: **Theorem** = proved here, from the stated assumptions and cited results. **Sketch** = the argument
is given but a step is not closed. **Conjecture** = supported by simulation at most. **Measured** = a number from a run, no proof behind it.

Sources read for this note. Deb et al. ICLR 2025 (Theorems 2, 3, Section 2.2). Howard et al. 2021 (Definition 1, Theorem 1, Eq. 10).
The link given in the task, arXiv:2302.03421, is **Chugg, Wang, Ramdas, "A unified recipe for deriving (time-uniform) PAC-Bayes bounds"**,
not Howard. Its Theorem 3.1 applies to forward supermartingales on any filtration, so it covers adaptive data. Its bounds built on
reverse submartingales assume exchangeable data, and adaptive rows are not assumed to be exchangeable, so I use none of those. Lemma A5
below is a direct instance of the forward recipe: a mixture, a supermartingale, Ville.

## Main results

1. **Tails are valid exactly under stated conditions (A1).** With a true predictive state, the tail window has the uniform law given the
   start, whatever the prefix was and however the start was chosen, if the start is a stopping time. Action-conditioned rows need only an
   open-loop tail, not a uniform one. **So a designed test sequence is legal for those rows.**
2. **An imperfect label breaks more than the code comment says (A2).** On a label that merges predictive states, one-step counts depend on the
   policy too. The docstring of `Episode` holds only for pure labels. Measured: p(o | label, a) = 0.788 under one prefix and 0.200 under another.
3. **The real `future_windows` has a live problem under active tails (A3c).** An event window is counted only if the change falls inside
   the episode. So its law depends on time left, which the planner now controls. One true state, tails starting at t = 5 and t = 16 of 20:
   false split with probability 1.00. A fixed horizon gives 0.00. MERGE_PAC refuted censoring as a cause under a uniform policy; that
   stands. Under active tails the start time varies by design, so the question reopens, and the answer here is yes.
4. **Adaptive data does not need martingales if rows are truncated (A4).** The first n0 entries of a row are the same multinomial under any
   policy. Theorems 2, 3 and 4 carry over at the same rates, and only coverage changes. Martingales are needed to use all visits and to stop
   on evidence, and the price is about 5x (A5, A6, A9).
5. **Lower bound (A10), and the limit of the gain (A11).** Prop 8 survives with expected counts: active learning lowers the *episodes* to reach n_e, never the
   n_e needed. Distinguishability gains at most A^L in KL, not the A^(2L) that 1/(d mu^2) suggests. Occupancy gains at most about 2T A^D, but on an
   unknown path any learner pays the search, so the gain is capped by min(2n*, A^D) (A11f).
6. **Novelty is not a greedy minimiser of the bound (A13, A14).** It coincides with one only when gaps and supports are equal. It maximises sum_e a(s_e) ln n_e: total
   parameter information. Structure information gain is the exact one-step minimiser of a bound on Bayes error.
7. **The chain rule is exact, and the naive form of it destroys the saving (A17).** With the per-edge term (m - 1)/(2 n_e), the no-weights sum behaves
   like count-based exploration: 1990 steps against 2013, where structure alone takes 310. With term 2 properly conditioned on structure, a decided
   edge shares a pooled row, and the sum takes 310. **The no-weights sum is right; the per-edge novelty term is the wrong term in it.**
   `src/active_k2.py` does not have that form: its novelty is on learned (s, a) rows, so decided merges are already pooled. Section 7 says what does apply to it.

## 1. Setting

A unifilar machine (Q, A, O, tau_next, theta, q0). The history h_t = (a_0, o_1, ..., a_{t-1}, o_t), the true predictive state q_t = g(h_t). F_t = sigma(all past episodes,
h_t, the agent's internal randomness up to t). A *policy* picks a_t from F_t, so it may use all past data. The environment sees only (q_t, a_t):

    P(o_{t+1} = o | F_t, a_t) = theta(o | q_t, a_t).    (*)

Everything below follows from (*). It fails when the machine is not time-homogeneous unless t is part of the state, and when q_0 is unknown unless the first observation fixes it.

Write P_q^unif for the law of a window (a_1, o_1, ..., a_L, o_L) started at q under uniform actions, and P_q^{a^L} for the law of o^L when a^L is played.

## 2. Question 1: tail validity

**Theorem A1 (exact state).** Let sigma be a stopping time of (F_t), let L be F_sigma-measurable, and let the actions a_sigma, ..., a_{sigma+L-1} be chosen as below. On {sigma < infinity, q_sigma = q}:

(a) if they are uniform and independent of F_sigma and of each other, then conditional on F_sigma the window has law P_q^unif;

(b) if they are *open loop*, meaning a^L is F_sigma-measurable or drawn from any F_sigma-measurable law and independent of the tail's observations, then conditional on (F_sigma, a^L), the observations have law P_q^{a^L}.

So the row "observations given the action sequence" is valid under (b), and any statistic that marginalises actions (an event window) is valid only under (a), or under weights that undo the action law.

(c) If actions inside the tail depend on the tail's observations, (b) fails: conditioning on a later action selects on earlier observations.

*Proof.* (a), (b): multiply (*) along the window. Each factor is theta at a state fixed by q and the window so far. No factor involves F_sigma other than through q, and {sigma = t} is in F_t, so conditioning on it adds nothing. (c): P(o_1 | a_1, a_2) is proportional to theta(o_1 | q, a_1) pi(a_2 | o_1), and it equals theta only if pi ignores o_1. QED

No condition is placed on the prefix: it may be adaptive inside the episode and across episodes.

Measured (T1a, T1b): adaptive prefix, data-dependent start, 400,000 windows: chi2 p = 0.29, largest TV 0.025 on the rarest state, as noise predicts. Closed-loop tail: TV 0.454. Open loop, non-uniform: TV 0.0045, chi2 p = 0.39, and the action-marginal row is off by TV 0.039 as (b) says.

**Theorem A2 (imperfect label).** Let the label be l_t = f(h_t), with f fixed before the episode. Let w_pi(q | l) = P_pi(q_sigma = q | l_sigma = l). Then

    P_pi(window | l_sigma = l) = sum_q w_pi(q | l) P_q^unif,
    P_pi(o | l_t = l, a_t = a) = sum_q w_pi(q | l, a) theta(o | q, a).

(a) Both laws are free of pi for all pi iff all q reachable under l share P_q^unif (for the first, at this L) or theta(. | q, .) (for the second), or the weights happen not to move.

(b) TV(law under pi, law under pi') <= TV(w_pi, w_pi') max_{q, q' in l} TV(P_q, P_q').

(c) Impurity does not transfer between policies. If l is 1 - eps pure under one policy, it can be 0 pure under another. There is no bound in eps. The factor A^D of A11(a) bounds a joint probability, not this ratio, because a plan can avoid the main component altogether.

(d) If the planner uses more than the label, then w_pi(q | l, a) differs from w_pi(q | l), and the two rows of one label under two actions are mixed differently.

*Proof.* (a), the first display, (b): total probability on q_sigma, then A1 and convexity. (c): the T1c machine is an example: two predictive states share a label, and a prefix that reaches only the rare one has impurity 1. (d): the weights condition on a, and a then carries information about q. QED

Measured (T1c): impurity 0.02 under the base prefix, 1.00 under the directed one. p(o | l, a) moves from 0.788 (formula 0.788) to 0.200. **The docstring of `Episode` ("one-step counts, whose law is p(o | s, a) under any policy") is true for a predictive s and false otherwise.**

This cuts both ways. A row that moves with the prefix is itself proof that the label is impure: under A2(a), pure labels cannot move. So the shift is evidence for a split, and what is tested is the *item*. In `split_round` items are edges, which is the right unit: an edge row is pure when its source is, whatever the prefix. This is the k = 2 picture: planning to the rare clean edge raises its weight from 0.02 to 1 in that row. It is the same mechanism as T1c, and I did not run it on k = 2.

**Proposition A3 (what stopping and choice do).**

(a) *Theorem.* Choosing the start from anything in F_sigma is safe, including which edge is uncertain and where the plan led. That is A1. Not starting at all ({sigma = infinity}) is also safe.

(b) *Theorem.* An end of episode that depends on the window biases "windows that fit". Rule: dropping a window is safe iff the decision is in F_t at its start. Fixed T qualifies. An end triggered by an observation does not. Pad instead, with an end symbol. Measured (T1e): truth 0.5, kept windows 0.000, end fixed in advance 0.500.

(c) *Theorem, for the statistic. Measured for the split.* `future_windows` counts event d only if the d-th change is inside the episode. Whether it is counted depends on the future, so A1 does not apply, and the row of q started with r steps left has law P_q(event | event within r). This depends on r. Under a uniform policy r is similar for rows being compared. Under active tails it is whatever the plan produced. Measured with the real function (T1d): one true state, tails from t = 5 and t = 16 of 20, n = 100: P(dF < -3) = 1.00. With a fixed horizon of 4 and the value "none": 0.00. The example is built to show the effect. **How much it matters on the mazes is not known.**

(d) *Theorem.* A label must be fixed before the tail it reads. If f is fitted on data that include the tail, then l_sigma is not in F_sigma and A1 and A2 do not apply. Online this means: rows at round k are labelled by f_{k-1}, or by a hold-out. Re-labelling old tails with a model fitted on them gives selection, not evidence. Measured (T1f): 32 items of one true state, split by their own rows: P(dF < -3) = 0.93 on the same data, 0.000 on held-out data. MERGE_PAC Theorem 2 avoids this by defining rows on the true partition, and Theorem 4 by hold-out. `learn_states` does neither.

(e) *Theorem.* Windows that overlap have the right marginal law and are not independent. Windows at sigma, sigma + L, sigma + 2L, ... are conditionally i.i.d. by A1. The overlapping family splits into L such phases, so bounds hold with a factor L on counts or ln L in the union. Measured (T1g): variance of a cell count 69.1 against 33.9, at the same number of windows.

(f) *Theorem.* Each draw being conditionally correct does not make a row at a data-chosen time multinomial. See A8.

## 3. Question 2: an adaptive PAC bound

### 3.1 Rows truncated in advance: no martingale needed

**Theorem A4 (stack, any policy).** In the stack construction of MERGE_PAC Section 4, outcomes are read from independent i.i.d. stacks, one per (v, a). That construction has the law of the process under any policy, because of (*). Rows defined as "the first n0 entries" are functions of the stacks alone. Therefore, for any adaptive policy, any stopping rule, and BISIM(n0) on the first n0 visits,

    P(fail) <= N_E p_S(n0) + sum_pairs p_M(u, v) + P(some (v, a) has fewer than n0 visits),

with p_S, p_M exactly as in Theorem 2. If sampling continues until every needed row has n0 entries the last term is 0. The same holds for Theorem 3 (n0 = 4), Theorem 3b, and Theorem 1 with fixed row sizes.

*Windows and GROW.* Read tails as blocks: at a tail start after edge e, draw one window from an i.i.d. stack indexed by (e, block type), with law P^unif of tau(e). By A1 this has the law of the process, for non-overlapping windows. So Theorem 4 holds under adaptive plans, with rows "the first n0 windows after edge e", a union over edge pairs in place of K (K A m), and **without the odd/even split**: rows of different edges come from different stacks, hence are independent. Prefix steps use the one-step stacks.

*Proof.* The policy decides which stack is read next. Each entry is independent of the fact it was reached, since the entry is fixed before the policy acts. The failure events of Theorem 2 are functions of the first n0 entries, so their probabilities do not change. QED

So under adaptive collection **the polynomial wrong-split rate, the log wrong-merge rate, and the vacuous constants are all unchanged**, and only the coverage time changes. What this does not cover: using all visits, n0 chosen from the data, and stopping a pair early once it looks decided. Since n0 must be set from the worst gap, truncation wastes what adaptivity could save. Measured (T2c): P(dF < 0) at n0 = 30, 100, 300, adaptive reading 0.0902, 0.0414, 0.0221, against i.i.d. rows 0.0923, 0.0404, 0.0203.

### 3.2 Using all visits: anytime bounds

Rows now have sizes n_e(t) that depend on the data. Below, alpha is fixed in advance. `Rows` refits alpha = 1/m' on the table; a union over m' <= m_max costs ln m_max, and A17(a) shows why the refit matters.

**Lemma A5 (anytime, equal laws). Theorem.** Let rows x, y have the same true law p, under any policy, with any interleaving. Then M_t = DM(x_t) DM(y_t) / p^{z_t} is a nonnegative supermartingale with M_0 = 1 (a martingale if p has full support), and

    P(exists t: ln M_t >= ln(1/delta)) <= delta,
    P(exists t: -dF_t >= rho(z_t) + ln(1/delta)) <= delta.

*Proof.* DM(x)/p^x = E_{theta ~ Dir}[prod (theta/p)^x], and by (*) each factor has conditional mean sum_{p > 0} theta <= 1 at the visits of that row. Mixing preserves the property (Tonelli), and rows update at different times, so the product is a supermartingale. Ville gives the first line. For the second, ln DM(z) = max_theta ln theta^z - rho(z) >= ln p^z - rho(z), so -dF_t <= ln M_t + rho(z_t). QED

**So a threshold that grows like rho(z), about kappa ln n, gives a wrong-split guarantee at every time at once, with ln(1/delta).** The fixed thresholds of `bmr_rows` do not have this property (A8).

**Lemma A6 (anytime, different laws). Theorem.** Let rows x, y have laws p, q, under any policy. Fix lambda in (0, 1), let D_a be the Renyi divergence of order a, and set

    J_lambda(w) = min_theta [ w D_{1-lambda}(p || theta) + (1 - w) D_{1-lambda}(q || theta) ].

With probability at least 1 - delta, for all t, with w_t = n1/n:

    dF_t <= rho(x_t) + rho(y_t) + ((1 - lambda)/lambda) rho(z_t) + (1/lambda) ln(1/delta) - n_t J_lambda(w_t).

J_lambda rises to JS_w as lambda -> 0, and J_lambda is about (1 - lambda) JS_w locally. If the supports are disjoint the event is impossible once C(n, n1) > e^tau (Lemma 3 holds for any rows).

*Proof.* Let c(r, theta) = -ln sum_o r^(1-lambda) theta^lambda and f(theta) = theta^z/(p^x q^y). By (*), f(theta)^lambda exp(n1 c(p, theta) + n2 c(q, theta)) is a martingale for each theta, for any interleaving. Mix over theta ~ Dir(alpha) to get a mean-1 martingale V_t. Since n1 c(p, theta) + n2 c(q, theta) >= n lambda J_lambda(w) for all theta, Ville gives ln integral f^lambda <= -n lambda J_lambda(w) + ln(1/delta), uniformly in t. Let A = ln integral f = ln DM(z) - ln p^x q^y, so f <= e^(A + rho(z)). Then integral f <= (max f)^(1 - lambda) integral f^lambda gives lambda A <= (1 - lambda) rho(z) - n lambda J_lambda + ln(1/delta). Finally dF <= A + rho(x) + rho(y), as ln DM(x) >= ln p^x - rho(x). QED

This is the anytime version of Lemma 5, at the same rate family. Note that w may be adaptive: the min over theta removes the need for a fixed r.

**The wrong-merge side of the real pair rule is covered.** A6 needs no change to the rule: with sigma = tau it says that once n J_lambda exceeds the right side plus tau, the veto fires at all later times. And by Theorem 1(c), one modality is enough, under any dependence. Only the wrong-split side needs a new threshold.

**Proposition A7 (language tests, Deb form). Theorem, using Howard et al. 2021.** For a test f in [0, 1] on non-overlapping tail windows of a pure row, S_n = sum (f - E f) is sub-Gaussian with variance process n/4, in the read order of the stack. By Theorem 1 and Eq. (10) of Howard et al., with eta = 2.04, s = 1.4, m = 1, l(v) = s ln(ln(eta v)/ln eta) + ln(zeta(s)/alpha) and k1 = (eta^(1/4) + eta^(-1/4))/sqrt 2 = 1.437,

    P(exists n: S_n >= k1 sqrt(max(n/4, 1) l(max(n/4, 1)))) <= alpha.

With |X| tests, Deb's condition becomes min(n1, n2) >= c k1^2 (ln(|X|/delta) + s ln ln n)/mu^2. The anytime price over the fixed-n sub-Gaussian boundary sqrt(2 v L) is k1^2/2 = 1.03 on the leading term, plus the ln ln n. This is far less than A6 pays, because a Hoeffding test is already crude.

Measured (T2a), adversary freezes the row that looks extreme, delta = 0.05: P(ever ln M >= ln(1/delta)) = 0.025 (m = 2), 0.035 (m = 4), so Ville is close to tight on M itself. P(ever -dF >= rho + ln(1/delta)) = 0.015, 0.0007. On the same data a 5% z test crosses with probability **0.625**, fixed-n Hoeffding **0.157**, stitched 0.000.

**Conjecture A8 (fixed threshold, data-dependent time).** For s = 2 on equal rows, P(exists n >= n0: dF_n < 0) <= C n0^(-kappa) up to slow factors, so a bounded multiple of the fixed-n rate. I have no proof. The peeling route needs a bound on sup n J^ near G ~ kappa ln n; the mixture of A5 gives rho(x) + rho(y), about twice G, which is not enough. Measured (T2c) over n0..8n0: 0.261, 0.149, 0.090 at n0 = 30, 100, 300; 2.8x, 3.7x, 4.4x the fixed-n rate; times sqrt(n0): 1.43, 1.49, 1.57. **The ratio is rising**, slowly, and the n0 = 300 line is tested every 10 draws, so it is an underestimate. The rate form fits; a constant factor is not shown.

### 3.3 Exact recovery

Use BISIM of Theorem 2 with fixed alpha, all visits, and the single-modality rule

    merge allowed <=> for every action a: dF_a >= -sigma_a,    sigma_a = rho(z_a) + ln(1/delta_S).

Let delta_S = delta/(2 N_E A) and delta_M = delta/(2 N_P), with N_P the number of separated pairs.

**Theorem A9 (exact recovery with realised counts).** With U a unifilar refinement as in Theorem 2, under any policy, with probability at least 1 - delta the following holds at all times t simultaneously, so at any stopping time. If every separated pair (u, v), at its separating round, has an action a for which the rows x, y over (o, block) satisfy

    n J_lambda(w) > need(x, y) := rho(x) + rho(y) + rho(z)/lambda + (1/lambda) ln(1/delta_M) + ln(1/delta_S),

then BISIM returns exactly the minimal machine. Lambda may depend on the pair, in advance.

Scaling, with rho about kappa ln n: for a pair of gap J and balanced rows,

    n* ~ [ kappa (2 + 1/lambda) ln n + (1/lambda) ln(1/delta_M) + ln(1/delta_S) ] / ((1 - lambda) J).

The ln(1/delta) coefficient has minimum (1 + lambda)/(lambda (1 - lambda)) = 5.83 at lambda = 0.414 when the two delta are of one size. **Unlike Theorem 2, the wrong-split term is logarithmic: polynomial in 1/delta has become ln(1/delta).** That is the gain from a threshold which grows, as MERGE_PAC conjectured; the price is the factor above, and a rule that is not the code's.

*Proof.* Events: A5 for each emission-equal pair and action at delta_S; A6 for each separated pair at its separating law and action at delta_M. The Moore partitions are fixed by the true machine, so these laws are deterministic and are relabellings of the rows of Theorem 2. Off these events: equal pairs are never split at any t. A separated pair satisfying the condition has dF_a < -sigma_a at its round. Induction on rounds as in Theorem 2. QED

**Corollary (episodes). Theorem.** If some plan traverses row e with conditional probability at least p_e per attempt given the past, then with probability at least 1 - delta the rows reach n*_e within

    N <= sum_e ( 2 n*_e + 8 ln(E/delta) ) / p_e.

*Proof.* The number of successes dominates Bin(N_e, p_e), by coupling attempts in order. Chernoff and a union over E rows. This serves one row per attempt, so it is loose whenever one path serves many. QED

The best any mixture of policies can do, given n*, is the design value

    N_design = min_{xi in Delta(policies)} max_e n*_e / d_e^xi,    against    N_fixed(pi) = max_e n*_e / d_e^pi.

Measured (T2d), ladder of 7 states, N in episodes, 6 seeds. All three are exact at the certificate.

| | Uniform | Count-directed, no gaps known | Pair-directed, gaps known |
|---|---|---|---|
| N at certificate (A9) | 51,200 | 5,625 | 1,925 |
| Gain | | 9.1x | 26.6x |
| Same rule first right on the grid (doubling) | 3,200 | 800 | 200 |
| Gain | | 4x | 16x |
| Certificate loose by | 16x | 7x | about 10x |

Tightness of A6 alone (T2b): certified at n = 1007 and 193, last accepted at n = 202 and 36: **about 5x**. All A6 violations: 0 of 600.

### 3.4 Lower bound for adaptive learners

**Theorem A10.** Let M, M' be machines, and let an algorithm with any adaptive policy and any stopping time stop with an answer that is right with probability at least 1 - delta on both. Then

(a) E_M[ sum_{episodes i} KL(P_M^{pi_i} || P_M'^{pi_i}) ] >= kl(delta, 1 - delta) >= ln(1/(2.4 delta)).

(b) Prop 8 for adaptive learners. If M, M' differ only in the target of edge e, and KL_e* is the largest KL per traversal over what follows, over all continuations up to the next traversal or episode end, then

    E_M[n_e] >= ln(1/(2.4 delta)) / KL_e*.

(c) Characteristic time. With Alt(M) the machines of different minimal structure,

    E_M[N] >= ln(1/(2.4 delta)) / sup_{xi} inf_{M' in Alt(M)} E_{pi ~ xi} KL(P_M^pi || P_M'^pi).

(d) Budget. Each step traverses one edge, so E[N] T >= sum_e ln(1/(2.4 delta)) / KL_e*.

*Proof.* (a) Chain rule for KL of the adaptive data law, then data processing on the event "answer M"; pi_i is F-measurable before episode i. This is the argument of Kaufmann, Cappe, Garivier 2016 (Lemma 1), and kl(d, 1 - d) >= ln(1/(2.4 d)) is checked numerically here on (0, 1/2), smallest margin 0.007. (b) Segments between traversals: data laws agree outside them, and each adds at most KL_e*. (c) Let xi be the empirical mix of pi_i, weighted by expected use. (d) Sum (b) over edges. QED

**Active learning lowers the episodes to reach n_e. It never lowers the n_e that is needed, beyond the choice of what follows the edge.** With a matched test (SPRT) on one row, E[n] / floor is 1.13 to 1.26 over six cases (T2e), so the floor is tight within about 25% in the simplest case. The gap to A9 is about 5x at the pair, plus Moore rounds. No matching upper bound for whole machines is claimed.

### 3.5 What depends on the policy, and the most that can be gained

| Term | Deb et al. Thm 3 | Here | Depends on the policy? |
|---|---|---|---|
| Occupancy | d*_m, with C* | n_e realised, or d_e^xi | **Yes** |
| Concentrability | C* = max d*/d^b | none | Offline term: target-versus-behaviour mismatch |
| Distinguishability | mu_0 of (R, pi), Section 2.2 | gamma, J of the row | **Yes** for windows. **No** for one-step JS given U |
| Confidence | ln(1/delta) | ln(1/delta) | No |
| Size | ln|X|, or H | ln N_P, kappa ln n | No |
| Rule | Hoeffding threshold | dF threshold | No |
| Validity | i.i.d. episodes, regular pi | any policy, by (*) | **Different**: their proof needs i.i.d. episodes |

**Theorem A11 (limits of the gain).** A actions, machine fixed, pi any history-dependent policy.

(a) P^pi(E) <= A^t P^unif(E) for any event E on t steps.

(b) *Distinguishability.* KL_L^pi(u || v) <= A^L KL_L^unif(u || v), for closed-loop pi, and this is tight. Open loop, KL_L^unif is exactly the average of KL over A^L sequences.

(c) *Occupancy.* If e is hit within D_e steps with probability at least half its best hit probability, d_e^pi / d_e^unif <= 2 T A^(D_e).

(d) *Budget.* sum_e d_e^pi <= T, so min_e d_e^xi <= T/E for every xi, and N_design >= sum_e n*_e / T. If uniform already has d_e near T/E with equal n*, there is no gain.

(e) *No agency.* If transitions and emissions do not depend on actions, d_e^pi is the same for all pi and the gain is 1.

(f) *Unknown path.* Take a lock of depth D with absorbing failure and a uniform secret. Each episode tests one sequence, so any learner has E[first hit] >= (A^D + 1)/2. So E[N] >= (A^D + 1)/2 + n* - 1, while uniform takes n* A^D. **The gain against uniform on this class is at most min(2 n*, A^D)**, against the A^D of (c).

*Proof.* (a) Each deterministic-action factor is at most 1 = A (1/A). Randomised pi is a mixture. (b) Chain rule: KL_L^pi = sum_{t<L} E_u^pi[f_t] with f_t >= 0 a function of (h_t, a_t), which holds t + 1 actions; apply (a). A lock whose bit shows under one action gives equality. (c) d_e^unif >= P^unif(hit within D_e) >= A^(-D_e) rho_e/2 with rho_e the best hit probability, and d_e^pi <= T rho_e. (d) One edge per step. (e) No dependence on a. (f) Random secret: distinct guesses, Yao. Uniform: negative binomial. Search-then-repeat attains the bound. QED

Measured: (b) on 120 random machines L = 3, largest KL*/KL_unif = 3.41, cap 8; lock D = 3: 16.000, exactly the cap. Hit ratio at most 4.48, cap 8 (T2f). (f): D = 6, n* = 50: 3200 against 81, gain 39x, cap 64; D = 10: 91x, cap 100; D = 6, n* = 5000: 64x, cap 64 (T2h).

**Corollary A12 (the form 1/(d mu^2) overstates the test gain).** On a lock, with L_inf^p and a uniform tail, mu_0 = A^(-L) g. So raising mu_0 to g by a designed test looks like A^(2L) in 1/(d mu^2). By A11(b) the information gain is A^L. The extra A^L belongs to Hoeffding on a rare event, as MERGE_PAC's rare event remark says. A Bernstein or KL test does not have it.

Measured (T2g), D = 3, L = 3, g = 0.4, SPRT on the informative episodes:

| | Uniform prefix, uniform tail | Directed prefix, uniform tail | Directed prefix, open-loop test |
|---|---|---|---|
| Episodes | 600 | 76 | 9 |
| Gain from the right column | 64x = A^(D+L) | 8x = A^L | 1 |
| Predicted by 1/(d mu^2) | 512x | 64x | 1 |

Three caveats. **The tail keeps its cost.** Under a uniform tail only d improves. The A^L needs the open-loop tail of A1(b), which event windows cannot use. **A test must be known.** Finding which sequence distinguishes costs what (f) says. **mu_0 is a minimum over pairs.** A test for one pair says nothing for others, so a policy-wide mu_0 gains less than one pair suggests, while an adaptive learner can run pair-specific tests.

## 4. Question 3: is novelty a greedy minimiser of the bound?

**Proposition A13.** For Dirichlet(c) with c0 = sum c, and p = c/c0,

    EIG(c) = H(p) - psi(c0 + 1) + sum_i p_i psi(c_i + 1)    (exact. Theorem)

(a) *Expansion. Sketch.* For s cells well populated and m - s empty, each with prior alpha,

    EIG ~ a(s)/n,    a(s) = (s - 1)/2 + (m - s) alpha (ln(1/alpha) + psi(1 + alpha)).

So (m - 1)/(2n) holds only for full support. A deterministic row at m = 8 has 0.42 of it. The steps are Stirling for psi; I did not bound the remainder. Measured (T3a): largest relative error 0.008 on 21 cases. *My first version had -gamma in place of psi(1 + alpha). The simulation rejected it, with error 0.84.*

(b) *Allocation. Sketch.* Greedy EIG equalises a(s_e)/n_e, so n_e is proportional to a(s_e). **It does not depend on any gap.** Measured (T3b): shares 0.178, 0.206, 0.264, 0.352; a(s) gives 0.178, 0.207, 0.264, 0.351; bare (s - 1) would give 0.071, 0.143, 0.286, 0.500. *My first claim was (s - 1), and it failed for the same reason.* `Rows` sets alpha = 1/m' from observed columns, so s = m' and bare (s - 1)/2 applies there, but then a row with one column has novelty exactly 0.

(c) *What it is greedy for. Theorem.* By the chain rule, sum_{k<n} EIG = I(o^n; theta_e), about a(s_e) ln n. So greedy novelty maximises sum_e a(s_e) ln n_e, whose optimum is n_e proportional to a(s_e). It is the expected one-step fall in sum_e h(theta_e | D). This is D-optimal design: total volume of the parameter posterior.

**Theorem A14 (where they agree and differ).** Consider routing, as in Prop 8: edges e with known candidates A_e, B_e, with R_e the route, and posterior pi_e.

(a) I(o; R_e | D) = JS_{pi_e}(A_e, B_e), and it is exactly the expected one-step fall of sum_e H(R_e | D). This sum bounds the Bayes error, as P(wrong) <= sum_e H(R_e | D) / (2 ln 2), since h(eps) >= 2 eps ln 2 on [0, 1/2]. **So structure information gain is the exact greedy minimiser of a bound on the failure probability.**

(b) As pi -> 1, JS_pi(A, B) ~ (1 - pi) KL(B || A). It has the form exp(-n KL) KL, the form of the greedy step on Phi = sum_e exp(-n_e C_e). That greedy equalises n_e C_e, so n_e is proportional to 1/C_e.

(c) If all C_e are equal and the rows have one law and one support, then count, static Phi and novelty pick the same edges, up to the fluctuation of EIG in row counts. If not, novelty allocates by a(s) and the bound by 1/C_e. The loss is unbounded in K: with one hard edge, novelty needs K n*_hard where the bound needs n*_hard + (K - 1) n*_easy.

*Proof.* (a) Definition of mutual information and of JS. (b) Expand. (c) Phi is symmetric and Schur-convex in n when C is constant, so balancing minimises it, and EIG is decreasing in n. The second part is arithmetic. QED

**Novelty is a greedy minimiser of the bound only when it cannot tell edges apart anyway.** It sees n and s. It does not see the gap.

Measured (T3c, d, e). All policies share one stopping rule (every posterior error below delta/E), 100 seeds, delta = 0.05, no policy barred from a decided edge except `elim`. Steps:

| Case | Count | Novelty | Static bound | Elimination | Structure EIG | Sum, per-edge term | Sum, term 2 given structure |
|---|---|---|---|---|---|---|---|
| 8 edges, equal gaps, g = 0.4 | 232 | 234 | 232 | 115 | 117 | 194 | 117 |
| 1 hard (g = 0.1), 7 easy (g = 0.9) | 2013 | 2143 | 346 | 296 | 310 | 1990 | 310 |
| Equal gaps, 4 with 16 outcomes, noise carrying nothing | 232 | **1621** | 232 | 115 | 117 | 1309 | 114 |

Errors are 0.00 to 0.06 across the table. With 100 seeds I would not rank them on that.

In the third case novelty gives the noisy rows 381 samples and the clean ones 24, with equal evidence per sample. That is the noisy-TV form Dirichlet EIG does have: it decays, but its allocation follows outcomes, and novelty alone takes 7x **as long as count**.

## 5. Question 4: when does structure uncertainty beat novelty?

**Theorem A15 (exact gain on routing).** With known candidates and a posterior threshold, read outcomes from stacks and let tau_e be the hitting time of edge e's evidence walk. With the same stacks,

    N_elim = sum_e tau_e,    N_count in (K (tau_max - 1), K tau_max].

So N_count/N_elim is about K tau_max / sum_e tau_e, and as delta -> 0 (Wald, tau_e ~ ln(1/delta_e)/KL_e),

    N_count / N_elim -> K max_e(1/KL_e) / sum_e (1/KL_e).

Also N_count >= K ln(1/(4 delta))/KL_hard for any test at equal counts (Prop 8(b) per edge).

*Proof.* Each edge reads its own stack, so tau_e does not depend on the order of reading. Elimination reads each stack to tau_e. Round robin reads every stack up to the largest. The limit is the strong law. QED

A process where it saves samples is 1 hard edge with K - 1 easy ones. Measured (T4a): count 2013 >= floor 642, elimination 296, structure EIG 310: **gain 6.8x against 6.5x for structure, limit 7.6x**. Most of the saving is **stopping on evidence**, not the information score: elimination, which has no score, does as well. Structure EIG reproducing elimination is measured, not proved. Also online, with rows unknown and pair posteriors (T4e, 8 rows, stop at P outside (0.1, 0.9)): count 6302, structure 3665, both with exact recovery at 0.90 or above on 30 seeds. See the warning in Section 9.

**Proposition A16 (where it does not help).**

(a) *Equal gaps. Theorem.* By A15, gain = tau_max / mean tau, which goes to 1 as delta -> 0. It stays above 1 at finite delta because walks spread. Measured (T4b): 2.01x, identity predicts 1.98x; by the identity 1.81x, 1.58x, 1.40x at delta_e = 1e-3, 1e-6, 1e-12. *My first cap, ln(K/delta)/ln(1/(2.4 delta)), was a wrong explanation of this number and is withdrawn. Convergence to 1 is slow.*

(b) *No agency. Theorem.* If d_e^pi is the same for all pi (A11e), or one path serves all edges, the counts cannot be allocated.

(c) *Undiscovered edges. Theorem, by construction.* A merge exists only between things seen. On an edge never traversed I(o; S) has no candidate, so a structure-only score on the known machine never goes there; EIG at n = 0 is large. Measured (T4d): 1.00 against 0.00. **Term 2 is needed for discovery.** This is the case for keeping both.

(d) *Equal rows. Measured, with a sketch.* If rows are truly equal, dF grows as kappa ln n, so 1 - P ~ n^(-kappa), and predictive rows differ by order 1/sqrt n. For s = 2: I(o; S) ~ n^(-1/2) n^(-1), so I(o; S) / I(o; theta | S) ~ n^(-1/2). Measured (T5h): 0.0273, 0.0118, 0.0038, 0.0010 at n = 300 to 300,000, consistent. Late on, merges that are true stay "open" at a cost no plan will pay. This is the polynomial wrong-split rate seen from the other side.

## 6. Question 5: the chain rule

    I(o; S, theta, q) = I(o; S) + I(o; theta | S) + I(o; q | theta, S)

**Theorem A17(a).** The sum is exact when one posterior over (S, theta, q) is used and the same o is meant in all terms. The merge posterior 1/(1 + exp(-dF)) is exactly that posterior for two rows with a fixed prior: the Bayes update of P by the predictive equals the P from the updated dF.

*Proof.* Chain rule; DM is a sequence marginal, so the predictive is its ratio. QED

Measured (T5a): against Monte Carlo on the joint, within 3.16 s.e. over 8 tables, and Bayes consistency to 1e-15. **With alpha = 1/m' refitted on a new column, the same update is off by 0.207.** The posterior is then not a martingale, and I(o; S) does not predict the update it scores.

So the question is which approximation breaks which term. In order of damage on these cases:

**(h) Term 2 conditioned on the wrong thing. Sketch for the formula, measured for the size. This is the main finding.** Correctly,

    I(o; theta | S) = sum_S P(S | D) EIG(row of o under S).

For a decided merge the row of o is the pool, so the term is about (m - 1)/(2 n_pool), and not (m - 1)/(2 n_e). Using the per-edge term is conditioning on "split" after the data have said "merge". Then the sum collapses to count. Greedy picks the hard edge over an easy one only if EIG(n_h) + I_S >= EIG(n_e), so at the last hard pick

    n_easy >= n_hard / (1 + 2 n_hard I_S),   I_S ~ (delta/K) KL at the end.

With I_S that small, n_easy approaches n_hard, and N approaches K n_hard. This uses EIG ~ 1/(2n) and I do not bound the remainder. Measured (T4c): prediction about 255, measured easy 245, hard 273 (`sum`); easy 4.5, hard 278 (`chain`). **Steps: count 2013, per-edge sum 1990, conditional sum 310, structure alone 310.** On noisy rows the per-edge sum is 11x the conditional one (1309 against 114).

Limits of this result. `chain` here sets term 2 from pools that are large and known, and with no "new state" hypothesis. With such a hypothesis it gains a term P(new) (m - 1)/(2 n_e), which decays with P(new). I did not test that. On unknown rows the online version is unstable: see Section 9.

**(c) MAP structure in term 2. Measured.** Replacing the mix by the MAP structure: 19% error at P = 0.73, 18% at 0.46, 0% once decided. The per-edge term: 49%, 76%, 81% at P = 0.73, 0.88, 0.93 (T5c).

**(b) The state term, plug-in or conditional. Theorem.** Let X, Y be two of the unknowns.

    [I(o; X | Y) + I(o; Y | X)] - I(o; X, Y) = I(o; X | Y) - I(o; X) = I(X; Y | o) - I(X; Y).

If X, Y are independent under the posterior (mean field), this is >= 0: computing both terms conditioned on the other **always overcounts**. Each order alone is exact. Also, with q independent of theta and p-bar the posterior means, I(o; q | theta = p-bar) = I(o; q), since the predictive is p-bar. So

    I(o; q | mean theta) + I(o; theta | q)   is exact, in the order q then theta.

So the usual "salience at the mean plus novelty given the state" is a valid chain, in the order S, q, theta. The stated order S, theta, q has term 3 = E_theta I(o; q | theta) >= I(o; q | mean theta), since mutual information is convex in the channel, and term 2 without q. **Use S, then q, then theta: all three terms are then closed form.** Measured (T5b): matches the joint within 1.38 s.e. Mixing the orders overcounts, never under in 12 tables, by up to 0.046 nats, 49% of the joint.

**(d) Pair terms in place of one structure. Theorem for the form, measured for the size.** Pairs share rows. sum_i I(o; S_i) - I(o; S_1..K) = TC(S) - E[TC(S | o)], with TC the total correlation, so either sign is possible. Pair posteriors need not be a joint law: transitivity is not enforced. The prior differs too: equal priors on partitions give P(i ~ j) = Bell(K - 1)/Bell(K): 0.4 at K = 3, not 1/2. Measured (T5d), 3 rows, 800 tables: sum of pairs / exact: median 1.41, range 0.63 to 6.15. Pair posterior against exact marginal: mean gap 0.082, largest 0.228.

**(e) Overlapping windows in dF. Measured.** Depth 1, 2, 3 windows reuse observations, so summed dF counts evidence up to three times, and 1/(1 + exp(-dF)) is overconfident. Expected calibration error: one modality 0.004, summed 0.157, summed and divided by 3: 0.113 (T5e). **Dividing by the overlap does not fix it.** Use one modality for the posterior, even if the veto reads all of them. A rule for the merge can be useful without being a posterior.

**(f) Steps of a plan. Theorem.** I(o_1, o_2; X) = I(o_1; X) + I(o_2; X | o_1), so per-step gains need beliefs updated along the rollout. Without that the sum passes H(S). Measured (T5f): at k = 27, sum 2.222 against exact 0.653 and the cap ln 2.

**(g) Which state. Theorem.** On a unifilar machine q_t is a function of h_t, so I(o; q_t | theta, S) = 0, for any model of known structure. If the term names q_{t+1}, as a POMDP with o emitted from the next state does, then I(o; q_{t+1} | h, a) = H(q_{t+1} | h, a) when the next state is a function of o: it is the entropy of a coin, for ever, with nothing to learn. Measured (T5g): 0 against 0.693 nats, against novelty 0.005. So term 3 must name the state that emitted o, and on a learned machine it is 0 unless the start is unknown or the machine is not unifilar. **The worth of a cue then sits in planning depth, not in term 3**: the secret is merged away into "knows left" and "knows right", and the transition to them is the coin.

**Also, the two windows.** The chain rule needs one o. Term 1 scored on an L-step window and term 2 on one step are different variables. If term 1 is a window gain, adding it at every step counts each observation up to L times: (f) again.

**What is left of "no weights".** All of it, as an identity. In practice it is a sum over one joint posterior, in the order S, q, theta, with term 2 mixed over structure, term 3 on the emitting state, beliefs updated in rollouts, and a prior fixed in advance. Each of these has a closed form. Under those conditions the unit "nat" is earned. Otherwise the terms are not the terms of the identity.

## 7. What this says about src/active_k2.py (read, not run)

`results/active_k2` holds no results yet, so nothing here is a statement about its outcome.

| Part of the code | Finding | Status |
|---|---|---|
| Tail at fixed t = 12 of 20, uniform | A1 holds on pure states: the start is a stopping time | Applies |
| `novelty` on learned (s, a), alpha = 1/O fixed | Term 2 at the MAP structure: A17c, not the collapse of A17h. A rare clean edge merged into a large state has almost no novelty, as the row it reads is the wrong pool | Applies |
| Same | Deterministic rows keep a floor (m - s) alpha (ln(1/alpha) + psi(1 + alpha)) / n, and allocation follows a(s), not gaps | Applies |
| `structure`: h(p_merge) / (n_e + 1) | Not I(o; S). A share of the entropy, not an information. So **`combined` is not the sum of the chain rule, and "nats, no weights" does not earn the identity** | See A18 |
| `p_merge = expit(margin)`, margin from the summed veto | Not a posterior: sums overlapping windows (A17e), uses Rows' refitted alpha (A17a), and mixes a veto distance with an evidence sum | Applies |
| `H[e] = max` over partners | Avoids the pair overcount of A17d | Good |
| DP in `plan`: fixed v, 12 steps, REPLAN = 50 | Sums per-step gains with no belief update in the rollout: a plan looping on one row is credited each time (A17f). Bounded by how much loops are possible | Applies; size unknown |
| One-step rows from the prefix | Valid on pure states; policy-dependent on impure labels (A2). This is the case the arm targets | Applies |
| First batch has tail 0, later batches tail 12 | Windows from the first batch have up to 20 steps left, later ones at most 8, and both enter one row. So event rows mix censoring levels by item: A3c applies, in a milder form than T1d | Applies; size unknown |
| `learn_states` refits on all data, tails included | A3d. Same in offline runs. It reads tails as evidence after they have shaped the labels | Applies |

**Proposition A18 (the proxy). Measured.** h(pi)/(n_e + 1) in place of I(o; S) in the routing setting, same stopping rule, 100 seeds. Steps, against structure EIG: equal gaps 132, 117; one hard edge **307, 310**; noisy rows 132, 117 (T3f). So it is not an information, and on this setting that costs at most about 13%. Its nats cannot be added to term 2 as the identity claims, though with pooled novelty that made no difference here. The routing setting has known candidates and a calibrated posterior, which the real `p_merge` is not, so this says nothing about how the real margin behaves.

## 8. What this suggests (none of it tested on a maze)

Listed by the size of the effect on the synthetic cases.

1. Term 2 mixed over structure. A decided merge reads the pool (A17h).
2. Stop sampling on decided edges. Most of the saving is here (A15).
3. Event windows with a fixed horizon, and a value "none" (A3c). The first thing to check on the real learner, given a false split of 1.00 in the built example.
4. Tails: label from the previous round's model (A3d), and consider open-loop tests for action-conditioned rows (A1b, A11b).
5. A fixed alpha, at least for scoring (A17a).
6. Keep novelty for unseen edges (A16c). Its term is what finds them.

## 9. Numerical checks

`python src/active_pac_check.py`: **44 of 44, 77 to 107 s by load**, seeds fixed, two threads. Several passed only after a claim or a check was changed.

| Claim | Result |
|---|---|
| A1 | chi2 p 0.29; closed loop TV 0.454; open loop 0.0045 |
| A2 | 0.788 against 0.200, formula 0.788 |
| A3b, c, d, e | 0.000 against 0.500; 1.00 against 0.00; 0.93 against 0.000; variance 69.1 against 33.9 |
| A4 | Adaptive and i.i.d. agree: 0.0902, 0.0923; 0.0414, 0.0404; 0.0221, 0.0203 |
| A5 | Crossings 0.025, 0.035 for M; 0.015, 0.0007 for -dF; delta = 0.05 |
| A6 | 0 violations; about 5x loose |
| A7 | Stitched 0.000; z test 0.625; Hoeffding 0.157 |
| A8 (conjecture) | Ratio to fixed n rises: 2.8x, 3.7x, 4.4x |
| A9 | Exact at certificate 1.00 in all modes; 7x to 16x loose |
| A10 | Ratio to floor 1.13 to 1.26 |
| A11 | Largest 3.41 of 8; lock 16.000 of 16; 4.48 of 8; unknown lock 39x, 91x, 64x against caps 64, 100, 64 |
| A12 | 64x and 8x measured, 512x and 64x from the form |
| A13 | Error 0.008; shares match a(s) to 0.001 |
| A14 | Section 4 table |
| A15 | Gain 6.8x, limit 7.6x, floor holds |
| A16 | 2.01x against 1.98x; 1.00 against 0.00; ratio falls to 0.0010 |
| A17 | Section 6 |
| A18 | 307 against 310; 132 against 117 |

**Corrections made because a run disagreed.**
(i) My first routing comparison barred structure, bound and sum from decided edges, and let count and novelty keep sampling them. That bar is itself structure knowledge, and it hid the collapse of the per-edge sum, which came out at 281 steps. Fixed by design: only `elim` has the bar, as its stated rule. **All Section 4 numbers are from the fixed version.**
(ii) The EIG expansion had -gamma; corrected to psi(1 + alpha).
(iii) Allocation by (s - 1) was wrong on sparse rows; corrected to a(s).
(iv) The A16a cap was a wrong explanation; replaced by the identity in A15.
(v) The first adversary in T2a froze a row at n < 5, where no boundary can be crossed, so every test passed at 0.000 and showed nothing. Replaced.
(vi) The first T2d uniform figure sat at the episode cap, so its gain was censored. It is now uncapped, and the limiting pair proved to be (3, 6), not (4, 5). T4e was also censored at its cap; lifted.
(vii) The T2h cap was min(n*, A^D); the exact value gives min(2 n*, A^D).
(viii) T5h first failed on noise in a median of 200 at n = 30. I raised the sample and compared from n = 300, where the claim is made.
(ix) Dividing summed dF by 3 was my candidate fix. It does not calibrate (0.113).

**Warning on T4e (online, unknown rows).** Medians move with the number of seeds, in steps of 5 draws, at 30 to 40 seeds, for the same code path:

| Seeds | Count | Structure | Sum, per-edge | Sum, soft-pooled |
|---|---|---|---|---|
| 40 | 7012 | 3762 | 4448 | 4022 |
| 30 | 6302 | 3665 | 6272 | 3880 |

Only "structure takes about 0.55x of count" is stable. **The order of the two sums on unknown rows is not established.** The pooled count there is a mean-field soft count, not term 2 proper, and exact recovery of 0.90 to 0.97 has a 2 s.e. of about 0.2.

## 10. What is proved, and what is not

| Claim | Status |
|---|---|
| A1: tail law given a true state; open loop for action-conditioned rows | Theorem |
| A2: mixture law on impure labels, one-step included | Theorem |
| A3a, b, d, e, f | Theorem |
| A3c: event windows depend on time left | Theorem for the statistic. Size on mazes unknown |
| A4: Theorems 2, 3, 3b, 4 under any policy with truncated rows | Theorem. Inherits their vacuous constants |
| A5, A6: anytime pair bounds | Theorem, fixed alpha |
| A7: stitched language test | Theorem, using Howard et al. |
| A8: fixed threshold at data-chosen times | **Conjecture.** Ratio to the fixed rate still rising |
| A9: exact recovery with realised counts | Theorem, given U, for a modified rule. **Not `bmr_rows`, not `learn_states`** |
| Wrong-merge side of the real pair rule, anytime | Theorem (A6 with Theorem 1c) |
| Wrong-split side of the real pair rule, anytime | Open |
| A10: lower bound | Theorem. Not matched for whole machines |
| A11, A12 | Theorem |
| A13: exact EIG, and its information identity | Theorem |
| A13a, b: expansion, allocation | Sketch, measured to 0.8% |
| A14: structure EIG is the greedy minimiser of a Bayes error bound | Theorem, on routing with known candidates |
| Same on merges with unknown rows | Measured only, unstable |
| A15: identity and limit | Theorem, on routing with known candidates |
| Structure EIG matches elimination | Measured |
| A16a, b, c | Theorem |
| A16d | Sketch and measured |
| A17a, b, f, g, form of d | Theorem |
| A17h: per-edge sum collapses to count | Sketch; measured on routing |
| A18: entropy-per-visit proxy behaves like structure EIG | Measured, on routing only |
| Any statement on k = 2 or on a real maze | **None.** Not run |

**Weaknesses.**
- The theorems on the whole machine assume U, or GROW's pure rows. Labels that start impure are the real case and A2 says the guarantees do not reach them. No theorem covers `learn_states`, as in MERGE_PAC.
- A9's rule is not the code's. The code has a fixed threshold, a sum veto, and refitted alpha. Only the wrong-merge side transfers.
- Certificates are 7x to 16x loose on one small machine. A6 costs about 5x on a pair. With Theorem 2's constants, A4 is as vacuous as before for stochastic rows.
- `N_design` assumes gaps are known. A learner that estimates them (track and stop) is not analysed. The count-directed column shows what is lost on one ladder: 4x instead of 16x.
- A10 is not matched, and (c) needs the alternatives to be in the class. Structure in the class, such as "a bit once known stays known", can escape (b), as for Prop 8.
- The routing setting has known candidates, so its optimal policies are close to bandit results. It does not include travel, where a plan to a hard edge costs steps. A11 and the corollary treat that part separately.
- The built examples show that an effect exists. T1d says censoring can split a state, not how often it does in a maze. Seed counts are 30 to 200 on the policy comparisons, and differences under about 10% should be ignored.
- A3c, A17h and the T4e direction need a run on the real learner. That is the next step, and none of those has one.

## References

- Deb, Cipollone, Jonsson, Ronca, Talebi 2025, Offline RL in Regular Decision Processes: Sample Efficiency via Language Metrics, ICLR.
- Howard, Ramdas, McAuliffe, Sekhon 2021, Time-uniform, nonparametric, nonasymptotic confidence sequences, Ann. Statist. 49(2), arXiv:1810.08240.
- Chugg, Wang, Ramdas 2023, A unified recipe for deriving (time-uniform) PAC-Bayes bounds, JMLR 24, arXiv:2302.03421.
- Kaufmann, Cappe, Garivier 2016, On the complexity of best-arm identification in multi-armed bandit models, JMLR 17. Cited from memory for Lemma 1 and the 2.4 constant; the constant is checked numerically here.
- Ville 1939, Etude critique de la notion de collectif. As given in Howard et al. and Chugg et al.
- MERGE_PAC.md and its references.
