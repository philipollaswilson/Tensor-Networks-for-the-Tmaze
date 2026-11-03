# PAC-style guarantees for BMR merging of histories

Checks: [src/merge_pac_check.py](src/merge_pac_check.py). Code read: `bmr_rows`, `Rows`,
`future_windows`, `learn_states`, `bisimulate` in [src/kmaze.py](src/kmaze.py), and
[src/merge_theory.py](src/merge_theory.py).

Status labels: **Theorem** = proved here, from the stated assumptions and cited results.
**Sketch** = the argument is given but a step is not fully closed. **Conjecture** = supported
by simulation at most. **Measured** = a number from a run, with no proof behind it.

## Main results

1. **One pair, exact.** dF = -n JS^ + G. G is an Occam term of order kappa ln h, with
   kappa = (s + 1)/2 - s alpha, and dF >= -ln C(n1 + n2, n1) always, with equality for disjoint rows.
2. **The two errors have different rates.** A wrong merge falls exponentially in n JS. A wrong split
   falls only polynomially, about h^-kappa. So BMR has a log(1/delta) cost on one side and a
   delta^(-1/kappa) cost on the other.
3. **The veto** makes the wrong-merge guarantee depend on the best single modality, under any dependence.
   Its form as a *sum* of negatives is what costs: on equal rows with M noisy modalities it fires more and
   more often. A veto on the single worst modality keeps the same proof.
4. **Whole machine.** Proved for idealised learners: given a unifilar refinement U (Theorems 2, 3), and from
   scratch given statistics that reach every difference (Theorem 4). Deterministic rows are the best case: N
   is the coverage time, with n0 = 4 visits per (state, action). **`learn_states` itself is not covered.**
5. **Rare edges (Proposition 8).** Routing an edge depends on that edge's own visits, for any learner on general
   unifilar machines. The relevant occupancy is per edge, weighted by KL, not d_min per (state, action). This is what
   the k = 2 failure runs into.
6. **Comparison.** The published O~(1/(d_min mu_0)) cannot hold as stated: the correct order is 1/(d_min mu_0^2).
   There is no exponential gain over L_X with the same statistic. Gains over L_inf^p are real.
7. **Why N* d_min varies.** Largely a rule effect, not coverage. On star k = 1, changing the sum veto to a
   single-modality veto takes N* from 2000 to 50. On the same grid, over D <= 3, the spread of N* d_min goes from
   108x to 5x. My censoring explanation for the corridors was tested and **failed**.

## 1. Setting and the rule

Two rows x, y of counts over the outcomes of one modality, with totals n1, n2, n = n1 + n2,
w = n1/n, h = n1 n2/n. Empirical laws p^, q^, pooled z = x + y.
`Rows` gives a modality with m' observed columns (over the **whole table**, not this pair) the
prior alpha = 1/m', total mass 1. Then

    dF = ln DM(x + y) - ln DM(x) - ln DM(y),   ln DM(x) = -ln Gamma(1 + n) + sum_i [ln Gamma(alpha + x_i) - ln Gamma(alpha)].

The rule of `bmr_rows`, for one pair with modalities j, is

    merge allowed  <=>  F := sum_j dF_j >= 0  and  F- := sum_j min(dF_j, 0) >= -tau,   tau = 3.

(The code breaks at `< -1e-9`, so ties merge.) Two facts follow by reading the rule.

- With one modality the veto does nothing: dF < -3 already implies dF < 0.
- The veto only removes merges. Compared with the plain sum rule it can only lower the wrong-merge
  risk and only raise the wrong-split risk.

JS_w(p, q) = w KL(p || r) + (1 - w) KL(q || r), r = wp + (1 - w)q, in nats. J = JS_1/2.
Since JS_w is concave in w and zero at 0 and 1, JS_w >= 2 min(w, 1 - w) J.

Checked against the current `kmaze.py`: `bmr_rows` and `Rows` are as above. Both rounds of `learn_states` stop
when a partition repeats (`partition_key`), and unseen end edges stay -1. Neither changes the pair rule. They
belong to the part of the learner that the theorems do not cover (Section 4.1, list of what is dropped).

## 2. Exact lemmas on one pair

**Lemma 1 (identity). Theorem.** With J^ = JS_w(p^, q^),

    dF = -n J^ + G,   G = rho(x) + rho(y) - rho(x + y),   rho(x) = max_theta ln theta^x - ln DM(x) >= 0.

Decomposition: with t(n) = ln Gamma(1 + n) - n ln n, r(x) = x ln x - ln Gamma(x + alpha) + ln Gamma(alpha), r(0) = 0,

    G = t(n1) + t(n2) - t(n) + sum_{i in C} c(x_i, y_i),   c(a, b) = r(a) + r(b) - r(a + b),

where C = {i: x_i > 0 and y_i > 0}.

*Proof.* n J^ is the log likelihood ratio of two fitted rows against one, so it equals
n H(z/n) - n1 H(p^) - n2 H(q^). Substituting into dF gives the first line. The second is algebra with
rho(x) = t(n) + sum_i r(x_i); cells outside C cancel because r(0) = 0. QED

So the merge evidence separates exactly. n J^ does not depend on alpha, and all prior dependence
sits in G. **Hence a data-dependent alpha (it depends on the whole table) is harmless:** any bound
that holds for G at every alpha in {1/m': m' <= m_max} holds for the actual alpha.

**Lemma 2 (Occam term). Theorem.** Let K(alpha) = ln Gamma(alpha) - (1/2) ln 2 pi and
e(x) = r(x) - x - (1/2 - alpha) ln x - K(alpha). Let e-(alpha) <= e(x) <= e+(alpha) for all integers x >= 1.

(a) Floor, valid for all rows. G >= (1/2) ln(2 pi h) + |C| ln((1 + alpha)/(4 alpha)) if alpha <= 1/3,
which is at least (1/2) ln(2 pi h). For alpha = 1/2, G >= (1/2) ln(2 pi h) - |C| (1/2) ln 2, which is at
least (1/2) ln(pi h / 2) as |C| <= 2. Modalities with m' = 1 have dF = 0.

(b) With h_i = x_i y_i/(x_i + y_i): G >= (1/2) ln(2 pi h) + sum_{i in C} [(1/2 - alpha) ln h_i + K + 2 e- - e+].

(c) G <= (1/2) ln(2 pi h) + 1/(12 n1) + 1/(12 n2) + max_{c <= |C|} c [(1/2 - alpha) ln(h/c) + K + 2 e+ - e-] =: U(h, |C|, alpha).

Rough size when all s = |C| cells are well populated: G ~ kappa ln h, where
**kappa = (s + 1)/2 - s alpha**. For s = m' this is the (m' - 1)/2 of the known formula; for s < m' it
is larger, so a Perks prior on a large alphabet favours merging sparse rows. Each common cell also
adds about K(alpha) ~ ln m'.

*Proof.* Stirling with 1/(12n + 1) < t(n) + n - (1/2) ln(2 pi n) < 1/(12n) gives
(1/2) ln(2 pi h) <= t(n1) + t(n2) - t(n) <= (1/2) ln(2 pi h) + 1/(12 n1) + 1/(12 n2). Writing r through e
gives c(a, b) = (1/2 - alpha) ln h_ab + K + e(a) + e(b) - e(a + b), hence (b), and (c) by
sum_i ln h_i <= c ln(h/c): (a, b) -> ab/(a + b) is concave and homogeneous, so sum h_i <= h.
For (a), let Delta(x) = r(x + 1) - r(x), so c(a + 1, b) - c(a, b) = Delta(a) - Delta(a + b). Now
Delta'(x) = f(x) = ln(1 + 1/x) - 1/(x + alpha). If alpha <= 1/3, f' >= 0 on x >= 1 (because
x(1 - 2 alpha) >= alpha^2) and f -> 0, so f <= 0, Delta decreases, c increases in each argument, and
c >= c(1, 1) = ln((1 + alpha)/(4 alpha)). If alpha = 1/2, f >= 0 as ln(1 + 1/x) >= 1/(x + 1/2), so c
decreases to its limit K(1/2) = -(1/2) ln 2. QED

The constants e± are defined as extrema. Evaluated, |e(x)| <= 0.082 at every alpha = 1/m', and the
elementary bounds -1/(12x) <= e(x) <= alpha follow from Stirling.

**Lemma 3 (hard floor). Theorem.** dF >= -ln C(n, n1) for all rows, with equality iff the supports are
disjoint, at any alpha.

*Proof.* With rising factorials, (alpha)_{a+b} = (alpha)_a (alpha + a)_b >= (alpha)_a (alpha)_b. So
DM(x + y)/(DM(x) DM(y)) >= n1! n2!/n!. Equality needs no cell with a, b > 0. QED

Two consequences, exact for any alpha:
- Disjoint rows split (dF < 0) as soon as n1, n2 >= 1.
- One modality can fire the veto only if C(n, n1) > e^3 = 20.09. So (3, 3) cannot (C = 20),
  (3, 4), (2, 5), (1, 20) can. **Below these counts a single disjoint modality does not veto.**

**Lemma 4 (upper tail, equal or known rows). Theorem, using Agrawal 2020.** Let x ~ Mult(n1, p),
y ~ Mult(n2, q) be independent, with supports of size s_p, s_q, M = s_p + s_q - 2. Then
n1 KL(p^ || p) + n2 KL(q^ || q) satisfies, for t > M,

    P(sum >= t) <= B_M(t) := exp(M - t) (t/M)^M.

If p = q then n J^ is at most that sum, so P(n J^ >= t) <= B_{2(s-1)}(t).

*Proof.* Agrawal (IEEE Trans IT 2020): E exp(lambda n KL(p^ || p)) <= (1 - lambda)^-(k-1), the MGF of a
gamma of shape k - 1 and rate 1. Multiply for independent rows, Chernoff at lambda = 1 - M/t. For
the second claim, n J^ = min_theta [n1 KL(p^ || theta) + n2 KL(q^ || theta)]. QED

If the rows are dependent, use the marginals: P(sum >= t) <= B_{s_p-1}(t/2) + B_{s_q-1}(t/2).

**Lemma 5 (lower tail, different rows). Theorem.** Independent rows, r = wp + (1 - w)q, and for
lambda in (0, 1), psi(lambda) = -w ln sum_o p^(1-lambda) r^lambda - (1 - w) ln sum_o q^(1-lambda) r^lambda. Then

    P(n J^ <= t) <= inf_{lambda, u > M} [ exp(lambda (t + u) - n psi(lambda)) + B_M(u) ].

psi is concave with psi(0) = 0, so J_lambda := psi(lambda)/lambda increases to JS_w(p, q) as lambda -> 0.

*Proof.* n J^ >= ln(p^x q^y) - max_theta ln theta^z = Lambda - n KL(z/n || r), with
Lambda = ln(p^x q^y / r^z). By joint convexity of KL, n KL(z/n || r) <= n1 KL(p^ || p) + n2 KL(q^ || q).
So {n J^ <= t} is contained in {Lambda <= t + u} or {that sum >= u}. Chernoff on Lambda, which is a sum
of independent terms, gives the first term; Lemma 4 the second. QED

## 3. Question 1: one pair of rows

**Theorem 1.** Rows as in Lemmas 4 and 5, rule of Section 1, any alpha = 1/m' with m' <= m_max.

(a) *Wrong split, one modality, equal laws p, support s.* With g the floor of Lemma 2(a) at |C| = 0
(m' >= 3) or |C| = 2 (m' = 2), so g >= (1/2) ln(pi h / 2),

    P(dF < 0) <= B_{2(s-1)}(g).

On the event E that every cell satisfies x_i >= n1 p_i/2 and y_i >= n2 p_i/2 (which fails with
probability at most sum_i [exp(-n1 p_i/8) + exp(-n2 p_i/8)]), the same holds with g replaced by the
bound of Lemma 2(b), about kappa ln h. If s = 1 a wrong split is impossible: J^ = 0 and G >= 0.

So the wrong-split probability falls **polynomially**, about h^(-kappa) up to logs, not exponentially.

(b) *Wrong merge, one modality, laws p != q.* With s_c = |supp p ∩ supp q|, U as in Lemma 2(c), and
tau = 0 for one modality:

    P(dF >= -tau) <= inf_{lambda, u} [ exp(lambda (U(h, s_c, alpha) + tau + u) - n psi(lambda)) + B_M(u) ].

It is enough that, for a free lambda and epsilon,
n >= [U + tau + u_eps + (1/lambda) ln(2/epsilon)] / J_lambda, with u_eps solving B_M(u) = epsilon/2.
This rate is JS_w, as the numerically known formula says, plus an explicit deviation cost. If
the supports are disjoint the probability is exactly 0 once C(n, n1) > e^tau (Lemma 3).

(c) *Wrong merge, many modalities, any dependence.* Let j be any single modality. Then
{merge allowed} is contained in {dF_j >= -tau}, because F- <= min(dF_j, 0). So the bound of (b) with this
tau applies with **no assumption on the other modalities or their overlap**, and the best modality
may be chosen. In particular a disjoint modality with C(n1 + n2, n1) > e^3 gives wrong-merge probability 0.

Without the veto no such statement exists: F can be pushed up by other modalities, each adding
about kappa ln h. The veto makes the guarantee depend on the best modality rather than on how many
uninformative ones are present. Its price is tau in the threshold, against 0 for one modality.

(d) *Wrong split, many modalities, any dependence.* Let all modalities have equal laws. Then

    P(wrong split) <= P(F- < -tau) + P(F < 0),
    P(F- < -tau) <= (1/tau) sum_j E[(dF_j)-],
    E[(dF_j)-] <= int_0^{ln C(n_j, n1_j)} min(1, B_{2(s_j-1)}(g_j + u)) du.

Linearity of expectation needs no independence, so this holds under any overlap. If modalities
are independent: F < 0 implies sum_j n_j J^_j > sum_j g_j, and the sum has a tail of B_{sum 2(s_j-1)}.

*Proof of (a).* dF < 0 means n J^ > G >= g; apply Lemma 4. *(b).* dF >= -tau means n J^ <= G + tau <= U + tau;
apply Lemma 5. *(d).* A wrong split is the union of the two events by the rule. Markov on -F-, then
(dF_j)- <= (n_j J^_j - g_j)+ and (dF_j)- <= ln C by Lemma 3. QED

**What overlap does (answer to the dependence question).** Merging decisions treat the
modalities as if they were separate evidence, and they are not. In `future_windows` one observation
at t + 1 enters depth 1, 2 and 3 windows, the `next` modality and up to three event windows, so up to
c = 7 modalities, and the same observation enters again for neighbouring start times.

- Wrong merge: no effect (part c).
- Wrong split, in the worst case: c perfect copies of a modality turn F- into c min(dF, 0), so
  the threshold is tau/c per copy.
- Wrong split, as measured: sharing samples is **not** the main effect. On iid noise with L = 3 and
  s = 3 symbols, P(veto) is about the same with shared and with unshared samples (Section 8). What
  matters is that F- is a **sum of negative parts**. Many weak negatives add up, positive evidence
  does not offset them, so P(veto) rises with the number of *noisy* modalities, on rows that are
  truly equal. Section 7 shows this is the main error in the k = 1 mazes.

**Remark (a cheaper veto that keeps the proof). Theorem.** The proof of (c) uses only one modality. So
the rule "veto iff min_j dF_j < -tau" has the same wrong-merge guarantee, at any overlap, and its
wrong-split term is P(exists j: dF_j < -tau) <= sum_j P(n_j J^_j >= g_j + tau). This still grows with the number of
modalities, by the union, but weak negatives no longer add up. Measured on 80 noisy modalities: 0.20
against 0.78 for the sum (Section 8).

**Sharp constant. Sketch.** For fixed p and n -> infinity, 2 n J^ -> chi2(s - 1) (Wilks), so
P(split) ~ P(chi2(s-1) > 2 G). This is a moderate-deviation use of Wilks, since G grows like ln n, and
I do not prove it. Measured, it is within a few percent in Section 8.

**Failure of the assumption.** Theorem 1 needs equal laws on equal states. If rows of one state have laws that differ
by J_c > 0, the false split is no longer rare: by (b) it becomes certain as n grows. Windows truncated by the episode
end could do this in principle. I tested that as the cause in the corridors and it is not (Section 7).

## 4. Question 2: a whole unifilar machine

Setting: a unifilar machine (Q, A, O, tau_next, theta, q0): next state = tau_next(q, a, o). The policy
is memoryless and picks all actions; episodes have T steps; N episodes. K_Q = |Q| after minimizing.

**Stack construction.** For each (v, a) draw an i.i.d. stack of outcomes from theta(. | v, a). A
trajectory reads the next unused entry of the stack of the current (v, a). This has the law of the
process. Row of (v, a) := **first n0 entries** of that stack. These rows are independent, and
exactly multinomial with fixed size, even when v is visited repeatedly in one episode. If (v, a) was
visited at least n0 times they are observed.

### 4.1 Theorem 2: refinement given a unifilar U (stage 3, idealised)

Let U be a known unifilar machine on V, K = |V|, that refines the minimal machine (Q is a quotient of
V). Moore partitions: P_0 = {V}; u, v stay together in P_{r+1} iff they are together in P_r and for
all a, l^r_{u,a} = l^r_{v,a}, where l^r_{u,a} is the law of (o, block in P_r of the next state). P_r* is the
fixed point, r* <= K_Q - 1. Define:

- gamma = minimum over pairs separated at some round r of max_a J(l^r_{u,a}, l^r_{v,a}): the **one-step
  JS along the refinement**. Next states count as outcomes, so a difference d steps away can have
  gamma = ln 2.
- s = max support of theta(. | v, a). Since next is a function of o, (o, block) has the same support: s
  does not grow with K.
- N_E = number of pairs with theta_u = theta_v at all actions; N_P <= C(K, 2).
- d = min over (v, a) of expected visits per episode; m_bar = m K_Q, a bound on the columns.

Learner BISIM(n0): rows of size n0; P^_0 = {V}; each round, inside each block test every pair with the
rule of Section 1 on one modality per action with outcome (o, block of next), and take connected components of
"merge allowed". Stop when nothing changes.

**Theorem 2.** Let

    p_S(n0) = A B_M(g + tau/A) + B_{AM}(A g),   M = 2(s - 1), g = (1/2) ln(pi n0 / 4)   [0 if s = 1]
    p_M(u, v) = the bound of Theorem 1(b) with n1 = n2 = n0, at the separating action   [0 if disjoint, n0 >= 4]

If N_E p_S(n0) + sum_{pairs} p_M(u, v) <= delta/2, and

    N >= max( 2 n0 / d,  8 T ln(2 K A / delta) / d ),

then with probability at least 1 - delta, BISIM returns exactly the minimal machine, in r* + 1 rounds.
The coverage condition can be replaced by N >= max(2 n0/p, 8 ln(2 K A/delta)/p), with p the smallest
probability per episode of visiting a (v, a) at all: this has no T, and is better when visits are repeated.

*Proof.* Coverage: visits to (v, a) in N episodes are a sum of i.i.d. variables in [0, T] of mean
Nd or more, so by multiplicative Chernoff at most delta/(2 K A) falls below Nd/2 >= n0. Define the
Moore partitions on the **true** machine (visits are at least a Binomial(N, p), which gives the second form): they are deterministic, so the following events are fixed
functions of the stacks, and no data-splitting is needed although every round reuses the data.
Wrong-merge event of a pair, at its unique separating round: the rule allows the merge. Wrong-split event of a pair with
equal emissions: exists a with n J^_a >= g (the row over (o, block) is a relabelled row over o at
every round where the pair is together, because equal law forces equal routing, so J^ does
not depend on the round). Off these events, induction on r: if P^_r = P_r the compatibility graph inside
each block is the union of cliques on P_{r+1}, and so P^_{r+1} = P_{r+1}. For p_S: dF_a >= g - n J^_a by
Lemma 2(a); a veto needs sum_a (n J^_a - g)+ > tau so some a has n J^_a > g + tau/A; F < 0 needs
sum_a n J^_a > A g, and the rows of one pair are independent, so Lemma 4 applies. Union bound. QED

**Corollary (scaling).** Up to logs in n0,

    N = O~( (1/d) max{ (s ln(m K_Q) + ln(K^2/delta) + tau) / gamma,   T ln(K A/delta),   [s >= 2] (N_E A / delta)^(1/kappa) } )

with kappa = (s + 1)/2 - s/m' from Lemma 2(b), at least 1/2 (this form also needs n0 >= 8 ln(..)/p_min for
the typical cells).

- gamma enters as 1/gamma: at most 1/ln 2 in deterministic worlds.
- A and K enter logarithmically through the first two terms.
- **The third term is polynomial in K and 1/delta.** It is the rate of Theorem 1(a), not an artefact of
  the union bound: Section 8 measures P(split) sqrt(n) nearly constant. For confidence 1 - delta BMR needs
  n ~ delta^(-1/kappa), where a calibrated test needs ln(1/delta). A threshold rule
  "merge iff F >= -sigma" with sigma ~ ln(K^2 A/delta) recovers the log, at the cost of sigma in
  the wrong-merge term (Conjecture for the machine, Theorem for the pair by the same proof).

**What this drops from the real learner.**
1. U is given. This is the largest gap. `learn_states` has to find a refinement first.
2. Fixed row size n0: the real rows use all visits, so their sizes depend on the data.
3. Pairwise tests plus connected components, instead of greedy pooling.
4. Only the one-step `next` modality: no L-step or event windows, which add modalities to the
   F- sum.
5. Nested blocks, whereas `bisimulate` regroups all items each iteration.
6. Episode ends (`fill_ends`, -1 labels), majority successors, and `minimize`.

Greedy pooling is **not covered**. A union over the subsets that pooling can reach is exponential in
K, and with a polynomial wrong-split rate that union is useless, so the proof does not transfer.
I expected pooling to help. Measured, it is **worse** than pairwise tests at small n0 and catches up
by n0 ~ 1000 (Section 8). I have no proof of why; the real `bmr_rows` regroups all items each
round and shares its prior across them, and either could be the cause.

### 4.2 Theorem 3: deterministic rows

**Theorem 3.** If every row is a point mass (theta(. | v, a) is deterministic for each v in V and a) and
the assumptions of Theorem 2 hold, then n0 = 4 gives **exact** recovery given coverage: on
{every (v, a) visited at least 4 times}, BISIM returns the minimal machine with probability 1. So

    N >= max(8/d, 8 T ln(K A / delta)/d)   suffices, and   N >= ln(1/(2 delta)) / -ln(1 - p)   is necessary,

p being the smallest probability per episode of visiting a needed (v, a). gamma = ln 2 here.

*Proof.* Equal-law rows are identical, dF >= 0 in every modality. Different laws are disjoint point
masses at the separating action, with dF = -ln C(8, 4) = -ln 70 < -3 by Lemma 3: the veto fires
regardless of the other modalities. Coverage as in Theorem 2. Necessity: if a needed pair is never
seen, its two possible continuations give the same data. QED

This is the best case: **N is the coverage time, up to T and a log, and nothing depends on gamma or
on the depth of the difference.**

"Deterministic" here is about rows of machine states. A deterministic world with hidden bits is
*not* this case: a state whose bit is unknown has stochastic rows even when nothing in the world is
random. In the k-mazes the state "centre, bit unknown" is such a state, and that matters (Section 7).

**Theorem 3b (near-deterministic, m = 2). Theorem.** Let rows be Bernoulli with minority
probability at most eps, and c0(n0) be the least a + b with dF < 0 over minority counts (a, b), found
exactly by enumeration. Then P(wrong split of a pair) <= P(Bin(2 n0, eps) >= c0(n0)), since dF >= 0
for all tables with fewer minority counts. Wrong merges obey Theorem 1(b) with
J >= (1 - 2 eps)^2/2 (Pinsker). c0 grows like ln n0, so at small eps the polynomial term of Theorem 2
is replaced by about (n0 eps)^c0.

### 4.3 Theorem 4: from scratch, with a statistic that reaches the difference

Theorems 2 and 3 assume U. Without U the learner must decide "is this candidate edge a state I
already have" with evidence that **reaches** the difference, as Moore's backward step
cannot work on labels that have already lost the distinction. So the from-scratch bound has to use a window or language.

Let Phi be a finite family of fixed-horizon statistics of the future: phi(a_t.., o_t+1..) of horizon H.
If the machine is in Moore form (the last observation is a function of the state) phi may also read the
last observation. gamma_Phi > 0 below is a real assumption: on CueCorridor my first statistic did not
separate the two absorbing states, and the all-go window does not separate positions D and D + 1,
until the last observation is added. A check failed on exactly this, then passed.
Examples: the observations under one action sequence of length L (applies with probability A^-L);
"go steps to the first change and its symbol", with a value "none". Let
gamma_Phi = min_{q != q'} max_phi J(P_q^phi, P_q'^phi), with kappa_phi the rate at which phi applies, and let
s_Phi be the largest support.

Learner GROW(n0): split episodes into odd and even. Start from q0. For each candidate edge (q^, a, o),
build the row of the first n0 odd episodes that traverse the edge with at least H steps left, and for each existing
state the row of the first n0 even episodes that enter it likewise. Route the edge to a compatible
state under the rule of Section 1, or else make a new state.

**Theorem 4.** Suppose every edge with theta > 0 is traversed with at least H steps left, with
probability at least p per episode, and n0 satisfies K (K A m) [p_S + p_M] <= delta/2 with the
terms of Theorem 2 at M = 2(s_Phi - 1), the gap gamma_Phi and one modality per phi (p_S with A replaced by |Phi|). Then if

    N >= max( 4 n0 / (p kappa_Phi),  16 ln(4 K A m / delta) / (p kappa_Phi) ),

GROW returns the minimal machine with probability at least 1 - delta.

*Proof.* The memoryless policy makes the first traversal of an edge a stopping time, so the window after it
is a draw from P^phi of the target state, independently across episodes (strong Markov).
Selection depends only on the past of each window. Odd against even rows are independent, so
Theorem 1 applies at full strength. Define the rows on the true machine; on the good event the
routing is true at each step, by induction over candidates. There are at most K A m
candidates and K existing states. Coverage by Chernoff on Bin(N/2, p kappa). QED

Idealisation: the odd/even hold-out costs a factor 2; the fixed horizon removes censoring; rows are pure
because the routing was correct before them. **That last one is the real gap to `learn_states`,**
which starts from observation labels. Its edges mix true states, so its rows are mixtures. Two
edges with the same states in different proportions look different, and two different sets can look
alike. Neither Theorem 1 nor its conclusion covers this, and I have no theorem for stages 1 and 2.
The mixture gap depends on time in the episode and on the policy, which is presumably part of
why stage 1 has no clean law.

Scaling: for action-sequence windows kappa_Phi = A^-L, so **N grows as A^L/p**: the exponential cost
of a direct test. For an event statistic kappa_Phi is near 1, and depth stops mattering, as far
as the language reaches. No theorem removes both U and the language, and none should: see Prop 6(c).

## 5. Question 3: comparison with Deb et al. 2024

### 5.1 A correction to the bound being compared with

**Already corrected by the authors.** The published version (Deb et al., "Offline RL in Regular
Decision Processes: Sample Efficiency via Language Metrics", ICLR 2025) says its new analysis
"uncovered a mistake in the proof of Cipollone et al. (2023)" and gives O~(H C* log(1/delta) /
(d*_m mu_0^2)) for CMS and O~(C* log(1/delta) log|X| / (d*_m mu_0^2)) for the language metric.
Below, "Deb" means the 2024 preprint (arXiv:2409.02747v1); this is not a new finding. The
"no erratum" remark is out of date. Use the ICLR form in any comparison. C* is the
concentrability of the behaviour policy, which section 5.3 does not include.

**Proposition 6. Theorem.** The bound |D| >= O~(1/(d_min mu_0)) in the preprint and in RegORL cannot
hold as stated. The correct order is 1/(d_min mu_0^2).

(a) *Where the proof breaks.* Deb Lemma 7 (and RegORL Lemma 14) needs min(|Z1|, |Z2|) >= 8 log(..)/mu_0^2.
Since |Z| is about N d, this asks for N >= 8 log(..)/(d mu_0^2). The proofs (Deb Eq. 4-8, RegORL
Eq. 18-22) instead impose the accuracy condition d/2 >= (sqrt(2 M l) + 14 l/3)/N, then put the *required*
M into it, which gives N of order sqrt(M l)/d = l/(d mu_0). But at that N the *actual* M is about Nd,
of order l/mu_0, far below the 8 l/mu_0^2 that the lemma needs. The substitution shows the
occupancy estimate is accurate; it does not show the sample is large enough. (This reading is mine and
comes from the two papers' text. I have not found a published erratum, and did not search for one exhaustively.)

(b) *Lower bound, for any learner.* Take two histories, each with occupancy d, followed by one bit. M0: both
Bern(1/2), so one state. M1: Bern(1/2 + mu/2) and Bern(1/2 - mu/2), so two states, with L_inf^p
distinguishability mu. KL between the data laws is N d (-ln(1 - mu^2)). By Bretagnolle-Huber,
any test errs with probability at least delta on one of them unless

    N >= ln(1/(4 delta)) / (d (-ln(1 - mu^2))) ~ ln(1/(4 delta)) / (d mu^2).

So at N = polylog/(d mu) learning fails as mu -> 0. The fix to both theorems is direct: N >= 2 M/d with M from the
lemma, so Theorem 2 of Deb becomes O~(ln|X| / (d mu_0^2)), and Theorem 1 O~(H / (d mu_0^2)).
Simulated in Section 8.

(c) *Exponential cost cannot be removed in general.* If a state is reached only by one action
sequence of length D under a uniform policy, then p <= A^-D and Theorem 3's necessity gives N >= A^D ln(1/2 delta).
So d_min itself carries the factor, for every learner. This is a coverage statement only. Noisy-parity PDFAs
(Kearns et al. 1994) show computational hardness at low distinguishability; I cite this and did not rework it.

The comparison below uses the corrected form.

### 5.2 Exact process: CueCorridor(D)

A fair bit b. Position 0 shows b; positions 1..D are blank; position D + 1 shows b again and absorbs.
Actions go and stay, uniform policy, T = 4(D + 2). Rows of (b, position) are point masses. K = 2(D + 2).
Exact values: L_inf^p mu_0 = 2^-D (a blank state must take D go steps to show the bit; a specific
string of that length has probability 2^-D). Moore rounds grow with D, gamma = ln 2 at all of them.

Exact numbers (delta = 0.05, N in episodes, from `corridor_table`). Layered columns are Deb's setting. mu_X is
for the single-symbol family X_111 only, minimised over layers and state pairs; higher levels of their hierarchy
were not computed, so the L_X column is an upper estimate for that family and says nothing about the others.
T3(U) assumes U = the machine itself. T4 uses the statistic (last observation, go steps to the first change,
symbol), at episode length 2T. The last column is Theorem 4's statistic with a Hoeffding test.

| D | mu L_inf^p | d layered | mu_X | Deb stated | Deb corrected | L_X corrected | d pooled | T3(U) | T4 | T4, Hoeffding |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5 | 1.2e-4 | 0.5 | 4.1e7 | 2.5e7 | 7.0e6 | 0.498 | 175 | 198 | 798 |
| 2 | 0.25 | 7.6e-6 | 8.5e-4 | 1.6e9 | 2.0e9 | 4.0e13 | 0.499 | 185 | 207 | 847 |
| 3 | 0.125 | 4.8e-7 | 1.3e-4 | 5.9e10 | 1.6e11 | 2.9e16 | 0.499 | 192 | 214 | 885 |
| 4 | 0.0625 | 3.0e-8 | 1.9e-5 | 2.1e12 | 1.2e13 | 2.2e19 | 0.500 | 198 | 220 | 917 |
| 5 | 0.031 | 1.9e-9 | 2.7e-6 | 7.5e13 | 8.6e14 | 1.7e22 | 0.500 | 203 | 225 | 944 |

Most of the gap to the layered columns is the layered d (2^-(T+2) for "still at the start after T stays"), so it is
time pooling, which needs stationarity. The comparison that isolates the test is the last two columns: same statistic,
same pooling, **BMR gains a constant of about 4, nothing exponential**.

### 5.3 Comparison

| | Deb et al. Thm 2, corrected | This note |
|---|---|---|
| Setting | Episodic RDP, layered by time, regular policy | Time-homogeneous unifilar machine, memoryless policy |
| Test | Hoeffding threshold on language probabilities, set from delta | Bayes factor with fixed threshold, delta does not enter |
| Gap | mu_0: L_X, a difference of probabilities | gamma: JS, at least mu^2/2 by Pinsker, up to order mu |
| Separation cost | 8 ln(|X|/delta) / mu_0^2 | ~ (s ln m' + kappa ln n + ln(1/delta)) / gamma |
| Equal states | ln(1/delta) | n ~ delta^(-1/kappa): **polynomial, worse** |
| Number of states | ln | ln for merges, polynomial for splits (s >= 2) |
| Occupancy | d per layer, min over t | d per episode, pooled over time: much larger |
| Horizon | ln|X| or H, plus layered d | T only in the coverage count |
| Guarantee from scratch | Yes, given X | Theorem 4 with statistics Phi (same idea); idealised |
| Given a refinement U | No analogue | Theorem 2, 3: one-step evidence, depth-free |

**Better.** (i) Rare events: when an event has probability mu in one row and 0 in the other, J ~ mu ln 2,
not mu^2. BMR needs n ~ 1/mu where the Hoeffding test needs 1/mu^2. This is a KL or Bernstein
effect, not a Bayesian one: a Bernstein test would have it too. (ii) Time pooling: on
a stationary machine one (state, action) pools over all t. In the layered bound the state "still at the
start after t stays" has occupancy 2^-(t+2), so layered d_min is exponentially small in T. This gain is
real, but only on time-homogeneous processes, and a layered learner handles non-stationary ones that
this note does not. (iii) Deterministic rows with U given: depth-free.

**Worse.** (i) delta dependence on equal states, polynomial instead of log. (ii) No calibration:
tau = 3 and the threshold 0 are fixed, so delta cannot be chosen. (iii) U is an extra input in Theorems 2 and 3.
(iv) Stationarity and a memoryless policy.

**Incomparable.** mu_0 and gamma are different quantities. Pinsker gives gamma >= mu^2/2 and no converse holds,
so neither bound dominates in general.

**Is the exponential-in-horizon gain real?** Three separate claims:
1. *Against L_inf^p:* yes. mu_0 = 2^-D in CueCorridor, so the corrected L_inf^p bound is about 4^D/d. Statistics
   that reach the difference cost no depth. This is Deb's own point, and it holds for Theorem 4 equally. Note it
   becomes 4^D, not 2^D, after the correction.
2. *Against L_X:* no exponential gain. Theorem 4 with the same statistic is Deb Theorem 2 with mu^2 replaced by J
   and d pooled over time. Also, Deb's mu_0 is a minimum over layers and pairs. For the single-symbol
   family X_111 on this process it is exponentially small (table above): near the end of an episode a state has
   only just enough steps left, and two blank positions differ only in the tail of a binomial. So on finite
   episodes the language gain can vanish in the layered bound, unless those layers are excluded, or a
   higher level of their hierarchy is used (I computed only X_111). Example 3 of Deb treats a mid-corridor
   state and long episodes, which avoids this. Time pooling with a fixed horizon, as in Theorem 4, avoids it too.
3. *From one-step JS (Theorems 2, 3):* real but conditional. gamma stays ln 2 while L_inf^p decays, on the
   class where a small refinement U with good occupancy is available. Passing evidence one edge
   at a time needs labels that already carry the distinction, and finding those is not covered. A
   history tree works as U but has occupancy exponentially small in its depth, which returns the
   factor. **So the coverage law N* = c/d_min is not a theorem for the real learner, and Theorem 3 gives it only
   given U, with c depending on T ln(KA/delta).**

## 6. Rare edges, and what the theorems say about k = 2

**Proposition 8. Theorem.** Let M1, M2 be unifilar machines that differ only in the target of one edge e = (q, a, o): A
in M1, B in M2. Let d_e be the expected traversals of e per episode, and KL_e the KL divergence, per traversal, of what
follows from A against B until the next traversal or the end of the episode.

(a) In a Dirichlet-count model with state sequences known, the log marginal likelihoods of M1 and M2 differ only by
rows reached after traversals of e. So the Bayes factor between the two routings is a function of those
futures, given the other rows. Evidence pooled elsewhere enters only through how well A and B are already known.

(b) For any learner, on any policy, one of M1, M2 is misrouted with probability at least delta unless

    N >= ln(1/(4 delta)) / (d_e KL_e).

(c) In particular, a pooled pair (A, B) with dF = -1000 says A and B differ. It does not say which of them e enters.

*Proof.* (a) Unifilarity makes the likelihood a product over (state, action) rows, and the state sequence up to the first
traversal is common. (b) Data laws agree until the first traversal. By the chain rule, KL between N-episode laws is
N d_e KL_e. Bretagnolle-Huber gives error sum at least (1/2) exp(-N d_e KL_e). QED

The bound needs the class of machines to leave the target free. A class with structure, such as "a bit once
known stays known", ties e to other edges and can escape it. If KL_e is infinite (disjoint futures), only coverage remains.

Compared with d_min: the coverage law takes the rarest (state, action). Proposition 8 says the unit is the **edge**, weighted
by how far apart its possible targets are. Rare and easily told apart is cheap. Rare and hard to tell apart is what costs.

Measured on a synthetic version (Section 8): pooled rows of 20,000 visits, Bern(0.6) against Bern(0.4), dF = -795. An edge
truly entering A with n_e = 20 is allowed to merge with B with probability 0.58, and the best possible test errs with
probability 0.13. That falls to 0.003 and 0.007 at n_e = 150.

### 6.1 The k = 2 diagnosis

Given (FUTURE_EXPERIMENTS.md A6): T = 20, 30k episodes, seed 2. With rows pooled by true predictive state, "knows both"
against "knows one" at L2 has -84 nats of veto evidence, nearly all from the event windows, and yet the states stay
merged. Clean "knows both" states exist, but the edge stepping out of the arm has ~150 visits, and it joins
"knows one", so the secret is forgotten.

What I computed exactly from the predictive machine (`--repo`, no learner run):

| | Visits at 30k (u, v) | F-, windows of depth <= 3 only |
|---|---|---|
| Pooled by true state | 3317, 27045 | -14.8 (one modality: back, back, go, at depth 3) |
| Scale of the step-out edge | 663, 5409 | -2.01 |

So windows alone give -14.8 of the -84, consistent with "nearly all from the event windows". I did not compute the
event-window part. And the exact expected count of (arm, back) is 663 at 30k, not ~150. My guess is that ~150 is the share
held by the clean copy, the rest being already filed under the merged state; I did not verify this. The conclusions below use
only the ratio of the two scales.

**What the theorems imply.**

1. *It is not a test failure at the pair level.* Theorem 1(c): on rows pooled by true state the veto fires with
   near certainty. Nothing about JS, the prior or the veto threshold needs fixing for that pair, and more accurate rows of
   the pair would change nothing.
2. *But that comparison is never made.* The -84 belongs to a pool defined by the true partition, which is the output
   being sought. The learner's items are edges, and by Proposition 8 the routing of an edge rests on the edge's own visits.
3. *The arithmetic fits, as a sketch.* -84 over ~3317 rare-row visits is about 0.025 nats per visit. At 150 visits that is
   about 3.8 nats of n JS, against an Occam floor (Lemma 2a) of (1/2) ln(2 pi 134) = 3.4. So dF is near -0.4, the veto is silent, and the
   edge merges. The floor alone puts the veto at about 270 visits, and the real G is larger. This scales a single
   measured number linearly, so read it as an order of magnitude: **about 2x more data at the very least**, likely more. The
   older note "still wrong at 50k" neither confirms nor contradicts it, as that was older code.
4. *An ideal classifier would not rescue it at 30k.* With JS near 0.0127 per visit, error of order exp(-150 x 0.0127) = 0.15 per
   edge, again as an estimate. This fits the note that hard-EM reassignment did not work, although that note has other causes
   too (thin rows, priors).

**Which assumption is violated.**

- Theorem 1: not violated. gamma_Phi > 0, and the pair is well separated.
- **Theorem 4, its coverage clause: violated. This is the root.** It requires n0 samples for *every candidate edge*, at
  the n0 set by the wrong-merge term. State-level coverage is met (d_min = 0.0221 per episode, or 663 visits), which is
  why d_min does not predict this. An edge with ~150 clean visits is below the n0 that its gap needs.
- **Theorem 4, its induction, then fails.** "Rows are pure because earlier routing was correct" stops holding after the first
  wrong edge, so errors cascade: this is the observed 97.9% main-copy purity rather than a clean miss.
- **Theorem 2, "U refines the minimal machine": violated as a consequence.** Once the two histories share a label, U is
  no longer a refinement. Moore refinement and `bisimulate` only coarsen items. They cannot split a label, so the distinction
  cannot come back, however many visits the pool has. Given a true refinement, by contrast, next-state evidence is disjoint for
  this pair and Theorem 3's count applies: **150 visits would be far more than enough**. So the problem is not that 150 is
  few in itself. It is few for a *window* test, and the *next-state* test is unavailable because its labels are merged.

**What this suggests (not tested).** Keep the distinction rather than try to recover it: start stage 3 from a deliberately
over-split U that carries "which observations have been seen", then merge down. Theorems 2 and 3 then apply, with the false-split
term as the cost. Or use structure to escape Proposition 8. Either would be a real change, and neither has been run.

**Testable prediction, cheap once the CPU is free:** at the edges that fail, per-edge dF should cross -3 at about twice the
visits. If those edges stay merged well past that, the linear scaling or the routing account is wrong.

## 7. Question 4: why N* d_min varies so much

Measured table: FUTURE_EXPERIMENTS.md A6, updated by `logs/merge_coverage2.out`. Everything below is checkable with
`python src/merge_pac_check.py --repo` (exact machines, fixed-point test) and `--variant <rule>` (whole learner, patched in memory,
`kmaze.py` untouched). The run labelled base reproduces the logged curves exactly where the grids overlap.

Exact quantities, uniform policy. gamma: smallest one-step JS along the refinement. Evidence: smallest (d_u + d_v) JS_w per episode
over separated pairs. Noisy windows: for the noisiest state, action sequences of length <= 3 with more than one outcome.

| Env | K | d_min | r* | gamma | Evidence | Noisy windows | N* | N* d_min | N* x evidence |
|---|---|---|---|---|---|---|---|---|---|
| hub k = 1 | 9 | 0.224 | 1 | 0.216 | 0.134 | 45/84 | 500 | 112 | 67 |
| star k = 1 | 9 | 0.302 | 1 | 0.216 | 0.135 | 81/84 | 2000 | 604 | 269 |
| corridor D = 1 | 10 | 0.112 | 1 | 0.693 | 0.155 | 0/84 | 50 | 5.6 | 7.7 |
| corridor D = 2 | 12 | 0.076 | 2 | 0.693 | 0.105 | 0/84 | 200 | 15 | 21 |
| corridor D = 3 | 14 | 0.051 | 2 | 0.693 | 0.071 | 0/84 | 200 | 10 | 14 |
| corridor D = 4 | 16 | 0.035 | 3 | 0.693 | 0.049 | 0/84 | 500 | 18 | 24 |

Note first what d_min leaves out. Hub and star have the same gamma and the same evidence, and star has the *larger* d_min, yet
needs 4x the data. Occupancy cannot explain this, as suspected. They differ in noisy windows: 45 against 81 of 84.

**Explanation 1: the sum veto fires on equal noisy rows. Supported.** Theorem 1(d) predicts this. Star's failures at large N are false
splits of (centre, bit unknown), the one state whose rows are stochastic: 5 of 6 seeds at N = 500, 2 of 6 at N = 1200, no wrong merges at
those N. Whole learner, N* on the grid 50, 120, 300, 500, 800, 1200, 2000, 3000 (8 seeds, 75% exact):

| Env | Sum veto (kmaze.py) | Single-modality veto | No veto | Single-modality, fixed-horizon events |
|---|---|---|---|---|
| hub k = 1 | 500 | 300 | 500 | 300 |
| star k = 1 | 2000 | **50** | 120 | 120 |
| corridor D = 1 | 50 | 120 | 120 | 120 |
| corridor D = 2 | 300 | 300 | 500 | 500 |
| corridor D = 3 | 300 | 300 | **2000** | 500 |
| N* d_min, spread over these five | 5.6 to 604 (108x) | 13 to 67 (5x) | 13 to 112 (8x) | 13 to 67 (5x) |

Corridor D = 4 did not finish under any variant. And one split round started at the truth, share of seeds that create no state:

| Env | N | Sum veto | Single-modality veto | Events off | Fixed-horizon events |
|---|---|---|---|---|---|
| hub k = 1 | 2000 | 0.25 | 0.88 | 0.25 | 0.25 |
| star k = 1 | 2000 | 0.00 | 1.00 | 0.00 | 0.00 |
| corridor D = 1 | 500 | 0.75 | 0.75 | 1.00 | 0.25 |
| corridor D = 3 | 500 | 0.25 | 0.25 | 1.00 | 0.00 |
| corridor D = 3 | 2000 | 0.50 | 0.50 | 1.00 | 0.62 |

With no veto at all, star is stable too (0.94 of 16 seeds at N = 2000), so the sum is the cause, not the threshold. The no-veto
column shows the veto is needed: corridor D = 3 goes from 300 to 2000 without it, as Theorem 1(c) would predict. **The single-modality
veto keeps the protection and loses the false splits.** It costs corridor D = 1 a step of the grid (50 to 120, on 6/8 against 4/8 at N = 50, two seeds apart), and it has not been tried on k = 2.

**Explanation 2: censoring of event windows in the corridors. Tested, refuted.** In corridors, truth is unstable only with event windows
on (1.00 with them off, at every N tried). I thought the law of an event window depended on time left in the episode. A fixed horizon
gives equal laws by construction, and does *not* restore stability: 0.25 against 0.75 at D = 1, N = 500. PLANS.md had already ruled this
out for another symptom. The small rise at D = 3 at large N in the whole learner (7/8 to 8/8) is one seed.

**Explanation 2b, the likely replacement: event windows bring policy noise. Not tested separately.** An action-sequence modality is
conditioned on actions, so a corridor row is a point mass. An event modality marginalises over them, so the *same* state gets a stochastic
row with many outcomes, and Theorem 1(a) applies with s > 1, where the alternative had s = 1. That fits "events off = 1.00", and the
stability rising with N under fixed horizon (D = 1: 0.25 to 0.88). A direct test: condition event windows on the action sequence of a small
abstraction, and see if false splits go, at the cost of evidence.
These one-round splits are mostly repaired by stage 3: corridor D = 3 at N = 500 has fixed-point share 0.25 and full recovery 1.00. So
they do not set corridor N*. One-round stability is not a proxy for recovery.

**Explanation 3: d_min measures visits, not evidence. Supported as far as it goes.** gamma ranges from 0.216 to 0.693, and pairs differ
in how much of the occupancy they share. In the corridors N* x evidence (8 to 24) and N* d_min (6 to 18) are about equally flat, so
evidence is **not** a better predictor than d_min there. After fixing the veto, N* x evidence spans 7 to 40 and N* d_min 13 to 67.
Neither collapses to a constant. On five points I would not claim either as a law.

**Explanation 4: mixture rows in stages 1 and 2. Open.** Stage 1 starts from observation labels, so its rows mix true states, and no
theorem here covers that. It probably accounts for some of the residual 5x. The diagnostics at low N (wrong merges of bit-different states
in corridor D = 4 at N = 120) are consistent with it, but I have no separate test.

**Explanation 5: measurement noise. Certain, of bounded size.** N* is the first grid point with 6 of 8 seeds, on a grid with steps of 1.5 to
2.4, and curves are not monotone: star under the single-modality veto reads 0.75, 0.875, 0.75, 0.875. Corridor D = 1 is 50 here at 75%
and 80 is 62.5% in the log. A factor of 2 in N* is within noise, so the residual 5x is at most moderately above it.

**Answer.** Most of the 108x spread is a rule effect on stochastic rows, not coverage. What remains, about 5x, is the size expected from
gamma, the evidence differences and noise, without a clean law. **So N* = c/d_min should not be stated, even after the fix.** What the theory
supports is the form N ~ n0(gamma, s, K, delta)/d, where n0 is not universal, and given U only. And k = 2 sits outside this: it is
limited by edge routing (Section 6), which d_min does not see.

## 8. Numerical checks

`python src/merge_pac_check.py`: **19 of 19 checks pass, in about 100 s** (machine under load, single thread). Several
passed only after a statement or a check was changed; those are listed below.

| Statement | Result |
|---|---|
| Lemmas 1-3 | 200,000 random tables, 0 violations; smallest Occam margin 0.0022 |
| Theorem 1a, split | Holds. Tightness 44x to 53,000x where non-vacuous; vacuous for s = 2 at n = 30. Example: s = 2, n = 3000, sim 0.0055, bound 0.57 |
| chi2 proxy, sketch | sim 0.00545 vs 0.00532; 0.00164 vs 0.00151; 0.00040 vs 0.00039. Far tighter than the bound |
| Theorem 1b, merge | Holds. n for 5% wrong merge, near case: bound 767, simulated 222, factor 3.5 |
| Theorem 1c, veto | n = 300: P(merge) sum rule 0.41 at 20 equal modalities, 1.00 at 80; with veto 0.0000; bound 0.0076 |
| Theorem 1d, Markov | P(veto) <= E[-F-]/3 holds with plug-in expectation (0.70 <= 2.15). **The analytic tail form is vacuous**: 27 to 531 |
| Accumulation | 80 modalities, n = 30: sum veto 0.78, single-modality veto 0.20; P(F < 0) is 0.00 |
| Sharing of samples | L = 3, s = 3, n = 200: P(veto) 0.70 shared, 0.73 unshared, 40 reps. **Sharing is not the main effect**; the number of noisy outcomes is (0.075 at s = 2) |
| Polynomial rate | P(dF < 0) sqrt(n): 0.40, 0.32, 0.28, 0.24, 0.22 over n = 100 to 1,000,000. Near constant, slow log decline |
| Theorem 2 | Failure <= bound, but **the bound is vacuous until n0 ~ 2e9**; simulated 0.97 at n0 = 100 |
| Theorem 2, source of the looseness | Union with simulated pair rate: 0.78 at n0 = 100, 0.98 at 1000 (Ring(4,2,0.1)). So the union is nearly right and the loss is the Lemma 4 constant, shape 2(s-1) against the true 1/2 per degree of freedom |
| Greedy `bmr_rows` | **Worse than pairwise** at low n0: 0.27 vs 0.70 at n0 = 10, 0.90 vs 0.97 at n0 = 100. Not covered by any theorem |
| Theorem 3 | Exact on Ring(5,2,0) at n0 >= 4 (n0 = 1 fails; 2, 3 work by luck of this machine). Real `bisimulate` on two-copy U: 12/12 exact at N = 396, 414, 428 for D = 1, 2, 3: N flat in D |
| Theorem 3b | Holds, tightness 2x to 48x |
| Theorem 4 | Success 1.00 at N(Thm 4) in all 6 cases (10 reps), including window N = 791, 1655, 3423, doubling with D, against event N = 199, 208, 215, flat. Ratios of 1 to 4 are at the theorem's fixed n0, so they measure coverage only, **not tightness in n0** |
| Prop 6 | Detection at N = O(1/(d mu)) with the paper's constant: 1.00, 0.88, 0.08, 0.00 at mu = 0.3, 0.1, 0.03, 0.003; at mu = 0.003 that N is below the any-test bound (216,540 < 715,303). At 1/(d mu^2): 1.00 throughout |
| Rare event remark | n BMR 79, 316, 1259; n Hoeffding 1585, 15849, 125893, at mu = 0.1, 0.03, 0.01 |
| Prop 8 | Floor holds. Pooled dF -795; P(merge with wrong state) 0.65, 0.58, 0.23, 0.003 at n_e = 5, 20, 50, 150 |

**Corrections made because a run disagreed.** (i) The T2 wrong-split union first counted only pairs equal at the end; it must count all
emission-equal pairs. (ii) The T4 check failed at first, for a valid reason: gamma_Phi was 0 for the statistics chosen, so the assumption
now has an explicit caveat. (iii) "Greedy does better" was wrong and is reversed. (iv) "The single-modality veto does not accumulate" was too
strong and now says it grows by the union. (v) "Overlap is the main false-split effect" was wrong. (vi) Censoring as an explanation was refuted.

## 9. What is proved, and what is not

| Claim | Status |
|---|---|
| Lemmas 1-5, Theorem 1 (a-d) and its Remark | Theorem. Uses Agrawal 2020. Constants e± evaluated numerically |
| Theorem 2, given U, fixed rows, pairwise + components | Theorem. Numerically vacuous for stochastic rows |
| Theorem 3, 3b | Theorem. Tight up to a log and T |
| Theorem 4, hold-out, fixed horizon, pure rows by induction | Theorem. Assumes gamma_Phi > 0 and coverage of every edge |
| Prop 6: corrected order 1/(d mu^2) | Theorem for the lower bound. The reading of where the proof breaks is mine; I found no erratum, and have not contacted the authors |
| Prop 8: edge-level lower bound | Theorem, on classes that leave the target free |
| chi2 form of the false-split rate | Sketch; measured within a few percent |
| Threshold rule recovering ln(1/delta) on a machine | Conjecture (pair level follows by the same proof) |
| Greedy pooling has a guarantee | Open. Measured worse than pairwise at low n0 |
| Any guarantee for `learn_states` | **None.** Stages 1 and 2 use mixture rows |
| N* = c/d_min | Not supported, even after the veto change |
| k = 2 needs about 2x more data at least | Order-of-magnitude sketch from one measured number; not tested |
| Policy noise in event windows drives corridor splits | Hypothesis, not tested separately |
| Single-modality veto improves the learner | Measured on 5 environments at 8 seeds; D = 4 and k = 2 not run |

**Weaknesses.**
- The stochastic bounds are valid but loose: Theorem 2 needs n0 ~ 2e9 where ~100 suffices.
- The theorems cover idealised learners. The hardest part of the real one, getting from observations to a refinement, has no theorem.
- All guarantees assume a time-homogeneous unifilar machine and a memoryless policy; the Deb setting is wider on both.
- Rule-variant results come from 8 seeds on a coarse grid, on 5 environments. Corridor D = 4 is missing.
- The -84, ~150 and 3317 figures in Section 6 do not fully line up: I could not reproduce -84 (the event part was not computed), and exact
  edge visits are 663 rather than ~150.
- Theorem 1d's dependence-free bound is vacuous in its analytic form. Its useful form needs E[-F-], which is itself simulated.

## References

- Deb, Cipollone, Jonsson, Ronca, Talebi 2024, Tractable Offline Learning of Regular Decision Processes, arXiv:2409.02747v1.
- Cipollone, Jonsson, Ronca, Talebi 2023, Provably Efficient Offline RL in Regular Decision Processes, NeurIPS.
- Clark, Thollard 2004, PAC-learnability of Probabilistic Deterministic Finite State Automata, JMLR 5.
  Sample bound polynomial in n, |Sigma|, 1/mu, L, and accuracy; their m0 has 1/mu^2 (Eq. 6), consistent with
  Prop 6.
- Agrawal 2020, Finite-Sample Concentration of the Multinomial in Relative Entropy, IEEE Trans IT 66(10).
- Kearns, Mansour, Ron, Rubinfeld, Schapire, Sellie 1994, On the Learnability of Discrete Distributions, STOC.
