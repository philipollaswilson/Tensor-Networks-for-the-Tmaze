# Future experiments

**Focus changed 2026-10-30:** only tensor-network and quantum-inspired structure learning.
Track B, planning, EFE and empowerment are dropped; see [PLANS.md](PLANS.md). The text
below is kept as the record.

Two tracks, kept apart on purpose:

- **Track A — offline structure learning, explained.** Show exactly how hidden states are
  recovered from a trained tensor network (MPS → histories → BMR → A/B → EFE; SVD
  geometry; factor test), and what the quantum (Born) form adds over a classical one.
- **Track B — an efficient online agent.** Act, learn every step, grow the state space on
  surprise, explore with the active-learning term, scale to long horizons.

Status keys: `next` (being built), `queued`, `idea`. Newest decisions at the top of each track.

---

## Track A — offline structure learning

### A4. Compare with renormalising generative models (RGMs) `next`
Friston, Heins, Verbelen, Da Costa et al., "From pixels to planning: scale-free active
inference" (arXiv:2407.20292). RGMs build hierarchy by grouping neighbouring pixels and
time steps (a locality prior), then learn states within it ("fast structure learning" +
BMR); dynamics are "paths" (a latent factor). Python port verified against SPM:
github.com/atomresearch/RGMs (arXiv:2608.09512), demos `DEM_AtariIII` (simplified Pong)
and `DEM_drone_VI`. Plan: run their simplest demo, read how it builds structure, then run
our pipeline (A0) on the same trajectories and ask whether it *finds* the factors RGMs
assume (ball, direction as memory, paddle). Needs first: a start below raw pixels (patches
or their lowest-level symbols), and a polynomial factor search (closure of state pairs
into SP partitions) instead of enumerating partitions.

Done so far (simplest examples, states only): `rgm_lite.py` re-implements SPM's
`spm_rgb2O` + `spm_MB_structure_learning` state generation (deterministic read-out instead
of VB; non-overlapping tiles). `factor_search.py`: pair-closure SP search + stream
predictive states (k-frame contexts, BMR, then refinement to a unifilar machine).
`make_fig_rgm_compare.py` → `fig_rgm_compare.png`.
- MNIST (DEM_MNIST_RGM setup, 130 images): 16 → 4 → 1 groups; from level 2 every image
  is its own state (top: one variable, 130 states). No dynamics, so our transition-based
  factorization does not apply; the open question there is tile grouping by dependence.
- Synthetic flapping bird (Karl's robin.mp4 is not distributed): wing period 8 × bob
  period 3. RGM: level 1 4 groups × 11 states, top 1 group × 6 states (4-frame steps).
  Ours: 15 distinct frames → 24 predictive states (up/downstroke need 2-frame memory) →
  SP partitions of sizes 2,3,4,6,8,12 → one product 3 × 8, ARI 1.00 with true bob and wing.
  Transition entries 576 (one variable) vs 73 (factored); search 0.02 s vs Bell(24) ≈ 4e17.
- Lesson: one-step BMR can merge histories whose later futures differ; refinement is
  required (not yet ported to `structure_pipeline.py`; the T-maze results were unaffected).
- Our learner on MNIST (`mnist_structure.py`, `make_fig_mnist_structure.py` →
  `fig_mnist_structure.png`): images read as 64 tile symbols in raster order (first SVD
  mode, 7 levels), label last; layered unifilar machine, BMR on [next tile, tile after,
  label]. 10,000 train images, 3 s: 1–72 states per position (2167 total; peaks
  mid-stroke), test accuracy 75.0% on 2000 held-out images. A unifilar MPS with bond
  ≤ 72. States read as partial-stroke prototypes with sharp p(digit | state).
  Weak points: raster order is arbitrary; one SVD mode per tile; unseen (state, tile)
  pairs fall back to the most visited state.
- Without sequences or tiles: a latent tree over pixels (`latent_tree.py`,
  `make_fig_latent_tree.py` → `fig_latent_tree.png`). MNIST 14×14 binarised, 146
  non-constant pixels, no order or neighbourhood given, no labels. Repeatedly join the two
  frontier nodes with the largest dependence Bayes factor; the new node's states are its
  children's state pairs merged by BMR on what they predict about the outside frontier
  (each outside node one Dirichlet modality; m = how many, "all" = no budget). Every node is
  a cut, its states the tree-tensor-network bond. Why not a PEPS: its bonds are not cuts,
  have no canonical gauge, and contraction is hard; a tree keeps all three.
  Held-out bits/image (2000 test, m = all): independent 71.1, Chow-Liu visible tree 58.2,
  k-d spatial tree (locality prior) 52.6, learned (evidence rule) 49.3, learned (redundancy
  rule = BF per nat of the smaller side) 48.3, raster chain 47.4. Best m = 16 (chain 46.5,
  learned 47.6); "all" costs ≤ 1 bit. ~10 s per tree on CPU (no GPU needed).
  Findings: locality emerges (mean within-subtree distance 2.2 px vs 6.3 random); the plain
  evidence rule snowballs into a chain (depth 126/145) with a data-chosen order; the k-d
  locality prior is worst on both density and digit information. The digit is the
  information shared across a mid-sized cut (50–80 pixels): best-node test accuracy, no
  labels used in learning, 64% (evidence) / 60% (redundancy) / 60% chain / 44% k-d. The
  root itself carries nothing (its last cuts separate rare border pixels).
  Open: the raster chain still wins on density, so greedy joining is not yet finding the
  best shape; outside is summarised by frontier states (lossy); no refinement pass;
  joins are greedy and never undone.
- k-secret maze (`kmaze.py`, in progress): hub C, per secret a cue room Q_i and arms
  L_i/R_i; truth = location (3k+1) × k bits, a product of k+1 factors. Offline, random
  policy. Learner (current): start with state = last observation; each round split the
  edges (prev state, a, o) entering a state by BMR on pooled futures (1–3-step windows,
  Perks prior over observed outcomes, ties merge, veto when the disagreeing modalities sum
  below −3, plus an (observation, next state) modality); then minimize: group states by
  BMR on L-step futures, majority successor per group, re-route all episodes.
  Best result so far (minimize without the next-state modality): k = 1 exact in 2 rounds,
  9 predictive states → 8 vertices = bit (2) × location (4), ARI 1.0 each, 0 conflicts.
  k = 2: 29 states, all correct (the 17 "mixed" ones are genuine one-secret-known states),
  but 16 "knows both" states at the arms are missing (12 of 28 found): from an arm the
  other secret only shows after back, back, go, and minimization merges them back.
  Tried and dropped: red-blue merging on the history tree (rare histories, too little
  evidence); unvetoed sum of dF (Occam gains outvote real differences); Perks prior over
  all O^L sequences (fresh rows too costly); next-state modality inside minimization,
  with old labels or iterated to a fixed point (over-splits: stale successors).
  The last revert was not re-verified (the run was stopped); rerun k = 1, 2 first.
  Update (later session). Learner now: stage 1 split + minimize until stable; then
  alternate one split round with bisimulation (group by L-step futures, then regroup with
  "which group comes next" to a fixed point). Hub k = 1 exact (verified); hub k = 2 33
  states, all cue-room and centre "knows both" states found, arm ones still missing: the
  "knows both" cue-room states get ~500 visits per 10k random episodes, the arms after
  them ~100, too few (50k episodes: still 29, 21 min). Tried and dropped: split-only rounds
  without minimizing (duplicates multiply, 1,589 states); redirecting only on a clear
  (90%) majority (duplicate chains return).
- Planning (`kmaze_plan.py`): value iteration on the learned machine (posterior-mean
  transitions, prior mass 1) vs fitted Q-iteration on windows of the last W (action, obs)
  pairs, same offline random data, same prior. New layouts in `kmaze.py`: 'star' (all rooms
  off the centre, as the T-maze) and `Corridor(D)` (cue, D identical cells, junction).
  Reward +1 cheese / −4 empty arm per step (with −1, guessing beats reading the cue).
  Star k = 1: window 2 is as data-efficient as ours (the task needs 2 steps of memory);
  ours optimal from ~200 episodes, occasional confident errors below (a sparse group
  mixes "knows" with "doesn't know" and the majority successor makes it certain).
  Corridor (optimum 11; mean of 3 seeds; `logs/kmaze_plan.out`):
    D = 2: ours 7.5 / 11.0 / 11.0 / 11.0 at 100 / 300 / 1k / 3k episodes; best window
           6.6–7.7 / 10.2 / 11.0 / 11.0 (W = 4). States 12 = truth from 1k.
    D = 4: ours −9.0 / −0.1 / 11.0 / 11.0; best window 3.1 / 8.0 / 8.3 / 8.1. Ours
           optimal from 1k where no window is; but over-split (36 at 100, 31 at 3k; truth 16).
    D = 8: ours −8.5 / −3.3 / −7.3 at 100 / 300 / 1k (63 states, truth 24); window 2 7.4
           at 1k (guess and switch). Ours fails. Sweep stopped before 3k and the star rerun.
  Why D = 8 fails: corridor cells more than L = 3 steps from the junction look alike and
  have the same 3-step futures, so states mix positions and also mix the two secrets;
  the memory of the cue is lost in those groups. Needs either successor-based
  distinctions that survive minimization (bisimulation on the full machine) or a test of
  longer futures; the next-state modality should carry "secret 0 vs 1" along the corridor
  but minimization merges it first.
Next: fix the long-corridor failure (keep the secret through aliased cells), rerun the
corridor sweep to D = 8 and the star k = 1 sweep with the Bayesian planner, then k = 3
and DreamerV3 on the same data budgets.

### A0. From trajectories to a factored generative model (T-maze) `done`
Scripts: `make_fig_hidden_states.py`, `generative_model.py`, `make_fig_generative_model.py`,
`structure_pipeline.py` (any symbolic data, any history length), `make_fig_structure_stress.py`.
Figures: `fig_hidden_states`, `fig_hidden_states_data`, `fig_generative_model`,
`fig_structure_learning_math` (the steps as equations), `fig_structure_stress`.

Pipeline: count prefixes → predictive rows → rank → BMR merges (Perks prior) → unifilar
machine → unmix (Bayes factor: is a row a mixture of the others?) → generator →
factorize (partitions with the substitution property, Hartmanis–Stearns; a pair with a
trivial meet and product size = |S| gives S = S1 × S2, B = B1 ⊗ B2).

Results, full pymdp T-maze, 3276 sampled trajectories:
- 24 histories → 7 predictive states, ARI 1.000; ~800 trajectories suffice (20 seeds).
- Prediction table rank 6, not 7: the centre row is exactly ½ cue-right + ½ cue-left
  (a belief). Unmixing: ln BF +62 for the centre, ≤ −192 for every other state.
- Generator: 6 hidden states, D = (½, ½). Smallest model with hidden states; 7 is the
  smallest whose state is read off the history; pymdp's 8 is the smallest where p(o | s)
  does not depend on the action (location must then be hidden).
- Factorization: 27 SP partitions, exactly 1 product: context (2, static) × place (3:
  not in an arm, right arm, left arm), B = B_ctx ⊗ B_place exactly. Location depends on
  place only; reward and cue bit on both (pymdp's A factor list, found from counts).
- Beyond the data (step 3): the flat 7-state model forgets the cue on returning to the
  centre (p(cheese) 0.50); the generator keeps it (1.00). TV at step 3 at 30k
  trajectories: 0.036 flat vs 0.013 generator. With 2-step data this rests on an inertia
  prior; with 3-step data it is confirmed by the data (centre after cue merges with cue).
- Stress test (3 variants, 20k trajectories): original maze 6 = 2 × 3; arms reachable only
  from the centre 8 = 2 × 4 (pymdp's own structure) but only with 3-step data (2-step
  gives 7, no product: the history that shows the centre remembers the cue has no
  future in the data); a third "no cheese" context gives 7 states and no product
  (54 SP partitions, so serial/hierarchical splits only).
- Quantum: the centre is a mixture, not a superposition; the classical generator already
  reaches the rank-6 bound, so no state-count advantage here (memory entropy: see A1c).

Open: serial (hierarchical) decompositions from a single SP partition; products of 3+
factors; planning with the factored model (as `learned_pomdp.py`); polynomial SP search.

### A1. What does quantum give us? `queued`
Same structure-learning pipeline on a **non-negative** (classical) tensor and on the
**Born** MPS, same data. Per Glasser et al. (arXiv:1907.03741) a non-negative MPS *is* an
HMM with bond = hidden state; a Born MPS can need a smaller bond. Quantum models of
stochastic processes can store less memory than the minimal classical one (the ε-machine;
Gu, Wiesner, Rieper & Vedral 2012, "Quantum mechanics can reduce the complexity of
classical models"). Experiment: processes where the quantum memory advantage is known to
be strict vs. where it is zero (deterministic, Markov — frozen lake, minimal maze), and
compare bond dimension, sample efficiency, and whether states are still readable.
Expected: no advantage on the lake/maze (deterministic, unifilar), a measurable one on
stochastic processes with overlapping futures.

### A1c. When does the quantum (Born) form help? Separate experiment `next`
Result so far, clock (`clock_born.py`, `results/clock/`, 3 seeds, 2,000 sequences of 60):
qubit instrument K0 = R(theta) diag(1, 0.81), K1 = sqrt(1 - 0.81^2) |0><1|, with the
outcome-0 map's eigenvalues all of modulus 0.81 and cos(phi) = 3/5, so no finite HMM is
exact (Niven + Perron-Frobenius). True rate 0.55 bits/symbol. Held-out gap above truth:
Born D = 1 / 2 / 3: +0.105 / +0.00002 / +0.00008 (D = 4 did not finish); Baum-Welch Mealy
HMM (4 restarts, 300 steps) D = 2 / 3 / 4 / 6 / 8 / 12 / 16: +0.051 / 0.021 / 0.011 /
0.0051 / 0.0033 / 0.0023 / 0.0026; age chain K = 8 / 16 / 24 / 32 / 48: +0.020 / 0.0027 /
0.00057 / 0.00024 / 0.00021; `kmaze.py` learner, no size given: 15, 15, 18 states, +0.00037.
Hazard error at n = 20-39: Born 0.005, learner 0.055, HMM 0.07+. Hankel singular values
0.172, 0.029, 0.021, then ~0.0009 (rank 3).
Reading: exact Born advantage in bond (2 vs 15+), tiny in bits, because the separation is
in a tail of 1.4% survival. The EM HMM is optimizer-limited (train gap = test gap, and the
learner's unifilar model is a better HMM of the same size), so EM rows are upper bounds.
Rank 3 vs bond 2 raises the next question: does a signed dimension-3 model match, which
would make this a non-negativity result rather than a quantum one?
Against current methods (`clock_compare.py`, `results/clock/compare_s*.json`,
`make_fig_clock.py` → `fig_clock.png`; 100 / 300 / 1k / 2k sequences, size chosen on 10%
validation): Born +0.00055 / 0.00029 / 0.00009 / 0.00002; warm-started HMM +0.0079 / 0.0013
/ 0.00083 / 0.00026; count learner +0.0050 / 0.0018 / 0.00049 / 0.00037; GRU +0.0034 /
0.0016 / 0.00067 / 0.00048; spectral +0.016 / 0.0073 / 0.0027 / 0.0016; CSSR-style +0.023 /
0.012 / 0.0039 / 0.0044; transformer +0.057 / 0.051 / 0.030 / 0.017 (untuned). Spectral
polish (Adam, signed, rank 3) mostly stayed at its start: +0.014 / 0.0022 / 0.0024 / 0.0033.
SVD recovers the eigenvalues from 100 sequences (modulus 0.80-0.82, angle 0.58-0.60 pi)
yet predicts well behind Born (negative probabilities are the likely cause, not measured).
Born is home-advantaged here: the
process is a Born model and the bond was given. Original plan:

Kept apart from the main agent comparison, which is chosen for online structure learning
and planning, not for quantum. Established cases where a Born / quantum model is smaller
than any classical one:
- Probability clock (Jaeger 2000, Neural Computation): an observable operator model of
  dimension 3 (a qubit, Born bond 2) that no finite HMM reproduces. It is the benchmark in
  hidden quantum Markov model papers (Srinivasan et al. 2017; Adhikary et al. 2020); an
  8-state HMM only roughly matches a small quantum model.
- Resettable stochastic clock / renewal processes (Elliott, Gu, Garner & Thompson, PRX
  2022): the classical memory entropy diverges as the time step shrinks, while the quantum
  agent's stays finite. This is entropy, not bond dimension.
- Glasser et al. (NeurIPS 2019): unbounded Born-vs-HMM bond separations, and complex
  tensors can be arbitrarily smaller than real ones; other distributions favour HMMs.
Plan: a Born MPS against a general HMM (non-negative MPS) at equal bond D, plus our
unifilar learner, on these processes and on the main-task data. Report the bond needed
to reach a given log-likelihood, to show where the quantum form pays and where it does
not.

### A6. The math of merging `next`
`merge_theory.py`, `results/merge_theory.json`, `logs/merge_coverage.out`.
Merge test, confirmed: dF ~ -(n1 + n2) JS + (m - 1)/2 ln(n1 n2 / (n1 + n2)) + c. Exact vs
formula, m = 3, alpha = 1/2: within 0.03 nats at n = 100, 0.00 from n = 1,000 (c = 0 is a
coincidence of m = 3 with this prior). So equal rows gain ln n for merging, different rows
lose JS per sample: separating two states takes about (m - 1) ln n / (2 JS) samples,
whatever the number of histories. Determinism gives the largest JS, which is why merging
is strongest there.
k = 2 diagnosis (T = 20, 30k episodes, seed 2). Not a data limit: with rows pooled by the true
predictive state, "knows both" vs "knows one" at L2 has -84 nats of veto evidence (-3 is
enough), nearly all from the event-window modality. Clean "knows both" states exist, but
stepping back out of an arm is an edge of ~150 visits, too few for a merge test, so it joins
the "knows one" state and the secret is forgotten; the evidence exists only pooled over
the state. Tried and dropped: hard-EM reassignment of edges to the best-fitting existing
state. Flat prior: 45 -> 29 states (thin rows lose). With backoff to pooled rows: 39 states,
main-copy share 0.9787 -> 0.9826, 5 states still own two true states, unchanged for prior
mass 1 to 500. Code removed. Open: how to route rare edges using pooled evidence.
Update after the stop-rule fix (`same_states`, now `partition_key`, `logs/merge_coverage2.out`): corridor D = 1
N* 50 (product 5.6), all other rows unchanged, so the law is still not confirmed. Recovery now
rises with data, except D = 3 (7/8) and D = 4 (6/8) at 3,000. The table and the "falls with
more data" note below are from before the fix.
Coverage law N* = c / d_min, not confirmed (8 seeds, 75% exact, default event windows):
| Env | States | d_min | N* | N* d_min |
| hub k = 1 | 9 | 0.224 | 500 | 112 |
| star k = 1 | 9 | 0.302 | 2000 | 604 |
| corridor D = 1 | 10 | 0.112 | 300 | 33 |
| corridor D = 2 | 12 | 0.076 | 200 | 15 |
| corridor D = 3 | 14 | 0.051 | 200 | 10 |
| corridor D = 4 | 16 | 0.035 | 500 | 18 |
Hub k = 2: 45 states, d_min 0.022; hub k = 3: 189 states, d_min 0.0028. The product spans a
factor of 60, and the k = 2 notes (still wrong at 50k) sit above any c here. Within
corridors D = 2-4 the data needed grows far less than the A^d (16×) a direct long-future
test would need, but event windows are on, and the older D = 4 run without them failed, so
this comes from event windows, not from next-state evidence. Event windows are a language
set in the sense of Deb et al. 2024, so this is close to their fix, found independently.
Recovery falls with more data (corridor D = 1: 7/8 at 300, 1/8 at 3,000): cause unknown.

### A5. What history merging is: the cone picture `idea`
Notes from reading (2026-11-01), not yet an experiment. Past-future rows are points. Merge
removes duplicates (causal states), unmix finds the corners (HMM states), factor splits the
corners into a product. Rank, nonnegative rank and psd rank are factorizations through a
subspace, a simplex and the psd cone (Gouveia et al.; Glasser et al. for tensor networks).
Exact k-maze, all futures up to L (throwaway script, positive square roots only, so the
Born figure is an upper bound; phases might lower it):
| | Predictive states | Vertices | rank P | rank sqrt(P) |
| k = 1, hub or star | 9 | 8 | 8 | 9 |
| k = 2, hub or star | 45 | 28 | 28 | 45 |
Operator Schmidt rank of every A_{a,o} across location | bits: 1, for k = 1 and 2.
Reading: a mixture of beliefs is linear in P, not in sqrt(P), so a pure Born state with
positive amplitudes needs one dimension per predictive state; ignorance needs a mixed
state. This fits the observable fidelity (rho = bb^dagger works, plain fidelity fails).
Data check at k = 2 (10k episodes, T = 20, whitened full-history Hankel, L = 3 and 5): no
spectral gap at 28, singular values near 1 past 40, because most history rows are nearly
unique. So plain SVD does not by itself find the rare states.
Candidate next test, factor completion: find the product from well-visited states, then
generate the unvisited arm states as products, and score them on held-out data.

### A1b. Why structure learning is easy here and hard in general `queued`
The online learner (B1) only learns **unifilar** models: the next state is a function
of (state, action, observation), so the state is a function of the history and Bayesian
counting is exact. These are ε-machines / predictive states, the same objects BMR finds
offline (cf. CSSR, Shalizi & Klinkner 2004; Bayesian structural inference, Strelioff &
Crutchfield 2014).

General POMDP/HMM structure learning, as in Friston, Da Costa et al., "Supervised
structure learning" (2023) and Smith et al. (2020), allows latent states that the
history never pins down. It must infer them, which brings local optima, and learning
HMMs is hard in the worst case. The tractable general route is spectral / predictive-
state learning: the SVD of the past–future (Hankel) matrix, as in `fig_svd_states`.
That is what an MPS bond analysis computes.

Experiment: a task whose minimal unifilar model is large but whose non-unifilar or
quantum model is small. That task should separate B1 from the MPS route.

### A2. Per-action-normalized Born model (quantum transducer) `idea`
Constrain Σ_o K_{a,o}^† K_{a,o} = I for each action, so the Born MPS is a proper
conditional p(o | history, a) rather than a joint over actions. Removes the policy
dependence of the current joint model and lets the model stream indefinitely.

### A3. Offline pipeline write-up on frozen lake `queued`
Already have: `frozenlake_train.py`, `frozenlake_pomdp.py`, `make_fig_frozenlake.py`
(9 states recovered, rank-9 SVD, row x column factor test ranks 1 of 5040). Known leak:
the 4-step history cut and T = 5 were chosen with map knowledge; redo with the cut chosen
by the data (every bond, as in the online run) and a variable episode length.

---

## Track B — efficient online agent

### B1. Classical shared tensor, per-step learning, grow on surprise `done`
**Protocol (no task knowledge).** The agent observes (cell, reward) as one symbol and
prefers only reward. Episodes always run gymnasium's 100 steps, so the agent stays stuck
after reaching the hole or the goal; both become learned states. There is no terminal
flag and no hand-set hole penalty. RL baselines see the identical stream but are told
that the cell is the state (`src/frozenlake_rl.py`).

**Result (5 seeds, 150 episodes, `paper/figs/fig_frozenlake_sl.png`).** The EFE structure
learner recovers the minimal machine (10 states: 7 floor cells, hole, goal-entered,
goal-stuck; all pure) in episodes 0–2. Learned transitions reach TV 0.035 from the truth.
Goal rate over the last 50 episodes is 88–90%, with 20–22 falls into the hole in total.
- PSRL: 80–84%, 38–42 holes.
- Q-learning (ε = 0.1): 84–98%, 18–63 holes.
- Random: ~15%, ~125 holes.

**Strongest RL and the knob-free comparison (same stream, 5 seeds).**
- PSRL with the same sparse prior as our agent (Dirichlet 0.1): 92–98%, 20–23 holes.
  This is the best RL here, and it has no exploration weight.
- PSRL with Dirichlet(1): 80–84%, ~40 holes.
- OPSRL (Tiapkin et al. 2022, Algorithm 1, J = 10, κ = 1): 40–64%, 67–87 holes. Its
  optimistic pseudo-state pays r0 per step and assumes reward could come every step;
  this task pays at most 1 per episode.
- Posterior sampling on our learned states (`psrl_learned`, also knob-free): 86–94%,
  23–27 holes.

So learning the structure costs ~4–6 points against PSRL that is told the state. With
the novelty term, EFE reaches the full 10 states by episode ~5, against ~20 for
posterior sampling on learned states.

Our agent returns to the hole in short bursts. Transitions into it stay the least
counted, so their information gain periodically outweighs the reward (2 nats, once per
episode). The reward-preference magnitude sets this trade-off, as ε does for
Q-learning. Without the novelty term the agent never explores past its first fall (not
plotted). Design choices that mattered:
- count the entry observation (A side) in the split/merge evidence;
- one prior α = 0.1 everywhere (a prior of 1 caps novelty below the hole risk).

Original plan:
Uniform non-negative MPS with M^{a,o}[s', s] = p(o | s, a)·[s' = f(s, a, o)] (an
input-driven unifilar HMM / ε-machine transducer). Every step: Dirichlet count on
p(o | s, a). Grow: split a state when one incoming edge predicts a different future
(Bayes factor, the reverse of the BMR merge). Shrink: BMR merge with cascade so the
machine stays deterministic. Planning: sophisticated EFE with the novelty (parameter
information gain) term on the tensor's own counts. Benchmark: 3x3 lake, 100-step limit
(gym default). Conditions: EFE, EFE without novelty, random.

### B1b. Knob-free active inference: planning to learn `queued`
EFE's trade-off between reward and information is set by the preference precision
(REWARD_PREF). The principled way to remove it is Bayes-adaptive planning, where the
belief over parameters is part of the planning state, so exploration has value only
through the reward it later enables.
- RL side: BAMDP / BAMCP (Guez et al. 2012), approximated by posterior sampling.
- Active inference side: Sophisticated Learning (Hodson et al. 2023/2025), parameter
  updates inside the sophisticated-inference tree.
- Da Costa et al. (2023, Neural Computation) show that sophisticated inference with
  precise preferences gives Bellman-optimal actions over any finite horizon.

Test sophisticated learning with precise reward preferences on the learned machine
against PSRL. Fallback if it is intractable: an adaptive explore-to-exploit schedule for
both RL and AIF.

### B2. Active-learning term on the tensor's parameters (non-Dirichlet) `idea`
For models without conjugate counts (Born MPS, gradient-trained tensors): put a posterior
on the transfer tensor (Laplace approximation, or an ensemble of tensors) and compute the
expected information gain about its parameters directly. In the classical model of B1 the
Dirichlet counts already live on the tensor, so this is only needed for A1/A2-type models.

### B3. Memory task: cue + long corridor `queued`
The cue must be carried across many steps, so states that look the same now differ only in
the far future. One-step split tests (B1) cannot separate them; needs longer-context splits
(CSSR-style suffix contexts) or splits propagated backwards from where futures diverge.
This is where the bond has to grow as position x cue.

### B4. Factorized states `idea`
If the agent must remember several independent facts, the state count is their product
and the bond grows exponentially. Factor the transfer tensor (several small chains, or an
MPO-structured transfer). The factor test (row x column ranks first on the lake) is the
offline diagnostic for when this pays.

### B5. Hierarchical planner for long horizons `idea`
The model does not need a hierarchy (a shared tensor's size does not grow with T);
planning over 100+ steps does. Two levels: fast shared tensor for step dynamics, slow
model whose states are regions/subgoals (BMR on the fast states), one slow step = a
variable stretch of fast steps (deep temporal models). Tree tensor networks fit offline
multi-scale modelling but not streaming; MERA is polynomial (~χ^7–χ^9) but also does not
stream.

### B6. Bigger lakes `queued`
8x8 with the gym 200-step limit, then slippery. After B1 works on 3x3.

### B7. Baselines for scaling beyond tables `idea`
Tabular: PSRL (sparse prior) is the bar; OPSRL/Bayes-UCBVI are minimax-optimal but
tuned for per-step rewards. General deep RL to compare against once observations
are not symbols:
- EfficientZero V2 (ICML 2024): reports beating DreamerV3 on 50 of 66 tasks and BBF
  on Atari 100k.
- DreamerV3: a general world-model agent.
- BBF: Atari 100k.
- TD-MPC2: continuous control.
For memory tasks (B3), recurrent or world-model agents (DreamerV3) are the fair bar,
because they must learn state as we do.

### B8. Position-only Mountain Car `paused`
`mountaincar_env.py` (60 position bins + reward bit, velocity hidden, gym 200-step limit;
moving-flag option), `mountaincar_sl.py`, `mountaincar_born.py`, `mountaincar_rl.py`,
`mountaincar_dreamer.py`, `make_fig_mountaincar.py`. Seed 0, 500 episodes:

| agent | first flag | return, last 20 | total regret | time |
|---|---|---|---|---|
| PSRL told the state (20x20 grid) | ep 75 | -159.8 | 37497 | 513 s |
| Q-learning told the state | ep 361 | -200 | 49744 | 28 s |
| PSRL on the observation | never | -200 | 49757 | 21 s |
| classical tensor, EFE + novelty | never | -200 | 49757 | 5672 s, 123 states |
| classical tensor, posterior sampling | never | -200 | 49757 | 5822 s, 288 states |
| Born D=8 | stopped after 4.7 h with no output (profiled at 8.5 s/ep early on, then slowed down) | | | |
| DreamerV3 size1m, CPU | stopped (~58 ep / 70 min; GPU needs WSL) | | | |

Only the agents told the state reached the flag. The tabular learners grow states but
explore too slowly: nothing carries over between neighbouring bins (Tschantz et al. 2020
solved it with neural ensembles). Before more runs:
- speed up planning: re-plan every k steps or on a split, sparse DP over reachable
  states, and cap the Born belief graph's rebuild cost;
- per-episode progress prints and periodic result saves in every script;
- generalization across bins (factorized/ordinal states, B4) or a hierarchical planner (B5).
Planned next once fast enough: BSuite `mountain_car` protocol (20 seeds x 1000 episodes,
`ave_regret_score`) against the published DreamerV3 0.949 / Muesli 0.797
(arXiv 2301.04104, Table L.1).

### Done / baselines
- **Growing per-site Born chain (frozenlake_online.py).** Appends one site per step,
  re-trains between rounds, BMR over all prefixes. EFE agent reached the goal on the
  shortest path from round 1 (random: ~65% hole); learned B to TV 0.10. Does not scale:
  one tensor per time step, rounds slow down as the chain memorizes near-unique
  sequences. Lessons kept: repeat the terminal observation (a shared End token makes hole
  and goal indistinguishable), learn B every step, clip the cumulant gradient.
  Results: `results/frozenlake_online/`, `paper/figs/fig_frozenlake_online.png`.
