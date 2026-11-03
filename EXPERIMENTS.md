# Experiments so far

Written 2026-11-03 for Mao, coming back to the repo. This lists every experiment we've run, in
roughly the order we ran it: the question, what we did, what came out, and where the code and
results are. For why we ran them and what we think they mean, see [THINKING.md](THINKING.md).

The detailed records are still the source of truth. This file summarises them.
- [DEVLOG.md](DEVLOG.md): the July log (Parts 1 and 2 below).
- [FUTURE_EXPERIMENTS.md](FUTURE_EXPERIMENTS.md): the full record from September to November, items A0 to B8.
- [MERGE_PAC.md](MERGE_PAC.md), [ACTIVE_PAC.md](ACTIVE_PAC.md): the theory, with numerical checks.
- [ENV_CHECK.md](ENV_CHECK.md): our benchmarks checked against the ICLR 2025 RDP paper.
- [PLANS.md](PLANS.md): what's next, and the latest handover.

**Status words.** *Done*: finished and checked. *Partial*: ran, but with open problems. *Stopped*:
we killed the run or the line of work. *Refuted*: we tested the hypothesis and it failed.

**Numbers not to quote.** Learner results from before the stop-rule fix (older corridor and k = 2
figures in FUTURE_EXPERIMENTS.md, and the first coverage table). The clock comparison's "4–13×"
for Born, because that setup favoured Born.

**Note on git.** Everything from section 6 on is **not committed yet**. It exists only in the
local working copy. That covers `src/kmaze.py`, `src/rdp_*.py`, `src/gen_model.py`, `src/active_k2.py`,
the PAC notes, PLANS.md and `results/`. Section 2's work is on the branch
`analysis/structure-learning`, which isn't merged into `main`.

---

## Part 1. Learning the T-maze world model with an MPS (July)

### 1. MPS fit to the T-maze, and structure read off the fit. *Done.*
**Question.** Can we recover the T-maze's generative model (hidden states, A, B, factors) from a
Born-machine MPS trained on rollouts? Wauthier et al. (arXiv:2208.08713) said the MPS bonds "are
not normalized and therefore not usable" as states.

**Setup.** Two mazes: the minimal maze (4 observations) and the full pymdp T-maze (24 composite
observations). Trained on exhaustive rollouts with uniform actions. The full-maze fit uses a Han
learning-rate scheduler and 200 epochs, which took fit L1 from 0.30 to 0.005. The checkpoint is
`Saved_Models/FullTmaze.pt`.

**Results** (scripts in `src/`, figures in `paper/figs/`):

| What | Script | Result |
|---|---|---|
| Hidden states | `structure_recovery.py` | 4 (minimal), 7 (full). Raw bond-ray fidelity over-separates to 9, because it writes the uninformative context bit into the bond. Predictive equivalence is the robust criterion. |
| A = p(o\|s), B = p(s'\|s,a) | `structure_recovery.py` | Emission TV 0.000 on all 7 states; transition exactly 0.50/0.50. One bug fix mattered here: read A from the Born-squared joint, not by summing amplitudes, since summing amplitudes interferes on the superposed "centre" state. |
| Dependency graph | `structure_recovery.py` | Chow–Liu plus conditional MI gives the right chain; I(a1; o3 \| o2) = 0. |
| Observation factorisation | `factorization_test.py` | Context is nearly independent (subsystem quantum MI ≈ 0.05); position and reward are coupled (I = 1.0). |
| State count from evidence | `model_selection.py` | Fit knee at χ = 4 on the minimal maze. BIC/MDL is off by one (picks 3), because the naive parameter count ignores gauge freedom. |
| Symbolic states | `gauge_fix_states.py` | A unitary gauge into the cluster basis keeps the distribution to 4e-14; ARI 1.000. |
| Empowerment on the learned model | `empowerment_learned_model.py` | Post-cue 1.585 against an analytic 1.585. |
| p(o2\|s2) check (Philip's question) | `make_extension_figures.py` | A blind arm leaves context at 0.50/0.50, 0 bits. The cue gives 1.000 bit. Only the cue is informative. |

**Caveat we found afterwards.** In this dataset the context bit is drawn i.i.d. at each step, so it
is not a persistent latent. The agent experiments in Part 2 fixed that.

### 2. Seed protocol and a threshold-free state count. *Done (Oct, merged in PR #2 and on `analysis/structure-learning`).*
**Question.** Are the numbers in section 1 robust to the seed, and do they hold on held-out data?

**Setup.** A held-out 20% by trajectory, 5 seeds on the minimal maze, 3 on the full maze, 40 epochs.
`seed_protocol.py`, `seed_protocol_aggregate.py`, `bmr_states.py`, `observable_embedding.py`.

**Result.** Minimal maze: fidelity clustering counts 4.4 ± 0.5 states (it sometimes over-separates),
while BMR (Dirichlet-categorical evidence) counts 4 in every seed. The full maze gives 7 in every
seed, for alpha from 1e-3 to 0.5. Held-out NLL is 2.909 for the model against 2.890 for the
analytic model, and empowerment is 1.585 in every seed. We also added an "observable fidelity"
between histories that doesn't depend on gauge or phase (a four-copy contraction).

---

## Part 2. Phenotyping individual agents (July, `src/paper3/`)

### 3. Blind discrimination of agents from their recovered model. *Refuted (circular).*
**Setup.** A persistent-latent T-maze (`persistent_tmaze.py`), where context is a hidden episode
variable. The cue reads it at 1.000 bit, and an arm is a noisy readout (fidelity 0.85, 0.390 bit;
"go noisy, we shouldn't cheat"). Three EFE agents share A, B and D, and differ only in C, gamma and
horizon: an info-seeker, a reward-gambler and a habitual agent. One MPS per agent, then the three
agency criteria, then blind classification.

**Result.** Blind accuracy 1.00. Your adversarial audit then showed it was circular. The features
came straight from the agents' defining actions, the MPS was never consulted, and a bare P(a1)
histogram also scores 1.00. Recorded in the DEVLOG and `src/paper3/README.md`.

### 4. What limits recovery of an agent's world model. *Done, but it isn't a tensor-network result.*
Per-state recovery error: the L1 distance between the predictive signature read off the agent's
MPS and the analytic one.

- **Action entropy controls recovery, not visit rate.** error ~ H(a3): R² 0.807. error ~ log
  visits: R² 0.093. Both together: 0.904. The info-seeker spends 40% of its data at the cue and
  recovers the cue worst, because it always makes the same move there (Simpson's-paradox masking).
- **Gamma sweep** (`gamma_sweep.py`). Precision is the only knob, 0 to 16. Weighted H(a3) falls from
  1.99 to 0.02 bits, and error rises from 0.33 to 3.97. corr(gamma, error) = +0.987. The
  pre-registered falsifier did not fire.
- **Convergence confound closed.** The gap survives 25 vs 100 epochs. The loss was already flat at 25.
- **Fit and recovery are anti-correlated.** The deterministic agent's model fits its own data much
  better (loss 1.66 vs 3.69) and recovers the world much worse. So training loss can't stand in for
  recovery quality.
- **Floor at full support** (`floor_diagnosis.py`). More epochs make recovery worse (0.19 → 0.24),
  and bond 8 helps only 13%. This looks like overfitting to the empirical distribution.
- **Ceiling test** (training on the exact enumeration). The metric is unbiased: 0.01 to 0.12 on
  centre and cue. But it exposed a bug: `enumerate_persistent()` keeps the arm reward fixed across
  steps, while `agents.rollout()` resamples it. The agent results are internally consistent; the
  ceiling test itself is invalid as run. **Still to fix:** pick one convention and rerun.
- **Certificate** (`certificate.py`). A data-only identifiability verdict per conditional, AUC 0.997.
  The IDENTIFIED/UNIDENTIFIED split is close to a tautology. The WEAK tier, which falls in between,
  is the part with real content.
- **Confabulation hypothesis refuted** (`confabulation_check.py`). The MPS puts about zero mass on
  actions the agent never took. It's honest.
- **Is the error just a count?** About 74% of it is the number of unexercised (state, action) rows
  (R² 0.738). The other 26% is estimation quality on rare rows.

**Bottom line from July.** The finding is about data, not about tensor networks: you can only
identify a conditional where the agent varies its action, which is the positivity/coverage
assumption from causal inference and offline RL. You corrected the framing: epistemic value drives
*state disambiguation*, not *action coverage*, and that is the objective working as designed, not an
irony. One test is still open (see THINKING.md, open questions).

### 5. Analytic EFE and empowerment on the T-maze (yours, merged in PR #3). *Done.*
`src/efe_analysis/`: a per-action EFE decomposition, horizon-2 planners (open-loop and
sophisticated), and a noisy-arm sweep. Under the standard preferences (+2/−4) the preference term
alone already selects the cue, at every arm and cue reliability. The epistemic term does the work
only under mild preferences (+1/−1), where it breaks ties.

---

## Part 3. Structure learning straight from trajectories, no MPS (September–October)

### 6. Histories → predictive states → generator → factors (T-maze). *Done.*
**Pipeline.** Count prefixes, build predictive rows, BMR merges with a Perks prior, a unifilar
machine, unmix (Bayes factor: is this row a mixture of the others?), a generator, then a
factorisation (substitution-property partitions, Hartmanis–Stearns). Scripts:
`make_fig_hidden_states.py`, `generative_model.py`, `make_fig_generative_model.py`,
`structure_pipeline.py`, `make_fig_structure_stress.py`. The equations are in
`paper/figs/fig_structure_learning_math.png`.

**Results** (full pymdp T-maze, 3,276 trajectories):
- 24 histories → 7 predictive states, ARI 1.000. About 800 trajectories are enough (20 seeds).
- Rank 6, not 7. The centre row is exactly ½ cue-right + ½ cue-left, a belief. Unmixing gives
  ln BF +62 for the centre and ≤ −192 for every other state.
- Generator: 6 hidden states, D = (½, ½). Factorisation: of 27 SP partitions exactly 1 is a product,
  context (2) × place (3), with B = B_ctx ⊗ B_place exact.
- Beyond the data: the flat 7-state model forgets the cue on returning to the centre, and the
  generator keeps it. TV at step 3 is 0.036 vs 0.013.
- Stress test on three variants: original 2 × 3; arms reachable only from the centre gives 2 × 4
  (pymdp's own structure), but only with 3-step data; a third context gives 7 states and no product.
- Quantum: the centre is a mixture, not a superposition, so the classical generator already meets
  the rank bound here.

### 7. Frozen lake. *Done (offline and online), with one known leak.*
- **Offline MPS** (`frozenlake_train.py`, `frozenlake_pomdp.py`): 9 states recovered, SVD rank 9,
  and the row × column factor test ranks first of 5,040 partitions. **Leak:** the 4-step history
  cut and T = 5 were chosen knowing the map.
- **Online, growing Born chain** (`frozenlake_online.py`): one site per step. The EFE agent took the
  shortest path from round 1, and B was learned to TV 0.10. It doesn't scale, because there's one
  tensor per time step.
- **Online classical agent** (`frozenlake_sl.py`, `frozenlake_rl.py`, 5 seeds, 150 episodes): a shared
  non-negative unifilar tensor with Dirichlet counts, split on surprise, BMR merge, EFE plus novelty.
  It finds the minimal 10-state machine in episodes 0 to 2. Goal rate 88–90%, 20–22 falls.
  Baselines on the same stream, but told the state: PSRL with a sparse prior 92–98% (the best);
  Q-learning 84–98%; OPSRL 40–64%; PSRL on our learned states 86–94%. So learning the structure costs
  about 4 to 6 points. Without the novelty term the agent never explores past its first fall.

### 8. Position-only Mountain Car. *Stopped.*
60 position bins, velocity hidden. Only agents told the state reached the flag. Our tabular learners
grew 123 to 288 states but explored too slowly, since nothing carries over between neighbouring
bins. Born D = 8 and DreamerV3 on CPU were stopped for speed. Table in FUTURE_EXPERIMENTS.md B8.

### 9. Quantum vs classical: the probability clock. *Partial; the comparison was unfair.*
`clock_born.py`, `clock_compare.py`. Jaeger's clock is a qubit process that no finite HMM matches
exactly. Born at bond 2 is near-exact (+0.00002 bits above truth); Baum–Welch needs D ≥ 8 to get
within 0.003; our count learner uses 15 to 18 states (+0.00037). So Born wins clearly in bond
dimension and only slightly in bits, since the separation sits in a tail with 1.4% survival. **But**
the process is itself a Born model and the bond was given, so Born had a home advantage. A fair
version exists (`fair_clock_*`, `fair_hmm_*` logs) but didn't finish, and `clock_compare.py` no longer
matches `compare_s*.json`. Open question: does a signed rank-3 model match, which would make this a
result about non-negativity rather than about quantum?

### 10. Renormalising generative models (RGMs), MNIST, latent trees. *Partial.*
- `rgm_lite.py` re-implements SPM's structure-learning state generation. `factor_search.py` does a
  polynomial SP-partition search.
- Synthetic flapping bird: ours finds 3 × 8 (bob × wing), ARI 1.00, with 73 transition entries
  instead of 576. Searching Bell(24) ≈ 4e17 partitions was not needed. Lesson: one-step BMR can merge
  histories whose later futures differ, so a refinement pass is needed.
- MNIST as 64 tile symbols in raster order (`mnist_structure.py`): 75.0% test accuracy with a
  unifilar MPS of bond ≤ 72.
- Latent tree over pixels, with no order or neighbourhood given (`latent_tree.py`). Held-out
  bits/image: independent 71.1, Chow–Liu 58.2, k-d locality tree 52.6, ours 48.3–49.3, raster chain
  47.4. Locality emerges by itself, but the chain still wins, so greedy joining isn't finding the
  best shape.

---

## Part 4. The k-secret maze and the math of merging (October–November)

### 11. A learner for hidden bits: `src/kmaze.py`. *Partial.*
**Environment.** A hub, and per secret a cue room and two arms. Truth = location × k bits. Layouts:
hub, star (like the T-maze) and `Corridor(D)` (cue, D identical cells, junction).

**Learner** (`learn_states`). Start with state = last observation. Split edges entering a state by BMR
on pooled futures (1–3 step windows, event windows, a veto when the disagreeing modalities sum below
−3). Then minimise and bisimulate, and repeat until the partition repeats.

**Results.**
- k = 1: exact. 9 predictive states → 8 vertices = bit (2) × location (4), ARI 1.0 on each factor.
- k = 2: 44–45 states of 45 at 30,000 episodes, never exact (`logs/kmaze_k2_fixed.out`). The 45 is
  partly luck (97.9% of visits on the right state).
- Corridor: recovery rises with data, except D = 3 (7/8) and D = 4 (6/8) at 3,000 episodes.

**Planning on the learned machine** (`kmaze_plan.py`, value iteration vs fitted Q on windows of the
last W steps). Corridor D = 2: ours optimal from 300 episodes. D = 4: ours optimal from 1k where no
window size is. D = 8: ours fails, because cells more than 3 steps from the junction look alike.
(These are older numbers, from before the stop-rule fix.)

### 12. The math of merging. *Done for pairs and idealised learners; nothing covers `learn_states`.*
`merge_theory.py`, `merge_pac_check.py` (19/19 checks pass), MERGE_PAC.md.
- **Merge test.** dF ≈ −n·JS + (m−1)/2 · ln(n1 n2 / n) + c. Matches the exact value to 0.03 nats at
  n = 100.
- **Asymmetric errors.** Wrong merges fall exponentially in n·JS; wrong splits fall only
  polynomially (about h^−κ). Measured: P(split)·√n stays nearly constant from n = 100 to 10⁶.
- **The veto.** A veto on any single modality gives a wrong-merge guarantee under any dependence. The
  code's *sum* veto fires on equal noisy rows. Switching to a single-modality veto took star k = 1
  from N* = 2000 to 50, and cut the spread of N*·d_min from 108× to 5×. **Measured but not yet
  applied in `kmaze.py`.**
- **Coverage law N* = c/d_min.** Not supported, even after the fix.
- **Rare edges (Prop 8).** Routing an edge depends on that edge's own visits, for any learner. This
  is the k = 2 failure: pooled by true state, the evidence is −84 nats (−3 is enough), but the edge
  stepping out of the arm has only about 150 clean visits. It gets filed under "knows one", and later
  stages can only merge, so the distinction can't come back.
- **Deb et al.'s bound.** The 1/(d_min μ₀) bound in the 2024 preprint is wrong; the correct order
  is 1/(d μ₀²). The ICLR 2025 version already corrects it, so use that version.
- **Censoring refuted.** Event windows cut off by the episode end don't explain the corridor splits
  under a uniform policy. Policy noise in event windows is the untested replacement hypothesis.

### 13. Comparison with RDP learners (offline). *Done.*
`rdp_envs.py` (six benchmarks from Deb et al., ICLR 2025), `rdp_baselines.py` (AdaCT-H, FlexFringe),
`rdp_language.py` (their language metric), `rdp_compare.py`, `rdp_summary.py` →
`logs/rdp_summary.out`. 150 cells: 6 environments × 5 data sizes × 5 seeds, 15 minutes per fit,
rules written before the runs.

Gap above the Bayes floor, nats per step, at 30,000 episodes:

| Env | Ours | Language AdaCT-H | AdaCT-H | FlexFringe | HMM | GRU | States: ours / language / FlexFringe / truth |
|---|---|---|---|---|---|---|---|
| T-maze | 0.0006 | 0.0136 | 0.0241 | 0.0012 | 0.0061 | 0.0002 | 15 / 14 / 53 / 11 |
| Cookie | 0.0008 (1 of 5 timed out) | 0.0192 | 0.66 | 0.0156 | 0.0032 | 0.0002 | 17 / 41 / 214 / 14 |
| Corridor | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0001 | 9 / 11 / 13 / 12 |
| Cheese | 0.0014 | 0.11 | 0.41 | 0.030 | 0.0010 | 0.0020 | 27 / 61 / 488 / 14 |
| Mini-hall | 0.0002 | 0.25 | 0.48 | 0.027 | 0.0037 | 0.0014 | 22 / 123 / 448 / 18 |
| Tiger | timed out, 25/25 | 0.029 | 0.055 | 0.014 | 0.0002 | 0.0013 | – |

We match or beat every state-merging baseline at every size, usually with much smaller machines.
HMM and GRU predict better at low data, but give no readable states. Tiger times out, as predicted:
it has no finite unifilar model. Ours is slower than the merging baselines.

**Benchmark fidelity** (ENV_CHECK.md). Our T-maze junction observation differs from the paper's
figures. That changes results: minimal size 21 → 20. Uniform behaviour can't show the T-maze cue at
any practical n (the cues differ by 1/1024). We couldn't reproduce their Table 1. Their Cheese and
Mini-hall rows are 16× and 26× above the true minimum, which we never saw at any n.

### 14. Generative model and EFE planner on the learned states. *Done.*
`gen_model.py` → `results/gen_model/table.txt`. Return against the optimum at 10,000 episodes:
Corridor 5.00/5.00, Mini-hall 2.35/2.35, Cheese 1.19/1.19, Cookie 0.96/1.00, **T-maze 1.50/4.00**,
Tiger no model. The planner reaches the optimum on the true states, so the T-maze failure comes from
the learned states. The two cues merge, because the goal matters 5 steps later and shows up on only
about 1/256 of random paths. Epistemic value helped on Cookie in some settings and hurt in others,
and did little elsewhere. Learned states are readable, including a hidden "cookie exists" state
(`results/gen_model/show_cookie_n10000.txt`).

### 15. Active structure learning, theory on synthetic cases. *Done as theory; not run on a maze.*
`active_pac_check.py` (44/44 checks pass), ACTIVE_PAC.md. The headlines:
- **Tails stay valid under active data collection.** Given a true state, a tail started at a
  stopping time has the right law whatever the prefix was. On an impure label it does not:
  p(o | label, a) moved from 0.788 to 0.200 with the prefix.
- **Event windows under active tails** depend on time left: a built example gives a false split with
  probability 1.00; a fixed horizon gives 0.00. How much this matters on the mazes is unknown.
- **Limits.** Actions can't lower the visits an edge needs, only the episodes it takes to get them.
  Distinguishability gains at most A^L. On an unknown path the gain is capped at min(2n*, A^D).
- **Novelty vs structure information.** Novelty allocates samples by support, not by gap. With one
  hard edge and seven easy ones: count 2013 steps, novelty 2143, structure information 310,
  elimination 296. Most of the saving comes from stopping on evidence. On noisy rows, novelty takes
  7× as long as plain counting.
- **The chain rule** I(o; S, θ, q) = I(o; S) + I(o; θ | S) + I(o; q | θ, S) is exact. But only if
  the novelty term is mixed over structure (a decided merge reads the pooled row). The per-edge
  version collapses to count-based exploration (1990 vs 310). `combined` in `active_k2.py` is a
  proxy, not this sum.

### 16. Active run on k = 2: `src/active_k2.py`. *Stopped on purpose.*
Arms: uniform, novelty, structure, combined, oracle. Stopped at 11 of 48 checkpoints, because it was
too big and started at the slowest data sizes. So far it gives no comparison between arms. Partial
scores are in `results/active_k2/*.tmp`. Uniform, seed 1: 8,000 episodes, 30 states, 95.3% of visits
on the right state.

---

## Dead ends, so nobody retries them
- Red-blue merging on the history tree (rare histories have too little evidence).
- An unvetoed BMR sum (Occam gains outvote real differences).
- A Perks prior over all O^L sequences (fresh rows cost too much).
- A next-state modality inside minimisation (over-splits from stale successors).
- Split-only rounds without minimising (1,589 duplicate states).
- Redirecting only on a clear 90% majority (duplicate chains come back).
- Hard-EM reassignment of edges to the best-fitting state (thin rows lose).
- Censored event windows as the cause of corridor over-splitting (refuted).
- Thresholded binary coverage as a phenotype (the threshold is arbitrary and misses the effect).
- Blind agent discrimination from behavioural features (circular).

## Next experiment (agreed in outline, not started)
Does exploration help at all? k = 1 hub and star, 100 to 1,600 episodes, plus the RDP T-maze.
Arms: uniform, novelty, structure, oracle. One seed first. Then one active step on k = 2 from a
shared 8,000-episode base. Details in PLANS.md.
