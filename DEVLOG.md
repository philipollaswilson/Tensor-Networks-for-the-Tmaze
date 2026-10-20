# Devlog

## 2026-07-19 (evening: adversarial audit -- the "killer experiment" was circular)

Mao asked me to be adversarial about the Paper III result. The headline (blind
discrimination, accuracy 1.00) does not survive.

- **The tensor network was doing no work.** blind_validation's features
  (cue_visit, arm_first) are counted straight off `actions[1]` in the rollouts --
  i.e. the agents' *defining* behaviour (the info-seeker is the one built to visit
  the cue). The trained MPS is never consulted anywhere in the discrimination
  path. Docstrings claiming these were "marginals of the recovered model" were
  FALSE and are now corrected.
- **A trivial baseline ties it.** Raw P(a1) histogram, no criteria / EFE /
  inverse-C / tensor network: accuracy 1.00 (12/12). So none of the machinery
  contributes to the number.
- **"Held-out" was weak**: the same three specs re-seeded; near-deterministic
  agents put held-out points on the reference centroids, so 1.00 is guaranteed by
  construction.
- Why Paper II is untouched by this: its claims REQUIRED the latent model
  (7-state partition from bond rays, A/B to TV<=0.001, subsystem MI, empowerment
  on the learned model). None of those have a raw-behaviour analogue. The Paper
  III error was choosing a phenotype raw behaviour supplies for free.

Rebuild, in order (each checked rather than asserted):
- phenotype.recovered_action_marginal / coverage_features now read the action
  marginal off the CONTRACTED BORN JOINT of a trained MPS. Axis verified
  empirically (axis 2; MPS [0,.524,.438,.038] vs empirical [0,.52,.444,.036]),
  not trusted from the einsum string.
- **Negative result: thresholded coverage does not work.** Predicted the support
  floor might not bite; measured it and it does not. The gambler still visits the
  cue ~2% of the time (history weights 0.017/0.020), clearing any sane threshold,
  so it "supports" the cue states despite having almost no data there. The only
  history it loses is `center` (it never idles) -- incidental, not the cue
  blindness that matters. Binary coverage is threshold-arbitrary and misses the
  effect.
- **Pivot: per-state recovery FIDELITY** (recovery_fidelity.py). L1 between the
  predictive signature recovered from the agent's MPS (bond ray x T3) and the
  ANALYTIC signature of the true model. Continuous, threshold-free, and it needs
  the tensor network. Analytic truth sanity-checked in isolation first:
  q(K|cue)=[1,0], q(K|R,cheese)=[0.15,0.85], rows normalised, cue/ctx0 vs ctx1
  differ by L1 2.8.
- Bug caught: indexed T1 with three subscripts, feeding the obs index into the
  action slot (T1 is bond,action,obs,bond). Cost a full training run; shape
  asserts added so it fails fast.
- Gambler smoke map is the expected shape: arms (w~0.24) err 0.37-1.07, cue
  (w~0.02) err ~6, center (w=0) err 7.3 -- "you recover the world where you go".
- **Audit of the pivot is in flight** (fidelity_analysis.py): regress error on
  log10(weight) across all (agent,state) pairs. If R^2 ~ 1 the fidelity map is
  the visit rate restated and must be reported as such. Adding Spearman too,
  since a linear fit could under-fit a saturating relation and deflate R^2.
- Still open: structure_vs_visitrate.py -- holds the first-action distribution
  EXACTLY fixed while varying only second-step exploration. Largely superseded by
  the result below, which found the mechanism directly.

**RESULT: action entropy gates identifiability; data volume sets the rate.**

Per-state recovery error vs the analytic truth (1500 episodes, 35 epochs):

    agent            center  R/chee  R/shок  L/chee  L/shок  cue/c0  cue/c1
    info-seeker       3.000   3.496   2.039   3.489   2.904   6.001   6.015
    reward-gambler      nan   0.316   0.482   0.304   0.242   6.044   6.012
    habitual          0.288   0.318   0.453   0.452   0.310   0.275   0.246

Regressions over the 20 (agent,state) pairs:
    error ~ log10(weight)                R^2 = 0.093
    error ~ H(a3)                        R^2 = 0.807
    error ~ H(a3) + log10(weight)        R^2 = 0.904
    error ~ log10(weight) | H(a3) > 1.4  R^2 = 0.909

So recovery is NOT driven by how often a state is visited. Every state with
error ~6 has H(a3)=0 (exactly one third action ever taken); every state with
error <0.5 has H~2 (all four). The info-seeker holds 40% of its data at the cue
and recovers it WORST, while recovering its arm states -- 0.4% of its data, ~100x
less -- BETTER, because at the cue it is deterministic and at the arms it is not.
Marginally this masks the volume effect (R^2 0.09) because the highest-weight
states are exactly the zero-entropy ones; conditioned on the gate being open,
volume predicts recovery strongly (R^2 0.91). A Simpson's-paradox masking.

Interpretation, and the honest Paper III thesis: **agent epistemics and observer
epistemics are opposed.** The info-seeker resolves ITS OWN uncertainty at the
cue, which makes its subsequent behaviour deterministic, which destroys the
observer's ability to identify the conditionals. Competence removes the action
variation that identification requires. The RANDOM agent yields the best
recovered world model. This also explains Paper II: its exhaustive uniform-action
rollouts are the maximum-entropy best case, which is why it reached TV<=0.001.
Paper III is what happens when you drop that assumption and watch a real agent.

**GAMMA SWEEP: the tradeoff is continuous and monotone.** One agent family
(info-seeker's C and horizon fixed), precision gamma the only knob, 1200 episodes
/ 25 epochs each:

    gamma  cue_visit  H(a3) mean  H weighted  recovery err
      0.0       0.30        1.99        1.99         0.325
      0.5       0.34        1.84        1.85         0.299
      1.0       0.37        1.63        1.61         0.529
      2.0       0.45        1.41        1.26         1.139
      4.0       0.59        1.34        0.96         1.664
      8.0       0.84        1.18        0.38         2.228
     16.0       0.98        0.82        0.02         3.967

    corr(gamma, H_weighted) = -0.934
    corr(gamma, error)      = +0.987
    corr(H_weighted, error) = -0.962

The pre-registered falsifier (corr(gamma,error) ~ 0 => tradeoff does not bind)
did NOT fire. Recovery error degrades 12x across the competence range while
weighted action entropy collapses from 1.99 to 0.02 bits. This is the paper's
headline figure: competence is not free, it is paid for in the observer's ability
to identify the agent's model.

**CEILING TEST FOUND A REAL BUG: the two data generators implement DIFFERENT
ENVIRONMENTS.** Training on the exact weighted enumeration (zero sampling noise)
gave:

    mean err 1.1965      center   0.0822   cue/ctx0 0.0132   cue/ctx1 0.1204
                         R/cheese 2.0400   R/shock  2.0399
                         L/cheese 2.0398   L/shock  2.0400

Center and cue recover to 0.01-0.12, so THE METRIC IS NOT BIASED -- the
instrument works, which was the question the test was built to answer. But all
four ARM states sit at ~2.04, a flat signature rather than noise.

Cause: `persistent_tmaze.enumerate_persistent()` makes the arm reward PERSIST
across steps (r2 = r1 when you stay in an arm, inherited from the original
tmaze.py), while `agents.rollout()` RE-SAMPLES it fresh from the noisy channel at
each step. The analytic `recovery_fidelity.true_signature` matches the ROLLOUT
convention. So this test trained on one environment and scored against another,
and the mismatch shows up exactly at the states where persistence applies.

WHAT IS AND IS NOT AFFECTED:
  * NOT affected -- every agent-based result (gamma sweep, fidelity map, entropy
    decomposition, certificate, convergence and floor diagnoses). Those use
    agents.rollout throughout AND true_signature, which agree with each other, so
    they are internally consistent and valid for the re-sampling environment.
  * NOT affected -- the info-gain numbers (I(reward;ctx|arm)=0.390,
    I(cue;ctx)=1.000). They concern step-1 arm ENTRY, before persistence applies.
  * INVALID as run -- this ceiling test itself. It is the only place
    enumerate_persistent feeds the trainer.
  * TO FIX -- decide which convention is the real environment and make both
    generators agree, then re-run the ceiling test. The original tmaze.py used
    persistence, so that is probably the intended semantics and agents.rollout is
    the one that drifted.

Silver lining: the ceiling test did its job. It was built to ask "is my
measuring instrument biased?" and the answer is no -- the states it can score
cleanly it scores at 0.01-0.12, far below the ~0.2 floor. That means the floor
seen under sampled rollouts is a property of the DATA, not of the metric.

**FLOOR IS STRUCTURAL (floor_diagnosis.py).** Fixed N=3200, full support:

    epochs  40 bond 4   err 0.1920
    epochs 100 bond 4   err 0.2231
    epochs 200 bond 4   err 0.2377   <- MORE TRAINING IS WORSE (-24%)
    epochs 200 bond 8   err 0.2058   <- capacity helps only 13%

Neither budget nor capacity removes it, so the strict positive claim (error -> 0
under full support) is NOT established. My leading hypothesis (budget) was wrong.
Mechanism is probably ordinary overfitting: more epochs tightens the fit to the
EMPIRICAL distribution, which at finite N is a noisy sample. Same phenomenon as
the fit/recovery anti-correlation, now WITHIN one agent instead of across agents:
across agents concentrated data gives better fit + worse recovery; within an
agent more epochs gives tighter fit + worse recovery. Fitting your data is not
recovering the world.

OPEN: whether the error METRIC itself has a bias floor. Training on the exact
weighted enumeration (zero sampling noise, what Paper II did) is the ceiling
test -- ~0.01 means the floor was sampling variance; ~0.2 means the instrument is
biased and no absolute number here is quotable. Queued.

**CERTIFICATE VALIDATED, AUC 0.997 (certificate.py).** Per-conditional verdicts
computed from data + model ONLY (no ground truth, since none exists in the wild),
checked against the recovery error they claim to predict:

    IDENTIFIED    n=53  mean support 67.6  mean err 0.082
    WEAK          n=12  mean support  1.5  mean err 0.778
    UNIDENTIFIED  n=19  mean support  0.0  mean err 1.474
    AUC 0.997

Honest reading: the binary half is near-tautological -- zero-sample rows cannot be
learned, so flagging them predicts failure by construction. The non-trivial part
is the WEAK tier landing cleanly between the extremes, i.e. graded discounting
rather than presence/absence. Also clarifies the floor: identified rows average
0.082 PER ROW, so the ~0.2 PER-STATE aggregate is inflated by averaging
unidentified rows into it -- the well-supported parts recover better than the
aggregate suggested.

**CONCEPTUAL CORRECTION from Mao: drop the "irony" framing.** An agent ceasing to
explore once its uncertainty is resolved is epistemic value working AS DESIGNED,
not an ironic reversal. The precise, defensible claim is:

    Epistemic value drives STATE DISAMBIGUATION, not ACTION COVERAGE.

The info-seeker's epistemic term concerns its own posterior over the hidden
context. Visiting the cue resolves that -- one epistemic act -- after which
varying its action can only cost reward, so it never has reason to try
alternatives from the same state. But action coverage is exactly what an observer
needs to identify the dynamics. There is no term in EFE for "be legible to an
observer". Constructive corollary: making an agent's behaviour identifiable needs
an explicit objective for it (an observer-directed epistemic term, or DAgger's
trick of externally injecting the coverage that competence removes).

**CONFABULATION HYPOTHESIS: DEAD (confabulation_check.py).** Hypothesis was that
the MPS invents confident structure at conditionals the agent never exercised --
the one claim that would have made this a TENSOR-NETWORK result rather than an
identifiability note. The artifact control built to catch exactly this failure
mode caught it.

    gamma=16      n   mean err   mean H   mean rel-mass
    exercised    14      0.622     0.30          0.4999
    generalised   2      0.364     0.30          0.0005
    CONFABULATED  4      1.872     0.44          0.0001
    hedged        8      1.214     3.19          0.0000
    (gamma=0 control: all 28 rows exercised, mean err 0.114)

The "confabulated" rows carry relative mass 0.0001 vs 0.4999 for exercised rows.
The model assigns essentially ZERO probability to actions the agent never takes:
it is HONEST, saying "this does not happen". The apparent confidence was my own
row normalisation dividing a near-zero row by its near-zero sum. The model comes
out better than the hypothesis; the hypothesis is refuted.

**IS THE ERROR METRIC JUST A COUNT?** Follow-up, since error ~2 per unexercised
row suggested the whole apparatus might reduce to counting unsampled (state,
action) pairs -- computable with no MPS at all.

    gamma   unexercised   mean state err
      0.0        0/28              0.457
      2.0        5/28              1.191
      8.0        6/28              2.273
     16.0       14/28              3.805

    state-level  error ~ #unexercised:  slope 1.330, R^2 0.738, corr +0.859 (n=28)

R^2 = 0.738, NOT ~1.0. And gamma=2 vs gamma=8 have near-identical coverage (5 vs 6
unexercised) yet errors differ ~2x (1.19 vs 2.27), which counting cannot produce.
So recovery error decomposes into ~74% binary coverage (countable) + ~26% graded
estimation quality at rows that ARE sampled but rarely. The metric is more than a
count, but the count dominates.

**BOTTOM LINE: no tensor-network-specific claim survives.** The graded 26% is
sample efficiency at rare actions, which any estimator exhibits. With the
confabulation hypothesis refuted, the finding is about DATA, not about tensor
networks. It is real; it is not evidence for the method.

Options recorded for Mao's call: (a) reframe Paper III as a methodological study
of the limits of behavioural structure learning, stating plainly that the limit
is estimator-agnostic; (b) fold it into Paper II as the limitations section
quantifying what its exhaustive-uniform-action assumption buys; (c) one bounded
test of the only untried TN-specific route -- apply Paper II's ACTUAL machinery
per agent (bond-ray state partition, recovered A/B, subsystem MI) and ask whether
those differ informatively across agents rather than merely tracking coverage.
Do NOT spend compute on more mazes until this is settled.

**CONVERGENCE CONFOUND CLOSED (convergence_check.py).** The sweep fixed epochs,
not convergence, so the trend could have been uneven under-training. Same
rollouts trained at 25 vs 100 epochs:

    gamma   err@25   err@100    delta   loss@25 -> loss@100
      0.0    0.457     0.421   -0.036   3.689 -> 3.700
      4.0    1.950     1.852   -0.098   3.031 -> 3.029
     16.0    3.805     3.603   -0.202   1.661 -> 1.663

    error spread (gamma16 - gamma0): +3.349 at 25 epochs, +3.183 at 100 (95% kept)

The gap survives a 4x budget, so the sweep was not measuring convergence speed.
NOTE the test passed by a different route than designed: the intended proof was
"error flat WHILE loss keeps descending"; in fact the loss is flat to three
decimals, i.e. the models were ALREADY CONVERGED at 25 epochs. That closes the
confound just as well (nothing was under-trained) but the advertised corroboration
did not occur -- recorded here rather than quietly banked as a pass.

**Unplanned finding, and the sharpest form of the result: fit quality and
recovery quality are ANTI-CORRELATED.** Training loss across gamma runs
3.689 (gamma=0) -> 1.661 (gamma=16): the deterministic agent's model fits its own
data far BETTER while recovering the environment far WORSE (error 3.8 vs 0.46).
Its data is low-entropy and concentrated, hence easy to model. The high-gamma
model did not fail to learn -- it learned its data very well, and its data does
not contain the environment. Methodological consequence for the paper: TRAINING
LOSS CANNOT BE USED AS A PROXY FOR RECOVERY QUALITY. A well-fit model of
self-selected data is still a bad model of the world.

Minor honesty notes on the curve: gamma=0.5 dips slightly below gamma=0 (0.299 vs
0.325), within run-to-run noise -- do not over-read the very low end. And
error_mean is unweighted across the 7 states, so at high gamma it mixes the
entropy effect with reduced arm coverage; the entropy dominance is established
separately by the R^2 0.81 vs 0.09 decomposition, not by this curve alone.

Caveats to keep honest: (a) ~81% of the error variance is predicted by action
entropy, a BEHAVIOURAL statistic, so the fidelity map is not independent of
behaviour -- the contribution is the identifiability law, not a classifier;
(b) error ~6 means UNIDENTIFIED (unexercised rows hit the metric ceiling), not
"learned wrongly"; (c) the info-seeker's H=0 follows from gamma=16 -- a gamma
sweep would turn this into a continuous determinism-vs-identifiability curve, and
is the obvious next experiment.

## 2026-07-19 (Paper III kickoff: branch + persistent-latent environment)
- Started Paper III on branch `paper3/agent-phenotyping` (off `analysis/structure-recovery`).
  Scaffolded `src/paper3/`: persistent_tmaze, agents, agency_criteria,
  blind_validation, phenotype, README. Stubs raise NotImplementedError with TODOs.
- **Workstream 4 (persistent-latent env) built + verified.** The key realisation:
  in Paper II "context" is the 3rd observation modality, so it was directly
  OBSERVED every step; making it merely persistent would leave Philip's info-gain
  worry vacuous. Fix: context is now a genuine persistent HIDDEN latent, emitted
  only indirectly -- the cue reads it (I(cue;ctx)=1.000 bit, safe), an arm's
  reward is a NOISY readout (fidelity 0.85 -> I(reward;ctx|arm)=1-H(0.85)=0.390
  bit; leaks but does not resolve -- Mao said "go noisy, we shouldn't cheat").
  Keeps the [4,3,2]->24-obs shape so the whole Paper II pipeline transfers.
- Fixed a diagnostic bug: info-gain was measured against a context INFERRED from
  the reward (circular -> spurious 1.000 bit). Now measured against the true
  hidden context via _enumerate_with_context.
- Step 1 (trainer bridge) verified: to_memory_pool builds an mpstwo MemoryPool
  with obs (3,24)/action (3,4)/complex128, the shapes MPSTwo expects.
- Environment check: pymdp + MPS stack importable (MPS needs the standard
  gym/pymdp shim). Installed pymdp exposes pymdp.envs.TMaze (generate_A/B/D,
  reset, step) + pymdp.agent.Agent. Class is TMaze, not the old TMazeEnv that
  mpstwo.envs imports -- for Paper III we use pymdp directly.
- Step-2 architecture settled: drive a real pymdp Agent (genuine active
  inference) against OUR verified env; agents share A/B/D, differ only in C and
  gamma, so phenotype differences are pure character.
- Commits: 0ad3ead (skeleton), c53dd45 (env), 370586b (step-2 plan).
- **Step 2 done + verified (e3c4f96).** Installed pymdp is the JAX build (Agent
  is functional JAX; its TMaze is a foreign 5-location env), so instead of
  driving it we wrote a small transparent EFE agent on our own generative model
  (generative_model.py, verified against the env: 0.390 / 1.000 bit). Agents
  share A/B/D, differ only in C, gamma, horizon. Behaviour separates and
  rollout() asserts it: info-seeker cue-first 0.98, gambler arm-first 0.96,
  habitual ~random. Horizon is the distinguishing trait (cue-then-arm is
  reward-optimal, so the gambler differs by being myopic, not by a rigged C).
- **Step 3 done + verified.** phenotype.train_agent_mps reuses the full_tmaze
  Han-scheduler recipe on each agent's own rollout pool; run_experiment reports
  fit L1 to the agent's empirical joint. Smoke (300 ep, 15 epochs): info-seeker
  0.007, gambler 0.027, habitual 0.039 -- each MPS already reproduces its agent's
  behaviour. Full run (5000 ep, 200 epochs) will tighten these.
- **Step 4 done + verified.** agency_criteria operationalises all three Paper I
  criteria on the recovered model, reusing the agents' own EFE: intentionality =
  peakedness of the inverse-inferred C, rationality = 1 - normalised EFE-regret,
  explainability = model fidelity. Separates the roster:
  info-seeker (intent 0.96, ration 0.98, C=(5,-5)), gambler (0.11, 0.99, C=(1,0)),
  habitual (0.00, 0.51, C=(0,0)). Two honest properties documented in the module:
  (a) rationality is measured vs each agent's OWN inferred C, so the gambler is
  "consistent with shallow prefs", not irrational -- intentionality is what
  separates it; (b) inferring C at a fixed horizon-2 conflates myopia with weak
  preference (motivates inferring horizon as a 4th trait, future work).
- **Step 5 done + verified -- the killer experiment works.** blind_validation
  freezes a PreRegistration (features, decision rule, disjoint reference/holdout
  seeds, behavioural predictions), builds reference centroids on seeds (0,1,2),
  then blind-classifies held-out agents on seeds (100-103). **Blind accuracy
  1.00** (12/12), confusion matrix diagonal. Discriminates on
  (rationality, cue_visit, arm_first): rationality separates habitual, cue_visit
  the info-seeker, arm_first the gambler.
- Finding surfaced en route: INTENTIONALITY is under-identified. MLE inverse-C
  flips across seeds because cue-seeking is explained equally by peaked-C (reward)
  or flat-C (pure curiosity/epistemic). Stabilised to a posterior-expected,
  prior-relative goal-directedness (~0 when behaviour underdetermines C) and
  documented as a genuine identifiability result; discrimination deliberately
  does NOT use it. Resolving it needs inferring the epistemic weight / planning
  horizon as a separate trait (Paper III extension).
- Spine 1+2+3+4 now implemented end-to-end and verified. Commits this session:
  0ad3ead, c53dd45, 370586b, e3c4f96, a8dce31, be779b8, + step-5 commit.
- NEXT (open): read cue_visit/arm_first off the trained MPS directly (not the
  fit rollouts); full-scale training run; empowerment facet on recovered models;
  then write the Paper III draft. Optionally infer horizon to identify
  intentionality.

## 2026-07-17 (Philip's p(o2|s2) / info-gain question)
- Philip asked to see p(o2|s2), worried that seeing Cheese/Shock at a blind arm
  resolves context (info-gain), contradicting "only the cue is informative."
- Checked it directly against the model: at a blind Right arm the emission leaves
  context at 0.50/0.50 for all arm states, I(reward; start-context) = 0.000 bits, so
  the context posterior is unchanged. The cue gives I(cue bit; next arm reward) = 1.000
  bit. Learned model matches both to <1e-3. His concern does not bite; the recovery
  reproduces the intended information geometry. `R/cheese`/`R/shock` split on the reward
  outcome (distinct predictive states) but carry no context.
- Note for the meeting: in this dataset (enumerate_weighted/tmaze.py) the context bit is
  drawn i.i.d. at each observation and only the bit read AT the cue governs the next
  arm's reward, so context is not a persistent episode latent like the canonical Friston
  T-maze. Our recovery faithfully reflects that generation.
- Added `fig_emission()` to make_extension_figures.py -> paper/figs/fig_emission.png and
  an "Only the cue is informative" paragraph + figure to §4. Paper now 13 pp, compiles.
- The transition figure Philip screenshotted lives on analysis/structure-recovery (not
  the analysis/learned-empowerment branch he merged); he can't regenerate it from what
  he has.

## 2026-07-14 (wrap: sent to coauthors)
- Paper II draft + pipeline pushed to branch `analysis/structure-recovery`; sent the
  group a summary and three open questions (retitle, agency scope, author order, and
  whether Philip sees it as distinct from his Section 5). Awaiting coauthor feedback.

## 2026-07-14 (night: Paper II hardening)
- Retitled to *Structure Learning of Generative Models for Agency Phenotyping* — we
  structure-learn a generative (world) model; agency is the application, not the object.
- De-duplicated against Paper I (verbatim criteria/empowerment phrasing rewritten; only
  author block, refs, and Paper I's cited title now overlap) and removed AI-writing tells
  (em-dashes 25→~10, no appositive-dash cadence, no hedges). Prior work is now cited
  normally rather than labelled "Paper I".
- Added a "What is learned, and from what" subsection: the MPS is fit to exhaustive
  uniform-action rollouts, so it recovers the *environment's* generative model (checkable
  against q); agent-specific phenotyping would use that agent's rollouts (a limitation).
- Fixed a real method error: the state count is a predictive-future L1 threshold (τ=1.5),
  not a largest-gap dendrogram cut (which returns 5). Corrected structure_recovery.py,
  fig_states, and §3; redesigned fig_states around the over-separation→correction (novel)
  and cited Paper I for the fidelity method and the reused task schematic.
- Reference audit vs the web: fixed the AISTATS 2021 author order (Adhikary first, not
  Srinivasan); all other citations and the author/affiliation mapping verified correct.
- Scoped the agency claim: the contribution to agency is the recovered model (the
  belief/world-model substrate the three criteria read off), with empowerment as the
  controllability facet only.
- Updated README.md to document the pipeline and both papers.

## 2026-07-14 (evening: canonical model, figures, extension paper)
- Promoted the converged model to canonical: folded the Han-scheduler recipe into
  `full_tmaze_train.py` and replaced `Saved_Models/FullTmaze.pt` with it (fit L1 0.005);
  removed the v2 duplicates. All analyses default to the good model now. Regenerated fig8.
- Added `src/make_extension_figures.py` -> `paper/figs/`: state recovery (fidelity vs
  predictive equivalence), A/B recovery, MI dependency graph + factorization, model
  selection, empowerment phenotyping + gauge-fixed emitted states.
- Wrote the extension paper `paper/main.tex` (LNCS, 11 pp, compiles clean): "Structure
  Learning of Agency: Recovering Interpretable Generative Models from Behaviour with
  Tensor Networks." Promotes the learned-model pipeline from a companion section to the
  central contribution -- states, labeled A/B, dependency graph, factorization, principled
  state count, symbolic states, then empowerment phenotyping on the learned model. Closes
  the Wauthier et al. normalized-hidden-state gap.

## 2026-07-14 (afternoon: full-maze fit + weak-point fixes, branch analysis/structure-recovery)
- Added `src/full_tmaze_train_v2.py`: converged full-maze MPS with a Han LR scheduler
  (shrinks lr when the noisy DMRG/cumulant loss jumps up) + gentler lr + 200 epochs.
  **Fit L1 0.30 -> 0.0047** (60x). Empowerment now matches analytic essentially exactly:
  post-cue reward-only 1.5848 vs 1.5850 (old model 1.55, bond-4 checkpoint 0.65).
  Saved `Saved_Models/FullTmaze_v2.pt` (FullTmaze.pt kept intact).
- Fixed an extraction bug in `structure_recovery.py`: emission A=p(o|s) was read by
  coherently summing bond amplitudes over the unobserved start context, which INTERFERES
  on the one genuinely-superposed state ("center"). Now computed from the Born-squared
  joint (correct classical mixture). Center-state TV 0.18 -> 0.001; the other 6 states
  were already pure so unaffected. On v2, **all 7 states emission TV <= 0.001 (mean 0.000)**
  and the transition B is exactly 0.50/0.50 (was noisy 42/58). CI certificates now exactly 0.
- `structure_recovery.py` and `factorization_test.py` take an optional model filename arg
  so they run against either checkpoint. Item 3 sweep now uses the Han scheduler too.

## 2026-07-14
- Added `ROADMAP.md`: five ways to strengthen the "structure learning" claim beyond
  latent-dimensionality (labeled A/B, MI dependency graph, principled state-count selection,
  observation-factorization discovery, symbolic-state gauge-fixing), each grounded in a
  literature review with citations and a concrete recipe. Framing: Wauthier et al. (the
  precursor) could not extract normalized hidden states / A/B from the T-maze MPS; our
  ray-fidelity states supply the missing partition.
- Added `src/structure_recovery.py` (Phase A, no retraining):
  * Item 1 -- reconstruct emission A=p(o|s) and transition B=p(s'|s,a) from the learned MPS,
    pushing Born amplitudes to probabilities before forming stochastic objects. Minimal maze:
    emission matches analytic ground truth to **TV=0.0000** for all 4 states. Full maze:
    emission p(r3|a3,s) to **mean TV=0.025** (4 traps + 2 cue contexts near-exact; unresolved
    "center" carries the L1=0.30 fit error); transition recovers the 50/50 arm/cue branching.
    Finding: raw bond-ray fidelity OVER-separates (9 clusters) by writing the uninformative
    o2 context bit into the bond; predictive-equivalence (PSR/Hsu-Kakade) is the robust state
    criterion.
  * Item 2 -- MI dependency graph (Chow-Liu) + conditional-MI certificates, validated against
    the observable-projected POMDP structure. Minimal: recovers o2-o3, a1-o2, a2-o3 and
    I(a1;o3|o2)=0. Full: recovers the a2-o2-o3, a3-o3 chain.
- Added `src/factorization_test.py` (Item 4, no retraining): single-site RDM subsystem
  quantum MI. Model DISCOVERS that context is a nearly-independent factor
  (I(pos:ctx)=I(rew:ctx)~=0.05) while position-reward are coupled (I=1.0, causally real);
  grouping search picks (position,reward)x context (residual 0.17) over full separation (0.53).
- Added `src/gauge_fix_states.py` (Item 5, no retraining): unitary gauge into the cluster
  basis so the bond EMITS labeled states. Distribution preserved to 4e-14, mean max-prob
  0.907, ARI=1.000 vs fidelity clustering (cue states softer -- non-orthogonal-cluster effect
  of the complex Born machine).
- Added `src/model_selection.py` (Item 3, retraining): bond-dim sweep on the minimal maze,
  framed as active-inference model expansion + Bayesian model reduction. Fit/accuracy knee
  lands at chi=4 (the true state count); extra bond dimension is unused "spare capacity".
  MDL/BIC is conservative-by-one (picks 3) because the SGD trainer floors at ~0.14 fit and
  the naive param count ignores MPS gauge freedom -- reported honestly.
- ROADMAP.md now fully executed: all five items implemented and validated off the checkpoints
  (Items 1,2,4,5 no retraining; Item 3 a light sweep). Five research briefs (A/B extraction,
  MI/Chow-Liu, AIF model selection, factorization, gauge-fixing) informed the methods.

## 2026-07-13
- (uncommitted, work in progress) src/Utils.py rewritten (182+/115−): the matplotlib/qiskit density-matrix plotting helpers were replaced with tensor-network MPS utilities — init_mps builds an MPS_c from a dataset + config (cutoff, descent steps, batching) with left-canonicalization and cumulant init, and check_mps verifies left-canonical orthogonality tensor by tensor.
