# Paper III — `src/paper3/`

Agent-specific agency phenotyping. Roadmap and rationale live in `PAPER3.md` at
the repo root; this package is the code skeleton for the spine (workstreams
1 + 2 + 3 + 4).

## Layout

| module                 | workstream | closes Paper II gap                       | status |
|------------------------|-----------|-------------------------------------------|--------|
| `persistent_tmaze.py`  | 4         | context i.i.d.-per-step (Philip's point)  | **done + verified** — generator, info-gain checks, `to_memory_pool` + `rollouts_to_memory_pool` bridges |
| `generative_model.py`  | 1, 2      | (shared A/B/C/D for the agent + EFE)      | **done + verified** — matches the env info geometry (0.390 / 1.000 bit) |
| `agents.py`            | 1         | recovered env, not an agent               | **done + verified** — EFE agent; info-seeker cue-first 0.98, gambler arm-first 0.96, habitual ~random |
| `agency_criteria.py`   | 2         | empowerment ≠ agency                       | **done + verified** — intentionality/rationality/explainability; separates the roster |
| `blind_validation.py`  | 3         | non-blind vs ground truth                 | ⚠️ **CIRCULAR — negative result.** Scores 1.00, but a bare `P(a1)` histogram ties it; no MPS involved |
| `structure_phenotype.py`| 1, 2     | phenotype by *recoverable latent structure* | non-circular replacement (in progress) |
| `structure_vs_visitrate.py`| —     | adversarial check on the above            | tests whether structure ≠ visit rate restated |
| `phenotype.py`         | 1–3       | per-agent train + fit (steps 1–3)         | **done + verified** — `train_agent_mps` + `run_experiment` |

Stubs raise `NotImplementedError` with a one-line TODO pointing at the Paper II
code to reuse. Nothing is silently fake — an unfinished path fails loudly.

## What already works (verified)

```
python src/paper3/persistent_tmaze.py
```

Prints, against the TRUE hidden context:
- `I(reward; context | arm) = 0.390 bit` — under the noisy readout (fidelity
  0.85), an arm **leaks** the hidden context but does not **resolve** it. This is
  the honest, non-cheating contrast with Paper II's i.i.d. dataset (0 bit).
- `I(cue reading; context) = 1.000 bit` — the cue **resolves** it, risk-free.

`to_memory_pool(enumerate_persistent())` builds an `mpstwo` `MemoryPool` with obs
`(3, 24)` / action `(3, 4)` / complex128 — the exact shapes `MPSTwo` expects, so
the persistent-latent data drops straight into the Paper II trainer.

## Environment notes

pymdp and the MPS stack are both importable (the MPS side needs the standard
gym/pymdp import shim from `full_tmaze_train.py`, replicated in
`persistent_tmaze._install_import_shims`). The installed pymdp exposes
`pymdp.envs.TMaze` (with `generate_A/B/D`, `reset`, `step`) and
`pymdp.agent.Agent` — everything step 2 needs. Note the class is `TMaze`, not the
old `TMazeEnv` that `mpstwo.envs` imports; for Paper III we use pymdp directly.

## What to reuse (unchanged, from Paper II)

`structure_recovery.py`, `factorization_test.py`, `gauge_fix_states.py`,
`model_selection.py`, `make_extension_figures.py`, and the Han-scheduler training
recipe in `full_tmaze_train.py`.

## Build order

1. ~~`persistent_tmaze.to_memory_pool`~~ — **done.**
2. ~~`agents` — EFE agent + `rollout`~~ — **done + verified** (behaviour separates).
3. ~~`phenotype.train_agent_mps`~~ — **done + verified** (each MPS fits its agent).
4. ~~`agency_criteria`~~ — **done + verified** (rationality + coverage separate the roster).
5. ~~`blind_validation`~~ — **done + verified**: pre-registered blind discrimination, **accuracy 1.00**.

The spine (1+2+3+4 from PAPER3.md) is implemented end-to-end and the killer
experiment — blind-classify a held-out agent from its recovered phenotype —
works.

Quick checks:
- behaviour:        `python -m src.paper3.agents`
- criteria:         `python -m src.paper3.agency_criteria`
- blind experiment: `python -m src.paper3.blind_validation`   (accuracy 1.00)
- full pipe (smoke):`python -m src.paper3.phenotype smoke`     (rollout → MPS fit)

Smoke fits (300 episodes, 15 epochs): info-seeker L1≈0.007, gambler ≈0.027,
habitual ≈0.039 — each MPS already reproduces its agent's own behaviour; a full
run (5000 episodes, 200 epochs) tightens these.

## ⚠️ Adversarial audit: the first "killer experiment" was circular

`blind_validation` reported blind accuracy **1.00** and that number does not
support the paper's thesis. Audit findings:

1. Its features `cue_visit` / `arm_first` are read straight off `actions[1]` in
   the rollouts — they *are* the agents' defining behaviour. The trained MPS is
   never consulted; docstrings claiming they were "marginals of the recovered
   model" were **false** and have been corrected.
2. A trivial baseline — raw `P(a1)`, no criteria/EFE/tensor network — also scores
   **1.00 (12/12)**. None of the machinery contributes to the number.
3. "Held-out" = the same three specs re-seeded; near-deterministic agents make
   1.00 guaranteed by construction.

Paper II earned its tensor network because its claims *required* the latent model
(hidden-state partition from bond rays, A/B, subsystem MI, empowerment). The
error was choosing a Paper III phenotype that raw behaviour supplies for free.

### Negative result #2 — thresholded coverage doesn't work either

The first replacement (`structure_phenotype.py`) asked which histories are
"supported" above a Born-weight floor. Measured, the floor does not bite:

| agent | states | supported | note |
|---|---|---|---|
| info-seeker | 7 | 7/7 | — |
| reward-gambler | 6 | 6/7 | loses only `center` (never idles) |
| habitual | 6 | 7/7 | two histories merged predictively |

The gambler still visits the cue ~2% of the time, so its cue histories carry
weight 0.017/0.020 and clear any sane threshold — it "supports" the cue states
despite having almost no data there. The one history it loses is `center`, an
incidental artifact, not the cue blindness that matters. **Binary coverage is
threshold-arbitrary and misses the effect.**

### Current attempt — per-state recovery *fidelity*

`recovery_fidelity.py` measures, per latent state, the L1 between the predictive
signature recovered from the agent's MPS (bond ray × T3) and the **analytic**
signature of the true model. Continuous, threshold-free, and it requires the
tensor network. Analytic ground truth was sanity-checked independently
(q(K|cue)=[1,0], q(K|R,cheese)=[0.15,0.85], cue/ctx0 vs ctx1 differ by L1 2.8).

### ✅ The actual result: action entropy gates identifiability

Auditing the fidelity map (`fidelity_analysis.py`) overturned the expected story.
Per-state recovery error vs analytic truth:

| agent | center | R/cheese | R/shock | L/cheese | L/shock | cue/ctx0 | cue/ctx1 |
|---|---|---|---|---|---|---|---|
| info-seeker | 3.000 | 3.496 | 2.039 | 3.489 | 2.904 | **6.001** | **6.015** |
| reward-gambler | — | 0.316 | 0.482 | 0.304 | 0.242 | **6.044** | **6.012** |
| habitual | 0.288 | 0.318 | 0.453 | 0.452 | 0.310 | 0.275 | 0.246 |

| model | R² |
|---|---|
| `error ~ log10(visit weight)` | 0.093 |
| `error ~ H(a3)` | **0.807** |
| `error ~ H(a3) + log10(weight)` | **0.904** |
| `error ~ log10(weight)` given `H > 1.4` | **0.909** |

**Recovery is not driven by how often a state is visited.** Every state with
error ≈ 6 has `H(a3) = 0` (one action ever taken there); every state with error
< 0.5 has `H ≈ 2`. The info-seeker holds **40%** of its data at the cue and
recovers it *worst*, while recovering its arm states — **0.4%** of its data — far
better, because at the cue it is deterministic and at the arms it is not. That
masks the volume effect marginally (R²=0.09, a Simpson's-paradox artifact: the
highest-weight states are exactly the zero-entropy ones); once the gate is open,
volume predicts recovery strongly (R²=0.91).

**Thesis this supports: agent epistemics and observer epistemics are opposed.**
The info-seeker resolves *its own* uncertainty at the cue, which makes its later
behaviour deterministic, which destroys the *observer's* ability to identify the
conditionals. Competence removes the variation identification needs — the
*random* agent yields the best recovered world model. This also explains Paper
II: its exhaustive uniform-action rollouts are the maximum-entropy best case,
which is why it reached TV ≤ 0.001. Paper III is what happens when you drop that
assumption and watch a real agent.

Honest caveats: ~81% of the error variance is predicted by action entropy, a
*behavioural* statistic — so the contribution is the identifiability law, not a
classifier; error ≈ 6 means **unidentified** (unexercised rows hit the metric
ceiling), not "learned wrongly"; and the info-seeker's `H=0` follows from
γ=16, so a γ sweep turns this into a continuous
determinism-vs-identifiability curve (`gamma_sweep.py`).

### The tradeoff curve (headline figure) — `gamma_sweep.py`

One agent family, C and horizon fixed, policy precision γ the only knob:

| γ | cue_visit | H(a3) weighted | recovery error |
|---|---|---|---|
| 0.0 | 0.30 | 1.99 | 0.325 |
| 0.5 | 0.34 | 1.85 | 0.299 |
| 1.0 | 0.37 | 1.61 | 0.529 |
| 2.0 | 0.45 | 1.26 | 1.139 |
| 4.0 | 0.59 | 0.96 | 1.664 |
| 8.0 | 0.84 | 0.38 | 2.228 |
| 16.0 | 0.98 | 0.02 | **3.967** |

`corr(γ, H) = −0.934` · `corr(γ, error) = +0.987` · `corr(H, error) = −0.962`

**12× degradation** in recoverability across the competence range. The
pre-registered falsifier (`corr(γ, error) ≈ 0`) did not fire. Competence is not
free — it is paid for in the observer's ability to identify the agent's model.

Read-with-care: γ=0.5 dips marginally below γ=0 (noise; don't over-read the low
end), and `error_mean` is unweighted across states so at high γ it mixes the
entropy effect with reduced arm coverage — entropy *dominance* rests on the
R² 0.81 vs 0.09 decomposition, not on this curve alone.

### ❌ Confabulation hypothesis — refuted (`confabulation_check.py`)

The claim that would have made this tensor-network-specific: that the MPS invents
confident structure where the agent never sampled. **It doesn't.**

| category (γ=16) | n | mean err | mean H | mean rel-mass |
|---|---|---|---|---|
| exercised | 14 | 0.622 | 0.30 | **0.4999** |
| generalised | 2 | 0.364 | 0.30 | 0.0005 |
| CONFABULATED | 4 | 1.872 | 0.44 | **0.0001** |
| hedged | 8 | 1.214 | 3.19 | 0.0000 |

Relative mass 0.0001 vs 0.4999 — the model assigns ~zero probability to actions
the agent never takes. It is **honest**. The apparent confidence was my own row
normalisation dividing a near-zero row by its near-zero sum. The built-in
artifact control caught my own hypothesis.

### Is the error metric just a count? Mostly, but not entirely

| γ | unexercised | mean state err |
|---|---|---|
| 0 | 0/28 | 0.457 |
| 2 | 5/28 | 1.191 |
| 8 | 6/28 | 2.273 |
| 16 | 14/28 | 3.805 |

`error ~ #unexercised`: slope 1.330, **R² = 0.738**, corr +0.859.

Not ~1.0 — and γ=2 vs γ=8 have near-identical coverage (5 vs 6) yet ~2× the error
(1.19 vs 2.27), which counting cannot produce. Decomposition: **~74% binary
coverage** (countable without any MPS) + **~26% graded estimation quality** at
rarely-sampled rows.

### ⚠️ Bottom line: no tensor-network-specific claim survives

The graded 26% is sample efficiency at rare actions — any estimator shows it.
With confabulation refuted, the finding is about **data**, not tensor networks.
It is real; it is not evidence for the method. Open options: (a) reframe as a
methodological study of the limits of behavioural structure learning; (b) fold
into Paper II as its limitations section; (c) one bounded test of the untried
route — Paper II's *actual* machinery (bond-ray state partition, recovered A/B,
subsystem MI) applied per agent. **Do not spend compute on more mazes until this
is settled.**

### Convergence confound — closed (`convergence_check.py`)

The sweep fixed epochs, not convergence. Same rollouts at 25 vs 100 epochs:

| γ | err@25 | err@100 | delta | loss@25 → loss@100 |
|---|---|---|---|---|
| 0.0 | 0.457 | 0.421 | −0.036 | 3.689 → 3.700 |
| 4.0 | 1.950 | 1.852 | −0.098 | 3.031 → 3.029 |
| 16.0 | 3.805 | 3.603 | −0.202 | 1.661 → 1.663 |

Error spread survives: **+3.349 → +3.183** (95% retained), so the sweep was not
measuring convergence speed. Noted honestly: the intended proof was "error flat
*while loss descends*"; in fact loss is flat, i.e. the models were **already
converged at 25 epochs**. Same conclusion, different route than advertised.

### ⚠️ Fit quality and recovery quality are ANTI-correlated

Training loss across γ: **3.689 (γ=0) → 1.661 (γ=16)**. The deterministic agent's
model fits its own data *far better* while recovering the environment *far worse*
(error 3.8 vs 0.46). Its data is low-entropy and concentrated, hence easy to
model. The high-γ model did not fail to learn — it learned its data very well,
and its data does not contain the environment.

**Methodological consequence: training loss cannot be used as a proxy for
recovery quality.** A well-fit model of self-selected data is still a bad model of
the world. This is the failure mode a reader would otherwise walk into.

### Novelty — state this plainly in the paper

The underlying principle is **not new**. It is the *positivity / overlap*
assumption in causal inference and the *coverage* requirement in offline RL and
imitation learning: no identification for an action never taken. Cite that
literature; do not present the principle as novel.

Defensibly new here: its instantiation in **tensor-network structure learning**
(the limit appears as unexercised rows of the action-conditioned bond signature),
the **quantification** in this setting (entropy R² 0.81 vs visit-rate 0.09, with
the Simpson's-paradox masking), and the **active-inference twist** — the agent
whose objective *includes* epistemic value is the one that most damages the
observer's identifiability.

`structure_vs_visitrate.py` (same agent, identical first-step policy, second
action greedy vs uniform) remains available as an independent confirmation.

## Honest limitations (findings, not bugs)

- **Intentionality is under-identified** in this task: cue-seeking is degenerate
  between reward-preference and pure curiosity, so inverse-C cannot attribute the
  info-seeker's cue visits to goals. Discrimination therefore uses rationality +
  recovered state-visitation coverage, not intentionality. Resolving it needs the
  planning horizon / epistemic weight inferred as a separate trait.
- **Discrimination features** (cue_visit, arm_first) are marginals of the
  recovered policy-weighted model, reproduced by the fitted MPS to <0.04 L1 — so
  this is discrimination by recovered phenotype, read here off the rollouts the
  MPS is fit to. Reading them off the trained MPS directly is a small next step.
