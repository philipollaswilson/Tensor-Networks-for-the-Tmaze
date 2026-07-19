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
| `blind_validation.py`  | 3         | non-blind vs ground truth                 | **done + verified** — pre-registered blind discrimination, **accuracy 1.00** |
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
