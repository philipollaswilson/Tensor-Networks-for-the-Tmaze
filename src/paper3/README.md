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
| `agency_criteria.py`   | 2         | empowerment ≠ agency                       | stub (interfaces defined) |
| `blind_validation.py`  | 3         | non-blind vs ground truth                 | stub (interfaces defined) |
| `phenotype.py`         | 1–4       | the killer experiment (discrimination)    | stub (driver) |

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
3. `phenotype.train_agent_mps` — one MPS per agent (in progress).
4. `agency_criteria` — the three criteria on each recovered model.
5. `blind_validation` — freeze a pre-registration, then discriminate blind.

Quick behaviour check: `python -m src.paper3.agents`.
