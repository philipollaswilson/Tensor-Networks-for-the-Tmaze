# Paper III — `src/paper3/`

Agent-specific agency phenotyping. Roadmap and rationale live in `PAPER3.md` at
the repo root; this package is the code skeleton for the spine (workstreams
1 + 2 + 3 + 4).

## Layout

| module                 | workstream | closes Paper II gap                       | status |
|------------------------|-----------|-------------------------------------------|--------|
| `persistent_tmaze.py`  | 4         | context i.i.d.-per-step (Philip's point)  | **implemented** (generator + info-gain check); `to_memory_pool` stubbed |
| `agents.py`            | 1         | recovered env, not an agent               | stub (specs defined) |
| `agency_criteria.py`   | 2         | empowerment ≠ agency                       | stub (interfaces defined) |
| `blind_validation.py`  | 3         | non-blind vs ground truth                 | stub (interfaces defined) |
| `phenotype.py`         | 1–4       | the killer experiment (discrimination)    | stub (driver) |

Stubs raise `NotImplementedError` with a one-line TODO pointing at the Paper II
code to reuse. Nothing is silently fake — an unfinished path fails loudly.

## What already works

```
python src/paper3/persistent_tmaze.py
```

Enumerates the persistent-latent episodes and prints `I(reward; context)` at a
blind arm. Under a persistent latent this is ~1 bit (the arm now reveals the
hidden state), versus ~0 bits in the Paper II i.i.d. dataset — the concrete
difference behind Philip's critique.

## What to reuse (unchanged, from Paper II)

`structure_recovery.py`, `factorization_test.py`, `gauge_fix_states.py`,
`model_selection.py`, `make_extension_figures.py`, and the Han-scheduler training
recipe in `full_tmaze_train.py`.

## Build order

1. `persistent_tmaze.to_memory_pool` — drop the new data into the existing trainer.
2. `agents.build_agent` / `rollout` — per-agent pymdp rollouts.
3. `phenotype.train_agent_mps` — one MPS per agent.
4. `agency_criteria` — the three criteria on each recovered model.
5. `blind_validation` — freeze a pre-registration, then discriminate blind.
