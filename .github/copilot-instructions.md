# AI Coding Agent Instructions

This repo implements tensor-network models (MPS) for sequence modeling in T-maze–style environments. Use these rules to be productive quickly.

## Big Picture
- Core model: `MPSTwo` represents a Matrix Product State over alternating `action`/`observation` legs. See [mpstwo/model/mpstwo.py](mpstwo/model/mpstwo.py).
- Training loop: `MPSTrainer` orchestrates data loading, loss, optimizer, scheduling, logging, and checkpointing. See [mpstwo/model/mpstwo_trainer.py](mpstwo/model/mpstwo_trainer.py).
- Decision agents: `DiscreteFreeEnergyAgent` plans via expected free energy with `Q_obs`/`likelihood` utilities. See [mpstwo/agents/discrete_fe_agent.py](mpstwo/agents/discrete_fe_agent.py).
- Distributions & contractions: Probability utilities in [mpstwo/utils/distributions.py](mpstwo/utils/distributions.py); low-level ops in [mpstwo/utils/operations.py](mpstwo/utils/operations.py).
- Environments: T-maze wrapper around `pymdp` in [mpstwo/envs/tmaze_wrapper.py](mpstwo/envs/tmaze_wrapper.py).

## Data Shapes & Conventions
- Model legs: `physical_legs = time steps` (alternating `a_i, o_i`).
- Embedded observations: `embed_input()` expects `[batch, time]` and maps to `[batch, time, feature_dim_obs]` (default 2 ⇒ `[x, 1-x]`).
- Actions: `one_hot_input()` expects `[batch, time]` and maps to `[batch, time, feature_dim_act]`.
- Sequences in agents use `TensorDict` with keys `"action"` and `"observation"`; each entry is shaped `[batch, t, 1]` for unembedded values.
- Canonical forms: training and inference rely on right/left canonical gauges (`right_canonical()`, `left_canonical()`); many `Q_*` calls pass `canonical="right"`.
- Normalization: ops like `normalize()` and `transfer_normalize()` are applied during sweeps and contractions; don’t remove them.

## Training Workflow
- Minimal setup:
  - Create `MPSTwo(physical_legs, feature_dim_obs=2, feature_dim_act=K, bond_dim=B, device)`.
  - Prepare a `Dataset` that yields a `TensorDict` with `"observation"` and `"action"` tensors shaped `[batch, physical_legs]` (integers for actions, [0,1] for observations or pre-embedded).
  - Choose `Optimizer` and optional `Scheduler` and `CutoffScheduler` (see [mpstwo/model/optimizers](mpstwo/model/optimizers) and [mpstwo/model/schedulers](mpstwo/model/schedulers)).
  - Use `MPSTrainer(model, train_dataset, optimizer, ...)` and call `trainer.train(epochs)`.
- Logging/checkpoints:
  - TensorBoard writers created under a dated folder; models saved to `model/model-XXXX.pt`, optimizer/schedulers to `optimizer/*.pt` (see `MPSTrainer._save_model_callback`).
  - To resume, call `MPSTrainer.load(checkpoint_epoch)` after reinitializing trainer with matching hyperparameters.

## Decision/Planning Patterns
- Free energy agent computes next-action EFE via `Q_obs` and `likelihood` and plans recursively or iteratively with pruning. See `expected_free_energy_rec()` and `expected_free_energy_iter()` in [mpstwo/agents/discrete_fe_agent.py](mpstwo/agents/discrete_fe_agent.py).
- Utility functions accept raw sequences but internally use `feature_map_obs` and `feature_map_act` to embed/one-hot; pass these maps explicitly when building agents.

## Probability Utilities
- `prob_given_seq(model, obs, act, idx)` returns probabilities for a prefix plus `idx` open legs and a label string; most `Q_*` functions wrap it.
- `Q_obs(model, ...)` computes `Q(o_i | a_<=i)`; `likelihood(model, ...)` computes `Q(o_i | s_i, a_<=i)`; `marginal_prob(model, indices)` yields marginals like `Q(o_i)`.
- For density matrices (rdm) use `dens_given_seq`/`likelihood_`; eigenvalues of an RDM sum to 1, partial trace preserves trace.

## Environment Integration (T-maze)
- `TMaze` wraps `pymdp.envs.TMazeEnv`, sets `action_space = MultiDiscrete([4,1])` and `observation_space = MultiDiscrete(num_obs)`. See [mpstwo/envs/tmaze_wrapper.py](mpstwo/envs/tmaze_wrapper.py).
- `step()` enforces cue-follow behavior: if the first observation is a cue (`1` or `2`), it overwrites the first action accordingly before delegating to `super().step()`.

## Build & Setup
- Root README: run `make install` to create venv, install `requirements.txt`, and clone `lib` repos. See [README.md](README.md) and [Makefile](Makefile).
- Windows note: Makefile uses POSIX paths. Prefer native PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
- Optional integrations: TensorBoard (`torch.utils.tensorboard`), Weights & Biases (`wandb`) — trainer auto-selects based on `log_to` and availability.

## Common Patterns & Gotchas
- Always call `model.right_canonical()` before using agents; trainers do this during init/sweeps.
- When using `MPSTwo.update()`, provide batch tensors `[batch, time, feature_dim]` returned by trainer embeds; the optimizer may perform two-site SVD and adjust bond dims by `cutoff`.
- `transfer_normalize()` guards zero tensors; don’t assume plain `softmax` everywhere — follow existing normalization calls.
- Action cardinality (`feature_dim_act`) must match environment/agent maps; mismatches raise `ValueError` in trainer embedding.

## Quick Usage Example
```python
import torch
from mpstwo.model.mpstwo import MPSTwo
from mpstwo.model.mpstwo_trainer import MPSTrainer
# Create a toy model
model = MPSTwo(physical_legs=5, feature_dim_obs=2, feature_dim_act=4, bond_dim=8, device="cpu")
# Dummy batch: integers for actions, floats in [0,1] for observations
obs = torch.randint(0, 2, (32, 5)).float()
act = torch.randint(0, 4, (32, 5))
# Minimal trainer (assumes you provide a Dataset that yields these)
# trainer = MPSTrainer(model, train_dataset, optimizer, batch_size=32)
# trainer.train(epochs=10)
# Direct probability queries
# probs, label = model.prob_given_seq(model, model.embed_input(obs), model.one_hot_input(act))
```

If anything here is unclear or missing (e.g., dataset interfaces or optimizer configs you use), tell me and I’ll refine this file. 