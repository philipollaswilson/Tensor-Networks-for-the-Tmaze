# Tensor Networks for the T-maze

Learn a generative model of a T-maze POMDP from behaviour with a matrix product state
(MPS), then **recover the model's structure** and read agency-relevant quantities off it.

This is the companion code for two papers:

- **Paper I** — *Characterizing Agency in AI Systems: Active Inference and an
  Empowerment-Based Probe* (Wilson, Albarracin, Hinrichs, Moore, Polani, Friston,
  Constant). Defines agency by three criteria (intentionality, rationality,
  explainability), instantiates them in active inference, and uses empowerment as a
  controllability probe on **hand-specified** models.
- **Paper II** — *Structure Learning of Generative Models for Agency Phenotyping with
  Tensor Networks* (this repo, `paper/`). Delivers the model-learning half: the MPS is
  trained unsupervised on T-maze rollouts, and its structure is recovered and validated
  against the analytic environment.

## What the pipeline recovers

From MPS models trained on the minimal 4-observation maze and the full `pymdp` T-maze
(24 composite observations), each validated against analytic ground truth:

| Result | Script | Headline number |
|--------|--------|-----------------|
| Hidden states (predictive equivalence, robust where bond-ray fidelity over-separates) | `src/structure_recovery.py` | 4 (minimal), 7 (full) |
| Labelled emission `A=p(o\|s)` and transition `B=p(s'\|s,a)` vs ground truth | `src/structure_recovery.py` | emission TV `0.000`; branching exact |
| Conditional-independence graph (Chow–Liu + conditional MI) | `src/structure_recovery.py` | correct chain; `I(a₁;o₃\|o₂)=0` |
| Observation factorisation (subsystem quantum MI) | `src/factorization_test.py` | context near-independent (`I≈0.05`) |
| Principled state count (bond-dim sweep, AIF model expansion + BMR) | `src/model_selection.py` | fit-L1 knee at χ=4 |
| Symbolic states (gauge-fix the bond into the cluster basis) | `src/gauge_fix_states.py` | ARI `1.000`; dist. preserved to 4e-14 |
| Empowerment phenotype on the learned model | `src/empowerment_learned_model.py` | post-cue reward 1.585 vs analytic 1.585 |

The full maze fits to `‖p−q‖₁ = 0.005`. See `ROADMAP.md` for the methods and the
literature behind each item, and `DEVLOG.md` for the running log.

## Layout

```
src/
  minimal_tmaze_train.py     train + save MinimalTmaze.pt (4-obs maze)
  full_tmaze_train.py        train + save FullTmaze.pt (converged, Han-scheduled)
  structure_recovery.py      Items 1+2: A/B recovery, MI dependency graph
  factorization_test.py      Item 4: observation factorisation
  model_selection.py         Item 3: state-count model selection (bond-dim sweep)
  gauge_fix_states.py        Item 5: symbolic-state gauge fixing
  make_extension_figures.py  paper/figs/*.png for Paper II
Saved_Models/                MinimalTmaze.pt, FullTmaze.pt, SamuelModel.pt (bond-4 checkpoint)
paper/                       Paper II: main.tex, main.pdf, figs/, llncs.cls
mpstwo/                      the MPS package (model, trainer, optimizers, schedulers)
```

Analysis scripts take an optional checkpoint name, e.g. `python src/structure_recovery.py
FullTmaze.pt`. They default to the converged models.

## Install & run

```bash
make install                          # dependencies
python src/full_tmaze_train.py        # (re)train the converged full-maze MPS
python src/structure_recovery.py      # A/B recovery + dependency graph
python src/factorization_test.py      # observation factorisation
python src/model_selection.py         # state-count selection sweep
python src/gauge_fix_states.py        # symbolic states
python src/make_extension_figures.py  # regenerate paper figures
```

Training writes TensorBoard logs to `MPS_*/` (gitignored). Build the paper with
`pdflatex main.tex` in `paper/`.
