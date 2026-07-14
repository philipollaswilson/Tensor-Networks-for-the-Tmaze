# Devlog

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
