# Roadmap: strengthening the "structure learning" claim

Status of the current pipeline (branch `analysis/learned-empowerment`): the MPS models
`p(o, a)` on the minimal and full T-mazes, reproduces the empowerment regimes, and
recovers the **number** of hidden states plus a task-relevant **abstraction** by
gauge-invariant ray-fidelity clustering. That is genuine structure learning in the
*latent-dimensionality + abstraction* sense — but **not** in three other senses a
reviewer will ask about: the factorized dependency structure (A/B matrices), the
conditional-independence graph, and principled (evidence-based) state-count selection.

**Contribution framing (load-bearing).** Wauthier et al. (arXiv:2208.08713), the paper
this pipeline follows, explicitly *could not* extract normalized hidden states or A/B from
the T-maze MPS: "the bonds between tensors can be regarded as internal states, but they are
not normalized and therefore not usable." Our ray-fidelity clustering supplies exactly the
missing normalized-state partition. The roadmap below turns that partition into labeled,
ground-truth-comparable structure — closing the gap the precursor left open.

**Key technical fact.** All checkpoints are genuine *complex* Born machines
(`p = |ψ|²/Z`, ~50–64% negative/complex amplitudes), not non-negative MPS. Per Glasser et
al. (arXiv:1907.03741), a non-negative MPS *is* an HMM with bond index = hidden state, but a
generic Born machine has no such exact form and finding one is NP-hard. So every extraction
below **pushes amplitudes → probabilities first, then forms stochastic objects and
renormalizes** — never reads a stochastic matrix off the complex tensor. Residual
non-stochasticity ("Born leakage") is reported as a diagnostic, not hidden.

---

## Item 1 — Reconstruct labeled A = p(o|s) and B = p(s'|s,a)  [HEADLINE, no retraining]

**Goal.** Turn the clustered states into an actual factorized generative model and compare
to the pymdp ground-truth A/B up to a state permutation.

**Method** (transfer-operator / Zipper extraction; Han arXiv:1709.01662, Bradley–Stoudenmire–
Terilla arXiv:1910.07425, Adhikary et al. arXiv:2010.10653):
1. Left/mixed-canonicalize so bond vectors are comparable.
2. **Emission** Â(o|s) = Σ_{h∈H_s} p(h)·p(o|h) / Σ_{h∈H_s} p(h), with actions marginalized
   (pymdp A is action-independent). Bin Born probabilities, never square a sum.
3. **Transition** B̂(s'|s,a): for each history in cluster s and action a, append (a,o),
   re-contract the bond vector, classify it into s' by ray fidelity, accumulate Born weight,
   renormalize each (s,a) column.
4. **Resolve permutation** vs pymdp with the Hungarian algorithm (ray-fidelity already killed
   the continuous gauge; only a discrete permutation remains).

**Validation.** Per-column TV/KL and Frobenius ‖P·Â·Pᵀ − A‖ vs pymdp; row/column
stochasticity + non-negativity of Â, B̂ (deviation = Born leakage); held-out sequence
log-likelihood of the reconstructed HMM vs the MPS's own |ψ|²/Z. Semantic checks: cue emits
context-informative obs, arms are absorbing, actions move location deterministically.

**Pitfalls.** Clusters must be near rank-1 bond rays (check Schmidt spectrum per cluster);
keep cluster labels consistent across cuts; normalize with exact Z, not counts; marginalize
actions for A but condition for B or the two smear.

---

## Item 2 — Mutual-information dependency graph  [cheapest, no retraining]

**Goal.** Recover the dependency / conditional-independence structure over
{a₁,o₁,a₂,o₂,a₃,o₃} (and observation sub-factors) and compare to the true POMDP.

**Method** (Chow–Liu 1968; Rissler–Noack–White cond-mat/0508524; Legeza–Sólyom):
- Pairwise MI matrix I(Xᵢ;Xⱼ) from exact marginals → max-weight spanning tree (Chow–Liu) and
  an MI heatmap (the DMRG quantum-information diagnostic).
- Conditional MI I(X;Y|Z) for the Markov claims; with exact `p`, I≈0 to tolerance *is* the
  conditional-independence certificate (optionally G²=2N·I for a finite-N analogue). Small
  PC-style skeleton loop over conditioning sets (n is tiny).

**Validation.** Compare against the **observable-projected** ground-truth graph, i.e. the
d-separations implied by the T-maze DBN *after marginalizing the latent sₜ* — not the full
DBN. Latent marginalization induces real dependencies among observables; those are correct,
not model failure. Report structural Hamming distance / precision-recall and the numeric CMI.

**Pitfalls.** Pairwise blindness (Chow–Liu draws indirect edges — use CMI for claims); no
natural MI scale (use spectrum gaps + product-surrogate noise floor, not a hard threshold);
low-entropy near-deterministic variables destabilize MI (entropy floors, drop zero-prob
outcomes); consistent log base.

---

## Item 3 — Principled state-count model selection  [needs χ-sweep retraining]

**Goal.** Show the state count (4 minimal, 7 full) falls out of an evidence-vs-complexity
comparison, not the SVD cutoff.

**Method.** Sweep χ = 1…χ_max (cap the bond, don't use adaptive cutoff — χ is the controlled
variable), train to convergence, and plot, all expected to knee at the same χ*:
- **held-out NLL** (primary; MPS is normalizable so NLL = −mean log|ψ|²/Z);
- **Schmidt / discarded-weight spectrum** at the central bond (gap after the 4th/7th value —
  shows the original cutoff read a real gap);
- **BIC / MDL** = train-NLL + ½·k·log N (k = free MPS params);
- **Bayesian-model-reduction free energy** = accuracy − complexity, scored on reduced χ
  analytically from the full-χ posterior (Friston–Parr–Zeidman arXiv:1805.07092).

**AIF framing** (Smith et al. 2020 PMC7250191; Neacsu et al. 2026). Cast as *model expansion*
(large-χ "spare capacity") + *Bayesian model reduction* (commit capacity to states that raise
evidence, release the rest). The state count is then the fixed point of free-energy
minimization over model structure — the active-inference definition of structure learning.

**Pitfalls.** Training NLL is monotone in χ — never use it; only held-out/evidence knees.
On near-exactly-fit data the held-out curve plateaus (elbow, not minimum) — the Schmidt gap +
complexity penalty become decisive. Best-of-seeds per χ. Count real params (gauge freedom
slightly overcounts, biasing toward fewer states — conservative).

---

## Item 4 — Discover the observation factorization  [no retraining — corrected]

**Goal.** Show the 24-dim observation index factorizes as position(4)×reward(3)×context(2)
*from the learned tensors*, not by our decode convention.

**Codebase correction.** The full-maze MPS already feeds a **flat** 24-dim leg
(`MultiOneHotMap([4,3,2]).flatten()`); the factorization is assumed only in the *analysis*
(`reward_marginal`, `(o//2)%3`). So no re-encoding is needed — this is a physical-index
factorization test on the learned reduced density matrix.

**Method** (Convy et al. arXiv:2103.00105; quantum mutual information).
- Build the single-site RDM ρ (24×24) = Tr_rest|ψ⟩⟨ψ|, conditioned per history/regime.
- For candidate factorization 24≅A⊗B⊗C: partial-trace to ρ_A,ρ_B,ρ_C; compute
  I(A:B)=S(ρ_A)+S(ρ_B)−S(ρ_AB) etc., and the trace-distance residual ‖ρ − ρ_A⊗ρ_B⊗ρ_C‖₁.
- Small exhaustive search over factorizations of 24 (4×3×2, 6×4, 12×2…) and index
  permutations; the winner (near-zero cross-MI, smallest residual) is the discovered
  structure. Headline scalar: MIG (Chen et al. arXiv:1802.04942).

**Validation.** Product-surrogate noise floor for MI; expect **nonzero** reward↔context MI
(causally real — target is *sparse*, not empty cross-MI); validate against the near-exact
`exact_joint` fit; sweep max_bond to confirm the factorization is stable, not a bond-dim
artifact.

---

## Item 5 — Gauge-fix the MPS to emit symbolic states  [no retraining, polish]

**Goal.** Rotate the bond into the cluster basis so the bond index *is* a labeled state and
the model emits states, removing the "states are a post-hoc overlay" objection.

**Method** (Orús arXiv:1306.2164; Schollwöck arXiv:1008.3477; Aizpurua et al. arXiv:2401.00867):
1. Mixed-canonical form at the target bond (Schmidt basis).
2. Build an orthonormal frame from the k cluster representatives — reduced QR, or better the
   eigenbasis of the history-aggregated bond RDM M = E_h[|ψ_h⟩⟨ψ_h|] (robust to noisy reps).
3. Apply as a **unitary** gauge U (keeps canonical form, preserves the distribution exactly);
   the bond index then emits p_i(h) = |⟨q_i|ψ_h⟩|².
4. Optional: small Procrustes/Jacobi rotation within the top-k block for monosemanticity;
   project to a stochastic HMM form to make "bond = hidden state" literal.

**Validation.** Distribution unchanged to machine precision (unitary check); off-diagonal
mass of the emitted transfer operator drops vs a random-gauge baseline; per-history emitted
distribution near one-hot (low entropy, high max-prob); adjusted Rand index vs the original
fidelity clustering; RDM diagonal matches empirical cluster occupancies. Exactness is bounded
by how non-positive the Born machine is (measurable) — for these complex models it's an
interpretable approximation, quantified by the block-diagonality/monosemanticity metrics.

---

## Implementation plan

| Phase | Items | Retraining | Deliverable |
|-------|-------|------------|-------------|
| A | 1 + 2 | No | `src/structure_recovery.py` — A/B vs pymdp + MI graph, on Minimal & Full checkpoints |
| B | 4 + 5 | No | `src/factorization_test.py`, `src/gauge_fix_states.py` |
| C | 3 | Yes (χ-sweep) | `src/model_selection.py` — evidence-vs-χ curves + BMR |

Phases A and B run entirely off `MinimalTmaze.pt` / `FullTmaze.pt` / `SamuelModel.pt`;
Phase C is the only one that retrains. Order chosen so the two load-bearing senses
(dependency structure + conditional independence) land first with no retraining.
