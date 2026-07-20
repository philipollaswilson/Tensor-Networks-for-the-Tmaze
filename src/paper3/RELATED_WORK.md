# Related work: our finding is a known principle in a new setting

**Position honestly.** The principle behind Paper III's result — you cannot identify
what was never sampled — is well established in at least three literatures. The
contribution is the instantiation, quantification, and the practical diagnostic,
not the principle. Claiming novelty for the principle is the fastest route to a
desk rejection.

Verification status is marked on every entry. Do **not** put an UNVERIFIED entry
in the bibliography without checking it first.

---

## 1. Causal inference — positivity / overlap  ✅ concept verified

The positivity (a.k.a. overlap) assumption is *an identifiability condition*: the
probability of each treatment must be bounded away from 0 for every covariate
combination. Where it is violated, the data carry no information about the
counterfactual outcome, so the effect is **not identified** in that subgroup and
any estimate is extrapolation.

That is exactly our situation with `(state, action)` pairs the policy never
exercises. Our per-state recovery error is a measurement of a positivity
violation, and our "identifiability certificate" is a positivity check.

- Review of violations and their consequences:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC8492528/> ✅ verified
- Course notes stating positivity as an identifiability assumption:
  <https://www.stats.ox.ac.uk/~evans/APTS/causassmp.html> ✅ verified

⚠️ UNVERIFIED: the classical primary citation is usually given as Rosenbaum &
Rubin (1983), *The Central Role of the Propensity Score...*, Biometrika 70(1).
I did not verify this directly — check before citing.

## 2. Offline RL — concentrability / coverage  ✅ verified

Chen & Jiang, **"Information-Theoretic Considerations in Batch Reinforcement
Learning"**, ICML 2019, pp. 1042–1051. arXiv:1905.00360. ✅ verified via DBLP
(`dblp.org/rec/conf/icml/ChenJ19.html`) and Illinois Experts.

Establishes that offline RL guarantees rest on coverage (concentrability)
assumptions — the data must be exploratory enough to cover the relevant state
distributions — and that all-policy concentrability is a strong requirement that
implicitly constrains the dynamics. Later work relaxes it to single-policy
concentrability.

Relevance: our γ sweep is a direct measurement of coverage collapsing as policy
precision rises. The offline-RL framing ("historical data in real applications
often lacks exploration") is precisely our motivating scenario.

- <https://arxiv.org/pdf/1905.00360> ✅ verified

## 3. Imitation learning — covariate shift  ✅ concept verified, THE CLOSEST PRIOR ART

This is the nearest neighbour to our result and should be cited most prominently.

Behaviour cloning from expert demonstrations suffers because the expert's state
distribution is **narrow**: the expert never visits the states its own mistakes
would produce, so the learner has no data there. DAgger's fix is to interactively
query the expert on learner-induced states — i.e. to *manufacture the coverage
the expert's competence removed*.

**Our finding is the structure-learning analogue of this.** Imitation learning
says competent demonstrations are bad for learning the *policy*; we show
competent behaviour is bad for recovering the *world model*. Same mechanism
(competence narrows the data), different target.

- Ross & Bagnell, *A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning* (DAgger), AISTATS 2011 — ⚠️ venue/year from memory,
  VERIFY before citing. Concept verified via:
  <https://www.researchgate.net/publication/47646552_A_Reduction_of_Imitation_Learning_and_Structured_Prediction_to_No-RegretOnline_Learning>
- Spencer et al., *Feedback in Imitation Learning: The Three Regimes of Covariate
  Shift*, arXiv:2102.02872 ✅ verified URL:
  <https://arxiv.org/pdf/2102.02872>

---

## What this leaves as ours

1. **Instantiation in tensor-network structure learning.** The identifiability
   limit appears concretely as unexercised rows of the action-conditioned bond
   signature — a representation-level statement, not just a data-level one.
2. **Quantification.** Action entropy predicts recovery error (R² 0.807); visit
   frequency essentially does not (R² 0.093). The Simpson's-paradox masking
   (marginally the visit rate looks irrelevant because the highest-weight states
   are exactly the zero-entropy ones) is, as far as we know, not reported
   elsewhere in this form.
3. **The loss/recovery inversion.** Training loss *improves* (3.689 → 1.661) as
   recovery *degrades* (0.46 → 3.8), because a concentrated policy is easy to
   model. Practitioners' default health check points the wrong way. This is a
   concrete, actionable warning.
4. **A data-only certificate.** Which conditionals are supported is computable
   without ground truth, and we validate that it predicts recovery failure.
5. **The active-inference irony.** The agent whose objective *includes* epistemic
   value is the one that most damages the observer's identifiability. Agent
   epistemics and observer epistemics are opposed.

## How to phrase the claim

> "The requirement that behaviour cover the state–action space is a known
> identifiability condition, appearing as positivity in causal inference,
> concentrability in offline RL, and covariate shift in imitation learning. We
> characterise what it costs in tensor-network structure learning: which
> quantity actually governs recovery (action entropy, not visitation), how
> sharply it degrades with policy precision, why the usual convergence
> diagnostic misleads, and how to certify from data alone which parts of a
> recovered model can be trusted."

Not: "we discover that agents must explore."
