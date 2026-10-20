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

## 4. Active inference itself — ALREADY NOTES THE MECHANISM  ⚠️ read this first

**This is the most consequential entry. The core phenomenon is already in print in
the active-inference parameter-recovery literature.**

From the ActiveInference.jl paper (MDPI *Entropy* 27(1):62, 2025) ✅ verified:

> "The ability to recover parameters depends on the specific model and task, as
> well as on the specific values of the parameters (when α is very high, for
> example, **the behaviour becomes essentially deterministic; further increases in
> α would then not have any effect on the behaviour, and therefore, not be
> estimable**)."

That is our gamma result, stated in active inference: high precision =>
deterministic behaviour => not estimable. We must cite this and NOT present the
mechanism as new.

- <https://www.mdpi.com/1099-4300/27/1/62> ✅ verified

The distinction that survives, and it is genuine but narrow:
  * THEY recover the AGENT'S OWN PARAMETERS (gamma, C) -- computational
    phenotyping. Question: "can I fit this agent's precision?"
  * WE recover the ENVIRONMENT'S generative model from the agent's behaviour.
    Question: "can I learn how the world works by watching this agent?"
Same mechanism, different target.

Also: "As One and Many: Relating Individual and Emergent Group-Level Generative
Models in Active Inference", MDPI *Entropy* 27(2):143 ✅ verified. Names the
identifiability precondition for fitting active-inference models to behaviour
("different types of generative models can be distinguished based on the
behaviour") and observes the area is under-researched.
- <https://www.mdpi.com/1099-4300/27/2/143>

⚠️ CHECK: "Active Inference: A Method for Phenotyping Agency in AI Systems?",
arXiv:2604.23278 (2026). Title is close to this series -- establish whether it is
ours, adjacent, or independent work occupying the same framing.
- <https://arxiv.org/html/2604.23278v1>

## 5. Legibility — the constructive corollary already exists as a field  ✅ verified

The "if you wanted identifiable behaviour you would need an explicit objective for
it" corollary is not a new idea; it is the legibility literature.

Dragan & Srinivasa, **"Generating Legible Motion"**, RSS 2013 ✅ verified PDF:
<https://www.roboticsproceedings.org/rss09/p24.pdf>. Legible motion = motion that
lets an observer quickly and confidently infer the correct goal. Their central
result is STRONGER than our framing: legibility and predictability are formally
distinct and **often contradictory**.

- Dragan, Lee & Srinivasa, "Legibility and Predictability of Robot Motion",
  HRI 2013: <https://dl.acm.org/doi/10.5555/2447556.2447672> ✅ verified
  and <https://www.ri.cmu.edu/pub_files/2013/3/legiilitypredictabilityIEEE.pdf>
- "Active Legibility in Multiagent Reinforcement Learning", arXiv:2410.20954 ✅
  <https://arxiv.org/pdf/2410.20954>
- "Observer-Aware Probabilistic Planning Under Partial Observability",
  arXiv:2502.10568 ✅ <https://arxiv.org/pdf/2502.10568>

Note the target differs again: legibility is about an observer inferring the
agent's GOAL. Ours is an observer inferring the ENVIRONMENT's dynamics. Related
but not the same object -- worth stating explicitly rather than hoping nobody
notices.

---

## What this leaves as ours (revised down, twice)

The mechanism is NOT ours — §4 has it in print for active inference. The
constructive fix is NOT ours — §5 is an established field. What remains:

1. **A different target.** Prior work recovers the *agent's parameters* (§4) or
   the agent's *goal* (§5). We recover the *environment's generative model* from
   the agent's behaviour. Same mechanism, different object, and the consequences
   differ: an unidentifiable precision parameter is a nuisance, an unidentifiable
   world model means your recovered dynamics are partly fabricated.
2. **Quantification of WHICH quantity governs it.** Action entropy predicts
   recovery error (R² 0.807); visit frequency essentially does not (R² 0.093),
   with a Simpson's-paradox masking (marginally the visit rate looks irrelevant
   because the highest-weight states are exactly the zero-entropy ones). We have
   not found this decomposition reported elsewhere.
3. **The loss/recovery inversion.** Training loss *improves* (3.689 → 1.661) as
   recovery *degrades* (0.46 → 3.8), and within a single agent more epochs also
   worsens recovery. The practitioner's default health check points the wrong
   way. Concrete and actionable.
4. **A data-only certificate**, validated (AUC 0.997, with the graded WEAK tier
   doing the non-trivial work). §4 notes the identifiability problem; it does not
   give a per-conditional instrument for detecting it without ground truth.
5. **Instantiation in tensor-network structure learning** — the limit appears as
   unexercised rows of the action-conditioned bond signature.

Drop the "irony"/"epistemics are opposed" framing entirely. An agent that stops
exploring once its uncertainty is resolved is epistemic value working as
designed. The precise claim is that epistemic value drives STATE DISAMBIGUATION,
not ACTION COVERAGE, and only the latter supports identification.

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
