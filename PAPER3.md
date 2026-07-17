# Paper III — direction and roadmap

## Where the series stands

- **Paper I** asked *what agency is* and answered with hand-built models: three
  criteria (intentionality, rationality, explainability), instantiated in active
  inference, with empowerment as a controllability probe.
- **Paper II** showed the *world model can be recovered from behaviour*: normalised
  hidden states, labelled A/B, dependency graph, factorisation, state count, symbolic
  states, then controllability read off the learned model.

Paper II has four honest gaps, stated in its own Discussion:

1. It recovers the **environment's** model (uniform-action rollouts), not any particular
   **agent's** posterior. So it is not yet *phenotyping an agent*.
2. It computes only **empowerment** (controllability), not the three agency criteria.
   Empowerment on its own is not a measure of agency.
3. Validation is **against known ground truth**, non-blind, with the state-count
   criterion chosen after inspection. Nothing is predicted out-of-sample.
4. Two tiny POMDPs, and (Philip's point) the context bit is drawn i.i.d. each step
   rather than being a **persistent episode latent** — so "only the cue is informative"
   holds, but context is not the stable hidden variable the standard T-maze intends.

## The thesis for Paper III

**Phenotype actual agents from their own behaviour: recover the full agency profile
(all three criteria, not just control), discriminate agents by character, and validate
blind with a pre-registered predictive test on a persistent-latent environment.**

This closes the loop the series opened: Paper I defined the profile, Paper II recovered
the substrate, Paper III reads the profile off the recovered substrate for a *specific
agent* and shows it *distinguishes agents* — the thing "phenotyping" has been promising.

## The killer experiment: agent discrimination

Generate rollouts from N agents that differ in a controlled way:
- an **info-seeker** (high preference precision, visits the cue first),
- a **reward-gambler** (low precision / high risk, goes straight to an arm),
- a **habitual/random** agent (flat policy).

Fit one MPS per agent, on that agent's *own* rollouts. Recover each agent's structure
and full agency profile. Show the profiles **separate by character**, then blind-classify
a held-out agent from its recovered phenotype alone. A profile that can tell a curious
agent from a gambler, without being told which is which, is a genuine phenotyping result.

## Workstreams (each tied to the gap it closes)

### 1. Agent-specific phenotyping — closes gap 1
Train per-agent MPS on the agent's rollouts instead of exhaustive uniform-action ones.
The recovered model is now the agent's policy-weighted world model; state-space coverage
is no longer guaranteed, which is realistic and worth characterising. Reuses the whole
Paper II pipeline; the change is the training set. **High payoff, low-moderate effort.**

### 2. The full three criteria, not just empowerment — closes gap 2
Operationalise all three on the recovered model:
- **Intentionality**: inverse-infer the preference vector C from the recovered A, B and
  the behaviour; goal-directedness = whether recovered preferences render the observed
  policy an EFE minimiser.
- **Rationality**: the EFE-optimality gap (regret) of the behaviour under the recovered
  model.
- **Explainability**: fidelity and description length of the recovered model itself.
Empowerment stays as the controllability facet, now one axis of a full profile.
**This is the direct answer to "empowerment isn't agency." High payoff.**

### 3. Blind, pre-registered, predictive validation — closes gap 3
- Pre-register the state-count threshold and the structure hypotheses before fitting.
- Hold out trajectories / whole sub-branches the exact model cannot memorise; test that
  the recovered structure predicts held-out conditionals.
- Make one falsifiable behavioural prediction (e.g. "this agent will seek the cue") and
  confirm it out-of-sample. **Cheap, and the single biggest credibility upgrade.**

### 4. Persistent-latent, richer environment — closes gap 4 (Philip's point) + scale
- Regenerate the data so context is a **stable episode latent** (proper HMM-style
  emission), so the cue resolves a persistent hidden state and the info geometry is about
  a real latent. Directly answers Philip and matches the canonical T-maze.
- Then scale: longer horizons, more actions, larger mazes; stress-test where recovery
  breaks. Continuous observations via a feature map are a stretch. **Do the
  persistent-latent regen early — it is a small data-gen change with a big framing payoff.**

### 5. Actual tensor-network structure search — closes the "senses we skip" caveat
Use the topology-search literature we already cite (Li et al., TNALE) to search bond
structure / tree-vs-chain, *discovering* the dependency topology rather than assuming the
temporal chain. This is the one sense of structure learning Paper II explicitly does not
do. **Most novel, highest effort — stretch / future.**

### 6. Gauge-aware model selection — technical loose end
Replace the naive parameter count (which ignores MPS gauge freedom and makes MDL/BIC
conservative-by-one) with a gauge-aware count, so state selection stops relying on an
eyeballed knee. **Contained fix.**

## Suggested scope

Spine = **1 + 2 + 3 + 4**: phenotype specific agents, compute all three criteria, validate
blind, on a persistent-latent environment. That alone fixes every stated limitation and
delivers the discrimination result. Keep **5** (topology search) and **6** (gauge-aware
count) as strengtheners or explicit future work so the paper stays focused.

## What to reuse

Everything in `src/` transfers: `structure_recovery.py`, `factorization_test.py`,
`gauge_fix_states.py`, `model_selection.py`, `make_extension_figures.py`. The new pieces
are (a) an agent that generates rollouts (pymdp active-inference agents with different C /
precision), (b) inverse inference of C and the EFE-regret computation, and (c) a
persistent-latent version of the data generator.
