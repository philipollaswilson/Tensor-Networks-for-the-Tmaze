# Why we're doing this, and what we've learnt

Written 2026-11-03 for Mao. [EXPERIMENTS.md](EXPERIMENTS.md) lists what was run. This file covers
the reasoning: why the project moved where it did, the intuitions we now work from, what we think
we've learnt, and what's still open. Where something is a belief rather than a result, it says so.

## 1. How we got here

**The starting point** was agency defined through three criteria (intentionality, rationality,
explainability), with empowerment as a probe, all on hand-built models.

**First finding: the model can be learned from behaviour.** An MPS trained on T-maze rollouts gives
back normalised hidden states, A and B to TV ≤ 0.001, the dependency graph, the factorisation and
empowerment. Wauthier et al. had said the bonds weren't usable as states; they are.

**Second finding: doing the same for individual agents taught us the most important lesson so
far.** Blind discrimination at accuracy 1.00 was circular, as your audit showed. What survived is
that **an observer can only identify a conditional where the agent varies its action.** Competent
agents stop varying their actions, so their world models become unidentifiable from their
behaviour. The first finding worked because uniform-action rollouts are the best case.

That raised the question we're on now. **If the data decide what can be learned, the learner
should choose its data.** So the work moved from "recover structure offline from someone else's
behaviour" to "an agent that learns its own state space online, and explores where its model of
the *structure* is unsure."

**The current goal (set 2026-11-01).** Structure learning with active inference and active
learning: an efficient online agent that learns states, explores where its model is unsure, and
acts. Exploration works at three levels:
- **Active inference**, about states: where am I?
- **Active learning**, about parameters: what does this state do?
- **Active structure learning**, about structure: are these two histories the same state? Friston's
  papers call this "active selection". We've been calling it "active construction", and want to keep
  that name only if we can show the agent *builds* states rather than picks from candidates.

## 2. The core picture: hidden states are merged histories

Almost everything we do now rests on one idea.

**A hidden state is a set of histories that predict the same future.** Two histories that make the
same predictions should be one state, and two that don't must be two. This is the predictive state,
or causal state, of computational mechanics. It is also what an MPS bond represents, and what BMR
decides when it compares "one row" with "two rows".

So **structure learning is mostly merging**, and the geometry has three steps (FUTURE_EXPERIMENTS.md
A5):
1. **Merge** removes duplicate rows of the past–future table. That gives causal states, the 7 of the
   T-maze.
2. **Unmix** finds the corners: rows that are mixtures of other rows are beliefs, not states. The
   T-maze centre is ½ cue-left + ½ cue-right, which gives 6 states and rank 6.
3. **Factor** splits the corners into a product: context 2 × place 3, with B = B_ctx ⊗ B_place.

Rank, non-negative rank and psd rank are the same question asked through a subspace, a simplex and
the psd cone. This links HMMs, PSRs and Born machines in one frame. We find this the most useful
lens on "what does quantum add?" (section 4).

**Why BMR is the right merge test.** The Bayes factor for merging two count rows is, to good
accuracy,

    dF ≈ −n · JS(p, q) + (m − 1)/2 · ln(n₁n₂/n)

The first term is evidence that the rows differ; it grows linearly. The second is Occam's razor,
which favours merging; it grows like log n. So two different rows separate after about
(m − 1) ln n / (2 JS) samples, however many histories there are. Deterministic worlds have the
largest JS, which is why merging works so well on mazes.

**The two errors are not symmetric.** This one fact explains much of what we see:
- A wrong merge becomes exponentially unlikely with data.
- A wrong split decays only polynomially, about n^−κ.

So BMR is cautious about merging and loose about splitting. A learner built on it has to be able
to *undo* splits cheaply. It mostly can. What it can't do is undo a merge (next section).

## 3. What we've learnt about why learners fail

**Once a distinction is merged away, it can't come back.** This is the k = 2 lesson, and we think
it's the central problem for any merge-based learner. Pooled by the true state, "knows both secrets"
vs "knows one" has −84 nats of evidence, when −3 is enough. But the learner never compares those
pools. It routes one *edge* at a time, and the edge out of the arm has about 150 clean visits, too
few on its own. It gets filed under "knows one". Every later stage only coarsens, so the secret is
forgotten for good. Prop 8 in MERGE_PAC.md makes this a lower bound for *any* learner: what matters
is the rarest edge weighted by how different its possible targets are, not the rarest state.

Two consequences we believe but haven't tested:
- **Keep distinctions rather than try to recover them.** Start from an over-split machine that
  remembers what has been seen, then merge down. Splits are cheap to undo (section 2); merges aren't.
- **This is exactly where acting should help.** An agent that knows which edge is undecided can go
  back and traverse it. Active learning doesn't lower the visits an edge needs. It lowers the
  episodes it takes to collect them (ACTIVE_PAC.md A10). That's the whole bet.

**The rule matters more than coverage.** We expected data needs to follow coverage (N* ∝ 1/d_min).
They don't. Most of the 108× spread was the *sum* veto firing on equal noisy rows: many weak
negatives add up and nothing offsets them. A veto on the single worst modality keeps the same
safety proof and cut the spread to 5×. We haven't applied it in `kmaze.py` yet; it would be a
separately labelled version.

**Fitting your data is not learning the world.** The agent experiments showed this twice: across agents
(concentrated data fit better and recover worse) and within one agent (more epochs fit better and
recover worse). Training loss can't be used as a proxy for structure recovery. This is also why we
score learners by gap above the Bayes floor *and* by readable states, not by fit alone.

**Unifilar is the easy case, and we're mostly in it.** Our learner finds unifilar machines: the
next state is a function of (state, action, observation), so counting is exact Bayes. That's why it
is fast and its states are readable. General POMDPs have states the history never pins down (Tiger
is the example; ours times out on it). There you need inference or spectral methods, and you get
local optima. We expect a task that's small non-unifilar and large unifilar to be where the MPS
route and the count route separate. We haven't built one yet.

## 4. What role do tensor networks play?

This is worth being plain about, because the project's name suggests more than we've shown.

- **The MPS is the right language.** The bond is the hidden state, cuts are where structure lives,
  and a non-negative unifilar MPS is exactly the machine our learner builds. The latent tree on
  MNIST is a tree tensor network learned with no order given.
- **The quantum (Born) form has not paid off on our tasks.** The T-maze centre is a mixture, not a
  superposition, and the classical generator already meets the rank bound. On the probability clock
  Born wins clearly in bond dimension (2 vs 15+) but only slightly in bits, and that test favoured
  Born. A pure Born state with positive amplitudes needs one dimension per predictive state; it takes
  mixed states to represent ignorance.
- **The coverage finding is estimator-agnostic.** Any estimator would show the coverage
  limit, and the confabulation hypothesis, the one tensor-network-specific claim, was refuted.

So our honest position: tensor networks give us the representation and a clean link to the
physics literature on structure. The working method is Bayesian merging of predictive states. The
quantum question (A1, A1b) is parked, not dead. The test that would settle it is a signed rank-3
model on the clock, or a task whose minimal unifilar model is large but whose Born model is small.

## 5. Where we stand against the literature

The closest published work is **Regular Decision Process learning** (AdaCT-H, RegORL, and Deb et
al., ICLR 2025, the language-metric paper). It also merges histories into states. Our offline
comparison:

- **Better.** We match or beat every state-merging baseline on all five tractable benchmarks, with
  usually much smaller machines (Cheese: 27 states vs 61 and 488; truth 14). On rare events a KL-based test
  needs about 1/μ samples where a Hoeffding test needs 1/μ². Pooling over time is exponentially
  better than their layered occupancy on stationary processes. And our learner makes cycles, which
  their stated future work (online RDPs, RDPs with cycles) asks for.
- **Worse.** Ours is slower. There's no calibrated δ: BMR has fixed thresholds, and its wrong-split
  rate is polynomial where theirs is logarithmic. A growing threshold fixes this in theory (ACTIVE_PAC
  A9). We assume stationarity and a memoryless policy, and they don't.
- **Not ours to beat.** HMMs and GRUs predict better at low data, and give no readable states.

We also found that their preprint bound has the wrong order: 1/(dμ₀) should be 1/(dμ₀²). The ICLR
version already fixes this, so it isn't a contribution, but it's a reason to cite the ICLR version
only. And we couldn't reproduce their Table 1 (ENV_CHECK.md).

On the active-inference side, the RGM work (Friston et al., "From pixels to planning") assumes the
factors through a locality prior. Our pipeline *finds* factors, for example 3 × 8 on the synthetic
bird. Whether it finds the factors RGMs assume on their own demos is still open.

## 6. What we think active learning can and can't buy

From ACTIVE_PAC.md. All of this is on synthetic cases; none of it has run on a maze.

- **Most of the gain is stopping on evidence**, not a clever score. Elimination, which has no score
  at all, does as well as structure information gain (296 vs 310 steps against 2013 for counting).
- **Novelty is the wrong signal for structure.** It allocates by support, not by how hard a
  distinction is. It's fooled by noisy rows (7× worse than counting there), and it does nothing for
  a rare edge already merged into a big state, because it reads the wrong pool.
- **But novelty is needed for discovery.** A structure score can only rank distinctions between
  things already seen. On a never-visited edge only novelty points there.
- **"No weights" is earned, if done right.** The chain rule I(o; S) + I(o; θ | S) + I(o; q | θ, S)
  is an identity, so summing the terms needs no trade-off weight. It only holds in the order S,
  then q, then θ, with the novelty term mixed over structure. The per-edge version quietly collapses
  to count-based exploration. Our current `combined` arm is a proxy for this, not the identity.
- **The ceiling.** Gains are bounded by A^L in distinguishability and by min(2n*, A^D) on unknown
  paths. If actions don't change what you see, there's no gain at all.

How this links back to the agent experiments: there, the agent's epistemics (resolve *my* uncertainty about
context) were opposed to the observer's (vary actions so I can identify you). Here the agent *is*
the observer of its own world, so the epistemic term has to be about structure, not just about
states. EFE has no term for "make the world identifiable". Active structure learning is our attempt
to add one.

## 7. How we've been working, and why

These habits came out of mistakes, and we'd like to keep them.

- **Adversarial checks before celebrating.** Your audit of the blind-discrimination result is the template. Ask what a
  trivial baseline scores, and whether the method is doing any work at all.
- **Fair by design.** Write the comparison rules before the runs. Give every arm the same data,
  budget and stopping rule. The clock comparison is the example of what goes wrong otherwise.
- **Small steps, saved runs.** About 4 runs at a time, checkpoints, resumable, and a result to look
  at after each step. The k = 2 active run is the example: 48 checkpoints, started at the slowest
  sizes, stopped with nothing to compare.
- **Record corrections.** MERGE_PAC and ACTIVE_PAC each list the claims that changed because a run
  disagreed, about 15 in all. We trust those notes more because of it.

## 8. Open questions, and where your input would help most

1. **Is anything tensor-network-specific left in the coverage finding?** One bounded test is still
   untried: run the full recovery machinery (bond-ray states, recovered A and B, subsystem MI) per
   agent, and ask whether it differs across agents in ways coverage alone doesn't predict. If not,
   the finding stands as a limit of learning from behaviour, and it is the motivation for active
   structure learning.
2. **Does exploration help at all?** This is the next experiment, and it's cheap: k = 1 hub and
   star, plus the RDP T-maze, where learned states give return 1.50 of 4.00 because the two cues
   merge. If structure-directed exploration doesn't beat uniform there, the bet in section 3 is wrong.
3. **Theory for the real learner.** Every theorem covers an idealised learner that is given a
   refinement, or one with pure rows. `learn_states` starts from observation labels, whose rows are
   mixtures. No theorem covers that, and it's the hard part.
4. **The active-inference framing.** Is "active structure learning" a fair reading of Friston's
   active selection? Is the chain-rule decomposition (structure, then state, then parameters) the
   right EFE for it? You know this literature better than we do.
5. **Back to agency.** If the agent builds its own states, are its learned states a better substrate
   for the three criteria than a hand-built model? We haven't touched this since July.
