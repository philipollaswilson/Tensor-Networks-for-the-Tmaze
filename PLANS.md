# Plans and handover

Short-term and long-term plans for this repo, plus the latest handover. Detailed results
and experiment write-ups live in [FUTURE_EXPERIMENTS.md](FUTURE_EXPERIMENTS.md); this file
says what to do next and where things stand. On "ho", Claude rewrites the Handover
section (and updates the plan lists) before the session is cleared.

## Focus (changed 2026-11-01, replaces the 2026-10-30 "no agency" focus)

**The goal is structure learning with active inference and active learning: an efficient
online agent** that learns states, explores where its model is unsure, and acts. The offline
work is the foundation for that, not the end point. Three levels of exploration: active
inference (states), active learning (parameters), active structure learning (structure;
"active selection" in Friston's papers, and "active construction" is the user's name for it,
worth keeping if we show we build states rather than pick from candidates).

The closest published work is Regular Decision Process learning, which also merges states:
AdaCT-H, RegORL, and Deb et al., "Offline RL in Regular Decision Processes: Sample Efficiency
via Language Metrics", ICLR 2025
([PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/6e3ff406118173f18dc42a4c11c37ac9-Paper-Conference.pdf)).
Use this version, not the 2024 preprint. Its stated future work is online RDP learning and
RDPs with cycles; our learner already makes cycles.

**Working rules.** Small steps: about 4 runs at a time, saved progress, a result to look at
after each step. Comparisons fair by design, rules written first.

## Short term (next sessions)

The next experiment is agreed in outline, but the user has not said go on it yet. Ask first.

1. **Does exploration help at all? (likely minutes.)** k = 1, hub and star, 100 to 1,600
   episodes, plus T-maze (`rdp_envs`), where learned states give return 1.50 of 4.00 because
   the two cues merge. Arms: uniform, novelty, structure, oracle. One seed first. T-maze has
   two measures: states recovered, and return rising toward 4.0.
2. **One active step on k = 2.** All arms share 8,000 uniform episodes; each active arm
   collects one batch of 8,000; compare with uniform at 16,000. Don't start below about 8,000:
   small k = 2 datasets are the slowest fits and tell us little.
3. **Continue to 32,000 only if step 2 shows something.**
4. **If exploration works:** run the online loop with the generative model, so the agent both
   explores and acts, and count the experience needed. Compare structure against novelty on
   the real learner, which is the new claim.
5. **PAC under informed actions.** [ACTIVE_PAC.md](ACTIVE_PAC.md) has the first results.
   Decide what is worth turning into a paper section once real data exists.

`src/active_k2.py` needs small edits for steps 1 and 2: `K_SECRETS` and `CHECKS` are constants,
and T-maze needs its own episode loop.

## Long term

- **Learner:** k = 2 arm states (see Handover); corridor D = 4 dips to 6/8 at 3,000; the
  single-modality veto cut the spread in N* d_min from 108× to 5× (D ≤ 3), measured but not
  applied, so try it as a separately labelled version.
- **Event windows under planning:** they depend on the time left in the episode. If active
  runs show false splits, add a fixed-horizon version.
- **Benchmark fidelity:** T-maze observations differ from the ICLR paper; see
  [ENV_CHECK.md](ENV_CHECK.md). Their Table 1 could not be reproduced.
- **Prediction at low data:** HMM and GRU beat us there; worth understanding.
- **Tree tensor networks:** the latent tree ([src/latent_tree.py](src/latent_tree.py)) still
  loses to a raster chain on density; try refinement and undoing greedy joins.
- **Parked side threads:** quantum vs classical, the clock, Born vs non-negative on the maze
  (A1), a task separating the tensor route (A1b), signed fit on the clock, factors from the
  tensor, factor completion (A5), quantum transducer (A2), RGMs.
- **Not planned now:** `kmaze_plan.py` sweeps, DreamerV3 and Q-learning baselines, MiniGrid
  Memory, Mountain Car and BSuite (B8), port of refinement into `structure_pipeline.py`.

## Handover

*Last updated: 2026-11-02. The user is taking a break from experiments.*

**State of the code.** Nothing is committed, and nothing is running. Use the project venv
(`.venv\Scripts\python.exe`). `lib/flexfringe/` holds a GPL-3 binary of about 5 MB: decide before
committing it.
- [src/kmaze.py](src/kmaze.py): learner `learn_states`; `Episode` and `tail_of` (planned prefix,
  random tail); `partition_key` stops rounds and ends cycles.
- [src/rdp_envs.py](src/rdp_envs.py), [src/rdp_baselines.py](src/rdp_baselines.py),
  [src/rdp_language.py](src/rdp_language.py): six benchmarks, AdaCT-H, FlexFringe, language metric.
- [src/rdp_compare.py](src/rdp_compare.py), [src/rdp_summary.py](src/rdp_summary.py): the comparison,
  `logs/rdp_summary.out`.
- [src/gen_model.py](src/gen_model.py): generative model and EFE planner; `results/gen_model/table.txt`.
- [src/active_k2.py](src/active_k2.py): the active loop; rules in its header; resumes from `.pkl`.
- [MERGE_PAC.md](MERGE_PAC.md) with `src/merge_pac_check.py` (19/19) and
  [ACTIVE_PAC.md](ACTIVE_PAC.md) with `src/active_pac_check.py` (44/44).

**Offline comparison (150 cells, 5 seeds, 15-minute budget per fit).** Gap above the Bayes
floor, nats per step, at 30,000 episodes (ours / language AdaCT-H / FlexFringe / HMM / GRU):
T-maze 0.0006 / 0.014 / 0.0012 / 0.0061 / 0.0002; Cookie ~0.0009 (1 of 5 timed out) / 0.019 / 0.016 /
0.0032 / 0.0002; Corridor 0.0001 / 0.0001 / 0.0002 / 0.0002 / 0.0001; Cheese 0.0014 / 0.11 / 0.030 / 0.0010 /
0.0020; Mini-hall 0.0002 / 0.25 / 0.027 / 0.0037 / 0.0014. Ours matches or beats every state-merging
baseline at every size, tying on Corridor from 1,000 up, usually with much smaller machines (Cheese
27 states; language 61, FlexFringe 488; truth 14). HMM and GRU predict better at low data, and
they give no readable states. Tiger: ours timed out on all 25 cells, as predicted. Ours is slower
than the merging baselines.

**Generative model.** Return vs optimum at 10,000 episodes: Corridor 5.00/5.00, Mini-hall
2.35/2.35, Cheese 1.19/1.19, Cookie 0.96/1.00, T-maze 1.50/4.00, Tiger no model. The planner reaches
the optimum on true states. T-maze fails because the two cues merge: the goal matters 5 steps on,
on about 1/256 of random paths; true states get 4.0 on the same data. Epistemic value helped and
hurt on Cookie, and did little elsewhere. Return depends on preference strength, so "no weights"
holds only between epistemic terms. Learned states are readable, including a hidden "cookie
exists" state on Cookie (`results/gen_model/show_cookie_n10000.txt`).

**Active run.** Stopped on purpose at 11 of 48 checkpoints: too big, and started at the slowest
data sizes. It gave no comparison between arms. Partial scores: `results/active_k2/*.tmp`.
Uniform, seed 1: 8,000 episodes, 30 states, 95.3% of visits on the right state.

**Why k = 2 fails.** Not a data limit in total: at 30,000 episodes, rows pooled by true state have
-84 nats against merging (-3 is enough). But a rare doorway is routed on its own visits, so it
files "knows both" under "knows one", and later stages only merge. 44–45 states of 45, never
exact; the 45 is partly luck (97.9% of visits on the right state). Edge reassignment failed
(A6 in FUTURE_EXPERIMENTS.md).

**Math in one paragraph.** The merge test follows dF ~ -n JS + (m-1)/2 ln(...), confirmed. Wrong merges
fall exponentially, wrong splits polynomially. N* = c/d_min is not a law. No theorem covers
`learn_states` itself. A 1/(d mu_0^2) bound is right, and the ICLR paper already corrected it.
Actions change both d and mu_0, which is the research opening. Theory says tails stay valid given a
true state, and that structure information beat count-based exploration on small synthetic cases,
mostly by stopping on evidence. Novelty is not a greedy minimizer of the bound. `combined` in
`active_k2.py` is a proxy, not the exact chain-rule sum (ACTIVE_PAC.md, section 7).

**Dead ends already tried.** Details in FUTURE_EXPERIMENTS.md (A4, A6): red-blue merging on the
history tree; unvetoed BMR sum; Perks prior over all O^L sequences; next-state modality inside
minimization; split-only rounds without minimizing; redirecting only on a clear majority;
hard-EM edge reassignment; censored event windows as the cause of over-splitting.

**Stale numbers.** Learner results from before the stop-rule fix, including older corridor and k = 2
figures in FUTURE_EXPERIMENTS.md and the coverage table's first version. The clock comparison was
tilted toward Born; don't quote "4–13×". `clock_compare.py` holds unrun fair code and no longer
matches `compare_s*.json`.

**Where to start.** Ask whether the user wants step 1. The one-line summary: exploration has
not been tested yet, and step 1 is the cheap way to find out.

**Open question for the user.** Which paper this targets (`paper_mm/` or `paper_aaai/`) and whether
there is a deadline. Both drafts predate this work.
