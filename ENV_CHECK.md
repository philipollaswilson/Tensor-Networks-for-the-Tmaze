# rdp_envs.py against the ICLR 2025 paper

`src/rdp_envs.py` was checked against Deb et al., ICLR 2025, Appendix D, plus Examples 1 and 3, Table 1 and the figures. `rdp_envs.py` was not edited. The numbers come from [src/rdp_language_test.py](src/rdp_language_test.py), and the log is [logs/rdp_language_test.out](logs/rdp_language_test.out).

Horizons all match Table 1: T-maze 5, Cookie 9, Corridor 5, Cheese 6, Mini-hall 15.

## Differences that can change results

| # | Domain | Difference | Effect |
|---|---|---|---|
| 1 | T-maze(c) | Observation at the junction. `rdp_envs` shows 010 on arriving at the junction and a fifth code 000 on the N/S step. Figures 4 and 6 show 101 on arrival and 010 on the N/S step. Example 1 says "at the T-junction and beyond the observation is 010", and Example 3 emits from the cell left, which fits the figures. The docstring chose the text over the figure, but the ICLR wording resolves it in favour of the figures. | **Changes results.** Minimal layered size 21 becomes 20 (exact count; my reading is that cells 2 and 3 at t = 4 stop being different). Excess log-loss at n = 2000 goes 0.060 to 0.049, at n = 30000 0.012 to 0.0065. The extra code also gives the language metric a "reached the junction" symbol the paper's domain does not have. |
| 2 | All | Behaviour policy. `rdp_envs` is uniform; the paper gives none for Table 1. Under uniform T-maze(c), the two cues differ by only L = 1/1024 at t = 0, which needs about 3.5e7 episodes. Table 1 reports r = 4.0 and 0.26 s. | **Changes results, inference.** Their data almost certainly favoured East or was far larger. Uniform data cannot reproduce that row at any n we can afford. |
| 3 | All | Dataset size is not stated for Table 1. Only Figure 2 gives K = 100. | **Changes results.** State counts depend strongly on n (see below). |
| 4 | Cheese, Mini-hall | The reset is visible only through r = 1. The paper's automaton reads AO, not r, so with the paper's `key='ao'` a reset and a plain move with the same (a, code) share an edge. | **Can change results.** Measured at n = 2000: Cheese excess 0.714 with `ao`, 0.739 with `aor`, states 26 both. Mini-hall 1.309 vs 1.313. Small here, but it means these two domains are not strictly RDPs over AO as encoded. |
| 5 | T-maze(c) | Corridor action set in the figures. Figure 6 has no W edge at q1 or q12 (cell 0), yet q5 has both E and W going to q8. No single rule produces it. | **Open.** `ew` gives 21, 20 with figure observations, 16 with no W at S. Full actions with no W at S gives 19. None reproduces Figure 6's structure or Table 1's 18. |

## T-maze(c) vs the Example 1 T-maze

The paper names one difference: corridor observation 101 always in T-maze(c), 101 or 111 uniformly in Example 1. `rdp_envs` has T-maze(c) only, so nothing in the benchmark exercises this.

- It is the case that matters for the language metric. In Example 1 the L-infinity prefix distance decays as 2^-N (Theorem 1) and the language metric does not. With deterministic 101 that gap is absent, so T-maze(c) does not test the paper's main claim.
- Example 1 and 3 also differ from T-maze(c) in two ways the paper does not mention. Rewards are 1 and 0, not +4, -1 and -1 for standing still. The horizon is H = N + 2, while Figure 4 is H = N + 1 (four E, one N/S, H = 5).
- `TMazeEx1` in the test file builds this variant without touching `rdp_envs`. Section 6 of the test confirms the gap: at K = 1000 the language metric is exact for N = 2 to 18 (5/5 seeds), and L-infinity stays at N + 4 states at every K tried, which means it merged the cue.

## Differences that cannot change results

| Domain | Difference | Why harmless |
|---|---|---|
| T-maze(c) | Cue meaning. `CUES = (1, 0)` makes 110 mean goal north. ICLR Example 1, Example 3 and Figures 4 and 6 all give 011 = north. | Relabelling. Figure 4 red edges agree with the ICLR text. |
| All | The docstring's figure numbers are from v1. ICLR: Corridor automaton is Figure 3, T-maze restricted Figure 4, environments Figure 5, partially restricted Figure 6. D.2 itself says "Figure 1" where it means Figure 4. | Documentation only. |
| T-maze(c) | Example 3 and D.2 put cue at S. `cue_once=True` shows 101 on returning to S. | Figure 6 W edges show 101, and Example 3 agrees. Already correct. |
| Corridor | D.1 prints p_i^i, a typo for p_i^1. | `rdp_envs` is right. |

## Checked and matching

- **Corridor.** Every edge of Figure 3 decoded by hand fits `rdp_envs` with m = 6, p0 = 1, p1 = 0, start row uniform, no enemy in column 0, swap on encounter only. Layered size 11, the same as Table 1.
- **T-maze(c), restricted.** Figure 4 has 12 states, and so does `actions='east'`, with either observation scheme.
- **T-maze(c) rewards.** +4 correct, -1 otherwise and end, -1 standing still. E or W at the junction as "otherwise" is still an assumption.
- **Cookie.** Rooms, Figure 5b layout, actions, 6 observations, reward 1, pressing again after eating. The figure's cookie at the start is still read as illustration. Pressing with a cookie out replaces it (Icarte 2019), and the ICLR text is silent. Without replacement the minimal size stays 77, so this does not matter.
- **Cheese.** 10 states, 6 observations, 4 actions, reset uniform on non-goal cells, H = 6. There is no figure in the paper. Goal as non-state is still the only reading giving 10 and 6.
- **Mini-hall.** 12 states, 3 actions, 6 observations. Every label of Figure 5a matches `OBS`, and doors match. D.5 says "reset" without saying uniform.

## Paper statements that do not check out

- **Mini-hall.** D.5 says that from observation 3, 3 actions suffice if the agent starts in room B or C. In `rdp_envs`, (C, N) needs 3 but (B, S) needs 4: turn, forward, turn, forward. Figure 5a gives the same geometry, so I read this as loose wording, not an environment error.
- **Corridor.** Table 1 gives r = 1.0, and "the maximum reward for all the domains is 1". `rdp_envs` pays +1 per column, return 5.0, which D.1 and Figure 3's red edges support. T-maze's r is a sum, so per-step normalisation does not explain it. It does not affect automaton learning, since r follows from e.
- **Table 1 vs its text.** The text says language automata are smaller than FlexFringe and CMS "in all domains except Mini-hall". Printed Cheese is 1178 vs 669 and 1158. The v1 HTML has 326 for Cheese and 5134 for Mini-hall with X_{1,1,1}, so the sentence seems left over from v1. The `rdp_envs` docstring quotes those v1 numbers. ICLR values are 1178 and 6098.
- **Figure 2.** The paper reports ~2H states at K = 100 on the Example 1 T-maze. With the printed test we got none of 5 seeds exact at K = 100 for any N, and all 5 at K = 1000. On expected counts it cannot work: at t = 1 a candidate holds about K / 4 = 25 episodes (cue times 101 or 111) and the cues differ by L = 0.5, while the threshold is 0.64 even at delta = 1 (|X| = 89). So Figure 2's setup differs from the printed test or from the K stated.

## Table 1 vs what these environments give

Minimal layered size against AdaCT-H with X_{3,1} (`key='ao'`, delta = 0.05, uniform behaviour, one seed):

| Domain | Table 1 | Minimal | n = 2000 | n = 30000 | n = 300000 |
|---|---|---|---|---|---|
| Corridor | 11 | 11 | 11 | 11 | 11 |
| T-maze(c) | 18 | 21 | 11 | 14 | 16 |
| Cookie | 91 | 77 | 34 | 41 | 56 |
| Cheese | 1178 | 74 | 26 | 63 | 70 |
| Mini-hall | 6098 | 236 | 39 | 123 | 179 |

Our runs approach the minimum from below and never exceeded it at any n tested, as a correct test should. Table 1's Cookie, Cheese and Mini-hall are above the minimum, Cheese and Mini-hall by 16 and 26 times. A wrong merge early on can force extra splits later, so being above the minimum is possible, but we never saw it, at any n from 300 to 300000. I conclude those rows are unlikely to come from this test on these domains with this behaviour. Their domains, data or code differ in a way the paper does not state, and I could not identify which.

## Worth doing, in rdp_envs and only after the comparison ends

1. Fix observation item 1, as an option so existing results stay reproducible.
2. Add the Example 1 T-maze, which is the only domain that tests Theorem 1.
3. Add an East-biased behaviour for T-maze, since uniform cannot show the cue at any practical n.
4. Update the docstring's figure numbers and Table 1 values.
