# Horizon-2 expected free energy, honest context model (nats)

Command: `python efe_two_step.py` (numpy; pymdp cross-check appended if importable).

## Preferences paper (+2/-4): C over (Cheese, Shock, SeeLeft, SeeRight, Null) = [2.0, -4.0, 0.0, 0.0, 0.0]

### Horizon 1

| action | risk | epistemic | G | q(pi), gamma=16 |
|---|---|---|---|---|
| Left | 3.343 | 0.693 | 2.649 | 0.000 |
| Right | 3.343 | 0.693 | 2.649 | 0.000 |
| Cue | 2.343 | 0.693 | 1.649 | 1.000 |

### Horizon 2, open-loop (pymdp policy_len=2 convention)

| policy | risk t1 | epist. t1 | risk t2 | epist. t2 | G | q(pi) |
|---|---|---|---|---|---|---|
| Left-Left | 3.343 | 0.693 | 3.343 | 0.693 | 5.299 | 0.000 |
| Left-Right | 3.343 | 0.693 | 3.343 | 0.693 | 5.299 | 0.000 |
| Left-Cue | 3.343 | 0.693 | 3.343 | 0.693 | 5.299 | 0.000 |
| Right-Left | 3.343 | 0.693 | 3.343 | 0.693 | 5.299 | 0.000 |
| Right-Right | 3.343 | 0.693 | 3.343 | 0.693 | 5.299 | 0.000 |
| Right-Cue | 3.343 | 0.693 | 3.343 | 0.693 | 5.299 | 0.000 |
| Cue-Left | 2.343 | 0.693 | 3.343 | 0.693 | 4.299 | 0.000 |
| Cue-Right | 2.343 | 0.693 | 3.343 | 0.693 | 4.299 | 0.000 |
| Cue-Cue | 2.343 | 0.693 | 2.343 | 0.693 | 3.299 | 1.000 |

### Horizon 2, sophisticated (second action conditioned on the first observation)

| first action | risk t1 | epist. t1 | E[risk t2] | E[epist. t2] | G | q(a1) |
|---|---|---|---|---|---|---|
| Left | 3.343 | 0.693 | 3.343 | 0.000 | 5.992 | 0.000 |
| Right | 3.343 | 0.693 | 3.343 | 0.000 | 5.992 | 0.000 |
| Cue | 2.343 | 0.693 | 0.343 | 0.000 | 1.992 | 1.000 |

## Preferences flat (0/0): C over (Cheese, Shock, SeeLeft, SeeRight, Null) = [0.0, 0.0, 0.0, 0.0, 0.0]

### Horizon 1

| action | risk | epistemic | G | q(pi), gamma=16 |
|---|---|---|---|---|
| Left | 1.609 | 0.693 | 0.916 | 0.333 |
| Right | 1.609 | 0.693 | 0.916 | 0.333 |
| Cue | 1.609 | 0.693 | 0.916 | 0.333 |

### Horizon 2, open-loop (pymdp policy_len=2 convention)

| policy | risk t1 | epist. t1 | risk t2 | epist. t2 | G | q(pi) |
|---|---|---|---|---|---|---|
| Left-Left | 1.609 | 0.693 | 1.609 | 0.693 | 1.833 | 0.111 |
| Left-Right | 1.609 | 0.693 | 1.609 | 0.693 | 1.833 | 0.111 |
| Left-Cue | 1.609 | 0.693 | 1.609 | 0.693 | 1.833 | 0.111 |
| Right-Left | 1.609 | 0.693 | 1.609 | 0.693 | 1.833 | 0.111 |
| Right-Right | 1.609 | 0.693 | 1.609 | 0.693 | 1.833 | 0.111 |
| Right-Cue | 1.609 | 0.693 | 1.609 | 0.693 | 1.833 | 0.111 |
| Cue-Left | 1.609 | 0.693 | 1.609 | 0.693 | 1.833 | 0.111 |
| Cue-Right | 1.609 | 0.693 | 1.609 | 0.693 | 1.833 | 0.111 |
| Cue-Cue | 1.609 | 0.693 | 1.609 | 0.693 | 1.833 | 0.111 |

### Horizon 2, sophisticated (second action conditioned on the first observation)

| first action | risk t1 | epist. t1 | E[risk t2] | E[epist. t2] | G | q(a1) |
|---|---|---|---|---|---|---|
| Left | 1.609 | 0.693 | 1.609 | 0.000 | 2.526 | 0.333 |
| Right | 1.609 | 0.693 | 1.609 | 0.000 | 2.526 | 0.333 |
| Cue | 1.609 | 0.693 | 1.609 | 0.000 | 2.526 | 0.333 |

## Preferences mild (+1/-1): C over (Cheese, Shock, SeeLeft, SeeRight, Null) = [1.0, -1.0, 0.0, 0.0, 0.0]

### Horizon 1

| action | risk | epistemic | G | q(pi), gamma=16 |
|---|---|---|---|---|
| Left | 1.806 | 0.693 | 1.113 | 0.333 |
| Right | 1.806 | 0.693 | 1.113 | 0.333 |
| Cue | 1.806 | 0.693 | 1.113 | 0.333 |

### Horizon 2, open-loop (pymdp policy_len=2 convention)

| policy | risk t1 | epist. t1 | risk t2 | epist. t2 | G | q(pi) |
|---|---|---|---|---|---|---|
| Left-Left | 1.806 | 0.693 | 1.806 | 0.693 | 2.226 | 0.111 |
| Left-Right | 1.806 | 0.693 | 1.806 | 0.693 | 2.226 | 0.111 |
| Left-Cue | 1.806 | 0.693 | 1.806 | 0.693 | 2.226 | 0.111 |
| Right-Left | 1.806 | 0.693 | 1.806 | 0.693 | 2.226 | 0.111 |
| Right-Right | 1.806 | 0.693 | 1.806 | 0.693 | 2.226 | 0.111 |
| Right-Cue | 1.806 | 0.693 | 1.806 | 0.693 | 2.226 | 0.111 |
| Cue-Left | 1.806 | 0.693 | 1.806 | 0.693 | 2.226 | 0.111 |
| Cue-Right | 1.806 | 0.693 | 1.806 | 0.693 | 2.226 | 0.111 |
| Cue-Cue | 1.806 | 0.693 | 1.806 | 0.693 | 2.226 | 0.111 |

### Horizon 2, sophisticated (second action conditioned on the first observation)

| first action | risk t1 | epist. t1 | E[risk t2] | E[epist. t2] | G | q(a1) |
|---|---|---|---|---|---|---|
| Left | 1.806 | 0.693 | 1.806 | 0.000 | 2.919 | 0.000 |
| Right | 1.806 | 0.693 | 1.806 | 0.000 | 2.919 | 0.000 |
| Cue | 1.806 | 0.693 | 0.806 | 0.000 | 1.919 | 1.000 |

## Reading

Every first action resolves the context, so the state information gain at t1 is ln 2 for all three,
and after any first observation nothing remains to be learned, so the sophisticated t2 epistemic term is 0
for every branch. The epistemic term never separates the actions at either horizon. What separates them at
horizon 2 is the decision value of the resolved belief: after the Cue the agent can still choose the cheese
arm (low t2 risk), after an arm it is absorbed (t2 risk equals t1 risk). Under the mild profile the three
actions tie at horizon 1 and the Cue wins at horizon 2 through this term alone. The open-loop convention
counts the same bit at both steps and cannot exploit it, which is why it prefers Cue-Cue.

(pymdp cross-check skipped: ModuleNotFoundError("No module named 'pymdp'"))

