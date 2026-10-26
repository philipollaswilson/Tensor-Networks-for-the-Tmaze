# Noisy-arm T-maze: when the epistemic term does the work

Command: `python noisy_arm.py` (numpy, matplotlib; pymdp cross-check appended if importable). Nats unless stated.

Arms reveal the context with reliability r, the Cue with reliability c. Each step inside an arm draws a fresh outcome. Sophisticated horizon-2 planner as in efe_two_step.py. Selection = argmin G at precision gamma = 16 (ties listed).

## 1. Selection maps

Codes: a = epistemic term alone selects the Cue (flat preferences, full EFE); t = the preference term ties and the epistemic term breaks the tie toward the Cue; b = the preference term alone selects an arm and the epistemic term flips the choice to the Cue; c = the preference term alone and the full EFE both select the Cue; d = the full EFE does not select the Cue (tie or arm). The flat profile can only show a or d.

### paper (+2/-4), horizon 1

| r \ c | 0.7 | 0.9 | 1.0 |
|---|---|---|---|
| 0.5 | c | c | c |
| 0.6 | c | c | c |
| 0.7 | c | c | c |
| 0.8 | c | c | c |
| 0.9 | c | c | c |
| 1.0 | c | c | c |

### paper (+2/-4), horizon 2

| r \ c | 0.7 | 0.9 | 1.0 |
|---|---|---|---|
| 0.5 | c | c | c |
| 0.6 | c | c | c |
| 0.7 | c | c | c |
| 0.8 | c | c | c |
| 0.9 | c | c | c |
| 1.0 | c | c | c |

### mild (+1/-1), horizon 1

| r \ c | 0.7 | 0.9 | 1.0 |
|---|---|---|---|
| 0.5 | t | t | t |
| 0.6 | t | t | t |
| 0.7 | d | t | t |
| 0.8 | d | t | t |
| 0.9 | d | d | t |
| 1.0 | d | d | d |

### mild (+1/-1), horizon 2

| r \ c | 0.7 | 0.9 | 1.0 |
|---|---|---|---|
| 0.5 | t | t | t |
| 0.6 | c | c | c |
| 0.7 | c | c | c |
| 0.8 | c | c | c |
| 0.9 | c | c | c |
| 1.0 | c | c | c |

### flat (0/0), horizon 1

| r \ c | 0.7 | 0.9 | 1.0 |
|---|---|---|---|
| 0.5 | a | a | a |
| 0.6 | a | a | a |
| 0.7 | d | a | a |
| 0.8 | d | a | a |
| 0.9 | d | d | a |
| 1.0 | d | d | d |

### flat (0/0), horizon 2

| r \ c | 0.7 | 0.9 | 1.0 |
|---|---|---|---|
| 0.5 | a | a | a |
| 0.6 | a | a | a |
| 0.7 | d | a | a |
| 0.8 | d | a | a |
| 0.9 | d | d | a |
| 1.0 | d | d | d |

### mild gambler (+2/-1), horizon 1

| r \ c | 0.7 | 0.9 | 1.0 |
|---|---|---|---|
| 0.5 | d | d | b |
| 0.6 | d | d | b |
| 0.7 | d | d | b |
| 0.8 | d | d | b |
| 0.9 | d | d | d |
| 1.0 | d | d | d |

### mild gambler (+2/-1), horizon 2

| r \ c | 0.7 | 0.9 | 1.0 |
|---|---|---|---|
| 0.5 | d | d | b |
| 0.6 | d | b | b |
| 0.7 | d | b | c |
| 0.8 | d | c | c |
| 0.9 | d | c | c |
| 1.0 | c | c | c |

### gambler (+4/-2), horizon 1

| r \ c | 0.7 | 0.9 | 1.0 |
|---|---|---|---|
| 0.5 | d | d | d |
| 0.6 | d | d | d |
| 0.7 | d | d | d |
| 0.8 | d | d | d |
| 0.9 | d | d | d |
| 1.0 | d | d | d |

### gambler (+4/-2), horizon 2

| r \ c | 0.7 | 0.9 | 1.0 |
|---|---|---|---|
| 0.5 | d | d | d |
| 0.6 | d | d | b |
| 0.7 | d | b | c |
| 0.8 | d | c | c |
| 0.9 | d | c | c |
| 1.0 | c | c | c |

## 2. Per-action terms at selected grid points

### flat (0/0), r = 0.5, c = 1.0

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.609 | 0.000 | 1.609 | 1.609 | 1.609 | 0.000 | 3.219 | 3.219 |
| Right | 1.609 | 0.000 | 1.609 | 1.609 | 1.609 | 0.000 | 3.219 | 3.219 |
| Cue | 1.609 | 0.693 | 0.916 | 1.609 | 1.609 | 0.000 | 2.526 | 3.219 |

Selected h1 full: ['Cue']; h1 pref only: ['Left', 'Right', 'Cue']; h2 full: ['Cue']; h2 pref only: ['Left', 'Right', 'Cue'].

### flat (0/0), r = 0.7, c = 0.9

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.609 | 0.082 | 1.527 | 1.609 | 1.609 | 0.069 | 3.067 | 3.219 |
| Right | 1.609 | 0.082 | 1.527 | 1.609 | 1.609 | 0.069 | 3.067 | 3.219 |
| Cue | 1.609 | 0.368 | 1.241 | 1.609 | 1.609 | 0.119 | 2.732 | 3.219 |

Selected h1 full: ['Cue']; h1 pref only: ['Left', 'Right', 'Cue']; h2 full: ['Cue']; h2 pref only: ['Left', 'Right', 'Cue'].

### flat (0/0), r = 0.8, c = 1.0

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.609 | 0.193 | 1.417 | 1.609 | 1.609 | 0.126 | 2.900 | 3.219 |
| Right | 1.609 | 0.193 | 1.417 | 1.609 | 1.609 | 0.126 | 2.900 | 3.219 |
| Cue | 1.609 | 0.693 | 0.916 | 1.609 | 1.609 | 0.000 | 2.526 | 3.219 |

Selected h1 full: ['Cue']; h1 pref only: ['Left', 'Right', 'Cue']; h2 full: ['Cue']; h2 pref only: ['Left', 'Right', 'Cue'].

### flat (0/0), r = 0.9, c = 0.7

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.609 | 0.368 | 1.241 | 1.609 | 1.609 | 0.146 | 2.705 | 3.219 |
| Right | 1.609 | 0.368 | 1.241 | 1.609 | 1.609 | 0.146 | 2.705 | 3.219 |
| Cue | 1.609 | 0.082 | 1.527 | 1.609 | 1.609 | 0.314 | 2.823 | 3.219 |

Selected h1 full: ['Left', 'Right']; h1 pref only: ['Left', 'Right', 'Cue']; h2 full: ['Left', 'Right']; h2 pref only: ['Left', 'Right', 'Cue'].

### flat (0/0), r = 1.0, c = 1.0

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.609 | 0.693 | 0.916 | 1.609 | 1.609 | 0.000 | 2.526 | 3.219 |
| Right | 1.609 | 0.693 | 0.916 | 1.609 | 1.609 | 0.000 | 2.526 | 3.219 |
| Cue | 1.609 | 0.693 | 0.916 | 1.609 | 1.609 | 0.000 | 2.526 | 3.219 |

Selected h1 full: ['Left', 'Right', 'Cue']; h1 pref only: ['Left', 'Right', 'Cue']; h2 full: ['Left', 'Right', 'Cue']; h2 pref only: ['Left', 'Right', 'Cue'].

### mild (+1/-1), r = 0.5, c = 1.0

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.806 | 0.000 | 1.806 | 1.806 | 1.806 | 0.000 | 3.612 | 3.612 |
| Right | 1.806 | 0.000 | 1.806 | 1.806 | 1.806 | 0.000 | 3.612 | 3.612 |
| Cue | 1.806 | 0.693 | 1.113 | 1.806 | 1.806 | 0.000 | 2.919 | 3.612 |

Selected h1 full: ['Cue']; h1 pref only: ['Left', 'Right', 'Cue']; h2 full: ['Cue']; h2 pref only: ['Left', 'Right', 'Cue'].

### mild (+1/-1), r = 0.7, c = 0.9

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.806 | 0.082 | 1.724 | 1.806 | 1.806 | 0.069 | 3.460 | 3.612 |
| Right | 1.806 | 0.082 | 1.724 | 1.806 | 1.806 | 0.069 | 3.460 | 3.612 |
| Cue | 1.806 | 0.368 | 1.438 | 1.806 | 1.498 | 0.034 | 2.901 | 3.294 |

Selected h1 full: ['Cue']; h1 pref only: ['Left', 'Right', 'Cue']; h2 full: ['Cue']; h2 pref only: ['Cue'].

### mild (+1/-1), r = 0.8, c = 1.0

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.806 | 0.193 | 1.613 | 1.806 | 1.806 | 0.126 | 3.293 | 3.612 |
| Right | 1.806 | 0.193 | 1.613 | 1.806 | 1.806 | 0.126 | 3.293 | 3.612 |
| Cue | 1.806 | 0.693 | 1.113 | 1.806 | 1.206 | 0.000 | 2.319 | 3.012 |

Selected h1 full: ['Cue']; h1 pref only: ['Left', 'Right', 'Cue']; h2 full: ['Cue']; h2 pref only: ['Cue'].

### mild (+1/-1), r = 0.9, c = 0.7

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.806 | 0.368 | 1.438 | 1.806 | 1.806 | 0.146 | 3.098 | 3.612 |
| Right | 1.806 | 0.368 | 1.438 | 1.806 | 1.806 | 0.146 | 3.098 | 3.612 |
| Cue | 1.806 | 0.082 | 1.724 | 1.806 | 1.486 | 0.316 | 2.894 | 3.294 |

Selected h1 full: ['Left', 'Right']; h1 pref only: ['Left', 'Right', 'Cue']; h2 full: ['Cue']; h2 pref only: ['Cue'].

### mild (+1/-1), r = 1.0, c = 1.0

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.806 | 0.693 | 1.113 | 1.806 | 1.806 | 0.000 | 2.919 | 3.612 |
| Right | 1.806 | 0.693 | 1.113 | 1.806 | 1.806 | 0.000 | 2.919 | 3.612 |
| Cue | 1.806 | 0.693 | 1.113 | 1.806 | 0.806 | 0.000 | 1.919 | 2.612 |

Selected h1 full: ['Left', 'Right', 'Cue']; h1 pref only: ['Left', 'Right', 'Cue']; h2 full: ['Cue']; h2 pref only: ['Cue'].

### paper (+2/-4), r = 0.5, c = 1.0

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 3.343 | 0.000 | 3.343 | 3.343 | 3.343 | 0.000 | 6.685 | 6.685 |
| Right | 3.343 | 0.000 | 3.343 | 3.343 | 3.343 | 0.000 | 6.685 | 6.685 |
| Cue | 2.343 | 0.693 | 1.649 | 2.343 | 2.343 | 0.000 | 3.992 | 4.685 |

Selected h1 full: ['Cue']; h1 pref only: ['Cue']; h2 full: ['Cue']; h2 pref only: ['Cue'].

### paper (+2/-4), r = 0.7, c = 0.9

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 3.343 | 0.082 | 3.260 | 3.343 | 3.343 | 0.069 | 6.533 | 6.685 |
| Right | 3.343 | 0.082 | 3.260 | 3.343 | 3.343 | 0.069 | 6.533 | 6.685 |
| Cue | 2.343 | 0.368 | 1.974 | 2.343 | 2.346 | 0.137 | 4.183 | 4.699 |

Selected h1 full: ['Cue']; h1 pref only: ['Cue']; h2 full: ['Cue']; h2 pref only: ['Cue'].

### paper (+2/-4), r = 0.8, c = 1.0

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 3.343 | 0.193 | 3.150 | 3.343 | 3.343 | 0.126 | 6.366 | 6.685 |
| Right | 3.343 | 0.193 | 3.150 | 3.343 | 3.343 | 0.126 | 6.366 | 6.685 |
| Cue | 2.343 | 0.693 | 1.649 | 2.343 | 1.543 | 0.000 | 3.192 | 3.885 |

Selected h1 full: ['Cue']; h1 pref only: ['Cue']; h2 full: ['Cue']; h2 pref only: ['Cue'].

### paper (+2/-4), r = 0.9, c = 0.7

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 3.343 | 0.368 | 2.974 | 3.343 | 3.343 | 0.146 | 6.171 | 6.685 |
| Right | 3.343 | 0.368 | 2.974 | 3.343 | 3.343 | 0.146 | 6.171 | 6.685 |
| Cue | 2.343 | 0.082 | 2.260 | 2.343 | 2.381 | 0.307 | 4.334 | 4.699 |

Selected h1 full: ['Cue']; h1 pref only: ['Cue']; h2 full: ['Cue']; h2 pref only: ['Cue'].

### paper (+2/-4), r = 1.0, c = 1.0

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 3.343 | 0.693 | 2.649 | 3.343 | 3.343 | 0.000 | 5.992 | 6.685 |
| Right | 3.343 | 0.693 | 2.649 | 3.343 | 3.343 | 0.000 | 5.992 | 6.685 |
| Cue | 2.343 | 0.693 | 1.649 | 2.343 | 0.343 | 0.000 | 1.992 | 2.685 |

Selected h1 full: ['Cue']; h1 pref only: ['Cue']; h2 full: ['Cue']; h2 pref only: ['Cue'].

### mild gambler (+2/-1), r = 0.5, c = 1.0

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.876 | 0.000 | 1.876 | 1.876 | 1.876 | 0.000 | 3.751 | 3.751 |
| Right | 1.876 | 0.000 | 1.876 | 1.876 | 1.876 | 0.000 | 3.751 | 3.751 |
| Cue | 2.376 | 0.693 | 1.682 | 2.376 | 1.876 | 0.000 | 3.558 | 4.251 |

Selected h1 full: ['Cue']; h1 pref only: ['Left', 'Right']; h2 full: ['Cue']; h2 pref only: ['Left', 'Right'].

### mild gambler (+2/-1), r = 0.7, c = 0.9

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.876 | 0.082 | 1.793 | 1.876 | 1.876 | 0.069 | 3.599 | 3.751 |
| Right | 1.876 | 0.082 | 1.793 | 1.876 | 1.876 | 0.069 | 3.599 | 3.751 |
| Cue | 2.376 | 0.368 | 2.007 | 2.376 | 1.396 | 0.030 | 3.373 | 3.771 |

Selected h1 full: ['Left', 'Right']; h1 pref only: ['Left', 'Right']; h2 full: ['Cue']; h2 pref only: ['Left', 'Right'].

### mild gambler (+2/-1), r = 0.8, c = 1.0

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.876 | 0.193 | 1.683 | 1.876 | 1.876 | 0.126 | 3.432 | 3.751 |
| Right | 1.876 | 0.193 | 1.683 | 1.876 | 1.876 | 0.126 | 3.432 | 3.751 |
| Cue | 2.376 | 0.693 | 1.682 | 2.376 | 0.976 | 0.000 | 2.658 | 3.351 |

Selected h1 full: ['Cue']; h1 pref only: ['Left', 'Right']; h2 full: ['Cue']; h2 pref only: ['Cue'].

### mild gambler (+2/-1), r = 0.9, c = 0.7

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.876 | 0.368 | 1.507 | 1.876 | 1.876 | 0.146 | 3.237 | 3.751 |
| Right | 1.876 | 0.368 | 1.507 | 1.876 | 1.876 | 0.146 | 3.237 | 3.751 |
| Cue | 2.376 | 0.082 | 2.293 | 2.376 | 1.396 | 0.316 | 3.373 | 3.771 |

Selected h1 full: ['Left', 'Right']; h1 pref only: ['Left', 'Right']; h2 full: ['Left', 'Right']; h2 pref only: ['Left', 'Right'].

### mild gambler (+2/-1), r = 1.0, c = 1.0

| action | h1 risk | h1 epistemic | h1 G | h1 G (pref only) | h2 E[risk t2] | h2 E[epist t2] | h2 G | h2 G (pref only) |
|---|---|---|---|---|---|---|---|---|
| Left | 1.876 | 0.693 | 1.182 | 1.876 | 1.876 | 0.000 | 3.058 | 3.751 |
| Right | 1.876 | 0.693 | 1.182 | 1.876 | 1.876 | 0.000 | 3.058 | 3.751 |
| Cue | 2.376 | 0.693 | 1.682 | 2.376 | 0.376 | 0.000 | 2.058 | 2.751 |

Selected h1 full: ['Left', 'Right']; h1 pref only: ['Left', 'Right']; h2 full: ['Cue']; h2 pref only: ['Cue'].

Closed form at horizon 1: epistemic(arm) = ln 2 - H(r), epistemic(Cue) = ln 2 - H(c), with H the binary entropy in nats, because under a uniform context prior each channel is binary symmetric. risk(arm) - risk(Cue) = (C_SeeLeft + C_SeeRight)/2 - (C_Cheese + C_Shock)/2 for every r and c, since the predicted outcome is a fair coin either way. So at horizon 1 the epistemic term favours the Cue exactly when c > r, by H(r) - H(c) nats, and the preference term is independent of the reliabilities.

## 3. Empowerment after each first action (bits), Proposition 1 decomposition

Subjective potential empowerment E0 at the choice point under the uniform context belief, then for each first action the expected empowerment after acting, split into the information term (belief updated, interface kept) and the interface term (move to the new interface at the updated belief). Cue returns to the choice point; arms are absorbing.

| r | c | E0 | action | E after | information term | interface term | net |
|---|---|---|---|---|---|---|---|
| 0.6 | 0.9 | 1.000 | Left | 0.000 | +0.001 | -1.001 | -1.000 |
| 0.6 | 0.9 | 1.000 | Right | 0.000 | +0.001 | -1.001 | -1.000 |
| 0.6 | 0.9 | 1.000 | Cue | 1.009 | +0.009 | +0.000 | +0.009 |
| 0.6 | 1.0 | 1.000 | Left | 0.000 | +0.001 | -1.001 | -1.000 |
| 0.6 | 1.0 | 1.000 | Right | 0.000 | +0.001 | -1.001 | -1.000 |
| 0.6 | 1.0 | 1.000 | Cue | 1.015 | +0.015 | +0.000 | +0.015 |
| 0.8 | 0.9 | 1.000 | Left | 0.000 | +0.049 | -1.049 | -1.000 |
| 0.8 | 0.9 | 1.000 | Right | 0.000 | +0.049 | -1.049 | -1.000 |
| 0.8 | 0.9 | 1.000 | Cue | 1.089 | +0.089 | +0.000 | +0.089 |
| 0.8 | 1.0 | 1.000 | Left | 0.000 | +0.049 | -1.049 | -1.000 |
| 0.8 | 1.0 | 1.000 | Right | 0.000 | +0.049 | -1.049 | -1.000 |
| 0.8 | 1.0 | 1.000 | Cue | 1.146 | +0.146 | +0.000 | +0.146 |
| 1.0 | 0.9 | 1.000 | Left | 0.000 | +0.585 | -1.585 | -1.000 |
| 1.0 | 0.9 | 1.000 | Right | 0.000 | +0.585 | -1.585 | -1.000 |
| 1.0 | 0.9 | 1.000 | Cue | 1.290 | +0.290 | +0.000 | +0.290 |
| 1.0 | 1.0 | 1.000 | Left | 0.000 | +0.585 | -1.585 | -1.000 |
| 1.0 | 1.0 | 1.000 | Right | 0.000 | +0.585 | -1.585 | -1.000 |
| 1.0 | 1.0 | 1.000 | Cue | 1.585 | +0.585 | +0.000 | +0.585 |

Decomposition check over the 11 x 11 grid r, c in [0.5, 1]: minimum information term = +0.00e+00 (non-negative required); Cue interface term identically zero: yes. Arms: interface term = -(empowerment at the choice point under the updated belief), net change negative whenever the arm is informative.

## 4. Silent-arm control: n-step empowerment

Arms show Null on entry and the outcome only on the following step (delayed information). Cue reliability c = 1, arm reliability r = 1. n-step empowerment = capacity from open-loop action sequences of length n to the observation sequence, in bits.

| situation | belief over context | 1-step | 2-step |
|---|---|---|---|
| choice point, before acting | [0.5, 0.5] | 1.000 | 1.585 |
| after Cue (context resolved), at the choice point | [1.0, 0.0] | 1.000 | 2.000 |
| after Left (silent antechamber), context unresolved | [0.5, 0.5] | 0.000 | 0.000 |
| after Left (silent antechamber), context resolved | [1.0, 0.0] | 0.000 | -0.000 |

Standard (non-silent) maze for comparison, r = c = 1:

| situation | 1-step | 2-step |
|---|---|---|
| choice point, unresolved | 1.000 | 2.000 |
| after Cue, resolved | 1.585 | 2.322 |
| inside an arm | 0.000 | -0.000 |

Figure written: figures\fig_noisy_arm.png

(pymdp cross-check skipped: ModuleNotFoundError("No module named 'pymdp'"))

