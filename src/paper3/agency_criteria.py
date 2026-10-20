"""The full three agency criteria on a recovered model (Paper III, workstream 2).

Paper II reported only empowerment (controllability). This is the direct answer
to "empowerment isn't agency": operationalise all three Paper I criteria on the
recovered generative model, with empowerment kept as the controllability facet of
a fuller profile.

  intentionality -- inverse-infer the preference vector C from behaviour under
      the recovered model; goal-directedness = how PEAKED those preferences are
      (a flat C means no goals). Fitted jointly with precision by maximum
      likelihood of the agent's observed first actions.
  rationality    -- the EFE-optimality gap (regret) of the behaviour under the
      recovered model and the inferred C, normalised to [0,1]. A rational agent
      minimises expected free energy; an impulsive or random one does not.
  explainability -- fidelity of the recovered model to the agent's own behaviour
      (1 - fit L1) and its description length (MPS parameter count).

empowerment stays as the controllability facet (Blahut-Arimoto), carried when a
recovered joint is available.

The EFE machinery is reused from agents.ActiveInferenceAgent, so the criteria
score behaviour with the SAME expected-free-energy the agents plan by -- there is
one definition of value in the paper, not two.

Three honest findings of this operationalisation, worth stating in the paper:
  * rationality is measured against each agent's OWN inferred C, so a CONSISTENT
    agent scores high regardless of what it wants. The reward-gambler is not
    "irrational" -- it has shallow, risk-neutral preferences under which going
    straight to an arm really is near-optimal. What separates the habitual agent
    is its inconsistency (chance-level rationality).
  * INTENTIONALITY IS UNDER-IDENTIFIED HERE. Cue-seeking is explained about
    equally well by a peaked reward preference OR by pure epistemic drive under a
    flat C, so behaviour alone cannot attribute the info-seeker's cue visits to
    reward goals: the prior-relative goal-directedness is ~0 for it. This is a
    genuine identifiability result, not a coding artifact. Pinning it down needs
    the epistemic weight / planning horizon inferred as a separate trait (a clean
    Paper III extension). We therefore do NOT discriminate agents by
    intentionality; we use rationality plus recovered state-visitation coverage
    (visitation_signature), which are stable and identifiable.
  * inferring C under a fixed horizon-2 observer also conflates MYOPIA with WEAK
    PREFERENCE: the gambler's short horizon is read as a small |C|.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from . import agents
from . import generative_model as gm

_EPS = 1e-12


@dataclass
class AgencyProfile:
    """The full phenotype for one agent, read off its behaviour + recovered model."""

    agent: str
    intentionality: float          # goal-directedness (peakedness of inferred C), [0,1]
    rationality: float             # 1 - normalised EFE regret, [0,1]
    explainability: float          # model fidelity to the agent's behaviour, [0,1]
    recovered_C: tuple             # inverse-inferred (none, cheese, shock) preference
    empowerment: float | None      # controllability facet, if a joint was given

    def vector(self) -> np.ndarray:
        """The 3-criteria coordinate used for agent discrimination (step 5)."""
        return np.array([self.intentionality, self.rationality, self.explainability])


def behavioural_first_action(rollouts) -> np.ndarray:
    """Empirical P(first real action | start) from an agent's rollouts."""
    counts = np.zeros(gm.N_ACT)
    for actions, _obs in rollouts:
        counts[int(actions[1])] += 1.0     # actions[0] is the null a1
    return counts / counts.sum()


def _reference_agent() -> "agents.ActiveInferenceAgent":
    """A rational-observer planner (horizon 2) used to score behaviour. Its C and
    gamma are overridden per call; only its A/B and EFE code are used."""
    spec = agents.AgentSpec("reference", C_reward=(0.0, 0.0, 0.0), gamma=1.0, horizon=2)
    return agents.ActiveInferenceAgent(spec)


def infer_preferences(rollouts, c_grid=None, gamma_grid=None):
    """Infer preferences (C_reward, gamma) explaining the observed first actions
    under the reference planner.

    Returns (C_map, gamma_map, loglik_map, intentionality_post) where the first
    three are the maximum-likelihood point and the last is the POSTERIOR-EXPECTED
    goal-directedness. The posterior expectation is used for the intentionality
    criterion because the MLE argmax is unstable: cue-seeking is explained about
    equally well by a peaked reward preference OR by pure epistemic drive under a
    flat C, so the argmax flips between them across seeds. Averaging peakedness
    over the likelihood is stable and honestly reflects that ambiguity.
    """
    ref = _reference_agent()
    phat = behavioural_first_action(rollouts)
    n = sum(1 for _ in rollouts)
    c_grid = c_grid if c_grid is not None else np.linspace(0.0, 6.0, 13)
    gamma_grid = gamma_grid if gamma_grid is not None else np.array([0.0, 0.5, 1, 2, 4, 8, 16])
    lls, Cs, peaks = [], [], []
    best = (-np.inf, (0.0, 0.0, 0.0), 0.0)
    for c_ch in c_grid:
        for c_sh in -c_grid:                # shock preference in [-6, 0]
            C = (0.0, float(c_ch), float(c_sh))
            for g in gamma_grid:
                qpi = ref.first_action_dist(gm.CENTER, steps_left=2, gamma=float(g), C_rew=C)
                ll = n * float(np.sum(phat * np.log(np.clip(qpi, _EPS, 1.0))))
                lls.append(ll); Cs.append(C); peaks.append(_intentionality(C))
                if ll > best[0]:
                    best = (ll, C, float(g))
    peaks = np.asarray(peaks)
    w = agents._softmax(np.asarray(lls))    # posterior over the grid, uniform prior
    prior_peak = float(peaks.mean())        # what peakedness we'd expect a priori
    post_peak = float(np.sum(w * peaks))    # peakedness the behaviour actually implies
    # prior-relative goal-directedness: how much the behaviour shifts C toward
    # peaked beyond the prior. ~0 when behaviour underdetermines C (random agent
    # reverts to the prior); positive when behaviour genuinely implies goals.
    denom = max(peaks.max() - prior_peak, _EPS)
    intentionality = float(np.clip((post_peak - prior_peak) / denom, 0.0, 1.0))
    ll, C, g = best
    return C, g, ll, intentionality


def _efe_by_policy(C_reward) -> np.ndarray:
    """EFE G(a1,a2) for every 2-step policy from the start belief, under C."""
    ref = _reference_agent()
    ref.C = gm.build_C(C_reward)
    qK0 = gm.build_D()[1]
    G = np.zeros((gm.N_ACT, gm.N_ACT))
    for a1, a2 in product(range(gm.N_ACT), repeat=2):
        G[a1, a2] = ref._efe(qK0, gm.CENTER, (a1, a2))
    return G


def efe_regret(rollouts, C_reward) -> float:
    """Normalised EFE-optimality gap of the behaviour in [0,1] (0 = optimal).

    Uses the best continuation a2 for each first action (isolating first-choice
    quality), weighted by the empirical P(a1). Normalised by the spread between
    the best and worst first choice so it is comparable across agents.
    """
    G = _efe_by_policy(C_reward)
    best_cont = G.min(axis=1)               # min over a2 for each a1
    phat = behavioural_first_action(rollouts)
    g_behaviour = float(np.sum(phat * best_cont))
    g_opt = float(best_cont.min())
    g_worst = float(best_cont.max())
    if g_worst - g_opt < _EPS:
        return 0.0
    return (g_behaviour - g_opt) / (g_worst - g_opt)


def _intentionality(C_reward) -> float:
    """Goal-directedness = how far the softmaxed preferences are from uniform,
    normalised so a flat C scores 0 and a maximally peaked one scores 1."""
    p = agents._softmax(np.asarray(C_reward, float))
    h = -np.sum(p * np.log(np.clip(p, _EPS, 1.0)))
    return float(1.0 - h / np.log(len(p)))


def visitation_signature(rollouts) -> dict:
    """RAW BEHAVIOURAL statistics: how often the agent's FIRST action was the cue
    or an arm, counted straight off the rollouts.

    HONESTY NOTE (this docstring previously claimed these were "marginals of the
    recovered model" -- they are not). Nothing here touches a trained MPS. These
    are the agents' defining behaviour, so classifying agents with them is
    circular: a bare first-action histogram, with no criteria, no EFE and no
    tensor network, separates the roster equally well.

    For the genuine recovered-model version, read the action marginal off a
    trained MPS's contracted joint: phenotype.coverage_features(joint). For a
    phenotype that raw behaviour cannot supply at all, see structure_phenotype.
    """
    cue = arm_first = 0
    n = 0
    for actions, _obs in rollouts:
        n += 1
        if int(actions[1]) == gm.CUE:
            cue += 1
        if int(actions[1]) in gm.ARMS:
            arm_first += 1
    return {"cue_visit": cue / n, "arm_first": arm_first / n}


def profile_agent(agent_name, rollouts, fit_L1=None, empowerment=None) -> AgencyProfile:
    """Assemble the full AgencyProfile from an agent's rollouts (and optionally
    its recovered-model fit L1 and empowerment facet)."""
    C, gamma, _ll, intentionality = infer_preferences(rollouts)
    rationality = 1.0 - efe_regret(rollouts, C)
    explainability = (1.0 - fit_L1) if fit_L1 is not None else float("nan")
    return AgencyProfile(agent_name, intentionality, rationality, explainability,
                         tuple(round(x, 2) for x in C), empowerment)


if __name__ == "__main__":
    # verification: the three criteria should separate the roster by character
    print(f"{'agent':16s} {'intent':>7s} {'ration':>7s} {'cue_vis':>8s} {'arm_1st':>8s} "
          f"{'C(cheese,shock)':>18s}")
    for spec in agents.ROSTER:
        rollouts, _ = agents.rollout(spec, n_episodes=3000, seed=0, verify=False)
        C, gamma, _, _ = infer_preferences(rollouts)
        prof = profile_agent(spec.name, rollouts)
        vis = visitation_signature(rollouts)
        print(f"{spec.name:16s} {prof.intentionality:7.2f} {prof.rationality:7.2f} "
              f"{vis['cue_visit']:8.2f} {vis['arm_first']:8.2f} {str((C[1], C[2])):>18s}")
