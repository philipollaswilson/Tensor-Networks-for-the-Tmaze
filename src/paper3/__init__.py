"""Paper III — agent-specific agency phenotyping with tensor networks.

Spine workstreams (see PAPER3.md at the repo root):

  1. agents.py            per-agent rollout generators        (closes gap 1)
  2. agency_criteria.py   the full three criteria, not just    (closes gap 2)
                          empowerment
  3. blind_validation.py  pre-registered predictive test       (closes gap 3)
  4. persistent_tmaze.py  persistent-latent data generator     (closes gap 4,
                          Philip's point)

  phenotype.py            assembles the full profile per agent and runs the
                          killer experiment: discriminate info-seeker vs
                          reward-gambler vs random, then blind-classify.

Everything under src/ from Paper II transfers unchanged: structure_recovery.py,
factorization_test.py, gauge_fix_states.py, model_selection.py,
make_extension_figures.py. The new work is the three pieces above plus the
per-agent training loop.
"""
