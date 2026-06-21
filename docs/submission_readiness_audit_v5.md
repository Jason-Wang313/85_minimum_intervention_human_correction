# Paper 85 Submission Readiness Audit v5

Last update: 2026-06-21 23:35:31 +08:00

Terminal decision: `KILL_ARCHIVE`

ICLR main ready: no

## Evidence Added

- Frozen plan-first protocol.
- New method under test: `minimum_intervention_boundary_learner_v5`.
- Main methods: 13.
- Seeds: 10.
- Tasks: 6.
- Splits: 8.
- Main rollout rows: 199,680.
- Dataset summary rows: 15,360.
- Ablation rollout rows: 33,600.
- Stress raw rows: 302,400.
- Fixed-risk raw rows: 69,120.
- Negative cases: 24.
- Manuscript pages: 28.

## Main Hard-Aggregate Gate

- Proposed v5 success: `0.48958 +/- 0.01057`.
- Best success reference: `robust_mpc_correction` at `0.59010 +/- 0.01426`.
- Proposed v5 correction efficiency: `0.69909`.
- Best efficiency reference: `robust_mpc_correction` at `0.88048`.
- Paired success lower95: `-0.11788`.
- Paired efficiency lower95: `-0.20944`.

Gate result: failed.

## Mechanism Gate

- Full v5 robust utility: `0.67199`.
- `minus_human_effort_cost` robust utility: `0.77648`.
- `minus_minimum_norm_objective` robust utility: `0.76607`.
- `minus_safety_override` robust utility: `0.75679`.

Gate result: failed.

## Stress Gate

At maximum combined stress, v5 was not Pareto-dominated on success, efficiency, damage, and unsafe override.

Gate result: passed.

## Fixed-Risk Gate

At risk budget `0.05`, accepted coverage was zero on both `dynamics_mismatch` and `combined_hard_shift`.

Gate result: failed.

## Scope Gate

No real robot, real human-in-the-loop, accepted high-fidelity simulator, or external benchmark evidence is available.

Gate result: failed.

## Final Decision

Archive. The expanded audit is stronger and more transparent, but the paper is not submission-ready.
