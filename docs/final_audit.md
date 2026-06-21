# Final Audit

Last update: 2026-06-21 23:35:31 +08:00

Paper: `85_minimum_intervention_human_correction`

Terminal decision: `KILL_ARCHIVE`

ICLR main ready: no

## v5 Expanded Evidence

- Main rollout rows: 199,680.
- Dataset summary rows: 15,360.
- Main seed-metric rows: 1,040.
- Main aggregate metric rows: 1,352.
- Main pairwise rows: 768.
- Hard aggregate seed rows: 130.
- Hard aggregate metric rows: 169.
- Hard aggregate pairwise rows: 96.
- Ablation rollout rows: 33,600.
- Stress raw rows: 302,400.
- Fixed-risk raw rows: 69,120.
- Negative cases: 24.
- Manuscript: 28 pages.
- Canonical PDF: `C:/Users/wangz/Downloads/85.pdf`.
- PDF SHA256: `3A689EE96ED760908FCE9678AA3A3667B9C768F2218D06272EE382EDDD121406`.
- Desktop PDF: absent.

## Gate Findings

- Best success reference: `robust_mpc_correction`.
- Best efficiency reference: `robust_mpc_correction`.
- v5 hard-aggregate success: `0.48958 +/- 0.01057`.
- robust MPC hard-aggregate success: `0.59010 +/- 0.01426`.
- v5 hard-aggregate correction efficiency: `0.69909`.
- robust MPC hard-aggregate correction efficiency: `0.88048`.
- Paired success lower95: `-0.11788`.
- Paired efficiency lower95: `-0.20944`.
- Mechanism gate: failed because ablations beat the full v5 method.
- Stress gate: passed; v5 was not Pareto-dominated at maximum combined stress.
- Fixed-risk gate: failed because budget `0.05` coverage is zero.
- Scope gate: failed because there is no real robot or accepted high-fidelity benchmark evidence.

## Submission Decision

Paper 85 is not ICLR-main ready. It should remain archived unless future work adds real human-in-the-loop robot correction traces or accepted high-fidelity external benchmarks and a learned method that decisively beats robust MPC, active querying, residual correction, and preference-learning baselines under fixed-risk deployment constraints.
