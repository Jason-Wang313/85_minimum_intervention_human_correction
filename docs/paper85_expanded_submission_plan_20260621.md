# Paper 85 Expanded Submission-Readiness Plan

Date: 2026-06-21

Paper: `85_minimum_intervention_human_correction`

Target venue posture: ICLR-main hostile-review readiness audit.

Execution constraint: CPU-only, RAM-light, deterministic, no visible Desktop PDFs, canonical numbered PDF only at `C:/Users/wangz/Downloads/85.pdf`.

## Objective

Rebuild Paper 85 from a 5-page v4 archive memo into a 25+ page ICLR-style evidence package without padding. The rebuild must add real methodological substance, a stricter empirical protocol, theory, stronger baselines, fixed-risk deployment tests, stress tests, negative cases, clickable bright-box citations, reproducible scripts, and honest terminal decision logic.

The central hypothesis remains:

> A robot should learn from the smallest human correction that changes the physical outcome, because such corrections identify a local decision boundary with less human burden, less overcorrection damage, and better transfer under correction shift than full demonstrations or generic residual corrections.

The v5 method under test will be `minimum_intervention_boundary_learner_v5`: a deterministic CPU-light proxy for a minimum-intervention boundary learner with counterfactual boundary projection, intent-preserving tangent regularization, robust query throttling, safety override, and calibrated risk scoring.

## Non-Negotiable Honesty Rules

- Do not claim real robot validation.
- Do not claim high-fidelity simulation.
- Do not claim learned neural SOTA.
- Do not claim ICLR-main readiness unless all frozen gates pass and the evidence limitation is explicitly discussed.
- Report all predefined results, including failures.
- Treat synthetic-only evidence as a fatal deployment blocker for ICLR main unless the empirical margins are exceptional and the paper is framed as a diagnostic benchmark contribution.

## Frozen Experimental Design

### Seeds, Tasks, Splits, and Episodes

- Seeds: `0..9` (10 seeds).
- Tasks: 6 tasks:
  - `peg_insertion`
  - `drawer_alignment`
  - `cloth_corner_place`
  - `cup_handoff_pose`
  - `cable_hook_alignment`
  - `tool_handover_alignment`
- Splits: 8 splits:
  - `nominal_correction`
  - `overcorrection_bias`
  - `delayed_feedback`
  - `ambiguous_intent`
  - `sparse_corrections`
  - `adversarial_helpfulness`
  - `dynamics_mismatch`
  - `combined_hard_shift`
- Episodes per task/split/seed: 32.
- Main methods: 13 methods:
  - `no_human_baseline`
  - `full_demo_imitation`
  - `dagger_full_correction`
  - `residual_correction_learner`
  - `preference_only_ranker`
  - `uncertainty_query_policy`
  - `active_entropy_query_policy`
  - `safety_filtered_residual`
  - `robust_mpc_correction`
  - `inverse_rl_correction_proxy`
  - `minimum_intervention_learner_v4`
  - `minimum_intervention_boundary_learner_v5`
  - `oracle_minimal_correction`

Expected main rollout rows: `10 * 6 * 8 * 32 * 13 = 199680`.

Expected scene summary rows: `10 * 6 * 8 * 32 = 15360`.

### Metrics

Primary metrics:

- `task_success`
- `correction_efficiency`
- `damage`
- `intervention_magnitude`
- `boundary_error`
- `intent_preservation`

Secondary and deployment metrics:

- `query_rate`
- `human_time`
- `unsafe_override`
- `calibration_error`
- `regret_to_oracle`
- `intervention_sparsity`
- `robust_utility`

### Hard Aggregate

The hard aggregate is predeclared as:

- `adversarial_helpfulness`
- `dynamics_mismatch`
- `combined_hard_shift`

Hard aggregate decision metrics are computed over seed means from these three splits, not selected after seeing results.

### Ablations

Ablations are evaluated on `dynamics_mismatch` and `combined_hard_shift` with 28 episodes per task/seed.

Ten ablations:

- `full_minimum_intervention_boundary_learner_v5`
- `minus_minimum_norm_objective`
- `minus_counterfactual_boundary`
- `minus_intent_preservation`
- `minus_human_effort_cost`
- `minus_safety_override`
- `minus_query_throttling`
- `minus_calibration`
- `all_corrections_imitation`
- `preference_only_objective`

Expected ablation rollout rows: `10 * 6 * 2 * 28 * 10 = 33600`.

### Stress Sweeps

Stress sweeps are evaluated on `combined_hard_shift`, 20 episodes per task/seed, 6 axes, 6 levels, and 7 methods:

- Stress axes:
  - `human_noise`
  - `correction_delay`
  - `overcorrection_bias`
  - `intent_ambiguity`
  - `helpfulness_shift`
  - `combined`
- Levels: `0.0, 0.2, 0.4, 0.6, 0.8, 1.0`.
- Methods:
  - `full_demo_imitation`
  - `residual_correction_learner`
  - `uncertainty_query_policy`
  - `active_entropy_query_policy`
  - `robust_mpc_correction`
  - `minimum_intervention_boundary_learner_v5`
  - `oracle_minimal_correction`

Expected stress raw rows: `10 * 6 * 20 * 6 * 6 * 7 = 302400`.

### Fixed-Risk Deployment Tests

Fixed-risk tests evaluate whether each policy can maintain a deployment budget on unsafe correction/damage risk while preserving coverage.

- Splits: `dynamics_mismatch`, `combined_hard_shift`.
- Budgets: `0.02, 0.05, 0.08, 0.10`.
- Methods:
  - `uncertainty_query_policy`
  - `active_entropy_query_policy`
  - `safety_filtered_residual`
  - `robust_mpc_correction`
  - `minimum_intervention_boundary_learner_v5`
  - `oracle_minimal_correction`
- Episodes per task/seed: 24.

Expected fixed-risk raw rows: `10 * 6 * 2 * 24 * 4 * 6 = 69120`.

### Negative Cases

Create 24 negative cases covering:

- human correction conflicts with safety;
- semantic intent ambiguity;
- local correction hides nonlocal dynamics;
- helpful human becomes adversarial under shift;
- delayed correction changes the boundary;
- minimal intervention under-corrects in deformable/contact tasks.

## Frozen Decision Gates

The terminal decision will be `STRONG_REVISE` only if all recoverable evidence gates pass.

### Main Hard-Aggregate Gate

Let `best_success_reference` be the strongest non-oracle baseline by hard-aggregate `task_success`.

The v5 method must satisfy:

- hard-aggregate `task_success >= best_success_reference + 0.030`;
- hard-aggregate `correction_efficiency >= best_efficiency_reference + 0.080`;
- hard-aggregate `damage <= best_success_reference_damage - 0.010`;
- paired lower-95 seed difference for `task_success` versus `best_success_reference` is positive;
- paired lower-95 seed difference for `correction_efficiency` versus `best_efficiency_reference` is positive.

### Mechanism Gate

The full v5 method must beat every non-full ablation on hard-split robust utility by at least `0.015`, and no ablation may improve task success while also improving damage and efficiency.

### Stress Gate

At maximum combined stress, v5 must not be dominated by any non-oracle method on the tuple:

`task_success`, `correction_efficiency`, `damage`, `unsafe_override`.

### Fixed-Risk Gate

At risk budget `0.05`, v5 must have:

- nonzero accepted-deployment coverage on both fixed-risk splits;
- coverage at least as high as the best non-oracle method satisfying the same budget;
- task success among accepted episodes no worse than the best non-oracle feasible method by more than `0.010`.

### Scope Gate

Even if all synthetic gates pass, mark as at most `STRONG_REVISE` unless real robot or accepted high-fidelity evidence exists. If any of the main, mechanism, stress, or fixed-risk gates fail, mark `KILL_ARCHIVE`.

## Manuscript Requirements

- Generate a 25+ page ICLR-style manuscript.
- Include theorem/proposition section covering:
  - why minimum interventions estimate a local boundary under monotone intervention response;
  - when minimum-norm corrections fail under ambiguity/delay;
  - a fixed-risk acceptance bound tied to calibrated risk scores.
- Include full tables for main results, hard aggregate, paired tests, ablations, stress, fixed-risk, negative cases, and prior-work threat map.
- Use bright boxed clickable citations via `hyperref` color boxes.
- Include references in `paper/references.bib`; citations must resolve and route to bibliography entries.
- Build a clean PDF and copy only to `C:/Users/wangz/Downloads/85.pdf`.
- Do not copy any PDF to the visible Desktop.

## Validation Requirements

The final validator must check:

- expected CSV row counts;
- required columns and methods;
- terminal decision tokens in `results/summary.txt`;
- `paper/main.log` has no unresolved citations/references or rerun warnings;
- `paper/main.pdf` has at least 25 pages;
- `C:/Users/wangz/Downloads/85.pdf` exists and matches the built PDF hash;
- `C:/Users/wangz/Desktop/85.pdf` does not exist;
- PDF text contains terminal decision and key gate failures or passes;
- bright boxed citation settings are present in `paper/main.tex`.

## Repository and Ledger Requirements

- Commit and push the expanded Paper 85 repo to the existing public GitHub repository.
- Update root ledgers:
  - `GLOBAL_POOL_STATUS.md`
  - `BATCH_STATUS.md`
  - `SUBMISSION_STATUS.md`
  - `MASTER_REPORT.md`
  - `MASTER_SUBMISSION_REPORT.md`
  - `SUBMISSION_AUDIT_MATRIX.csv`
- Advance the expanded-standard frontier from Papers 61-84 to Papers 61-85.
