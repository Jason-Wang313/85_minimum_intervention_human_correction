# Paper 85 Terminal Audit

Date: 2026-06-15 09:30:48 +01:00

Paper: `85_minimum_intervention_human_correction`

Terminal decision: `KILL_ARCHIVE`

## Rerun Command

```powershell
python -m py_compile src\run_experiment.py
python src\run_experiment.py
```

The experiment runner completed successfully and printed `terminal=KILL_ARCHIVE`.

## Evidence Coverage

- `rollouts.csv`: 53,760 rows, 21 columns.
- `raw_seed_metrics.csv`: 280 rows, 13 columns.
- `metrics.csv`: 360 rows, 7 columns.
- `pairwise_stats.csv`: 210 rows, 6 columns.
- `ablation_rollouts.csv`: 9,408 rows, 21 columns.
- `ablation_seed_metrics.csv`: 49 rows, 13 columns.
- `ablation_metrics.csv`: 7 rows, 9 columns.
- `stress_sweep_raw.csv`: 117,600 rows, 23 columns.
- `stress_sweep.csv`: 150 rows, 9 columns.
- `negative_cases.csv`: 4 rows, 4 columns.

Verified seeds: `0, 1, 2, 3, 4, 5, 6`.

Verified splits: `nominal_correction`, `overcorrection_bias`, `delayed_feedback`, `ambiguous_intent`, `combined_hard_shift`.

Verified tasks: `peg_insertion`, `drawer_alignment`, `cloth_corner_place`, `cup_handoff_pose`.

Verified methods: `no_human_baseline`, `full_demo_imitation`, `dagger_full_correction`, `residual_correction_learner`, `preference_only_ranker`, `uncertainty_query_policy`, `minimum_intervention_learner`, `oracle_minimal_correction`.

Verified ablations: `full_minimum_intervention_learner`, `minus_minimum_norm_objective`, `minus_counterfactual_boundary`, `minus_intent_preservation`, `minus_human_effort_cost`, `all_corrections_imitation`, `preference_only_objective`.

Verified stress axes: `human_noise`, `correction_delay`, `overcorrection_bias`, `intent_ambiguity`, `combined`.

## Main Gate

Combined hard-shift task success:

- `minimum_intervention_learner`: `0.56919 +/- 0.02468`.
- `uncertainty_query_policy`: `0.53720 +/- 0.02611`.
- Paired task-success difference versus uncertainty querying: `0.03199 +/- 0.04562`.
- Paired correction-efficiency difference versus preference-only ranker: `0.12153 +/- 0.04126`.
- `oracle_minimal_correction`: `0.65774 +/- 0.02258`.

The proposed method improves correction efficiency and damage, but the primary success margin is not decisive enough for an ICLR-main-target claim.

## Ablation Gate

- Full learner: `0.56919 +/- 0.02468` task success, `0.76851` efficiency, `0.02976` damage.
- `minus_human_effort_cost`: `0.56994 +/- 0.03428` task success, `0.71189` efficiency, `0.06176` damage.

This supports an efficiency/damage tradeoff, but not a success-dominant mechanism claim.

## Stress Gate

At maximum combined stress:

- `minimum_intervention_learner`: `0.53571 +/- 0.03285` task success, `0.55623` efficiency, `0.03189` damage.
- `uncertainty_query_policy`: `0.48469 +/- 0.02151` task success, `0.36353` efficiency, `0.08929` damage.
- `oracle_minimal_correction`: `0.64923 +/- 0.02746` task success.

Stress evidence is promising but local synthetic only and does not remove the main paired-success uncertainty.

## Submission Decision

Paper 85 is not ICLR-main ready. It should remain an archived negative result unless future work adds robot or recognized high-fidelity human-correction evidence, real human correction traces, stronger active-learning/preference/residual baselines, and decisive paired gains in task success while preserving burden and damage advantages.

## PDF Artifact

- Canonical PDF: `C:/Users/wangz/Downloads/85.pdf`.
- SHA256: `7FFA178EBE0655EC7AB2BB915E81012196CACFD55F9A2A17589A49198AA5D4D9`.
- Desktop copy: absent.
