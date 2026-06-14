# Paper 85 Rebuild Plan

Last update: 2026-06-14 11:11:47 +01:00

## Target Claim

Human corrections should be represented as the smallest physical intervention that changes the outcome, not as another full demonstration. A minimum-intervention learner should identify the local physical decision boundary more efficiently than behavior cloning, DAgger-style full corrections, residual imitation, or preference-only feedback.

## Evidence To Build

Replace the v3 template scaffold with a deterministic local human-correction benchmark for contact-rich manipulation.

### Splits

- `nominal_correction`: clean small corrective nudges.
- `overcorrection_bias`: humans tend to provide larger-than-needed corrections.
- `delayed_feedback`: correction arrives after the failure boundary has moved.
- `ambiguous_intent`: multiple corrections can change outcome but only one preserves task intent.
- `combined_hard_shift`: noise, delay, bias, and intent ambiguity together.

### Tasks

- peg insertion.
- drawer alignment.
- cloth corner placement.
- cup handoff pose.

### Methods

- `no_human_baseline`
- `full_demo_imitation`
- `dagger_full_correction`
- `residual_correction_learner`
- `preference_only_ranker`
- `uncertainty_query_policy`
- `minimum_intervention_learner` (proposed)
- `oracle_minimal_correction`

### Main Metrics

- task success.
- human intervention magnitude.
- correction efficiency: success per unit intervention.
- overcorrection damage.
- intent preservation.
- boundary estimation error.
- query rate.
- calibration error.
- paired seed-level differences versus strongest baselines.

### Ablations

- full minimum-intervention learner.
- minus minimum-norm objective.
- minus counterfactual boundary model.
- minus intent-preservation term.
- minus human-effort cost.
- all-corrections imitation.
- preference-only objective.

### Stress Tests

- human correction noise.
- correction delay.
- overcorrection bias.
- intent ambiguity.
- combined stress.

### Terminal Gate

Mark `STRONG_REVISE` only if the proposed learner beats the strongest non-oracle baseline on combined hard-shift task success and correction efficiency, reduces damage/human burden, and ablations degrade the mechanism. Otherwise mark `KILL_ARCHIVE`.

Even a `STRONG_REVISE` result is not ICLR-main ready without robot or accepted high-fidelity benchmark validation.
