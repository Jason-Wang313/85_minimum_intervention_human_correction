# Experiment Rigor Checklist

## v2 Synthetic Rigor
- [x] Multiple seeds.
- [x] Error bars.
- [x] Stronger synthetic baselines.
- [x] Ablations.
- [x] Stress tests.
- [x] Negative cases.

## v4 Local Human-Correction Rigor
- [x] Paper-specific correction-vector benchmark.
- [x] Four contact-rich manipulation tasks and five correction-shift splits.
- [x] Eight methods including uncertainty querying and oracle minimal correction.
- [x] Seed-level paired comparisons.
- [x] Ablations for minimum-norm, counterfactual boundary, intent, effort, all-corrections imitation, and preference-only objectives.
- [x] Stress sweeps for human noise, delay, overcorrection, intent ambiguity, and combined stress.
- [x] Negative cases documented.

## ICLR Main Bar
- [ ] Real-robot validation.
- [ ] High-fidelity simulator benchmark.
- [ ] Implemented learned model.
- [ ] Implemented real competing baselines.
- [ ] Manual related-work synthesis.
- [ ] Paper-specific qualitative figures.

Decision: fail ICLR main empirical-rigor gate because the v4 result is non-decisive and still local-only; archive.
