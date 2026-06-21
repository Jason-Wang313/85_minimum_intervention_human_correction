# Claims

- Mechanism claim tested: the smallest human correction that changes outcome should identify a local physical decision boundary more efficiently than full demonstrations, residual correction, active querying, preference ranking, and robust MPC-style correction.
- Evidence claim: the v5 local benchmark tests task success, correction efficiency, damage, intervention magnitude, boundary error, intent preservation, unsafe override, calibration, regret, robust utility, ablations, stress sweeps, fixed-risk deployment, and negative cases.
- Negative result: `minimum_intervention_boundary_learner_v5` does not beat `robust_mpc_correction` on hard-aggregate task success or correction efficiency.
- Gate result: paired lower95 bounds are negative, mechanism ablations beat the full method, and fixed-risk coverage at budget 0.05 collapses to zero.
- Scope claim: results support an archive-quality negative audit, not real-robot deployment or ICLR-main readiness.
- Unsupported claim explicitly avoided: no claim of SOTA robot performance, real human-in-the-loop robot validation, high-fidelity simulation validation, or ICLR-main readiness.
