# Final Audit

1. Chosen thesis: Minimum-Intervention Human Correction explores `Learn from the smallest human correction that changes physical outcome.` for interactive robot learning.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v4.
4. Reason: a local human-correction benchmark was added, but the success gain over uncertainty querying is non-decisive and the full objective is supported by efficiency/damage tradeoffs rather than task-success dominance.
5. Closest hostile prior work: see `docs/hostile_prior_work.md`, `docs/hostile_prior_work_100_cards.csv`, and `docs/hostile_reviewer_response.md`.
6. Reproducibility: v4 benchmark code runs and regenerates metrics/figures, but no real robot or high-fidelity benchmark is reproduced.
7. Claim-validity status: positive main-conference claims killed; v4 negative evidence audit retained.
8. Exact Downloads PDF path: `C:/Users/wangz/Downloads/85.pdf`
9. GitHub URL: https://github.com/Jason-Wang313/85_minimum_intervention_human_correction
10. Confirmation: no visible Desktop copy was requested or made.
11. 2026-06-15 rerun: 53,760 main rollouts, 9,408 ablation rollouts, and 117,600 stress rollouts reproduced `KILL_ARCHIVE`.
12. Hard-split gate: `minimum_intervention_learner` vs `uncertainty_query_policy` paired task-success difference is `0.03199 +/- 0.04562`.
13. Tradeoff gate: correction-efficiency paired gain is `0.12153 +/- 0.04126`, but the success claim remains non-decisive.
14. Mechanism gate: `minus_human_effort_cost` improves task success to `0.56994`, above full at `0.56919`, while worsening efficiency and damage.
