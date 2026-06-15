# Child Status 85

Current stage: ICLR main v4 evidence audit terminal
Last update: 2026-06-15 09:30:48 +01:00
PDF: C:/Users/wangz/Downloads/85.pdf
GitHub: https://github.com/Jason-Wang313/85_minimum_intervention_human_correction
Submission-hardening version: v4
Terminal decision: KILL_ARCHIVE
ICLR main ready: no

Reason: the 2026-06-15 full rerun regenerated 53,760 main rollouts, 9,408 ablation rollouts, and 117,600 stress rollouts. The minimum-intervention learner improves correction efficiency (`0.12153 +/- 0.04126` paired gain over preference-only ranker) and lowers damage, but the task-success gain over uncertainty querying is not decisive (`0.03199 +/- 0.04562`) and the minus-human-effort-cost ablation slightly improves success while worsening efficiency/damage. No robot hardware or high-fidelity simulator validation is available.
