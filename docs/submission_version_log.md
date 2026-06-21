# Submission Version Log

## v1 - Generated Draft
- Original continuation-batch generated paper and toy single-seed experiment.

## v2 - Submission Hardening
- Added hostile reviewer attack log and response docs.
- Replaced the toy experiment with seven-seed metrics, stronger baselines, ablations, stress tests, and negative cases.
- Narrowed claims to synthetic diagnostic evidence.
- Recompiled canonical PDF at `C:/Users/wangz/Downloads/85.pdf`.
- Terminal decision: WORKSHOP_ONLY.

## v3 - ICLR Main Gate Archive
- Applied the stricter ICLR-main-conference standard.
- Re-read local paper, docs, experiments, prior-work artifacts, PDF state, and repo state.
- Determined that missing real-robot/high-fidelity evidence, template-generated experiments, and unresolved novelty threats are not recoverable from local artifacts.
- Recompiled the canonical PDF with `Submission-hardening version: v3`.
- Terminal decision: KILL_ARCHIVE.

## v4 - Minimum-Intervention Evidence Audit
- Replaced the template scaffold with a deterministic local human-correction benchmark.
- Added four tasks, eight methods, five correction-shift splits, ablations, stress sweeps, negative cases, and figures.
- Main result: minimum intervention improves efficiency and damage but has non-decisive task-success gain over uncertainty querying.
- Ablation result: removing human-effort cost slightly improves task success while worsening efficiency/damage.
- Recompiled the canonical PDF with `Submission-hardening version: v4`.
- Terminal decision: KILL_ARCHIVE.

## v4.1 - 2026-06-15 Rerun Audit
- Added the paper-specific ICLR-main execution plan before running any new evidence.
- Re-ran `python src\run_experiment.py` from source and reproduced `terminal=KILL_ARCHIVE`.
- Verified 53,760 main rollouts, 9,408 ablation rollouts, 117,600 stress rollouts, seven seeds, eight methods, seven ablations, five stress axes, four tasks, and four negative cases.
- Preserved the terminal decision because the primary task-success gain remains non-decisive and the objective ablation still slightly improves success.

## v5 - 2026-06-21 Expanded Submission-Readiness Audit
- Wrote `docs/paper85_expanded_submission_plan_20260621.md` before modifying code.
- Replaced the v4 runner with a frozen CPU-only, RAM-light protocol over 10 seeds, 6 tasks, 8 splits, 13 methods, 10 ablations, 6 stress axes, fixed-risk deployment budgets, and 24 negative cases.
- Generated 199,680 main rollouts, 15,360 scene summaries, 33,600 ablation rows, 302,400 stress rows, and 69,120 fixed-risk rows.
- Generated a 28-page ICLR-style manuscript with theory, full appendices, bright boxed clickable citation links, and a clean citation/reference log.
- Validated `C:/Users/wangz/Downloads/85.pdf`; SHA256 `3A689EE96ED760908FCE9678AA3A3667B9C768F2218D06272EE382EDDD121406`; no Desktop PDF.
- Terminal decision remains `KILL_ARCHIVE` because v5 loses to `robust_mpc_correction`, paired lower95 bounds are negative, mechanism ablations beat the full method, fixed-risk coverage at budget 0.05 is zero, and no real robot or accepted high-fidelity benchmark evidence exists.
