# Paper 85 ICLR-Main Submission-Readiness Execution Plan

Date: 2026-06-15

Paper: `85_minimum_intervention_human_correction`

Target venue standard: ICLR main conference, with an evidence-first gate. The paper can advance only if the rebuilt evidence shows decisive task success, correction-efficiency, burden, damage, and ablation support against strong human-feedback baselines. A good tradeoff is not enough for main-conference readiness if the primary success margin is swallowed by uncertainty.

## Current State

The repository currently reports a v4 terminal decision of `KILL_ARCHIVE`. The existing claim is that robots should learn from the smallest human correction that changes physical outcome rather than from full demonstrations, residual corrections, or preference-only feedback. The prior audit found promising efficiency and damage benefits, but a non-decisive task-success margin over `uncertainty_query_policy` and an ablation where removing the human-effort cost slightly improves task success. The evidence remains local synthetic simulation, not robot hardware or an accepted high-fidelity benchmark.

## Execution Order

1. Verify repository hygiene before touching evidence.
   - Confirm the worktree is clean except for this plan.
   - Record the pre-audit commit.
   - Confirm the GitHub remote exists and is public.

2. Re-run the full evidence generator from source.
   - Compile-check `src/run_experiment.py`.
   - Run `python src/run_experiment.py`.
   - Preserve all generated CSVs, figures, and `results/summary.txt`.

3. Audit evidence completeness.
   - Confirm seven seeds are present.
   - Confirm all splits, tasks, methods, ablations, stress axes, and negative cases are represented.
   - Confirm row counts and schemas for rollout, seed metric, aggregate metric, pairwise, ablation, stress, and negative-case files.

4. Apply the ICLR-main decision gate.
   - Require the proposed learner to beat the strongest non-oracle baseline on combined hard-shift task success with paired uncertainty that supports the claim.
   - Require correction efficiency, intervention magnitude, damage, intent preservation, and boundary error to improve without hiding query cost.
   - Require ablations to degrade when minimum-norm, counterfactual boundary, intent-preservation, or human-effort terms are removed.
   - Require stress tests to support the same conclusion under human noise, correction delay, overcorrection bias, intent ambiguity, and combined stress.

5. Decide honestly.
   - If all gates pass but evidence remains local synthetic only, mark at most `STRONG_REVISE`.
   - If task-success separation is non-decisive or an objective ablation improves the primary metric, preserve `KILL_ARCHIVE`.
   - Do not claim ICLR-main readiness without robot or recognized high-fidelity benchmark evidence.

6. Update the paper and child documentation.
   - Make `README.md`, `child_status.md`, `plan.md`, audit docs, attack log, readiness decision, hostile reviewer response, and version log match the rerun.
   - Add a terminal audit document with exact row counts, seed coverage, metric conclusions, and PDF hash.

7. Build and verify the PDF.
   - Build `paper/main.pdf` with LaTeX.
   - Copy only the numbered PDF to `C:/Users/wangz/Downloads/85.pdf`.
   - Do not copy any PDF to the visible Desktop.
   - Scan logs for LaTeX/BibTeX warnings that affect submission quality.

8. Update root reports.
   - Update `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `SUBMISSION_STATUS.md`, `MASTER_REPORT.md`, and `MASTER_SUBMISSION_REPORT.md`.
   - Mark Paper 85 with the final terminal decision, commit hash, PDF hash, GitHub URL, and concise evidence.

9. Commit, push, and verify.
   - Commit only Paper 85 files inside its child repo.
   - Push `main` to the public GitHub repo.
   - Verify local `HEAD` equals `origin/main`.
   - Verify `C:/Users/wangz/Downloads/85.pdf` exists and `C:/Users/wangz/Desktop/85.pdf` does not.

## Expected Outcome Risk

The likely outcome is `KILL_ARCHIVE`, because the prior v4 evidence reports a useful burden/damage tradeoff but not a decisive task-success win, and the `minus_human_effort_cost` ablation slightly improves success. The rerun will still be performed end-to-end; the decision will be evidence-bound, not assumed.
