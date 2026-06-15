# Plan

Paper 85 `minimum_intervention_human_correction` is in the 2026-06-15 ICLR-main submission-readiness audit pass.

Execution plan:

1. Rerun the full deterministic human-correction benchmark from source.
2. Audit all main, ablation, stress, pairwise, and negative-case outputs.
3. Apply the ICLR-main evidence gate without overclaiming local synthetic evidence.
4. Preserve the terminal decision as `KILL_ARCHIVE` unless minimum-intervention learning decisively beats active querying on task success while also improving correction efficiency, burden, damage, and ablations.
5. Rebuild `C:/Users/wangz/Downloads/85.pdf` only, update root reports, commit, push, and verify the public GitHub repo.
