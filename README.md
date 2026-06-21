# 85 Minimum-Intervention Human Correction

Submission-hardening version: v5 expanded audit

Terminal decision: **KILL_ARCHIVE** for ICLR main conference.

Latest audit rerun: 2026-06-21.

This repository contains a reproducible local evidence audit for the research bet:

> Learn from the smallest human correction that changes physical outcome.

The v5 rebuild expands the old 5-page v4 memo into a 28-page ICLR-style negative audit with new theory, stronger baselines, fixed-risk deployment tests, full appendices, and bright boxed clickable citation links.

## Why This Is Archived

- The frozen v5 protocol regenerated 199,680 main rollouts, 15,360 scene summaries, 33,600 ablation rollouts, 302,400 stress rows, 69,120 fixed-risk rows, and 24 negative cases.
- On the hard aggregate, `minimum_intervention_boundary_learner_v5` reaches `0.48958 +/- 0.01057` task success and `0.69909` correction efficiency.
- The strongest non-oracle baseline, `robust_mpc_correction`, reaches `0.59010 +/- 0.01426` task success and `0.88048` correction efficiency.
- Paired lower95 bounds are negative: task success `-0.11788`; correction efficiency `-0.20944`.
- Mechanism ablations beat the full method: `minus_human_effort_cost` reaches `0.77648` robust utility versus `0.67199` for the full v5 method.
- Fixed-risk deployment coverage at budget `0.05` collapses to zero on both fixed-risk splits.
- The evidence is still local and synthetic, with no real robot or accepted high-fidelity benchmark validation.

## Reproduce

```powershell
python src\run_experiment.py
python scripts\generate_manuscript.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
cd ..
Copy-Item -LiteralPath paper\main.pdf -Destination C:\Users\wangz\Downloads\85.pdf -Force
python scripts\validate_submission_artifacts.py
```

Canonical local PDF: `C:/Users/wangz/Downloads/85.pdf`

PDF SHA256: `3A689EE96ED760908FCE9678AA3A3667B9C768F2218D06272EE382EDDD121406`
