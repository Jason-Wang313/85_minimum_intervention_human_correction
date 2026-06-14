# 85 Minimum-Intervention Human Correction

Submission-hardening version: v4

Terminal decision: **KILL_ARCHIVE** for ICLR main conference.

This repository contains a reproducible local evidence audit for the research bet:

> Learn from the smallest human correction that changes physical outcome.

The v4 rebuild replaces the template scaffold with a deterministic local correction benchmark over four manipulation tasks, five correction-shift splits, eight methods, ablations, stress sweeps, and negative cases.

## Why This Is Archived

- On the combined hard-shift split, `minimum_intervention_learner` reaches `0.56919 +/- 0.02468` task success.
- The strongest success baseline, `uncertainty_query_policy`, reaches `0.53720 +/- 0.02611`.
- The paired task-success difference is only `0.03199 +/- 0.04562`.
- The proposed learner has better correction efficiency (`0.76851`) and lower damage (`0.02976`), but this is a tradeoff result rather than a decisive task-success win.
- The `minus_human_effort_cost` ablation slightly improves task success (`0.56994`) while losing efficiency/damage, so the full objective is not uniquely validated by success.
- The evidence is local and synthetic, not hardware or accepted high-fidelity benchmark validation.

## Reproduce

```powershell
python src\run_experiment.py
```

The runner writes:

- `results/rollouts.csv`
- `results/raw_seed_metrics.csv`
- `results/metrics.csv`
- `results/pairwise_stats.csv`
- `results/ablation_metrics.csv`
- `results/stress_sweep.csv`
- `results/negative_cases.csv`
- `results/summary.txt`
- `figures/minimum_intervention_*.png`

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/85.pdf`
