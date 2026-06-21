# Paper 85 Terminal Audit

Date: 2026-06-21 23:35:31 +08:00

Paper: `85_minimum_intervention_human_correction`

Terminal decision: `KILL_ARCHIVE`

## Commands

```powershell
python -m py_compile src\run_experiment.py
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

## Validation

Validator output:

```text
validated Paper 85 artifacts: pages=28, sha256=3A689EE96ED760908FCE9678AA3A3667B9C768F2218D06272EE382EDDD121406
```

Visual PDF QA sampled title page, figure page, prior-work citation table, dense appendices, and bibliography. The PDF had readable text/tables, no clipped content, and visible bright green citation boxes.

## Artifact Constraints

- Canonical PDF: `C:/Users/wangz/Downloads/85.pdf`.
- Desktop PDF: absent.
- Public repository target: `https://github.com/Jason-Wang313/85_minimum_intervention_human_correction`.
