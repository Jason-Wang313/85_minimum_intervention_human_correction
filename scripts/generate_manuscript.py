import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
DOCS = ROOT / "docs"


METHOD_LABELS = {
    "no_human_baseline": "No human",
    "full_demo_imitation": "Full demo",
    "dagger_full_correction": "DAgger full",
    "residual_correction_learner": "Residual",
    "preference_only_ranker": "Preference",
    "uncertainty_query_policy": "Uncertainty query",
    "active_entropy_query_policy": "Active entropy",
    "safety_filtered_residual": "Safety residual",
    "robust_mpc_correction": "Robust MPC",
    "inverse_rl_correction_proxy": "Inverse-RL proxy",
    "minimum_intervention_learner_v4": "Min-int v4",
    "minimum_intervention_boundary_learner_v5": "MIBL v5",
    "oracle_minimal_correction": "Oracle minimal",
}

FOCUS_METHODS = [
    "preference_only_ranker",
    "uncertainty_query_policy",
    "active_entropy_query_policy",
    "safety_filtered_residual",
    "robust_mpc_correction",
    "minimum_intervention_learner_v4",
    "minimum_intervention_boundary_learner_v5",
    "oracle_minimal_correction",
]

SPLIT_LABELS = {
    "nominal_correction": "Nominal",
    "overcorrection_bias": "Overcorrect",
    "delayed_feedback": "Delay",
    "ambiguous_intent": "Ambiguous",
    "sparse_corrections": "Sparse",
    "adversarial_helpfulness": "Adversarial",
    "dynamics_mismatch": "Dynamics",
    "combined_hard_shift": "Combined",
    "hard_aggregate": "Hard agg.",
}


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_rows(name):
    return len(read_csv(RESULTS / name))


def ascii_clean(text):
    text = str(text or "")
    text = text.replace("–", "-").replace("—", "-").replace("“", '"').replace("”", '"').replace("’", "'")
    return text.encode("ascii", "ignore").decode("ascii")


def tex_escape(text):
    text = ascii_clean(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def method_name(method):
    return tex_escape(METHOD_LABELS.get(method, method))


def split_name(split):
    return tex_escape(SPLIT_LABELS.get(split, split))


def metric_lookup(rows, selectors, metric):
    for row in rows:
        if row.get("metric") != metric:
            continue
        if all(row.get(k) == v for k, v in selectors.items()):
            return float(row["mean"]), float(row["ci95"])
    raise KeyError((selectors, metric))


def fmt_pm(mean, ci):
    return f"{mean:.3f} $\\pm$ {ci:.3f}"


def parse_summary(summary_text):
    out = {}
    for line in summary_text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
        elif ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def bib_key(i):
    return f"pool85_{i:02d}"


def write_references():
    rows = read_csv(DOCS / "deep_read_250.csv")[:48]
    entries = []
    for i, row in enumerate(rows, start=1):
        title = tex_escape(row.get("title") or "Untitled prior work")
        authors_raw = ascii_clean(row.get("authors") or "Local Prior Work Pool")
        parts = [p.strip() for p in re.split(r";| and ", authors_raw) if p.strip()]
        authors = " and ".join(tex_escape(p) for p in parts[:8]) or "Local Prior Work Pool"
        year_raw = ascii_clean(row.get("year") or "")
        match = re.search(r"(19|20)\d{2}", year_raw)
        year = match.group(0) if match else "2026"
        venue = tex_escape(row.get("venue") or row.get("source") or "prior-work pool")
        link = tex_escape(row.get("doi") or row.get("url") or row.get("arxiv_id") or row.get("uid") or "local pool record")
        entries.append(
            "\n".join(
                [
                    f"@misc{{{bib_key(i)},",
                    f"  author={{{authors}}},",
                    f"  title={{{title}}},",
                    f"  year={{{year}}},",
                    f"  note={{{venue}; {link}}}",
                    "}",
                ]
            )
        )
    (PAPER / "references.bib").write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    return [bib_key(i) for i in range(1, len(rows) + 1)], rows


def longtable(header, rows, spec, caption, label, fontsize=r"\scriptsize"):
    lines = [
        r"\begin{center}",
        fontsize,
        f"\\begin{{longtable}}{{{spec}}}",
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
        r"\toprule",
        header + r"\\",
        r"\midrule",
        r"\endfirsthead",
        f"\\caption[]{{{caption} (continued)}}\\\\",
        r"\toprule",
        header + r"\\",
        r"\midrule",
        r"\endhead",
    ]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize", r"\end{center}"])
    return "\n".join(lines)


def hard_table(hard_metrics):
    rows = []
    for method in METHOD_LABELS:
        success = metric_lookup(hard_metrics, {"split": "hard_aggregate", "method": method}, "task_success")
        eff = metric_lookup(hard_metrics, {"split": "hard_aggregate", "method": method}, "correction_efficiency")
        damage = metric_lookup(hard_metrics, {"split": "hard_aggregate", "method": method}, "damage")
        unsafe = metric_lookup(hard_metrics, {"split": "hard_aggregate", "method": method}, "unsafe_override")
        utility = metric_lookup(hard_metrics, {"split": "hard_aggregate", "method": method}, "robust_utility")
        rows.append(
            f"{method_name(method)} & {fmt_pm(*success)} & {eff[0]:.3f} & {damage[0]:.3f} & {unsafe[0]:.3f} & {utility[0]:.3f}\\\\"
        )
    return longtable(
        "Method & Success & Efficiency & Damage & Unsafe & Utility",
        rows,
        "p{0.26\\linewidth}rrrrr",
        "Predefined hard-aggregate results over adversarial helpfulness, dynamics mismatch, and combined hard shift.",
        "tab:hard",
    )


def split_table(metrics):
    rows = []
    methods = ["robust_mpc_correction", "minimum_intervention_learner_v4", "minimum_intervention_boundary_learner_v5", "oracle_minimal_correction"]
    for split in SPLIT_LABELS:
        if split == "hard_aggregate":
            continue
        for method in methods:
            success = metric_lookup(metrics, {"split": split, "method": method}, "task_success")
            eff = metric_lookup(metrics, {"split": split, "method": method}, "correction_efficiency")
            damage = metric_lookup(metrics, {"split": split, "method": method}, "damage")
            unsafe = metric_lookup(metrics, {"split": split, "method": method}, "unsafe_override")
            rows.append(f"{split_name(split)} & {method_name(method)} & {fmt_pm(*success)} & {eff[0]:.3f} & {damage[0]:.3f} & {unsafe[0]:.3f}\\\\")
    return longtable(
        "Split & Method & Success & Efficiency & Damage & Unsafe",
        rows,
        "p{0.15\\linewidth}p{0.25\\linewidth}rrrr",
        "Split-level evidence for the strongest baseline, v4, v5, and oracle.",
        "tab:split",
    )


def pairwise_table(hard_pairs):
    refs = ["robust_mpc_correction", "minimum_intervention_learner_v4", "active_entropy_query_policy", "preference_only_ranker"]
    metrics = ["task_success", "correction_efficiency", "damage", "robust_utility"]
    rows = []
    for ref in refs:
        for metric in metrics:
            matches = [r for r in hard_pairs if r["split"] == "hard_aggregate" and r["reference"] == ref and r["metric"] == metric]
            if not matches:
                continue
            row = matches[0]
            rows.append(
                f"{method_name(ref)} & {tex_escape(metric)} & {float(row['mean_diff']):.3f} & {float(row['ci95_diff']):.3f} & {float(row['lower95_diff']):.3f}\\\\"
            )
    return longtable(
        "Reference & Metric & Mean diff & CI95 & Lower95",
        rows,
        "p{0.29\\linewidth}p{0.22\\linewidth}rrr",
        "Paired seed-level hard-aggregate differences for MIBL v5 minus reference. Positive lower95 was required for primary success and efficiency.",
        "tab:paired",
    )


def ablation_table(ablations):
    rows = []
    for row in ablations:
        rows.append(
            f"{split_name(row['split'])} & {tex_escape(row['ablation'])} & {float(row['task_success']):.3f} & {float(row['correction_efficiency']):.3f} & {float(row['damage']):.3f} & {float(row['robust_utility']):.3f}\\\\"
        )
    return longtable(
        "Split & Ablation & Success & Efficiency & Damage & Utility",
        rows,
        "p{0.15\\linewidth}p{0.39\\linewidth}rrrr",
        "Mechanism ablations. The full mechanism was required to beat every non-full ablation on robust utility by at least 0.015.",
        "tab:ablation",
    )


def stress_table(stress):
    rows = []
    for row in stress:
        if row["stress_axis"] != "combined":
            continue
        rows.append(
            f"{row['stress_level']} & {method_name(row['method'])} & {float(row['task_success']):.3f} & {float(row['correction_efficiency']):.3f} & {float(row['damage']):.3f} & {float(row['unsafe_override']):.3f} & {float(row['robust_utility']):.3f}\\\\"
        )
    return longtable(
        "Level & Method & Success & Efficiency & Damage & Unsafe & Utility",
        rows,
        "rp{0.26\\linewidth}rrrrr",
        "Combined stress sweep. V5 was not Pareto-dominated at maximum stress, but that alone did not rescue the paper.",
        "tab:stress",
    )


def fixed_table(fixed):
    rows = []
    for row in fixed:
        if row["risk_budget"] not in {"0.02", "0.05", "0.10"}:
            continue
        rows.append(
            f"{split_name(row['split'])} & {row['risk_budget']} & {method_name(row['method'])} & {float(row['coverage']):.3f} & {float(row['accepted_success']):.3f} & {float(row['accepted_damage']):.3f} & {float(row['accepted_unsafe']):.3f}\\\\"
        )
    return longtable(
        "Split & Budget & Method & Coverage & Succ. & Damage & Unsafe",
        rows,
        "p{0.16\\linewidth}rp{0.26\\linewidth}rrrr",
        "Fixed-risk deployment tests. At budget 0.05 all non-oracle and oracle methods had zero accepted coverage, so the frozen fixed-risk gate failed.",
        "tab:fixed",
    )


def negative_table(negative):
    rows = []
    for row in negative:
        rows.append(
            f"{tex_escape(row['case_id'])} & {tex_escape(row['case_family'])} & {tex_escape(row['terminal_lesson'])}\\\\"
        )
    return longtable(
        "Case & Family & Terminal lesson",
        rows,
        "p{0.25\\linewidth}p{0.25\\linewidth}p{0.38\\linewidth}",
        "Retained negative cases used to keep the discussion honest.",
        "tab:negative",
    )


def full_main_table(metrics):
    rows = []
    for split in SPLIT_LABELS:
        if split == "hard_aggregate":
            continue
        for method in METHOD_LABELS:
            success = metric_lookup(metrics, {"split": split, "method": method}, "task_success")
            eff = metric_lookup(metrics, {"split": split, "method": method}, "correction_efficiency")
            damage = metric_lookup(metrics, {"split": split, "method": method}, "damage")
            unsafe = metric_lookup(metrics, {"split": split, "method": method}, "unsafe_override")
            utility = metric_lookup(metrics, {"split": split, "method": method}, "robust_utility")
            rows.append(
                f"{split_name(split)} & {method_name(method)} & {fmt_pm(*success)} & {eff[0]:.3f} & {damage[0]:.3f} & {unsafe[0]:.3f} & {utility[0]:.3f}\\\\"
            )
    return longtable(
        "Split & Method & Success & Efficiency & Damage & Unsafe & Utility",
        rows,
        "p{0.13\\linewidth}p{0.24\\linewidth}rrrrr",
        "Full split-by-method main metrics. This appendix table prevents cherry-picking by exposing every predefined split and method.",
        "tab:full-main",
        fontsize=r"\tiny",
    )


def full_pairwise_table(hard_pairs):
    rows = []
    for row in hard_pairs:
        rows.append(
            f"{method_name(row['reference'])} & {tex_escape(row['metric'])} & {float(row['mean_diff']):.3f} & {float(row['ci95_diff']):.3f} & {float(row['lower95_diff']):.3f}\\\\"
        )
    return longtable(
        "Reference & Metric & Mean diff & CI95 & Lower95",
        rows,
        "p{0.32\\linewidth}p{0.25\\linewidth}rrr",
        "All hard-aggregate paired differences for MIBL v5 minus each reference method.",
        "tab:full-pairwise",
        fontsize=r"\tiny",
    )


def full_stress_table(stress):
    rows = []
    for row in stress:
        rows.append(
            f"{tex_escape(row['stress_axis'])} & {row['stress_level']} & {method_name(row['method'])} & {float(row['task_success']):.3f} & {float(row['correction_efficiency']):.3f} & {float(row['damage']):.3f} & {float(row['unsafe_override']):.3f} & {float(row['robust_utility']):.3f}\\\\"
        )
    return longtable(
        "Axis & Level & Method & Success & Eff. & Damage & Unsafe & Utility",
        rows,
        "p{0.16\\linewidth}rp{0.23\\linewidth}rrrrr",
        "All stress-sweep metrics across six axes, six levels, and seven methods.",
        "tab:full-stress",
        fontsize=r"\tiny",
    )


def full_fixed_table(fixed):
    rows = []
    for row in fixed:
        rows.append(
            f"{split_name(row['split'])} & {row['risk_budget']} & {method_name(row['method'])} & {float(row['coverage']):.3f} & {float(row['accepted_success']):.3f} & {float(row['accepted_damage']):.3f} & {float(row['accepted_unsafe']):.3f} & {float(row['accepted_utility']):.3f}\\\\"
        )
    return longtable(
        "Split & Budget & Method & Coverage & Succ. & Damage & Unsafe & Utility",
        rows,
        "p{0.14\\linewidth}rp{0.23\\linewidth}rrrrr",
        "All fixed-risk deployment rows across both hard splits, four budgets, and six methods.",
        "tab:full-fixed",
        fontsize=r"\tiny",
    )


def fixed_pairwise_table(fixed_pairs):
    rows = []
    for row in fixed_pairs:
        rows.append(
            f"{split_name(row['split'])} & {row['risk_budget']} & {method_name(row['reference'])} & {tex_escape(row['metric'])} & {float(row['mean_diff']):.3f} & {float(row['ci95_diff']):.3f} & {float(row['lower95_diff']):.3f}\\\\"
        )
    return longtable(
        "Split & Budget & Reference & Metric & Mean diff & CI95 & Lower95",
        rows,
        "p{0.13\\linewidth}rp{0.22\\linewidth}p{0.19\\linewidth}rrr",
        "All fixed-risk paired comparisons for MIBL v5 minus each non-v5 reference across budgets.",
        "tab:fixed-pairwise",
        fontsize=r"\tiny",
    )


def hard_seed_table(hard_seed):
    rows = []
    for row in hard_seed:
        rows.append(
            f"{method_name(row['method'])} & {row['seed']} & {float(row['task_success']):.3f} & {float(row['correction_efficiency']):.3f} & {float(row['damage']):.3f} & {float(row['unsafe_override']):.3f} & {float(row['robust_utility']):.3f}\\\\"
        )
    return longtable(
        "Method & Seed & Success & Efficiency & Damage & Unsafe & Utility",
        rows,
        "p{0.31\\linewidth}rrrrrr",
        "Hard-aggregate seed means. These rows are the units used by paired tests.",
        "tab:hard-seed",
        fontsize=r"\tiny",
    )


def ablation_seed_table(ablation_seed):
    rows = []
    for row in ablation_seed:
        rows.append(
            f"{split_name(row['split'])} & {tex_escape(row['ablation'])} & {row['seed']} & {float(row['task_success']):.3f} & {float(row['correction_efficiency']):.3f} & {float(row['damage']):.3f} & {float(row['robust_utility']):.3f}\\\\"
        )
    return longtable(
        "Split & Ablation & Seed & Success & Efficiency & Damage & Utility",
        rows,
        "p{0.13\\linewidth}p{0.35\\linewidth}rrrrr",
        "Ablation seed means for the two hard ablation splits.",
        "tab:ablation-seed",
        fontsize=r"\tiny",
    )


def prior_work_table(prior_rows, cite_keys):
    rows = []
    for i, row in enumerate(prior_rows[:48], start=1):
        title = tex_escape(row.get("title") or "Untitled")
        year = tex_escape(row.get("year") or "")
        venue = tex_escape(row.get("venue") or row.get("source") or "pool")
        rows.append(f"\\citep{{{cite_keys[i-1]}}} & {title[:120]} & {year} & {venue[:60]}\\\\")
    return longtable(
        "Cite & Prior-work pressure & Year & Source",
        rows,
        "p{0.12\\linewidth}p{0.53\\linewidth}p{0.08\\linewidth}p{0.18\\linewidth}",
        "Local prior-work pressure map. The entries are used as threat coverage, not as exhaustive manual survey.",
        "tab:prior",
    )


def main():
    PAPER.mkdir(exist_ok=True)
    cite_keys, prior_rows = write_references()
    metrics = read_csv(RESULTS / "metrics.csv")
    hard_metrics = read_csv(RESULTS / "hard_aggregate_metrics.csv")
    hard_pairs = read_csv(RESULTS / "hard_aggregate_pairwise_stats.csv")
    hard_seed = read_csv(RESULTS / "hard_aggregate_seed_metrics.csv")
    ablations = read_csv(RESULTS / "ablation_metrics.csv")
    ablation_seed = read_csv(RESULTS / "ablation_seed_metrics.csv")
    stress = read_csv(RESULTS / "stress_sweep.csv")
    fixed = read_csv(RESULTS / "fixed_risk_metrics.csv")
    fixed_pairs = read_csv(RESULTS / "fixed_risk_pairwise.csv")
    negative = read_csv(RESULTS / "negative_cases.csv")
    summary = parse_summary((RESULTS / "summary.txt").read_text(encoding="utf-8"))

    proposal = "minimum_intervention_boundary_learner_v5"
    best_ref = summary["best_success_reference"]
    prop_success = metric_lookup(hard_metrics, {"split": "hard_aggregate", "method": proposal}, "task_success")
    prop_eff = metric_lookup(hard_metrics, {"split": "hard_aggregate", "method": proposal}, "correction_efficiency")
    prop_damage = metric_lookup(hard_metrics, {"split": "hard_aggregate", "method": proposal}, "damage")
    best_success = metric_lookup(hard_metrics, {"split": "hard_aggregate", "method": best_ref}, "task_success")
    best_eff = metric_lookup(hard_metrics, {"split": "hard_aggregate", "method": summary["best_efficiency_reference"]}, "correction_efficiency")

    rows_text = (
        f"{count_rows('rollouts.csv'):,} main rollouts, {count_rows('dataset_summary.csv'):,} scene summaries, "
        f"{count_rows('ablation_rollouts.csv'):,} ablation rollouts, {count_rows('stress_sweep_raw.csv'):,} stress rows, "
        f"{count_rows('fixed_risk_raw.csv'):,} fixed-risk rows, and {count_rows('negative_cases.csv')} negative cases"
    )

    lines = [
        r"\documentclass{article}",
        r"\usepackage{iclr2026_conference,times}",
        r"\input{math_commands.tex}",
        r"\usepackage{hyperref}",
        r"\usepackage{url}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{array}",
        r"\usepackage{longtable}",
        r"\usepackage{xcolor}",
        r"\usepackage{amsmath,amssymb}",
        r"\hypersetup{colorlinks=false,pdfborder={0 0 1.8},citebordercolor={0 1 0},linkbordercolor={1 0.55 0},urlbordercolor={0 0.45 1}}",
        r"\graphicspath{{../figures/}}",
        r"\newcommand{\methodname}{minimum-intervention boundary learner v5}",
        r"\title{Minimum-Intervention Human Correction:\\A 25+ Page Negative Submission-Readiness Audit}",
        r"\author{Anonymous Authors}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        (
            "Human correction data in robotics is often treated as another demonstration, preference, or residual action. "
            "This paper tests a sharper hypothesis: the smallest human intervention that changes the outcome should reveal a local physical decision boundary with lower burden and lower overcorrection damage. "
            f"We rebuilt the archived Paper 85 into a frozen CPU-only v5 audit with {rows_text}. "
            f"On the predefined hard aggregate, \\methodname{{}} reaches {fmt_pm(*prop_success)} task success, {prop_eff[0]:.3f} correction efficiency, and {prop_damage[0]:.3f} damage. "
            f"The strongest non-oracle baseline, {method_name(best_ref)}, reaches {fmt_pm(*best_success)} task success and {best_eff[0]:.3f} correction efficiency. "
            f"The paired lower95 success bound is {float(summary['paired_success_lower95']):.3f}, the paired lower95 efficiency bound is {float(summary['paired_efficiency_lower95']):.3f}, mechanism ablations beat the full method, fixed-risk coverage at budget 0.05 collapses to zero, and no real robot or accepted high-fidelity benchmark exists. "
            "The honest terminal decision is therefore \\textbf{KILL/ARCHIVE}, not ICLR-main submission."
        ),
        r"\end{abstract}",
        r"\section{Terminal Decision}",
        (
            "\\textbf{Decision: KILL/ARCHIVE for ICLR main.} "
            "This document is intentionally submission-shaped but not submission-claiming. "
            "The rebuild added a stronger method, stronger baselines, fixed gates, and a substantially larger evidence package. "
            "The result still does not survive hostile review because the proposed method loses the main hard-aggregate gate and the mechanism gate."
        ),
        (
            f"The frozen primary comparison is against {method_name(best_ref)}. "
            f"MIBL v5 has hard-aggregate success {fmt_pm(*prop_success)} while {method_name(best_ref)} has {fmt_pm(*best_success)}. "
            f"The mean paired success difference is negative enough that the lower95 bound is {float(summary['paired_success_lower95']):.3f}. "
            f"The same failure occurs for correction efficiency, where v5 has {prop_eff[0]:.3f} and the best efficiency baseline has {best_eff[0]:.3f}."
        ),
        (
            "The paper is valuable as an archive because it says something concrete: under this synthetic correction-shift model, a robust MPC-style correction policy is a harder baseline than the original minimum-intervention learner, and several ablations of v5 improve success or utility. "
            "That is precisely the kind of result a hostile reviewer would find, so it is reported rather than hidden."
        ),
        r"\section{Research Question And Threat Model}",
        (
            "The target claim is not merely that small corrections are cheaper. "
            "The target claim is that minimal corrections identify a local physical decision boundary: the smallest intervention crossing the failure/success boundary is expected to carry more causal information per unit human effort than a full demonstration. "
            "This connects to active learning, imitation learning, preference learning, residual policy correction, model-predictive control, and safe deployment under uncertainty "
            f"\\citep{{{cite_keys[0]},{cite_keys[1]},{cite_keys[2]},{cite_keys[3]},{cite_keys[4]},{cite_keys[5]}}}."
        ),
        (
            "The threat model is severe. "
            "A reviewer can attack the claim from six angles: the correction might be delayed, the human might overcorrect, the semantic intent might be ambiguous, the correction might be sparse, the human might become less helpful under distribution shift, and deformable/contact dynamics might make local boundary evidence nonlocal. "
            "The v5 benchmark encodes all six threats rather than selecting only the easy nominal split."
        ),
        r"\section{Formal Setup}",
        (
            "Let $x$ denote the local robot-scene state, $u$ the robot action, $c\\in\\mathbb{R}^d$ a human correction vector, and $Y(x,u,c)\\in\\{0,1\\}$ the task outcome. "
            "A minimum-intervention correction is the solution of"
        ),
        r"\begin{equation}",
        r"c^\star(x,u)=\arg\min_c \|c\|_2 \quad \mathrm{s.t.}\quad Y(x,u,c)=1,\quad S(x,u,c)\leq \rho,",
        r"\end{equation}",
        (
            "where $S$ is a latent safety risk and $\\rho$ is a deployment budget. "
            "The proposed v5 learner estimates a boundary normal, adds an intent-preserving tangent term, clips dangerous components, throttles queries under low uncertainty, and outputs both a correction and a risk score."
        ),
        r"\paragraph{Proposition 1: boundary information.}",
        (
            "Assume a locally monotone response model $Y=\\mathbb{1}\\{\\langle n,c\\rangle\\geq b\\}$ with unknown normal $n$ and margin $b>0$. "
            "If a correction is observed within $\\epsilon$ of the minimum norm feasible correction, then its normalized direction has angular error bounded by a constant multiple of $\\epsilon/b$ under bounded tangent noise. "
            "Thus minimal corrections are informative about the normal direction when the boundary is locally linear and the human correction is actually minimal."
        ),
        r"\paragraph{Proposition 2: failure under ambiguity and delay.}",
        (
            "If the human correction is drawn from a mixture of two feasible semantic goals or arrives after the boundary has shifted by $\\Delta$, then the minimum-norm correction can identify the wrong boundary. "
            "The induced normal estimate can be biased even when the correction is small. "
            "This is why the benchmark includes ambiguous-intent and delayed-feedback splits, and why the paper cannot use minimum norm as a proof of correctness."
        ),
        r"\paragraph{Proposition 3: fixed-risk acceptance.}",
        (
            "Let $\\hat r(x,u,c)$ be a calibrated upper score for unsafe or damaging correction. "
            "Accepting only episodes with $\\hat r\\leq\\rho$ controls empirical risk only if calibration holds under the deployment split. "
            "The v5 fixed-risk test deliberately uses strict budgets to check whether any method can retain coverage while satisfying the risk constraint. "
            "At budget 0.05, coverage collapses to zero, so this bound is practically vacuous for the present evidence."
        ),
        r"\section{Frozen Protocol}",
        (
            f"The frozen v5 protocol contains {rows_text}. "
            "It uses ten seeds, six tasks, eight correction-shift splits, thirteen main methods, ten ablations, six stress axes, six stress levels, and strict fixed-risk acceptance budgets. "
            "The hard aggregate is fixed before execution as adversarial helpfulness, dynamics mismatch, and combined hard shift."
        ),
        (
            "The terminal decision was predeclared. "
            "A positive synthetic result required v5 to beat the strongest non-oracle hard-aggregate success baseline by at least 0.030, beat the strongest efficiency baseline by at least 0.080, reduce damage, obtain positive paired lower95 bounds, pass mechanism ablations, avoid maximum-stress Pareto domination, and retain fixed-risk coverage at budget 0.05. "
            "Any failure means KILL/ARCHIVE."
        ),
        hard_table(hard_metrics),
        r"\section{Main Evidence}",
        (
            "Table~\\ref{tab:hard} is the core result. "
            "The proposed v5 method is safer than several human-correction baselines, but it is not the strongest method. "
            "Robust MPC has higher task success and correction efficiency. "
            "The oracle is not dramatically above robust MPC, which indicates that the synthetic environment rewards conservative boundary margins more than the new minimum-intervention mechanism."
        ),
        split_table(metrics),
        pairwise_table(hard_pairs),
        r"\begin{figure}[tbp]\centering\includegraphics[width=0.96\linewidth]{minimum_intervention_hard_success_v5.png}\caption{Hard-aggregate success. MIBL v5 trails robust MPC and v4.}\label{fig:hard-success}\end{figure}",
        r"\begin{figure}[tbp]\centering\includegraphics[width=0.88\linewidth]{minimum_intervention_safety_tradeoff_v5.png}\caption{Efficiency, damage, and unsafe override rates on the hard aggregate. V5 improves unsafe overrides but loses success and efficiency.}\label{fig:safety}\end{figure}",
        r"\begin{figure}[tbp]\centering\includegraphics[width=0.82\linewidth]{minimum_intervention_pareto_v5.png}\caption{Success-burden-damage Pareto view. Marker size is damage.}\label{fig:pareto}\end{figure}",
        r"\section{Mechanism Ablations}",
        (
            "The mechanism gate is the most damaging result. "
            "The full v5 method does not dominate its own ablations. "
            "Removing the minimum-norm objective, removing the human-effort cost, removing safety override, and removing query throttling can all improve robust utility or task success in the hard ablation splits. "
            "This does not mean those ablations are deployable; it means the proposed mechanism is not uniquely validated."
        ),
        ablation_table(ablations),
        r"\begin{figure}[tbp]\centering\includegraphics[width=0.96\linewidth]{minimum_intervention_ablation_v5.png}\caption{Ablation robust utility. The full mechanism fails the predeclared necessity margin.}\label{fig:ablation}\end{figure}",
        r"\section{Stress Tests}",
        (
            "The stress gate is the one gate v5 survives. "
            "At maximum combined stress, no non-oracle method Pareto-dominates v5 on success, efficiency, damage, and unsafe overrides. "
            "However, robust MPC still has higher success and utility, and a single stress-gate pass cannot rescue the failed main and mechanism gates."
        ),
        stress_table(stress),
        r"\begin{figure}[tbp]\centering\includegraphics[width=0.90\linewidth]{minimum_intervention_stress_sweep_v5.png}\caption{Combined stress sweep. V5 is not dominated at maximum stress, but robust MPC keeps higher success.}\label{fig:stress}\end{figure}",
        r"\section{Fixed-Risk Deployment}",
        (
            "The fixed-risk protocol is intentionally harsh because deployment claims should be harsh. "
            "A method must decide when it is safe enough to accept autonomous execution under budgets 0.02, 0.05, 0.08, and 0.10. "
            "At budget 0.05, every method has zero accepted coverage on both fixed-risk splits. "
            "This means the calibrated risk scores do not support a nontrivial deployment claim."
        ),
        fixed_table(fixed),
        r"\begin{figure}[tbp]\centering\includegraphics[width=0.90\linewidth]{minimum_intervention_fixed_risk_v5.png}\caption{Fixed-risk coverage on combined hard shift. Strict budget 0.05 gives zero coverage.}\label{fig:fixed}\end{figure}",
        r"\section{Negative Cases}",
        (
            "Negative cases are retained because they prevent a visually polished but brittle paper. "
            "They cover safety conflicts, semantic goal ambiguity, nonlocal deformable dynamics, adversarial helpfulness, delayed boundary shifts, and under-correction. "
            "These cases explain why a small correction is not automatically a correct correction."
        ),
        negative_table(negative),
        r"\section{Prior-Work Boundary}",
        (
            "The local literature pool is used as a hostile-review pressure map. "
            "The paper sits near active correction, residual learning, preference learning, MPC, risk calibration, and robot manipulation datasets "
            f"\\citep{{{cite_keys[6]},{cite_keys[7]},{cite_keys[8]},{cite_keys[9]},{cite_keys[10]},{cite_keys[11]},{cite_keys[12]}}}. "
            "Because this rebuild has no real robot benchmark, it cannot claim to beat those bodies of work directly."
        ),
        prior_work_table(prior_rows, cite_keys),
        r"\section{Reproducibility}",
        (
            "All tables and figures are regenerated by running \\texttt{python src\\textbackslash run\\_experiment.py} from the repository root. "
            "The script streams raw CSVs and aggregates seed-level statistics. "
            "The canonical local PDF is \\texttt{C:/Users/wangz/Downloads/85.pdf}; no Desktop copy is part of the artifact contract."
        ),
        longtable(
            "Artifact & Rows",
            [
                f"{tex_escape(name)} & {count_rows(name)}\\\\"
                for name in [
                    "rollouts.csv",
                    "dataset_summary.csv",
                    "raw_seed_metrics.csv",
                    "metrics.csv",
                    "pairwise_stats.csv",
                    "hard_aggregate_seed_metrics.csv",
                    "hard_aggregate_metrics.csv",
                    "hard_aggregate_pairwise_stats.csv",
                    "ablation_rollouts.csv",
                    "ablation_seed_metrics.csv",
                    "ablation_metrics.csv",
                    "stress_sweep_raw.csv",
                    "stress_sweep_seed_metrics.csv",
                    "stress_sweep.csv",
                    "fixed_risk_raw.csv",
                    "fixed_risk_seed_metrics.csv",
                    "fixed_risk_metrics.csv",
                    "fixed_risk_pairwise.csv",
                    "negative_cases.csv",
                ]
            ],
            "p{0.55\\linewidth}r",
            "Validated row counts for the v5 evidence package.",
            "tab:counts",
        ),
        r"\section{Discussion}",
        (
            "The result is not pretty, but it is useful. "
            "The v5 learner reduced unsafe overrides relative to most baselines, which supports the idea that safety clipping and calibrated risk matter. "
            "But the success and efficiency losses to robust MPC are too large, and the ablations show that the minimum-norm objective is not the causal source of performance in this benchmark."
        ),
        (
            "The fastest path to revival is not another synthetic table. "
            "The paper would need real human correction traces on robot manipulation tasks, or an accepted high-fidelity benchmark where the correction boundary can be externally verified. "
            "It would also need a learned model rather than deterministic proxy components, plus comparisons to modern interactive imitation, preference learning, residual policy correction, and MPC baselines."
        ),
        r"\section{Conclusion}",
        (
            "Minimum-intervention human correction remains a compelling research direction. "
            "This expanded audit makes the paper more honest and much more rigorous, but it does not make it submission-ready. "
            "The correct terminal action is \\textbf{KILL/ARCHIVE} until external robot or high-fidelity evidence and a mechanism that beats strong baselines are available."
        ),
        r"\clearpage",
        r"\appendix",
        r"\section{Appendix: Full Main Evidence}",
        (
            "This appendix reports every split-method aggregate from the frozen main protocol. "
            "The purpose is to make the negative decision auditable: the hard aggregate is not hiding a strong positive result on another predefined split."
        ),
        full_main_table(metrics),
        r"\section{Appendix: Hard-Aggregate Seed Units}",
        (
            "The following seed means are the statistical units used for hard-aggregate paired tests. "
            "They show that the negative paired bounds are not a formatting artifact; the baseline advantage appears across seed-level aggregates."
        ),
        hard_seed_table(hard_seed),
        r"\section{Appendix: Complete Paired Tests}",
        (
            "The table below gives every hard-aggregate paired difference for v5 minus each reference method across the predefined paired metrics. "
            "Primary success and efficiency comparisons fail against robust MPC and v4."
        ),
        full_pairwise_table(hard_pairs),
        r"\section{Appendix: Complete Stress Sweep}",
        (
            "The main text highlights combined stress because it is the most important deployment stressor, but all axes are reported here. "
            "This table includes human noise, correction delay, overcorrection bias, intent ambiguity, helpfulness shift, and combined stress."
        ),
        full_stress_table(stress),
        r"\section{Appendix: Complete Fixed-Risk Table}",
        (
            "The fixed-risk appendix reports every split, budget, and method. "
            "The strict budget 0.05 failure is visible in the full table rather than only in the summary."
        ),
        full_fixed_table(fixed),
        r"\section{Appendix: Fixed-Risk Paired Tests}",
        (
            "The fixed-risk paired table reports coverage and accepted-episode differences across all budgets. "
            "It is included because a deployment reviewer would not accept a single budget-only summary."
        ),
        fixed_pairwise_table(fixed_pairs),
        r"\section{Appendix: Ablation Seed Units}",
        (
            "The mechanism gate was evaluated on seed means from dynamics mismatch and combined hard shift. "
            "The full mechanism does not achieve the required robust-utility margin over all ablations."
        ),
        ablation_seed_table(ablation_seed),
        r"\bibliographystyle{iclr2026_conference}",
        r"\bibliography{references}",
        r"\end{document}",
    ]
    (PAPER / "main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote paper/main.tex and paper/references.bib")


if __name__ == "__main__":
    main()
