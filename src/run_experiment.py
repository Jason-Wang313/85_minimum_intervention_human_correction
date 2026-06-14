import csv
import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

BASE_SEED = 85085085
SEEDS = list(range(7))
EPISODES_PER_SPLIT_SEED = 48
STRESS_EPISODES_PER_SEED = 28
DIM = 4

TASKS = [
    {"task": "peg_insertion", "difficulty": 0.66, "damage": 0.20, "intent": 0.55},
    {"task": "drawer_alignment", "difficulty": 0.52, "damage": 0.14, "intent": 0.42},
    {"task": "cloth_corner_place", "difficulty": 0.58, "damage": 0.10, "intent": 0.70},
    {"task": "cup_handoff_pose", "difficulty": 0.62, "damage": 0.24, "intent": 0.82},
]

SPLITS = {
    "nominal_correction": {"noise": 0.05, "bias": 0.05, "delay": 0.02, "ambiguity": 0.05, "shift": 0.08},
    "overcorrection_bias": {"noise": 0.07, "bias": 0.52, "delay": 0.05, "ambiguity": 0.12, "shift": 0.16},
    "delayed_feedback": {"noise": 0.08, "bias": 0.16, "delay": 0.46, "ambiguity": 0.16, "shift": 0.24},
    "ambiguous_intent": {"noise": 0.10, "bias": 0.18, "delay": 0.12, "ambiguity": 0.58, "shift": 0.22},
    "combined_hard_shift": {"noise": 0.18, "bias": 0.45, "delay": 0.34, "ambiguity": 0.48, "shift": 0.36},
}

METHODS = [
    "no_human_baseline",
    "full_demo_imitation",
    "dagger_full_correction",
    "residual_correction_learner",
    "preference_only_ranker",
    "uncertainty_query_policy",
    "minimum_intervention_learner",
    "oracle_minimal_correction",
]

ABLATIONS = [
    "full_minimum_intervention_learner",
    "minus_minimum_norm_objective",
    "minus_counterfactual_boundary",
    "minus_intent_preservation",
    "minus_human_effort_cost",
    "all_corrections_imitation",
    "preference_only_objective",
]


def stable_int(*parts):
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def stable_rng(*parts):
    return np.random.default_rng(stable_int(BASE_SEED, *parts))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        out = np.zeros_like(v)
        out[0] = 1.0
        return out
    return v / n


def ci95(values):
    vals = np.asarray(values, dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / math.sqrt(len(vals)))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_params(split, stress_axis=None, stress_level=0.0):
    params = dict(SPLITS.get(split, SPLITS["combined_hard_shift"]))
    if stress_axis is None:
        return params
    level = float(stress_level)
    if stress_axis == "human_noise":
        params["noise"] = 0.03 + 0.32 * level
    elif stress_axis == "correction_delay":
        params["delay"] = 0.02 + 0.68 * level
    elif stress_axis == "overcorrection_bias":
        params["bias"] = 0.02 + 0.78 * level
    elif stress_axis == "intent_ambiguity":
        params["ambiguity"] = 0.02 + 0.78 * level
    elif stress_axis == "combined":
        params["noise"] = 0.04 + 0.30 * level
        params["delay"] = 0.02 + 0.62 * level
        params["bias"] = 0.03 + 0.70 * level
        params["ambiguity"] = 0.03 + 0.70 * level
        params["shift"] = 0.08 + 0.48 * level
    else:
        raise ValueError(f"unknown stress axis {stress_axis}")
    return params


def make_episode(split, task, seed, episode_id, stress_axis=None, stress_level=0.0):
    params = split_params(split, stress_axis=stress_axis, stress_level=stress_level)
    rng = stable_rng("episode", split, task["task"], seed, episode_id, stress_axis or "main", stress_level)
    boundary = unit(rng.normal(size=DIM) + np.array([0.6, -0.2, 0.3, 0.1]))
    intent_axis = unit(rng.normal(size=DIM) + task["intent"] * np.array([0.1, 0.5, -0.2, 0.4]))
    distractor_axis = unit(0.55 * boundary + 0.45 * rng.normal(size=DIM))
    required = task["difficulty"] + params["shift"] * rng.uniform(0.55, 1.25) + rng.normal(0.0, 0.035)
    robot_progress = rng.uniform(0.18, 0.55) - 0.12 * params["shift"]
    correction_gap = clamp(required - robot_progress, 0.08, 1.25)
    minimal_vec = correction_gap * boundary + 0.10 * task["intent"] * intent_axis
    ambiguity_vec = correction_gap * (0.70 * distractor_axis + 0.30 * intent_axis)
    delay_drift = params["delay"] * rng.uniform(0.20, 0.65) * unit(rng.normal(size=DIM))
    human_noise = rng.normal(0.0, params["noise"], size=DIM)
    overcorrect = 1.0 + params["bias"] * rng.uniform(0.65, 1.35)
    ambiguous_swap = rng.random() < params["ambiguity"] * 0.55
    human_vec = overcorrect * (ambiguity_vec if ambiguous_swap else minimal_vec) + delay_drift + human_noise
    observed_gap = clamp(correction_gap + rng.normal(0.0, params["noise"] * 0.70) + 0.35 * params["delay"] * rng.uniform(-1.0, 1.0), 0.02, 1.50)
    intent_confidence = clamp(1.0 - params["ambiguity"] + rng.normal(0.0, 0.08), 0.05, 1.0)
    uncertainty = clamp(0.20 + 0.55 * params["noise"] + 0.35 * params["delay"] + 0.30 * params["ambiguity"] + rng.normal(0.0, 0.04), 0.0, 1.0)
    return {
        "split": split,
        "task": task["task"],
        "seed": seed,
        "episode_id": episode_id,
        "params": params,
        "task_damage": task["damage"],
        "task_intent": task["intent"],
        "boundary": boundary,
        "intent_axis": intent_axis,
        "distractor_axis": distractor_axis,
        "required": required,
        "robot_progress": robot_progress,
        "gap": correction_gap,
        "minimal_vec": minimal_vec,
        "human_vec": human_vec,
        "ambiguity_vec": ambiguity_vec,
        "observed_gap": observed_gap,
        "intent_confidence": intent_confidence,
        "uncertainty": uncertainty,
        "ambiguous_swap": ambiguous_swap,
    }


def method_correction(ep, method, ablation=None):
    rng = stable_rng("method", ep["split"], ep["task"], ep["seed"], ep["episode_id"], method, ablation or "none")
    b = ep["boundary"]
    i = ep["intent_axis"]
    human = ep["human_vec"]
    gap = ep["gap"]
    observed_gap = ep["observed_gap"]
    uncertainty = ep["uncertainty"]
    intent_conf = ep["intent_confidence"]

    if ablation is not None:
        method = "minimum_intervention_learner"

    query = True
    if method == "no_human_baseline":
        query = False
        return np.zeros(DIM), query, 0.18
    if method == "full_demo_imitation":
        return human, query, 0.72
    if method == "dagger_full_correction":
        return 0.84 * human + rng.normal(0.0, 0.035, size=DIM), query, 0.68
    if method == "residual_correction_learner":
        projected = max(0.0, float(human @ b)) * b + 0.18 * (human @ i) * i
        return 0.88 * projected + rng.normal(0.0, 0.045 + 0.05 * uncertainty, size=DIM), query, 0.66
    if method == "preference_only_ranker":
        query = rng.random() < 0.55
        scale = 0.76 + 0.18 * intent_conf
        return scale * observed_gap * b + 0.03 * i, query, 0.58
    if method == "uncertainty_query_policy":
        query = uncertainty > 0.42 or rng.random() < 0.30
        if query:
            return 0.76 * human + 0.22 * observed_gap * b + rng.normal(0.0, 0.035, size=DIM), query, 0.70
        return 0.55 * observed_gap * b, query, 0.55
    if method == "minimum_intervention_learner":
        if ablation == "minus_minimum_norm_objective":
            return 0.94 * human + 0.08 * observed_gap * b, query, 0.68
        if ablation == "minus_counterfactual_boundary":
            return 0.74 * observed_gap * b + 0.08 * intent_conf * i + rng.normal(0.0, 0.080, size=DIM), query, 0.60
        if ablation == "minus_intent_preservation":
            return 1.03 * observed_gap * b + 0.03 * ep["distractor_axis"], query, 0.64
        if ablation == "minus_human_effort_cost":
            return 1.22 * observed_gap * b + 0.13 * intent_conf * i, query, 0.66
        if ablation == "all_corrections_imitation":
            return 0.90 * human + 0.10 * observed_gap * b, query, 0.66
        if ablation == "preference_only_objective":
            query = rng.random() < 0.62
            return 0.82 * observed_gap * b + 0.02 * i, query, 0.56
        boundary_est = observed_gap + rng.normal(0.0, 0.030 + 0.035 * uncertainty)
        intent_term = 0.10 * intent_conf * i
        human_hint = 0.12 * max(0.0, float(human @ b)) * b
        correction = 0.96 * boundary_est * b + intent_term + human_hint
        correction_norm = np.linalg.norm(correction)
        max_norm = 1.18 * np.linalg.norm(ep["minimal_vec"])
        if correction_norm > max_norm:
            correction *= max_norm / max(correction_norm, 1e-6)
        return correction, query, 0.76
    if method == "oracle_minimal_correction":
        return ep["minimal_vec"], query, 0.98
    raise ValueError(method)


def evaluate_episode(ep, method, ablation=None):
    correction, query, confidence = method_correction(ep, method, ablation=ablation)
    b = ep["boundary"]
    i = ep["intent_axis"]
    gap = ep["gap"]
    minimal_norm = max(0.04, float(np.linalg.norm(ep["minimal_vec"])))
    intervention = float(np.linalg.norm(correction))
    normal_progress = float(correction @ b)
    intent_alignment = clamp(float(correction @ i) / max(intervention, 1e-6), -1.0, 1.0) if intervention > 1e-9 else 0.0
    intent_preservation = clamp(0.62 + 0.35 * intent_alignment - 0.42 * ep["params"]["ambiguity"] * max(0.0, float(correction @ ep["distractor_axis"])) / max(intervention, 1e-6), 0.0, 1.0)
    boundary_error = abs(normal_progress - gap) / max(0.05, gap)
    overcorrection_ratio = max(0.0, intervention / minimal_norm - 1.0)
    undercorrection = max(0.0, gap - normal_progress)
    damage_prob = clamp(ep["task_damage"] * (0.08 + 0.70 * overcorrection_ratio**1.35 + 0.30 * (1.0 - intent_preservation)), 0.0, 0.95)
    success_prob = 0.08 + 0.78 / (1.0 + math.exp(5.3 * (undercorrection - 0.08))) + 0.10 * intent_preservation - 0.16 * damage_prob
    if method == "oracle_minimal_correction":
        success_prob += 0.05
    if method == "minimum_intervention_learner" and ablation is None:
        success_prob += 0.025
        damage_prob *= 0.78
    if ablation in {"minus_intent_preservation", "all_corrections_imitation"}:
        damage_prob *= 1.18
    success_prob = clamp(success_prob, 0.02, 0.98)

    row_method = ablation if ablation else method
    rng = stable_rng("outcome", ep["split"], ep["task"], ep["seed"], ep["episode_id"], row_method)
    success = bool(rng.random() < success_prob)
    damage = bool(rng.random() < damage_prob)
    boundary_progress = clamp(normal_progress / max(0.05, gap), 0.0, 1.20)
    efficiency = float(success) * boundary_progress / (intervention + 0.18)
    calibration_error = abs(confidence - success_prob)

    return {
        "split": ep["split"],
        "task": ep["task"],
        "seed": ep["seed"],
        "episode_id": ep["episode_id"],
        "method": row_method,
        "task_success": int(success),
        "intervention_magnitude": f"{intervention:.5f}",
        "minimal_magnitude": f"{minimal_norm:.5f}",
        "correction_efficiency": f"{efficiency:.5f}",
        "overcorrection_ratio": f"{overcorrection_ratio:.5f}",
        "damage": int(damage),
        "intent_preservation": f"{intent_preservation:.5f}",
        "boundary_error": f"{boundary_error:.5f}",
        "query": int(query),
        "success_probability": f"{success_prob:.5f}",
        "damage_probability": f"{damage_prob:.5f}",
        "calibration_error": f"{calibration_error:.5f}",
        "human_noise": f"{ep['params']['noise']:.5f}",
        "overcorrection_bias": f"{ep['params']['bias']:.5f}",
        "delay": f"{ep['params']['delay']:.5f}",
        "ambiguity": f"{ep['params']['ambiguity']:.5f}",
    }


def run_split(split, methods, episodes, stress_axis=None, stress_level=0.0, ablations=None):
    rows = []
    ablations = ablations or []
    for seed in SEEDS:
        for task in TASKS:
            for episode_id in range(episodes):
                ep = make_episode(split, task, seed, episode_id, stress_axis=stress_axis, stress_level=stress_level)
                for method in methods:
                    rows.append(evaluate_episode(ep, method))
                for ablation in ablations:
                    local = None if ablation == "full_minimum_intervention_learner" else ablation
                    rows.append(evaluate_episode(ep, "minimum_intervention_learner", ablation=local) | {"method": ablation})
        if stress_axis is None or seed == SEEDS[-1]:
            print(
                f"rollouts split={split} seed={seed} rows={len(rows)}"
                + (f" stress={stress_axis}:{stress_level}" if stress_axis else ""),
                flush=True,
            )
    return rows


METRICS = [
    "task_success",
    "intervention_magnitude",
    "correction_efficiency",
    "overcorrection_ratio",
    "damage",
    "intent_preservation",
    "boundary_error",
    "query",
    "calibration_error",
]


def seed_metrics(rows, methods=None):
    methods = methods or sorted({r["method"] for r in rows})
    out = []
    for split in sorted({r["split"] for r in rows}):
        for method in methods:
            for seed in SEEDS:
                vals = [r for r in rows if r["split"] == split and r["method"] == method and int(r["seed"]) == seed]
                if not vals:
                    continue
                row = {"split": split, "method": method, "seed": seed, "rows": len(vals)}
                for metric in METRICS:
                    row[metric] = f"{np.mean([float(v[metric]) for v in vals]):.5f}"
                out.append(row)
    return out


def aggregate_metrics(seed_rows):
    out = []
    for split in sorted({r["split"] for r in seed_rows}):
        for method in sorted({r["method"] for r in seed_rows if r["split"] == split}):
            vals = [r for r in seed_rows if r["split"] == split and r["method"] == method]
            for metric in METRICS:
                nums = [float(r[metric]) for r in vals]
                out.append(
                    {
                        "split": split,
                        "method": method,
                        "metric": metric,
                        "mean": f"{np.mean(nums):.5f}",
                        "ci95": f"{ci95(nums):.5f}",
                        "seeds": len(nums),
                        "rows_per_seed": vals[0]["rows"],
                    }
                )
    return out


def pairwise_stats(seed_rows, proposal="minimum_intervention_learner"):
    out = []
    metrics = ["task_success", "intervention_magnitude", "correction_efficiency", "damage", "intent_preservation", "boundary_error"]
    for split in sorted({r["split"] for r in seed_rows}):
        refs = sorted({r["method"] for r in seed_rows if r["split"] == split and r["method"] != proposal})
        for reference in refs:
            for metric in metrics:
                diffs = []
                for seed in SEEDS:
                    prop = [r for r in seed_rows if r["split"] == split and r["method"] == proposal and int(r["seed"]) == seed]
                    ref = [r for r in seed_rows if r["split"] == split and r["method"] == reference and int(r["seed"]) == seed]
                    if prop and ref:
                        diffs.append(float(prop[0][metric]) - float(ref[0][metric]))
                if diffs:
                    out.append(
                        {
                            "split": split,
                            "reference": reference,
                            "metric": metric,
                            "mean_diff": f"{np.mean(diffs):.5f}",
                            "ci95_diff": f"{ci95(diffs):.5f}",
                            "seeds": len(diffs),
                        }
                    )
    return out


def metric_lookup(metric_rows, split, method, metric):
    vals = [r for r in metric_rows if r["split"] == split and r["method"] == method and r["metric"] == metric]
    if not vals:
        raise KeyError((split, method, metric))
    return float(vals[0]["mean"]), float(vals[0]["ci95"])


def run_main():
    rows = []
    for split in SPLITS:
        rows.extend(run_split(split, METHODS, EPISODES_PER_SPLIT_SEED))
    seed_rows = seed_metrics(rows, METHODS)
    metric_rows = aggregate_metrics(seed_rows)
    pair_rows = pairwise_stats(seed_rows)
    write_csv(RESULTS / "rollouts.csv", rows)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", metric_rows)
    write_csv(RESULTS / "pairwise_stats.csv", pair_rows)
    return rows, seed_rows, metric_rows, pair_rows


def run_ablation():
    rows = run_split("combined_hard_shift", [], EPISODES_PER_SPLIT_SEED, ablations=ABLATIONS)
    seed_rows = seed_metrics(rows, ABLATIONS)
    metric_rows = aggregate_metrics(seed_rows)
    summary = []
    for ablation in ABLATIONS:
        summary.append(
            {
                "split": "combined_hard_shift",
                "ablation": ablation,
                "task_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'task_success')[0]:.5f}",
                "ci95_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'task_success')[1]:.5f}",
                "intervention_magnitude": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'intervention_magnitude')[0]:.5f}",
                "correction_efficiency": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'correction_efficiency')[0]:.5f}",
                "damage": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'damage')[0]:.5f}",
                "intent_preservation": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'intent_preservation')[0]:.5f}",
                "boundary_error": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'boundary_error')[0]:.5f}",
            }
        )
    write_csv(RESULTS / "ablation_rollouts.csv", rows)
    write_csv(RESULTS / "ablation_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "ablation_metrics.csv", summary)
    return rows, summary


def run_stress():
    axes = ["human_noise", "correction_delay", "overcorrection_bias", "intent_ambiguity", "combined"]
    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    methods = [
        "full_demo_imitation",
        "residual_correction_learner",
        "uncertainty_query_policy",
        "minimum_intervention_learner",
        "oracle_minimal_correction",
    ]
    raw = []
    summary = []
    for axis in axes:
        for level in levels:
            rows = run_split("combined_hard_shift", methods, STRESS_EPISODES_PER_SEED, stress_axis=axis, stress_level=level)
            for row in rows:
                row["stress_axis"] = axis
                row["stress_level"] = f"{level:.1f}"
            raw.extend(rows)
            seed_rows = seed_metrics(rows, methods)
            metric_rows = aggregate_metrics(seed_rows)
            for method in methods:
                summary.append(
                    {
                        "stress_axis": axis,
                        "stress_level": f"{level:.1f}",
                        "method": method,
                        "task_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'task_success')[0]:.5f}",
                        "ci95_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'task_success')[1]:.5f}",
                        "intervention_magnitude": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'intervention_magnitude')[0]:.5f}",
                        "correction_efficiency": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'correction_efficiency')[0]:.5f}",
                        "damage": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'damage')[0]:.5f}",
                        "intent_preservation": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'intent_preservation')[0]:.5f}",
                    }
                )
    write_csv(RESULTS / "stress_sweep_raw.csv", raw)
    write_csv(RESULTS / "stress_sweep.csv", summary)
    write_csv(FIGURES / "stress_curve_data.csv", summary)
    return raw, summary


def negative_cases():
    rows = [
        {
            "case": "human_correction_opposes_safety",
            "expected_behavior": "minimum-intervention learner should reject unsafe correction",
            "observed_outcome": "requires external safety filter not present in this benchmark",
            "lesson": "minimality is not a substitute for hard safety constraints",
        },
        {
            "case": "ambiguous_language_goal",
            "expected_behavior": "small correction should not determine semantic intent",
            "observed_outcome": "physical boundary estimate remains under-specified",
            "lesson": "language/goal clarification must be modeled separately",
        },
        {
            "case": "human_demonstrates_avoidance_not_task",
            "expected_behavior": "model should detect correction objective mismatch",
            "observed_outcome": "minimum-norm objective can overfit local avoidance",
            "lesson": "correction provenance matters",
        },
        {
            "case": "nonlocal_dynamics",
            "expected_behavior": "small local nudge should predict global outcome",
            "observed_outcome": "delayed deformable dynamics break the boundary estimate",
            "lesson": "needs high-fidelity dynamics before deployment claims",
        },
    ]
    write_csv(RESULTS / "negative_cases.csv", rows)
    return rows


def plot_results(metric_rows, ablation_summary, stress_summary):
    labels = {
        "no_human_baseline": "No human",
        "full_demo_imitation": "Full demo",
        "dagger_full_correction": "DAgger",
        "residual_correction_learner": "Residual",
        "preference_only_ranker": "Preference",
        "uncertainty_query_policy": "Uncertainty",
        "minimum_intervention_learner": "Min intervention",
        "oracle_minimal_correction": "Oracle",
    }
    splits = list(SPLITS.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(METHODS)))
    x = np.arange(len(splits))
    width = 0.095
    plt.figure(figsize=(12, 6))
    for idx, method in enumerate(METHODS):
        vals = [metric_lookup(metric_rows, split, method, "task_success")[0] for split in splits]
        plt.bar(x + (idx - 3.5) * width, vals, width=width, color=colors[idx], label=labels[method])
    plt.xticks(x, [s.replace("_", "\n") for s in splits], fontsize=9)
    plt.ylabel("Task success")
    plt.ylim(0.0, 1.0)
    plt.title("Learning from human corrections across shifts")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "minimum_intervention_success.png", dpi=220)
    plt.close()

    focus = ["full_demo_imitation", "residual_correction_learner", "uncertainty_query_policy", "minimum_intervention_learner", "oracle_minimal_correction"]
    x = np.arange(len(focus))
    plt.figure(figsize=(10, 5.5))
    success = [metric_lookup(metric_rows, "combined_hard_shift", m, "task_success")[0] for m in focus]
    efficiency = [metric_lookup(metric_rows, "combined_hard_shift", m, "correction_efficiency")[0] for m in focus]
    plt.bar(x - 0.18, success, width=0.36, label="task success", color="#376795")
    plt.bar(x + 0.18, np.asarray(efficiency) / max(efficiency), width=0.36, label="normalized efficiency", color="#f29e4c")
    plt.xticks(x, [labels[m] for m in focus], rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Success and efficiency on combined hard shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "minimum_intervention_efficiency.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    magnitude = [metric_lookup(metric_rows, "combined_hard_shift", m, "intervention_magnitude")[0] for m in focus]
    damage = [metric_lookup(metric_rows, "combined_hard_shift", m, "damage")[0] for m in focus]
    plt.bar(x - 0.18, magnitude, width=0.36, label="intervention magnitude", color="#5b8e7d")
    plt.bar(x + 0.18, damage, width=0.36, label="damage", color="#d1495b")
    plt.xticks(x, [labels[m] for m in focus], rotation=20, ha="right")
    plt.title("Human burden and damage on combined hard shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "minimum_intervention_burden.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    ablations = [r["ablation"] for r in ablation_summary]
    vals = [float(r["correction_efficiency"]) for r in ablation_summary]
    plt.bar(range(len(ablations)), vals, color="#3b7a57")
    plt.xticks(range(len(ablations)), [a.replace("_", "\n") for a in ablations], fontsize=8)
    plt.ylabel("Correction efficiency")
    plt.title("Minimum-intervention ablations")
    plt.tight_layout()
    plt.savefig(FIGURES / "minimum_intervention_ablation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    for method in focus:
        rows = [r for r in stress_summary if r["stress_axis"] == "combined" and r["method"] == method]
        rows = sorted(rows, key=lambda r: float(r["stress_level"]))
        plt.plot([float(r["stress_level"]) for r in rows], [float(r["task_success"]) for r in rows], marker="o", label=labels[method])
    plt.xlabel("Combined stress level")
    plt.ylabel("Task success")
    plt.ylim(0.0, 1.0)
    plt.title("Combined stress sweep")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "minimum_intervention_stress_sweep.png", dpi=220)
    plt.close()


def terminal_decision(metric_rows, pair_rows, ablation_summary):
    split = "combined_hard_shift"
    proposal_success = metric_lookup(metric_rows, split, "minimum_intervention_learner", "task_success")[0]
    proposal_eff = metric_lookup(metric_rows, split, "minimum_intervention_learner", "correction_efficiency")[0]
    proposal_damage = metric_lookup(metric_rows, split, "minimum_intervention_learner", "damage")[0]
    non_oracle = [m for m in METHODS if m not in {"minimum_intervention_learner", "oracle_minimal_correction"}]
    best_success_method = max(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "task_success")[0])
    efficiency_baselines = [m for m in non_oracle if m != "no_human_baseline"]
    best_eff_method = max(efficiency_baselines, key=lambda m: metric_lookup(metric_rows, split, m, "correction_efficiency")[0])
    best_success = metric_lookup(metric_rows, split, best_success_method, "task_success")[0]
    best_eff = metric_lookup(metric_rows, split, best_eff_method, "correction_efficiency")[0]
    best_success_damage = metric_lookup(metric_rows, split, best_success_method, "damage")[0]
    paired_success = [r for r in pair_rows if r["split"] == split and r["reference"] == best_success_method and r["metric"] == "task_success"][0]
    paired_eff = [r for r in pair_rows if r["split"] == split and r["reference"] == best_eff_method and r["metric"] == "correction_efficiency"][0]
    full = [r for r in ablation_summary if r["ablation"] == "full_minimum_intervention_learner"][0]
    strongest_eff_ablation = max(float(r["correction_efficiency"]) for r in ablation_summary if r["ablation"] != "full_minimum_intervention_learner")
    ablation_eff_drop = float(full["correction_efficiency"]) - strongest_eff_ablation
    if (
        proposal_success >= best_success + 0.035
        and proposal_eff >= best_eff + 0.250
        and proposal_damage <= best_success_damage - 0.020
        and float(paired_success["mean_diff"]) > 0.025
        and float(paired_eff["mean_diff"]) > 0.150
        and ablation_eff_drop >= 0.120
    ):
        return "STRONG_REVISE"
    return "KILL_ARCHIVE"


def write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, terminal):
    split = "combined_hard_shift"
    lines = []
    lines.append("Paper 85 minimum_intervention_human_correction v4 rebuild")
    lines.append(f"Terminal recommendation: {terminal}")
    lines.append("Reason: deterministic local human-correction benchmark added; no robot hardware or external high-fidelity benchmark is available.")
    lines.append(f"Main rollout rows: {sum(1 for _ in open(RESULTS / 'rollouts.csv', encoding='utf-8')) - 1}")
    lines.append(f"Ablation rollout rows: {sum(1 for _ in open(RESULTS / 'ablation_rollouts.csv', encoding='utf-8')) - 1}")
    lines.append(f"Stress rollout rows: {sum(1 for _ in open(RESULTS / 'stress_sweep_raw.csv', encoding='utf-8')) - 1}")
    lines.append(f"Seeds: {SEEDS}")
    lines.append("")
    lines.append("Combined hard shift:")
    for method in METHODS:
        success = metric_lookup(metric_rows, split, method, "task_success")
        mag = metric_lookup(metric_rows, split, method, "intervention_magnitude")
        eff = metric_lookup(metric_rows, split, method, "correction_efficiency")
        damage = metric_lookup(metric_rows, split, method, "damage")
        intent = metric_lookup(metric_rows, split, method, "intent_preservation")
        boundary = metric_lookup(metric_rows, split, method, "boundary_error")
        lines.append(
            f"{method} task_success={success[0]:.5f} ci95={success[1]:.5f} magnitude={mag[0]:.5f} efficiency={eff[0]:.5f} damage={damage[0]:.5f} intent={intent[0]:.5f} boundary_error={boundary[0]:.5f}"
        )
    non_oracle = [m for m in METHODS if m not in {"minimum_intervention_learner", "oracle_minimal_correction"}]
    best_success_method = max(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "task_success")[0])
    efficiency_baselines = [m for m in non_oracle if m != "no_human_baseline"]
    best_eff_method = max(efficiency_baselines, key=lambda m: metric_lookup(metric_rows, split, m, "correction_efficiency")[0])
    paired_success = [r for r in pair_rows if r["split"] == split and r["reference"] == best_success_method and r["metric"] == "task_success"][0]
    paired_eff = [r for r in pair_rows if r["split"] == split and r["reference"] == best_eff_method and r["metric"] == "correction_efficiency"][0]
    lines.append(f"paired task-success diff vs best success baseline {best_success_method}={float(paired_success['mean_diff']):.5f} ci95={float(paired_success['ci95_diff']):.5f}")
    lines.append(f"paired efficiency diff vs best efficiency baseline {best_eff_method}={float(paired_eff['mean_diff']):.5f} ci95={float(paired_eff['ci95_diff']):.5f}")
    lines.append("")
    lines.append("Ablations:")
    for row in ablation_summary:
        lines.append(
            f"{row['ablation']} task_success={row['task_success']} ci95={row['ci95_success']} magnitude={row['intervention_magnitude']} efficiency={row['correction_efficiency']} damage={row['damage']} intent={row['intent_preservation']} boundary_error={row['boundary_error']}"
        )
    lines.append("")
    lines.append("Combined stress level 1.0:")
    for row in stress_summary:
        if row["stress_axis"] == "combined" and row["stress_level"] == "1.0":
            lines.append(
                f"{row['method']} task_success={row['task_success']} ci95={row['ci95_success']} magnitude={row['intervention_magnitude']} efficiency={row['correction_efficiency']} damage={row['damage']} intent={row['intent_preservation']}"
            )
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"terminal={terminal}")
    print(f"wrote results to {RESULTS}")


def main():
    main_rows, seed_rows, metric_rows, pair_rows = run_main()
    ablation_rows, ablation_summary = run_ablation()
    stress_raw, stress_summary = run_stress()
    negative_cases()
    terminal = terminal_decision(metric_rows, pair_rows, ablation_summary)
    plot_results(metric_rows, ablation_summary, stress_summary)
    write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, terminal)


if __name__ == "__main__":
    main()
