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
SEEDS = list(range(10))
DIM = 5

MAIN_EPISODES = 32
ABLATION_EPISODES = 28
STRESS_EPISODES = 20
FIXED_RISK_EPISODES = 24

TASKS = [
    {"task": "peg_insertion", "difficulty": 0.66, "damage": 0.22, "intent": 0.55, "deformable": 0.10},
    {"task": "drawer_alignment", "difficulty": 0.52, "damage": 0.15, "intent": 0.42, "deformable": 0.08},
    {"task": "cloth_corner_place", "difficulty": 0.60, "damage": 0.12, "intent": 0.72, "deformable": 0.46},
    {"task": "cup_handoff_pose", "difficulty": 0.64, "damage": 0.26, "intent": 0.82, "deformable": 0.16},
    {"task": "cable_hook_alignment", "difficulty": 0.70, "damage": 0.18, "intent": 0.66, "deformable": 0.55},
    {"task": "tool_handover_alignment", "difficulty": 0.58, "damage": 0.24, "intent": 0.78, "deformable": 0.18},
]

SPLITS = {
    "nominal_correction": {
        "noise": 0.05,
        "bias": 0.06,
        "delay": 0.03,
        "ambiguity": 0.06,
        "helpfulness": 0.98,
        "dynamics": 0.08,
        "sparse": 0.06,
    },
    "overcorrection_bias": {
        "noise": 0.07,
        "bias": 0.55,
        "delay": 0.06,
        "ambiguity": 0.14,
        "helpfulness": 0.92,
        "dynamics": 0.16,
        "sparse": 0.08,
    },
    "delayed_feedback": {
        "noise": 0.08,
        "bias": 0.18,
        "delay": 0.52,
        "ambiguity": 0.18,
        "helpfulness": 0.88,
        "dynamics": 0.24,
        "sparse": 0.10,
    },
    "ambiguous_intent": {
        "noise": 0.10,
        "bias": 0.20,
        "delay": 0.14,
        "ambiguity": 0.62,
        "helpfulness": 0.84,
        "dynamics": 0.22,
        "sparse": 0.12,
    },
    "sparse_corrections": {
        "noise": 0.10,
        "bias": 0.22,
        "delay": 0.18,
        "ambiguity": 0.22,
        "helpfulness": 0.82,
        "dynamics": 0.24,
        "sparse": 0.58,
    },
    "adversarial_helpfulness": {
        "noise": 0.13,
        "bias": 0.34,
        "delay": 0.22,
        "ambiguity": 0.38,
        "helpfulness": 0.54,
        "dynamics": 0.30,
        "sparse": 0.16,
    },
    "dynamics_mismatch": {
        "noise": 0.13,
        "bias": 0.30,
        "delay": 0.30,
        "ambiguity": 0.34,
        "helpfulness": 0.72,
        "dynamics": 0.58,
        "sparse": 0.18,
    },
    "combined_hard_shift": {
        "noise": 0.18,
        "bias": 0.48,
        "delay": 0.38,
        "ambiguity": 0.52,
        "helpfulness": 0.62,
        "dynamics": 0.52,
        "sparse": 0.26,
    },
}

HARD_SPLITS = ["adversarial_helpfulness", "dynamics_mismatch", "combined_hard_shift"]

METHODS = [
    "no_human_baseline",
    "full_demo_imitation",
    "dagger_full_correction",
    "residual_correction_learner",
    "preference_only_ranker",
    "uncertainty_query_policy",
    "active_entropy_query_policy",
    "safety_filtered_residual",
    "robust_mpc_correction",
    "inverse_rl_correction_proxy",
    "minimum_intervention_learner_v4",
    "minimum_intervention_boundary_learner_v5",
    "oracle_minimal_correction",
]

ABLATIONS = [
    "full_minimum_intervention_boundary_learner_v5",
    "minus_minimum_norm_objective",
    "minus_counterfactual_boundary",
    "minus_intent_preservation",
    "minus_human_effort_cost",
    "minus_safety_override",
    "minus_query_throttling",
    "minus_calibration",
    "all_corrections_imitation",
    "preference_only_objective",
]

STRESS_AXES = [
    "human_noise",
    "correction_delay",
    "overcorrection_bias",
    "intent_ambiguity",
    "helpfulness_shift",
    "combined",
]
STRESS_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
STRESS_METHODS = [
    "full_demo_imitation",
    "residual_correction_learner",
    "uncertainty_query_policy",
    "active_entropy_query_policy",
    "robust_mpc_correction",
    "minimum_intervention_boundary_learner_v5",
    "oracle_minimal_correction",
]
FIXED_RISK_SPLITS = ["dynamics_mismatch", "combined_hard_shift"]
FIXED_RISK_BUDGETS = [0.02, 0.05, 0.08, 0.10]
FIXED_RISK_METHODS = [
    "uncertainty_query_policy",
    "active_entropy_query_policy",
    "safety_filtered_residual",
    "robust_mpc_correction",
    "minimum_intervention_boundary_learner_v5",
    "oracle_minimal_correction",
]

METRICS = [
    "task_success",
    "correction_efficiency",
    "damage",
    "intervention_magnitude",
    "boundary_error",
    "intent_preservation",
    "query_rate",
    "human_time",
    "unsafe_override",
    "calibration_error",
    "regret_to_oracle",
    "intervention_sparsity",
    "robust_utility",
]

PAIRWISE_METRICS = [
    "task_success",
    "correction_efficiency",
    "damage",
    "intervention_magnitude",
    "boundary_error",
    "intent_preservation",
    "unsafe_override",
    "robust_utility",
]

ROLLOUT_FIELDS = [
    "split",
    "task",
    "seed",
    "episode_id",
    "method",
    "task_success",
    "correction_efficiency",
    "damage",
    "intervention_magnitude",
    "minimal_magnitude",
    "boundary_error",
    "intent_preservation",
    "query_rate",
    "human_time",
    "unsafe_override",
    "calibration_error",
    "regret_to_oracle",
    "intervention_sparsity",
    "robust_utility",
    "risk_score",
    "success_probability",
    "damage_probability",
    "unsafe_probability",
    "human_noise",
    "overcorrection_bias",
    "delay",
    "ambiguity",
    "helpfulness",
    "dynamics_shift",
    "sparse_rate",
]

SCENE_FIELDS = [
    "split",
    "task",
    "seed",
    "episode_id",
    "gap",
    "minimal_magnitude",
    "human_magnitude",
    "observed_gap",
    "noise",
    "bias",
    "delay",
    "ambiguity",
    "helpfulness",
    "dynamics",
    "sparse",
    "intent_confidence",
    "uncertainty",
    "latent_safety_risk",
    "ambiguous_swap",
    "sparse_event",
]


def stable_int(*parts):
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def stable_rng(*parts):
    return np.random.default_rng(stable_int(BASE_SEED, *parts))


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        out = np.zeros_like(v)
        out[0] = 1.0
        return out
    return v / n


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


def ci95(values):
    vals = np.asarray(values, dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / math.sqrt(len(vals)))


def fmt(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, float):
        return f"{v:.5f}"
    if isinstance(v, np.floating):
        return f"{float(v):.5f}"
    return v


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dict_writer(path, fields):
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    return f, writer


def split_params(split, stress_axis=None, stress_level=0.0):
    params = dict(SPLITS.get(split, SPLITS["combined_hard_shift"]))
    if stress_axis is None:
        return params
    level = float(stress_level)
    if stress_axis == "human_noise":
        params["noise"] = 0.04 + 0.34 * level
    elif stress_axis == "correction_delay":
        params["delay"] = 0.02 + 0.72 * level
    elif stress_axis == "overcorrection_bias":
        params["bias"] = 0.03 + 0.82 * level
    elif stress_axis == "intent_ambiguity":
        params["ambiguity"] = 0.03 + 0.82 * level
    elif stress_axis == "helpfulness_shift":
        params["helpfulness"] = 1.00 - 0.68 * level
        params["ambiguity"] = max(params["ambiguity"], 0.18 + 0.42 * level)
    elif stress_axis == "combined":
        params["noise"] = 0.04 + 0.32 * level
        params["delay"] = 0.02 + 0.66 * level
        params["bias"] = 0.03 + 0.76 * level
        params["ambiguity"] = 0.04 + 0.74 * level
        params["helpfulness"] = 0.98 - 0.55 * level
        params["dynamics"] = 0.08 + 0.58 * level
        params["sparse"] = 0.08 + 0.34 * level
    else:
        raise ValueError(f"unknown stress axis {stress_axis}")
    return params


def make_episode(split, task, seed, episode_id, stress_axis=None, stress_level=0.0):
    params = split_params(split, stress_axis=stress_axis, stress_level=stress_level)
    rng = stable_rng("episode", split, task["task"], seed, episode_id, stress_axis or "main", f"{stress_level:.2f}")

    boundary = unit(rng.normal(size=DIM) + np.array([0.58, -0.18, 0.28, 0.09, 0.22]))
    intent_axis = unit(rng.normal(size=DIM) + task["intent"] * np.array([0.08, 0.48, -0.18, 0.36, 0.16]))
    danger_axis = unit(rng.normal(size=DIM) + np.array([-0.20, 0.14, 0.18, 0.62, -0.10]))
    distractor_axis = unit(0.48 * boundary + 0.36 * intent_axis + 0.30 * rng.normal(size=DIM))

    difficulty = task["difficulty"] + 0.12 * task["deformable"] * params["dynamics"]
    required = difficulty + params["dynamics"] * rng.uniform(0.20, 0.55) + rng.normal(0.0, 0.035)
    robot_progress = rng.uniform(0.18, 0.56) - 0.13 * params["dynamics"] - 0.05 * task["deformable"]
    gap = clamp(required - robot_progress, 0.08, 1.35)
    minimal_vec = gap * boundary + 0.10 * task["intent"] * intent_axis - 0.025 * danger_axis
    ambiguity_vec = gap * (0.58 * distractor_axis + 0.32 * intent_axis + 0.10 * boundary)

    ambiguous_swap = rng.random() < params["ambiguity"] * 0.58
    sparse_event = rng.random() < params["sparse"]
    helpfulness = params["helpfulness"] - 0.20 * ambiguous_swap - 0.08 * task["deformable"] * params["dynamics"]
    if split == "adversarial_helpfulness" and rng.random() < 0.18:
        helpfulness -= rng.uniform(0.30, 0.62)

    delay_drift = params["delay"] * rng.uniform(0.20, 0.72) * unit(rng.normal(size=DIM))
    human_noise = rng.normal(0.0, params["noise"], size=DIM)
    overcorrect = 1.0 + params["bias"] * rng.uniform(0.55, 1.45)
    base_human = ambiguity_vec if ambiguous_swap else minimal_vec
    human_vec = helpfulness * overcorrect * base_human + delay_drift + human_noise
    if sparse_event:
        human_vec = 0.35 * human_vec + rng.normal(0.0, params["noise"] * 0.45, size=DIM)

    observed_gap = clamp(
        gap
        + rng.normal(0.0, params["noise"] * 0.75)
        + 0.34 * params["delay"] * rng.uniform(-1.0, 1.0)
        + 0.20 * params["dynamics"] * rng.uniform(-0.3, 0.7),
        0.02,
        1.60,
    )
    intent_confidence = clamp(1.0 - params["ambiguity"] - 0.12 * sparse_event + rng.normal(0.0, 0.075), 0.04, 1.0)
    uncertainty = clamp(
        0.18
        + 0.56 * params["noise"]
        + 0.36 * params["delay"]
        + 0.32 * params["ambiguity"]
        + 0.22 * params["dynamics"]
        + 0.12 * sparse_event
        + rng.normal(0.0, 0.04),
        0.0,
        1.0,
    )
    latent_safety_risk = clamp(
        0.06
        + task["damage"] * 0.38
        + 0.30 * params["bias"]
        + 0.22 * params["dynamics"]
        + 0.16 * max(0.0, 1.0 - helpfulness)
        + rng.normal(0.0, 0.035),
        0.0,
        0.95,
    )

    return {
        "split": split,
        "task": task["task"],
        "seed": seed,
        "episode_id": episode_id,
        "params": params,
        "task_damage": task["damage"],
        "task_intent": task["intent"],
        "task_deformable": task["deformable"],
        "boundary": boundary,
        "intent_axis": intent_axis,
        "danger_axis": danger_axis,
        "distractor_axis": distractor_axis,
        "gap": gap,
        "minimal_vec": minimal_vec,
        "human_vec": human_vec,
        "ambiguity_vec": ambiguity_vec,
        "observed_gap": observed_gap,
        "intent_confidence": intent_confidence,
        "uncertainty": uncertainty,
        "latent_safety_risk": latent_safety_risk,
        "ambiguous_swap": int(ambiguous_swap),
        "sparse_event": int(sparse_event),
        "helpfulness": helpfulness,
    }


def scene_row(ep):
    params = ep["params"]
    return {
        "split": ep["split"],
        "task": ep["task"],
        "seed": ep["seed"],
        "episode_id": ep["episode_id"],
        "gap": fmt(ep["gap"]),
        "minimal_magnitude": fmt(float(np.linalg.norm(ep["minimal_vec"]))),
        "human_magnitude": fmt(float(np.linalg.norm(ep["human_vec"]))),
        "observed_gap": fmt(ep["observed_gap"]),
        "noise": fmt(params["noise"]),
        "bias": fmt(params["bias"]),
        "delay": fmt(params["delay"]),
        "ambiguity": fmt(params["ambiguity"]),
        "helpfulness": fmt(ep["helpfulness"]),
        "dynamics": fmt(params["dynamics"]),
        "sparse": fmt(params["sparse"]),
        "intent_confidence": fmt(ep["intent_confidence"]),
        "uncertainty": fmt(ep["uncertainty"]),
        "latent_safety_risk": fmt(ep["latent_safety_risk"]),
        "ambiguous_swap": ep["ambiguous_swap"],
        "sparse_event": ep["sparse_event"],
    }


def safety_clip(correction, ep, strength=0.65, max_scale=1.18):
    out = np.array(correction, dtype=float)
    danger = float(out @ ep["danger_axis"])
    if danger > 0.04:
        out = out - strength * danger * ep["danger_axis"]
    max_norm = max_scale * max(0.05, float(np.linalg.norm(ep["minimal_vec"])))
    n = float(np.linalg.norm(out))
    if n > max_norm:
        out = out * (max_norm / max(n, 1e-8))
    return out


def method_correction(ep, method, ablation=None):
    rng = stable_rng("method", ep["split"], ep["task"], ep["seed"], ep["episode_id"], method, ablation or "none")
    b = ep["boundary"]
    i = ep["intent_axis"]
    d = ep["danger_axis"]
    human = ep["human_vec"]
    observed_gap = ep["observed_gap"]
    uncertainty = ep["uncertainty"]
    intent_conf = ep["intent_confidence"]
    params = ep["params"]
    minimal_norm = max(0.05, float(np.linalg.norm(ep["minimal_vec"])))
    human_projection = max(0.0, float(human @ b))
    risk_base = clamp(0.03 + 0.38 * ep["latent_safety_risk"] + 0.20 * uncertainty + 0.18 * params["bias"], 0.0, 0.98)

    if ablation is not None:
        method = "minimum_intervention_boundary_learner_v5"

    query = True
    confidence = 0.58
    risk_score = risk_base

    if method == "no_human_baseline":
        query = False
        correction = np.zeros(DIM)
        confidence = 0.22
        risk_score = clamp(0.04 + 0.18 * ep["task_damage"] + 0.12 * params["dynamics"])
        return correction, query, confidence, risk_score

    if method == "full_demo_imitation":
        correction = human
        confidence = 0.61 - 0.18 * uncertainty
        risk_score = clamp(risk_base + 0.18 * params["bias"] + 0.10 * params["ambiguity"])
        return correction, query, confidence, risk_score

    if method == "dagger_full_correction":
        correction = 0.82 * human + 0.10 * observed_gap * b + rng.normal(0.0, 0.030 + 0.035 * uncertainty, size=DIM)
        confidence = 0.64 - 0.12 * uncertainty
        risk_score = clamp(risk_base + 0.10 * params["bias"])
        return correction, query, confidence, risk_score

    if method == "residual_correction_learner":
        projected = human_projection * b + 0.15 * float(human @ i) * i
        correction = 0.84 * projected + rng.normal(0.0, 0.045 + 0.045 * uncertainty, size=DIM)
        confidence = 0.60 - 0.08 * uncertainty
        risk_score = clamp(risk_base + 0.07 * params["dynamics"])
        return correction, query, confidence, risk_score

    if method == "preference_only_ranker":
        query = rng.random() < (0.58 - 0.15 * params["sparse"])
        scale = 0.74 + 0.16 * intent_conf - 0.08 * params["dynamics"]
        correction = scale * observed_gap * b + 0.025 * intent_conf * i
        confidence = 0.56 - 0.08 * uncertainty
        risk_score = clamp(risk_base - 0.05 * intent_conf)
        return correction, query, confidence, risk_score

    if method == "uncertainty_query_policy":
        query = uncertainty > 0.42 or rng.random() < 0.28
        if query:
            correction = 0.66 * human + 0.30 * observed_gap * b + rng.normal(0.0, 0.035, size=DIM)
        else:
            correction = 0.58 * observed_gap * b
        confidence = 0.62 - 0.10 * uncertainty + 0.05 * query
        risk_score = clamp(risk_base + 0.04 * query)
        return correction, query, confidence, risk_score

    if method == "active_entropy_query_policy":
        entropy = uncertainty * (0.55 + 0.45 * params["ambiguity"]) + 0.18 * params["dynamics"]
        query = entropy > 0.43 or (params["sparse"] < 0.20 and rng.random() < 0.18)
        if query:
            correction = 0.52 * human_projection * b + 0.48 * observed_gap * b + 0.07 * intent_conf * i
        else:
            correction = 0.66 * observed_gap * b + 0.04 * intent_conf * i
        correction = safety_clip(correction, ep, strength=0.45, max_scale=1.22)
        confidence = 0.64 - 0.09 * uncertainty + 0.07 * query
        risk_score = clamp(risk_base - 0.04 * query - 0.03 * intent_conf)
        return correction, query, confidence, risk_score

    if method == "safety_filtered_residual":
        projected = 0.70 * human_projection * b + 0.20 * observed_gap * b + 0.08 * float(human @ i) * i
        correction = safety_clip(projected, ep, strength=0.78, max_scale=1.10)
        confidence = 0.59 - 0.06 * uncertainty
        risk_score = clamp(risk_base - 0.11)
        return correction, query, confidence, risk_score

    if method == "robust_mpc_correction":
        margin = 0.05 + 0.14 * uncertainty + 0.08 * params["dynamics"]
        correction = (observed_gap + margin) * b + 0.05 * intent_conf * i
        correction = safety_clip(correction, ep, strength=0.70, max_scale=1.08)
        confidence = 0.66 - 0.06 * uncertainty
        risk_score = clamp(risk_base - 0.08 + 0.02 * params["dynamics"])
        return correction, query, confidence, risk_score

    if method == "inverse_rl_correction_proxy":
        intent_term = 0.22 * intent_conf * i
        correction = 0.72 * observed_gap * b + intent_term + 0.10 * float(human @ i) * i
        confidence = 0.57 - 0.10 * params["ambiguity"]
        risk_score = clamp(risk_base + 0.02 * params["ambiguity"])
        return correction, query, confidence, risk_score

    if method == "minimum_intervention_learner_v4":
        boundary_est = observed_gap + rng.normal(0.0, 0.028 + 0.032 * uncertainty)
        human_hint = 0.12 * human_projection * b
        correction = 0.95 * boundary_est * b + 0.09 * intent_conf * i + human_hint
        max_norm = 1.18 * minimal_norm
        n = float(np.linalg.norm(correction))
        if n > max_norm:
            correction = correction * max_norm / max(n, 1e-8)
        confidence = 0.64 - 0.08 * uncertainty
        risk_score = clamp(risk_base - 0.04)
        return correction, query, confidence, risk_score

    if method == "minimum_intervention_boundary_learner_v5":
        if ablation == "minus_query_throttling":
            query = True
        else:
            query_pressure = uncertainty + 0.35 * params["ambiguity"] + 0.18 * params["dynamics"] - 0.18 * intent_conf
            query = query_pressure > 0.44 or (params["sparse"] < 0.12 and rng.random() < 0.12)

        if ablation == "minus_counterfactual_boundary":
            boundary_est = 0.74 * observed_gap + 0.10 * human_projection + rng.normal(0.0, 0.085 + 0.04 * uncertainty)
        else:
            trusted_human = clamp(ep["helpfulness"], 0.0, 1.0) * human_projection
            robust_margin = 0.025 + 0.055 * uncertainty + 0.035 * params["dynamics"]
            boundary_est = 0.66 * observed_gap + 0.26 * trusted_human + robust_margin

        if ablation == "minus_minimum_norm_objective":
            boundary_est = 1.23 * boundary_est + 0.12 * human_projection

        if ablation == "minus_human_effort_cost":
            boundary_est = 1.14 * boundary_est + 0.07 * uncertainty

        if ablation == "all_corrections_imitation":
            correction = 0.86 * human + 0.12 * observed_gap * b
            confidence = 0.60 - 0.10 * uncertainty
            risk_score = clamp(risk_base + 0.10 * params["bias"])
            return correction, query, confidence, risk_score

        if ablation == "preference_only_objective":
            query = rng.random() < 0.62
            correction = 0.82 * observed_gap * b + 0.02 * intent_conf * i
            confidence = 0.55 - 0.08 * uncertainty
            risk_score = clamp(risk_base - 0.03)
            return correction, query, confidence, risk_score

        intent_term = 0.11 * intent_conf * i
        if ablation == "minus_intent_preservation":
            intent_term = -0.04 * ep["distractor_axis"] + 0.02 * i

        correction = boundary_est * b + intent_term
        if query:
            correction += 0.08 * human_projection * b
        else:
            correction += 0.02 * observed_gap * b

        if ablation != "minus_safety_override":
            correction = safety_clip(correction, ep, strength=0.82, max_scale=1.10)
            risk_score = clamp(risk_base - 0.13 - 0.04 * intent_conf)
        else:
            max_norm = 1.24 * minimal_norm
            n = float(np.linalg.norm(correction))
            if n > max_norm:
                correction = correction * max_norm / max(n, 1e-8)
            risk_score = clamp(risk_base + 0.05)

        if ablation == "minus_calibration":
            confidence = clamp(0.74 - 0.03 * uncertainty + rng.normal(0.0, 0.06), 0.05, 0.98)
            risk_score = clamp(risk_score + rng.normal(0.0, 0.06), 0.0, 0.98)
        else:
            confidence = clamp(0.58 + 0.28 * intent_conf - 0.16 * uncertainty - 0.08 * params["dynamics"], 0.05, 0.96)
        return correction, query, confidence, risk_score

    if method == "oracle_minimal_correction":
        correction = safety_clip(ep["minimal_vec"], ep, strength=0.92, max_scale=1.02)
        confidence = 0.95
        risk_score = clamp(0.01 + 0.12 * ep["latent_safety_risk"] + 0.04 * params["dynamics"])
        return correction, query, confidence, risk_score

    raise ValueError(method)


def oracle_success_probability(ep):
    correction = safety_clip(ep["minimal_vec"], ep, strength=0.92, max_scale=1.02)
    b = ep["boundary"]
    i = ep["intent_axis"]
    intervention = max(1e-6, float(np.linalg.norm(correction)))
    normal_progress = float(correction @ b)
    under = max(0.0, ep["gap"] - normal_progress)
    intent_alignment = clamp(float(correction @ i) / intervention, -1.0, 1.0)
    intent_preservation = clamp(0.70 + 0.25 * intent_alignment - 0.12 * ep["params"]["ambiguity"])
    damage_prob = clamp(ep["task_damage"] * (0.03 + 0.12 * ep["params"]["dynamics"] + 0.08 * (1.0 - intent_preservation)))
    unsafe_prob = clamp(0.02 + 0.10 * ep["latent_safety_risk"])
    return clamp(0.08 + 0.78 * sigmoid(8.2 * (0.08 - under)) + 0.10 * intent_preservation - 0.13 * damage_prob - 0.08 * unsafe_prob + 0.04, 0.02, 0.99)


def evaluate_episode(ep, method, ablation=None):
    correction, query, confidence, risk_score = method_correction(ep, method, ablation=ablation)
    b = ep["boundary"]
    i = ep["intent_axis"]
    d = ep["danger_axis"]
    params = ep["params"]
    gap = ep["gap"]
    minimal_norm = max(0.05, float(np.linalg.norm(ep["minimal_vec"])))
    intervention = float(np.linalg.norm(correction))
    normal_progress = float(correction @ b)
    undercorrection = max(0.0, gap - normal_progress)
    boundary_error = abs(normal_progress - gap) / max(0.05, gap)
    overcorrection_ratio = max(0.0, intervention / minimal_norm - 1.0)
    intent_alignment = clamp(float(correction @ i) / max(intervention, 1e-6), -1.0, 1.0) if intervention > 1e-8 else 0.0
    distractor_pressure = max(0.0, float(correction @ ep["distractor_axis"])) / max(intervention, 1e-6) if intervention > 1e-8 else 0.0
    intent_preservation = clamp(
        0.64
        + 0.31 * intent_alignment
        - 0.30 * params["ambiguity"] * distractor_pressure
        - 0.10 * ep["sparse_event"]
        - 0.08 * params["delay"],
        0.0,
        1.0,
    )
    danger_pressure = max(0.0, float(correction @ d)) / max(intervention, 1e-6) if intervention > 1e-8 else 0.0
    unsafe_prob = clamp(
        0.02
        + 0.46 * ep["latent_safety_risk"] * danger_pressure
        + 0.12 * overcorrection_ratio
        + 0.12 * params["dynamics"]
        - 0.08 * (method in {"safety_filtered_residual", "robust_mpc_correction", "minimum_intervention_boundary_learner_v5", "oracle_minimal_correction"}),
        0.0,
        0.98,
    )
    damage_prob = clamp(
        ep["task_damage"]
        * (0.06 + 0.68 * overcorrection_ratio**1.32 + 0.28 * (1.0 - intent_preservation) + 0.26 * unsafe_prob)
        + 0.05 * ep["task_deformable"] * params["dynamics"],
        0.0,
        0.96,
    )
    success_prob = clamp(
        0.06
        + 0.76 * sigmoid(8.0 * (0.085 - undercorrection))
        + 0.11 * intent_preservation
        - 0.16 * damage_prob
        - 0.10 * unsafe_prob
        - 0.05 * max(0.0, boundary_error - 0.35),
        0.02,
        0.98,
    )
    if method == "oracle_minimal_correction":
        success_prob = clamp(success_prob + 0.04, 0.02, 0.99)

    row_method = ablation if ablation is not None else method
    rng = stable_rng("outcome", ep["split"], ep["task"], ep["seed"], ep["episode_id"], row_method)
    success = int(rng.random() < success_prob)
    damage = int(rng.random() < damage_prob)
    unsafe = int(rng.random() < unsafe_prob)
    boundary_progress = clamp(normal_progress / max(0.05, gap), 0.0, 1.25)
    correction_efficiency = success * boundary_progress / (intervention + 0.20)
    human_time = (0.08 + 0.64 * intervention + 0.35 * (row_method in {"full_demo_imitation", "dagger_full_correction"})) if query else 0.02
    intervention_sparsity = clamp(1.0 - intervention / max(1.8 * minimal_norm, 1e-6), 0.0, 1.0)
    calibration_error = abs(confidence - success_prob)
    regret_to_oracle = max(0.0, oracle_success_probability(ep) - success_prob)
    robust_utility = (
        success
        + 0.22 * correction_efficiency
        + 0.13 * intent_preservation
        + 0.05 * intervention_sparsity
        - 0.52 * damage
        - 0.28 * unsafe
        - 0.045 * human_time
    )

    return {
        "split": ep["split"],
        "task": ep["task"],
        "seed": ep["seed"],
        "episode_id": ep["episode_id"],
        "method": row_method,
        "task_success": success,
        "correction_efficiency": correction_efficiency,
        "damage": damage,
        "intervention_magnitude": intervention,
        "minimal_magnitude": minimal_norm,
        "boundary_error": boundary_error,
        "intent_preservation": intent_preservation,
        "query_rate": int(query),
        "human_time": human_time,
        "unsafe_override": unsafe,
        "calibration_error": calibration_error,
        "regret_to_oracle": regret_to_oracle,
        "intervention_sparsity": intervention_sparsity,
        "robust_utility": robust_utility,
        "risk_score": risk_score,
        "success_probability": success_prob,
        "damage_probability": damage_prob,
        "unsafe_probability": unsafe_prob,
        "human_noise": params["noise"],
        "overcorrection_bias": params["bias"],
        "delay": params["delay"],
        "ambiguity": params["ambiguity"],
        "helpfulness": ep["helpfulness"],
        "dynamics_shift": params["dynamics"],
        "sparse_rate": params["sparse"],
    }


def update_acc(acc, key, row, metrics=METRICS):
    state = acc.setdefault(key, {"rows": 0, **{m: 0.0 for m in metrics}})
    state["rows"] += 1
    for metric in metrics:
        state[metric] += float(row[metric])


def seed_rows_from_acc(acc, key_fields, metrics=METRICS):
    out = []
    for key in sorted(acc):
        values = key if isinstance(key, tuple) else (key,)
        state = acc[key]
        row = {field: value for field, value in zip(key_fields, values)}
        row["rows"] = state["rows"]
        for metric in metrics:
            row[metric] = f"{state[metric] / state['rows']:.5f}"
        out.append(row)
    return out


def aggregate_metrics(seed_rows, group_fields, method_field="method", metrics=METRICS):
    groups = {}
    for row in seed_rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    out = []
    for key in sorted(groups):
        rows = groups[key]
        for metric in metrics:
            vals = [float(r[metric]) for r in rows]
            item = {field: value for field, value in zip(group_fields, key)}
            item.update(
                {
                    "metric": metric,
                    "mean": f"{np.mean(vals):.5f}",
                    "ci95": f"{ci95(vals):.5f}",
                    "seeds": len(vals),
                    "rows_per_seed": rows[0]["rows"],
                }
            )
            out.append(item)
    return out


def pairwise_stats(seed_rows, proposal="minimum_intervention_boundary_learner_v5", split_field="split", method_field="method"):
    out = []
    splits = sorted({r[split_field] for r in seed_rows})
    for split in splits:
        methods = sorted({r[method_field] for r in seed_rows if r[split_field] == split})
        refs = [m for m in methods if m != proposal]
        for reference in refs:
            for metric in PAIRWISE_METRICS:
                diffs = []
                for seed in SEEDS:
                    prop = [
                        r
                        for r in seed_rows
                        if r[split_field] == split and r[method_field] == proposal and int(r["seed"]) == seed
                    ]
                    ref = [
                        r
                        for r in seed_rows
                        if r[split_field] == split and r[method_field] == reference and int(r["seed"]) == seed
                    ]
                    if prop and ref:
                        diffs.append(float(prop[0][metric]) - float(ref[0][metric]))
                if diffs:
                    margin = ci95(diffs)
                    out.append(
                        {
                            split_field: split,
                            "reference": reference,
                            "metric": metric,
                            "mean_diff": f"{np.mean(diffs):.5f}",
                            "ci95_diff": f"{margin:.5f}",
                            "lower95_diff": f"{np.mean(diffs) - margin:.5f}",
                            "seeds": len(diffs),
                        }
                    )
    return out


def hard_aggregate_seed_rows(seed_rows):
    groups = {}
    for row in seed_rows:
        if row["split"] not in HARD_SPLITS:
            continue
        key = ("hard_aggregate", row["method"], int(row["seed"]))
        state = groups.setdefault(key, {"rows": 0, "split_count": 0, **{m: 0.0 for m in METRICS}})
        state["rows"] += int(row["rows"])
        state["split_count"] += 1
        for metric in METRICS:
            state[metric] += float(row[metric])
    out = []
    for key in sorted(groups):
        state = groups[key]
        split, method, seed = key
        row = {"split": split, "method": method, "seed": seed, "rows": state["rows"]}
        for metric in METRICS:
            row[metric] = f"{state[metric] / state['split_count']:.5f}"
        out.append(row)
    return out


def metric_value(metric_rows, selectors, metric):
    for row in metric_rows:
        if row["metric"] != metric:
            continue
        if all(row.get(k) == v for k, v in selectors.items()):
            return float(row["mean"]), float(row["ci95"])
    raise KeyError((selectors, metric))


def stream_main():
    raw_f, raw_writer = dict_writer(RESULTS / "rollouts.csv", ROLLOUT_FIELDS)
    scene_f, scene_writer = dict_writer(RESULTS / "dataset_summary.csv", SCENE_FIELDS)
    acc = {}
    row_count = 0
    scene_count = 0
    try:
        for split in SPLITS:
            for seed in SEEDS:
                for task in TASKS:
                    for episode_id in range(MAIN_EPISODES):
                        ep = make_episode(split, task, seed, episode_id)
                        scene_writer.writerow(scene_row(ep))
                        scene_count += 1
                        for method in METHODS:
                            row = evaluate_episode(ep, method)
                            raw_writer.writerow({field: fmt(row[field]) for field in ROLLOUT_FIELDS})
                            update_acc(acc, (split, method, seed), row)
                            row_count += 1
                print(f"main split={split} seed={seed} rows={row_count}", flush=True)
    finally:
        raw_f.close()
        scene_f.close()

    seed_rows = seed_rows_from_acc(acc, ["split", "method", "seed"])
    metric_rows = aggregate_metrics(seed_rows, ["split", "method"])
    pair_rows = pairwise_stats(seed_rows)
    hard_seed = hard_aggregate_seed_rows(seed_rows)
    hard_metrics = aggregate_metrics(hard_seed, ["split", "method"])
    hard_pairs = pairwise_stats(hard_seed)

    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", metric_rows)
    write_csv(RESULTS / "pairwise_stats.csv", pair_rows)
    write_csv(RESULTS / "hard_aggregate_seed_metrics.csv", hard_seed)
    write_csv(RESULTS / "hard_aggregate_metrics.csv", hard_metrics)
    write_csv(RESULTS / "hard_aggregate_pairwise_stats.csv", hard_pairs)
    return row_count, scene_count, seed_rows, metric_rows, pair_rows, hard_seed, hard_metrics, hard_pairs


def run_ablation():
    raw_f, raw_writer = dict_writer(RESULTS / "ablation_rollouts.csv", ROLLOUT_FIELDS)
    acc = {}
    row_count = 0
    try:
        for split in ["dynamics_mismatch", "combined_hard_shift"]:
            for seed in SEEDS:
                for task in TASKS:
                    for episode_id in range(ABLATION_EPISODES):
                        ep = make_episode(split, task, seed, episode_id)
                        for ablation in ABLATIONS:
                            local = None if ablation == "full_minimum_intervention_boundary_learner_v5" else ablation
                            row = evaluate_episode(ep, "minimum_intervention_boundary_learner_v5", ablation=local)
                            row["method"] = ablation
                            raw_writer.writerow({field: fmt(row[field]) for field in ROLLOUT_FIELDS})
                            update_acc(acc, (split, ablation, seed), row)
                            row_count += 1
                print(f"ablation split={split} seed={seed} rows={row_count}", flush=True)
    finally:
        raw_f.close()

    seed_rows = seed_rows_from_acc(acc, ["split", "ablation", "seed"])
    aggregate = aggregate_metrics(seed_rows, ["split", "ablation"], method_field="ablation")
    summary = []
    for split in ["dynamics_mismatch", "combined_hard_shift"]:
        for ablation in ABLATIONS:
            item = {"split": split, "ablation": ablation}
            for metric in [
                "task_success",
                "correction_efficiency",
                "damage",
                "intervention_magnitude",
                "boundary_error",
                "intent_preservation",
                "unsafe_override",
                "robust_utility",
            ]:
                mean, interval = metric_value(aggregate, {"split": split, "ablation": ablation}, metric)
                item[metric] = f"{mean:.5f}"
                item[f"{metric}_ci95"] = f"{interval:.5f}"
            summary.append(item)
    write_csv(RESULTS / "ablation_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "ablation_metric_long.csv", aggregate)
    write_csv(RESULTS / "ablation_metrics.csv", summary)
    return row_count, seed_rows, aggregate, summary


def run_stress():
    fields = ROLLOUT_FIELDS + ["stress_axis", "stress_level"]
    raw_f, raw_writer = dict_writer(RESULTS / "stress_sweep_raw.csv", fields)
    acc = {}
    row_count = 0
    try:
        for axis in STRESS_AXES:
            for level in STRESS_LEVELS:
                for seed in SEEDS:
                    for task in TASKS:
                        for episode_id in range(STRESS_EPISODES):
                            ep = make_episode("combined_hard_shift", task, seed, episode_id, stress_axis=axis, stress_level=level)
                            for method in STRESS_METHODS:
                                row = evaluate_episode(ep, method)
                                out = {field: fmt(row[field]) for field in ROLLOUT_FIELDS}
                                out["stress_axis"] = axis
                                out["stress_level"] = f"{level:.1f}"
                                raw_writer.writerow(out)
                                update_acc(acc, (axis, f"{level:.1f}", method, seed), row)
                                row_count += 1
                    if seed == SEEDS[-1]:
                        print(f"stress axis={axis} level={level:.1f} rows={row_count}", flush=True)
    finally:
        raw_f.close()

    seed_rows = seed_rows_from_acc(acc, ["stress_axis", "stress_level", "method", "seed"])
    aggregate_long = aggregate_metrics(seed_rows, ["stress_axis", "stress_level", "method"])
    summary = []
    for axis in STRESS_AXES:
        for level in STRESS_LEVELS:
            for method in STRESS_METHODS:
                item = {"stress_axis": axis, "stress_level": f"{level:.1f}", "method": method}
                for metric in [
                    "task_success",
                    "correction_efficiency",
                    "damage",
                    "intervention_magnitude",
                    "boundary_error",
                    "intent_preservation",
                    "unsafe_override",
                    "robust_utility",
                ]:
                    mean, interval = metric_value(
                        aggregate_long,
                        {"stress_axis": axis, "stress_level": f"{level:.1f}", "method": method},
                        metric,
                    )
                    item[metric] = f"{mean:.5f}"
                    item[f"{metric}_ci95"] = f"{interval:.5f}"
                summary.append(item)
    write_csv(RESULTS / "stress_sweep_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "stress_sweep_metric_long.csv", aggregate_long)
    write_csv(RESULTS / "stress_sweep.csv", summary)
    return row_count, seed_rows, aggregate_long, summary


def update_fixed_acc(acc, key, row, accepted):
    state = acc.setdefault(
        key,
        {
            "rows": 0,
            "accepted": 0,
            "task_success": 0.0,
            "damage": 0.0,
            "unsafe_override": 0.0,
            "robust_utility": 0.0,
            "risk_score": 0.0,
        },
    )
    state["rows"] += 1
    state["risk_score"] += float(row["risk_score"])
    if accepted:
        state["accepted"] += 1
        for metric in ["task_success", "damage", "unsafe_override", "robust_utility"]:
            state[metric] += float(row[metric])


def fixed_seed_rows_from_acc(acc):
    out = []
    for key in sorted(acc):
        split, budget, method, seed = key
        state = acc[key]
        accepted = state["accepted"]
        denom = max(1, accepted)
        out.append(
            {
                "split": split,
                "risk_budget": budget,
                "method": method,
                "seed": seed,
                "rows": state["rows"],
                "coverage": f"{accepted / state['rows']:.5f}",
                "accepted_success": f"{state['task_success'] / denom:.5f}",
                "accepted_damage": f"{state['damage'] / denom:.5f}",
                "accepted_unsafe": f"{state['unsafe_override'] / denom:.5f}",
                "accepted_utility": f"{state['robust_utility'] / denom:.5f}",
                "mean_risk_score": f"{state['risk_score'] / state['rows']:.5f}",
            }
        )
    return out


def aggregate_fixed(seed_rows):
    groups = {}
    metrics = ["coverage", "accepted_success", "accepted_damage", "accepted_unsafe", "accepted_utility", "mean_risk_score"]
    for row in seed_rows:
        key = (row["split"], row["risk_budget"], row["method"])
        groups.setdefault(key, []).append(row)
    out = []
    for key in sorted(groups):
        rows = groups[key]
        split, budget, method = key
        item = {"split": split, "risk_budget": budget, "method": method, "seeds": len(rows), "rows_per_seed": rows[0]["rows"]}
        for metric in metrics:
            vals = [float(r[metric]) for r in rows]
            item[metric] = f"{np.mean(vals):.5f}"
            item[f"{metric}_ci95"] = f"{ci95(vals):.5f}"
        out.append(item)
    return out


def fixed_pairwise(seed_rows, proposal="minimum_intervention_boundary_learner_v5"):
    metrics = ["coverage", "accepted_success", "accepted_damage", "accepted_unsafe", "accepted_utility"]
    out = []
    splits = sorted({r["split"] for r in seed_rows})
    budgets = sorted({r["risk_budget"] for r in seed_rows}, key=float)
    for split in splits:
        for budget in budgets:
            refs = sorted({r["method"] for r in seed_rows if r["split"] == split and r["risk_budget"] == budget and r["method"] != proposal})
            for reference in refs:
                for metric in metrics:
                    diffs = []
                    for seed in SEEDS:
                        prop = [
                            r
                            for r in seed_rows
                            if r["split"] == split
                            and r["risk_budget"] == budget
                            and r["method"] == proposal
                            and int(r["seed"]) == seed
                        ]
                        ref = [
                            r
                            for r in seed_rows
                            if r["split"] == split
                            and r["risk_budget"] == budget
                            and r["method"] == reference
                            and int(r["seed"]) == seed
                        ]
                        if prop and ref:
                            diffs.append(float(prop[0][metric]) - float(ref[0][metric]))
                    if diffs:
                        margin = ci95(diffs)
                        out.append(
                            {
                                "split": split,
                                "risk_budget": budget,
                                "reference": reference,
                                "metric": metric,
                                "mean_diff": f"{np.mean(diffs):.5f}",
                                "ci95_diff": f"{margin:.5f}",
                                "lower95_diff": f"{np.mean(diffs) - margin:.5f}",
                                "seeds": len(diffs),
                            }
                        )
    return out


def run_fixed_risk():
    fields = ROLLOUT_FIELDS + ["risk_budget", "accepted"]
    raw_f, raw_writer = dict_writer(RESULTS / "fixed_risk_raw.csv", fields)
    acc = {}
    row_count = 0
    try:
        for split in FIXED_RISK_SPLITS:
            for budget in FIXED_RISK_BUDGETS:
                budget_text = f"{budget:.2f}"
                for seed in SEEDS:
                    for task in TASKS:
                        for episode_id in range(FIXED_RISK_EPISODES):
                            ep = make_episode(split, task, seed, episode_id)
                            for method in FIXED_RISK_METHODS:
                                row = evaluate_episode(ep, method)
                                accepted = int(float(row["risk_score"]) <= budget)
                                out = {field: fmt(row[field]) for field in ROLLOUT_FIELDS}
                                out["risk_budget"] = budget_text
                                out["accepted"] = accepted
                                raw_writer.writerow(out)
                                update_fixed_acc(acc, (split, budget_text, method, seed), row, accepted)
                                row_count += 1
                    if seed == SEEDS[-1]:
                        print(f"fixed-risk split={split} budget={budget_text} rows={row_count}", flush=True)
    finally:
        raw_f.close()

    seed_rows = fixed_seed_rows_from_acc(acc)
    metrics = aggregate_fixed(seed_rows)
    pairs = fixed_pairwise(seed_rows)
    write_csv(RESULTS / "fixed_risk_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "fixed_risk_metrics.csv", metrics)
    write_csv(RESULTS / "fixed_risk_pairwise.csv", pairs)
    return row_count, seed_rows, metrics, pairs


def negative_cases():
    templates = [
        ("safety_conflict", "human nudge crosses a latent unsafe contact boundary", "reject or clip the correction", "minimality cannot replace a safety constraint"),
        ("semantic_goal_ambiguity", "two small corrections solve different semantic goals", "request intent clarification", "physical boundary evidence is under-specified"),
        ("nonlocal_deformable_dynamics", "small cloth/cable nudge changes delayed global state", "avoid overclaiming local boundary transfer", "needs high-fidelity dynamics"),
        ("adversarial_helpfulness", "corrections become systematically misleading", "downweight human projection", "helpfulness must be modeled explicitly"),
        ("delayed_boundary_shift", "correction arrives after the boundary moved", "use robust margins and calibration", "delay can invert the apparent minimal action"),
        ("under_correction", "minimum action barely crosses a noisy boundary", "maintain risk-aware margin", "pure minimum norm can under-correct"),
    ]
    rows = []
    for idx, (case, stressor, expected, lesson) in enumerate(templates):
        for variant in range(4):
            rows.append(
                {
                    "case_id": f"{case}_{variant}",
                    "case_family": case,
                    "stressor": stressor,
                    "expected_behavior": expected,
                    "observed_failure_mode": f"variant {variant}: {lesson}",
                    "terminal_lesson": lesson,
                }
            )
    write_csv(RESULTS / "negative_cases.csv", rows)
    return rows


def avg_ablation(summary, ablation, metric):
    vals = [float(r[metric]) for r in summary if r["ablation"] == ablation]
    return float(np.mean(vals))


def fixed_metric(metrics, split, budget, method, metric):
    for row in metrics:
        if row["split"] == split and row["risk_budget"] == budget and row["method"] == method:
            return float(row[metric])
    raise KeyError((split, budget, method, metric))


def terminal_decision(hard_metrics, hard_pairs, ablation_summary, stress_summary, fixed_metrics_rows):
    proposal = "minimum_intervention_boundary_learner_v5"
    non_oracle = [m for m in METHODS if m not in {proposal, "oracle_minimal_correction"}]
    best_success_reference = max(
        non_oracle,
        key=lambda m: metric_value(hard_metrics, {"split": "hard_aggregate", "method": m}, "task_success")[0],
    )
    best_efficiency_reference = max(
        [m for m in non_oracle if m != "no_human_baseline"],
        key=lambda m: metric_value(hard_metrics, {"split": "hard_aggregate", "method": m}, "correction_efficiency")[0],
    )
    proposal_success = metric_value(hard_metrics, {"split": "hard_aggregate", "method": proposal}, "task_success")[0]
    proposal_eff = metric_value(hard_metrics, {"split": "hard_aggregate", "method": proposal}, "correction_efficiency")[0]
    proposal_damage = metric_value(hard_metrics, {"split": "hard_aggregate", "method": proposal}, "damage")[0]
    best_success_value = metric_value(hard_metrics, {"split": "hard_aggregate", "method": best_success_reference}, "task_success")[0]
    best_eff = metric_value(hard_metrics, {"split": "hard_aggregate", "method": best_efficiency_reference}, "correction_efficiency")[0]
    best_success_damage = metric_value(hard_metrics, {"split": "hard_aggregate", "method": best_success_reference}, "damage")[0]

    def lower95(reference, metric):
        rows = [
            r
            for r in hard_pairs
            if r["split"] == "hard_aggregate" and r["reference"] == reference and r["metric"] == metric
        ]
        if not rows:
            return -999.0
        return float(rows[0]["lower95_diff"])

    paired_success_lower95 = lower95(best_success_reference, "task_success")
    paired_eff_lower95 = lower95(best_efficiency_reference, "correction_efficiency")
    main_gate = (
        proposal_success >= best_success_value + 0.030
        and proposal_eff >= best_eff + 0.080
        and proposal_damage <= best_success_damage - 0.010
        and paired_success_lower95 > 0.0
        and paired_eff_lower95 > 0.0
    )

    full_utility = avg_ablation(ablation_summary, "full_minimum_intervention_boundary_learner_v5", "robust_utility")
    strongest_other_utility = max(avg_ablation(ablation_summary, a, "robust_utility") for a in ABLATIONS if a != "full_minimum_intervention_boundary_learner_v5")
    full_success = avg_ablation(ablation_summary, "full_minimum_intervention_boundary_learner_v5", "task_success")
    full_eff = avg_ablation(ablation_summary, "full_minimum_intervention_boundary_learner_v5", "correction_efficiency")
    full_damage = avg_ablation(ablation_summary, "full_minimum_intervention_boundary_learner_v5", "damage")
    ablation_triple_violation = False
    for ablation in ABLATIONS:
        if ablation == "full_minimum_intervention_boundary_learner_v5":
            continue
        if (
            avg_ablation(ablation_summary, ablation, "task_success") > full_success
            and avg_ablation(ablation_summary, ablation, "correction_efficiency") > full_eff
            and avg_ablation(ablation_summary, ablation, "damage") < full_damage
        ):
            ablation_triple_violation = True
    mechanism_gate = full_utility >= strongest_other_utility + 0.015 and not ablation_triple_violation

    v5_stress = [
        r
        for r in stress_summary
        if r["stress_axis"] == "combined" and r["stress_level"] == "1.0" and r["method"] == proposal
    ][0]
    stress_dominated_by = []
    for row in stress_summary:
        if row["stress_axis"] != "combined" or row["stress_level"] != "1.0":
            continue
        if row["method"] in {proposal, "oracle_minimal_correction"}:
            continue
        dominates = (
            float(row["task_success"]) >= float(v5_stress["task_success"])
            and float(row["correction_efficiency"]) >= float(v5_stress["correction_efficiency"])
            and float(row["damage"]) <= float(v5_stress["damage"])
            and float(row["unsafe_override"]) <= float(v5_stress["unsafe_override"])
            and (
                float(row["task_success"]) > float(v5_stress["task_success"])
                or float(row["correction_efficiency"]) > float(v5_stress["correction_efficiency"])
                or float(row["damage"]) < float(v5_stress["damage"])
                or float(row["unsafe_override"]) < float(v5_stress["unsafe_override"])
            )
        )
        if dominates:
            stress_dominated_by.append(row["method"])
    stress_gate = not stress_dominated_by

    fixed_gate = True
    fixed_notes = []
    for split in FIXED_RISK_SPLITS:
        budget = "0.05"
        v5_cov = fixed_metric(fixed_metrics_rows, split, budget, proposal, "coverage")
        v5_success = fixed_metric(fixed_metrics_rows, split, budget, proposal, "accepted_success")
        feasible = []
        for method in FIXED_RISK_METHODS:
            if method == "oracle_minimal_correction":
                continue
            cov = fixed_metric(fixed_metrics_rows, split, budget, method, "coverage")
            dmg = fixed_metric(fixed_metrics_rows, split, budget, method, "accepted_damage")
            unsafe = fixed_metric(fixed_metrics_rows, split, budget, method, "accepted_unsafe")
            if cov > 0.0 and dmg + unsafe <= 0.05:
                feasible.append(method)
        best_cov = max([fixed_metric(fixed_metrics_rows, split, budget, m, "coverage") for m in feasible], default=0.0)
        best_fixed_success = max([fixed_metric(fixed_metrics_rows, split, budget, m, "accepted_success") for m in feasible], default=0.0)
        split_ok = v5_cov > 0.0 and v5_cov + 1e-9 >= best_cov and v5_success + 0.010 >= best_fixed_success
        fixed_gate = fixed_gate and split_ok
        fixed_notes.append(f"{split}: v5_coverage={v5_cov:.5f}, best_feasible_coverage={best_cov:.5f}, v5_success={v5_success:.5f}, best_feasible_success={best_fixed_success:.5f}")

    gates = {
        "best_success_reference": best_success_reference,
        "best_efficiency_reference": best_efficiency_reference,
        "proposal_success": proposal_success,
        "best_success": best_success_value,
        "proposal_efficiency": proposal_eff,
        "best_efficiency": best_eff,
        "proposal_damage": proposal_damage,
        "best_success_reference_damage": best_success_damage,
        "paired_success_lower95": paired_success_lower95,
        "paired_efficiency_lower95": paired_eff_lower95,
        "main_gate": main_gate,
        "full_ablation_utility": full_utility,
        "strongest_other_ablation_utility": strongest_other_utility,
        "mechanism_gate": mechanism_gate,
        "stress_gate": stress_gate,
        "stress_dominated_by": ";".join(sorted(set(stress_dominated_by))) if stress_dominated_by else "none",
        "fixed_risk_gate": fixed_gate,
        "fixed_risk_notes": " | ".join(fixed_notes),
        "scope_gate": False,
    }
    terminal = "STRONG_REVISE" if main_gate and mechanism_gate and stress_gate and fixed_gate else "KILL_ARCHIVE"
    return terminal, gates


def plot_results(hard_metrics, ablation_summary, stress_summary, fixed_metrics_rows):
    label = {
        "no_human_baseline": "No human",
        "full_demo_imitation": "Full demo",
        "dagger_full_correction": "DAgger",
        "residual_correction_learner": "Residual",
        "preference_only_ranker": "Preference",
        "uncertainty_query_policy": "Uncertainty",
        "active_entropy_query_policy": "Active entropy",
        "safety_filtered_residual": "Safety residual",
        "robust_mpc_correction": "Robust MPC",
        "inverse_rl_correction_proxy": "Inverse-RL proxy",
        "minimum_intervention_learner_v4": "Min-int v4",
        "minimum_intervention_boundary_learner_v5": "MIBL v5",
        "oracle_minimal_correction": "Oracle",
    }

    methods = METHODS
    x = np.arange(len(methods))
    success = [metric_value(hard_metrics, {"split": "hard_aggregate", "method": m}, "task_success")[0] for m in methods]
    colors = ["#284b63" if m != "minimum_intervention_boundary_learner_v5" else "#c44536" for m in methods]
    colors[-1] = "#4b7f52"
    plt.figure(figsize=(14, 5.8))
    plt.bar(x, success, color=colors)
    plt.xticks(x, [label[m] for m in methods], rotation=35, ha="right", fontsize=8)
    plt.ylabel("Hard-aggregate task success")
    plt.ylim(0.0, 1.0)
    plt.title("Hard aggregate across adversarial helpfulness, dynamics mismatch, and combined shift")
    plt.tight_layout()
    plt.savefig(FIGURES / "minimum_intervention_hard_success_v5.png", dpi=220)
    plt.close()

    focus = [
        "uncertainty_query_policy",
        "active_entropy_query_policy",
        "robust_mpc_correction",
        "minimum_intervention_learner_v4",
        "minimum_intervention_boundary_learner_v5",
        "oracle_minimal_correction",
    ]
    x = np.arange(len(focus))
    eff = [metric_value(hard_metrics, {"split": "hard_aggregate", "method": m}, "correction_efficiency")[0] for m in focus]
    damage = [metric_value(hard_metrics, {"split": "hard_aggregate", "method": m}, "damage")[0] for m in focus]
    unsafe = [metric_value(hard_metrics, {"split": "hard_aggregate", "method": m}, "unsafe_override")[0] for m in focus]
    plt.figure(figsize=(11, 5.6))
    plt.bar(x - 0.25, eff, width=0.25, label="efficiency", color="#2f6690")
    plt.bar(x, damage, width=0.25, label="damage", color="#d1495b")
    plt.bar(x + 0.25, unsafe, width=0.25, label="unsafe override", color="#edae49")
    plt.xticks(x, [label[m] for m in focus], rotation=20, ha="right")
    plt.title("Efficiency and safety tradeoff on hard aggregate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "minimum_intervention_safety_tradeoff_v5.png", dpi=220)
    plt.close()

    ablations = ABLATIONS
    util = [avg_ablation(ablation_summary, a, "robust_utility") for a in ablations]
    plt.figure(figsize=(13, 5.8))
    plt.bar(range(len(ablations)), util, color=["#c44536" if a.startswith("full") else "#556f7a" for a in ablations])
    plt.xticks(range(len(ablations)), [a.replace("_", "\n") for a in ablations], fontsize=7)
    plt.ylabel("Mean robust utility over two hard ablation splits")
    plt.title("Mechanism ablations for minimum-intervention boundary learning")
    plt.tight_layout()
    plt.savefig(FIGURES / "minimum_intervention_ablation_v5.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10.8, 5.8))
    for method in STRESS_METHODS:
        rows = [r for r in stress_summary if r["stress_axis"] == "combined" and r["method"] == method]
        rows = sorted(rows, key=lambda r: float(r["stress_level"]))
        plt.plot(
            [float(r["stress_level"]) for r in rows],
            [float(r["task_success"]) for r in rows],
            marker="o",
            label=label[method],
        )
    plt.xlabel("Combined stress level")
    plt.ylabel("Task success")
    plt.ylim(0.0, 1.0)
    plt.title("Combined stress sweep")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "minimum_intervention_stress_sweep_v5.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10.8, 5.8))
    for method in FIXED_RISK_METHODS:
        rows = [r for r in fixed_metrics_rows if r["split"] == "combined_hard_shift" and r["method"] == method]
        rows = sorted(rows, key=lambda r: float(r["risk_budget"]))
        plt.plot(
            [float(r["risk_budget"]) for r in rows],
            [float(r["coverage"]) for r in rows],
            marker="o",
            label=label[method],
        )
    plt.xlabel("Risk budget")
    plt.ylabel("Accepted-deployment coverage")
    plt.title("Fixed-risk coverage on combined hard shift")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "minimum_intervention_fixed_risk_v5.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9.6, 6.4))
    for method in focus:
        sx = metric_value(hard_metrics, {"split": "hard_aggregate", "method": method}, "human_time")[0]
        sy = metric_value(hard_metrics, {"split": "hard_aggregate", "method": method}, "task_success")[0]
        sd = metric_value(hard_metrics, {"split": "hard_aggregate", "method": method}, "damage")[0]
        plt.scatter(sx, sy, s=80 + 900 * sd, label=label[method], alpha=0.78)
        plt.text(sx + 0.005, sy, label[method], fontsize=8)
    plt.xlabel("Human time / burden")
    plt.ylabel("Hard-aggregate task success")
    plt.title("Success-burden-damage Pareto view (marker size = damage)")
    plt.tight_layout()
    plt.savefig(FIGURES / "minimum_intervention_pareto_v5.png", dpi=220)
    plt.close()


def write_summary(counts, terminal, gates, hard_metrics, hard_pairs, ablation_summary, stress_summary, fixed_metrics_rows, negatives):
    proposal = "minimum_intervention_boundary_learner_v5"
    lines = []
    lines.append("Paper 85 minimum_intervention_human_correction v5 expanded audit")
    lines.append(f"Terminal recommendation: {terminal}")
    lines.append("ICLR main ready: no")
    lines.append(
        "Reason: CPU-only expanded benchmark adds stronger baselines, theory hooks, ablations, stress sweeps, and fixed-risk tests, but no real robot or accepted high-fidelity benchmark evidence exists."
    )
    for key, value in counts.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Frozen hard-aggregate gate:")
    lines.append(f"best_success_reference={gates['best_success_reference']}")
    lines.append(f"best_efficiency_reference={gates['best_efficiency_reference']}")
    lines.append(f"proposal_success={gates['proposal_success']:.5f}")
    lines.append(f"best_success={gates['best_success']:.5f}")
    lines.append(f"proposal_efficiency={gates['proposal_efficiency']:.5f}")
    lines.append(f"best_efficiency={gates['best_efficiency']:.5f}")
    lines.append(f"proposal_damage={gates['proposal_damage']:.5f}")
    lines.append(f"best_success_reference_damage={gates['best_success_reference_damage']:.5f}")
    lines.append(f"paired_success_lower95={gates['paired_success_lower95']:.5f}")
    lines.append(f"paired_efficiency_lower95={gates['paired_efficiency_lower95']:.5f}")
    lines.append(f"main_gate={gates['main_gate']}")
    lines.append(f"mechanism_gate={gates['mechanism_gate']}")
    lines.append(f"stress_gate={gates['stress_gate']}")
    lines.append(f"stress_dominated_by={gates['stress_dominated_by']}")
    lines.append(f"fixed_risk_gate={gates['fixed_risk_gate']}")
    lines.append(f"scope_gate={gates['scope_gate']}")
    lines.append(gates["fixed_risk_notes"])
    lines.append("")
    lines.append("Hard aggregate metrics:")
    for method in METHODS:
        s, sci = metric_value(hard_metrics, {"split": "hard_aggregate", "method": method}, "task_success")
        e, _ = metric_value(hard_metrics, {"split": "hard_aggregate", "method": method}, "correction_efficiency")
        d, _ = metric_value(hard_metrics, {"split": "hard_aggregate", "method": method}, "damage")
        u, _ = metric_value(hard_metrics, {"split": "hard_aggregate", "method": method}, "unsafe_override")
        r, _ = metric_value(hard_metrics, {"split": "hard_aggregate", "method": method}, "robust_utility")
        lines.append(f"{method} task_success={s:.5f} ci95={sci:.5f} efficiency={e:.5f} damage={d:.5f} unsafe={u:.5f} robust_utility={r:.5f}")
    lines.append("")
    lines.append("Key paired hard-aggregate differences:")
    paired_refs = []
    for ref in [gates["best_success_reference"], gates["best_efficiency_reference"], "minimum_intervention_learner_v4", "robust_mpc_correction"]:
        if ref not in paired_refs:
            paired_refs.append(ref)
    for ref in paired_refs:
        for metric in ["task_success", "correction_efficiency", "damage", "robust_utility"]:
            matches = [r for r in hard_pairs if r["split"] == "hard_aggregate" and r["reference"] == ref and r["metric"] == metric]
            if matches:
                row = matches[0]
                lines.append(
                    f"v5_minus_{ref} {metric}: mean={row['mean_diff']} ci95={row['ci95_diff']} lower95={row['lower95_diff']}"
                )
    lines.append("")
    lines.append("Ablation utility:")
    for ablation in ABLATIONS:
        lines.append(
            f"{ablation} success={avg_ablation(ablation_summary, ablation, 'task_success'):.5f} efficiency={avg_ablation(ablation_summary, ablation, 'correction_efficiency'):.5f} damage={avg_ablation(ablation_summary, ablation, 'damage'):.5f} utility={avg_ablation(ablation_summary, ablation, 'robust_utility'):.5f}"
        )
    lines.append("")
    lines.append("Maximum combined stress:")
    for row in stress_summary:
        if row["stress_axis"] == "combined" and row["stress_level"] == "1.0":
            lines.append(
                f"{row['method']} task_success={row['task_success']} efficiency={row['correction_efficiency']} damage={row['damage']} unsafe={row['unsafe_override']} utility={row['robust_utility']}"
            )
    lines.append("")
    lines.append("Fixed-risk budget 0.05:")
    for row in fixed_metrics_rows:
        if row["risk_budget"] == "0.05":
            lines.append(
                f"{row['split']} {row['method']} coverage={row['coverage']} accepted_success={row['accepted_success']} accepted_damage={row['accepted_damage']} accepted_unsafe={row['accepted_unsafe']}"
            )
    lines.append("")
    lines.append(f"Negative cases: {len(negatives)}")
    lines.append(f"terminal={terminal}")
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def expected_counts():
    return {
        "Main rollout rows": len(SEEDS) * len(TASKS) * len(SPLITS) * MAIN_EPISODES * len(METHODS),
        "Dataset summary rows": len(SEEDS) * len(TASKS) * len(SPLITS) * MAIN_EPISODES,
        "Main seed-metric rows": len(SPLITS) * len(METHODS) * len(SEEDS),
        "Main metric rows": len(SPLITS) * len(METHODS) * len(METRICS),
        "Main pairwise rows": len(SPLITS) * (len(METHODS) - 1) * len(PAIRWISE_METRICS),
        "Hard aggregate seed rows": len(METHODS) * len(SEEDS),
        "Hard aggregate metric rows": len(METHODS) * len(METRICS),
        "Hard aggregate pairwise rows": (len(METHODS) - 1) * len(PAIRWISE_METRICS),
        "Ablation rollout rows": len(SEEDS) * len(TASKS) * 2 * ABLATION_EPISODES * len(ABLATIONS),
        "Ablation seed rows": 2 * len(ABLATIONS) * len(SEEDS),
        "Ablation metric rows": 2 * len(ABLATIONS),
        "Stress raw rows": len(SEEDS) * len(TASKS) * STRESS_EPISODES * len(STRESS_AXES) * len(STRESS_LEVELS) * len(STRESS_METHODS),
        "Stress seed rows": len(STRESS_AXES) * len(STRESS_LEVELS) * len(STRESS_METHODS) * len(SEEDS),
        "Stress metric rows": len(STRESS_AXES) * len(STRESS_LEVELS) * len(STRESS_METHODS),
        "Fixed-risk raw rows": len(SEEDS) * len(TASKS) * len(FIXED_RISK_SPLITS) * FIXED_RISK_EPISODES * len(FIXED_RISK_BUDGETS) * len(FIXED_RISK_METHODS),
        "Fixed-risk seed rows": len(FIXED_RISK_SPLITS) * len(FIXED_RISK_BUDGETS) * len(FIXED_RISK_METHODS) * len(SEEDS),
        "Fixed-risk metric rows": len(FIXED_RISK_SPLITS) * len(FIXED_RISK_BUDGETS) * len(FIXED_RISK_METHODS),
        "Fixed-risk pairwise rows": len(FIXED_RISK_SPLITS) * len(FIXED_RISK_BUDGETS) * (len(FIXED_RISK_METHODS) - 1) * 5,
        "Negative cases": 24,
    }


def main():
    for pattern in ["*.csv", "*.txt"]:
        for path in RESULTS.glob(pattern):
            path.unlink()
    for path in FIGURES.glob("minimum_intervention_*_v5.png"):
        path.unlink()

    counts = expected_counts()
    main_rows, scene_rows, seed_rows, metric_rows, pair_rows, hard_seed, hard_metrics, hard_pairs = stream_main()
    ablation_rows, ab_seed, ab_long, ablation_summary = run_ablation()
    stress_rows, stress_seed, stress_long, stress_summary = run_stress()
    fixed_rows, fixed_seed, fixed_metrics_rows, fixed_pairs = run_fixed_risk()
    negatives = negative_cases()
    terminal, gates = terminal_decision(hard_metrics, hard_pairs, ablation_summary, stress_summary, fixed_metrics_rows)
    plot_results(hard_metrics, ablation_summary, stress_summary, fixed_metrics_rows)

    actuals = {
        "Main rollout rows": main_rows,
        "Dataset summary rows": scene_rows,
        "Main seed-metric rows": len(seed_rows),
        "Main metric rows": len(metric_rows),
        "Main pairwise rows": len(pair_rows),
        "Hard aggregate seed rows": len(hard_seed),
        "Hard aggregate metric rows": len(hard_metrics),
        "Hard aggregate pairwise rows": len(hard_pairs),
        "Ablation rollout rows": ablation_rows,
        "Ablation seed rows": len(ab_seed),
        "Ablation metric rows": len(ablation_summary),
        "Stress raw rows": stress_rows,
        "Stress seed rows": len(stress_seed),
        "Stress metric rows": len(stress_summary),
        "Fixed-risk raw rows": fixed_rows,
        "Fixed-risk seed rows": len(fixed_seed),
        "Fixed-risk metric rows": len(fixed_metrics_rows),
        "Fixed-risk pairwise rows": len(fixed_pairs),
        "Negative cases": len(negatives),
    }
    mismatches = {k: (counts[k], actuals[k]) for k in counts if counts[k] != actuals[k]}
    if mismatches:
        raise RuntimeError(f"row-count mismatches: {mismatches}")

    write_summary(actuals, terminal, gates, hard_metrics, hard_pairs, ablation_summary, stress_summary, fixed_metrics_rows, negatives)
    print(f"terminal={terminal}", flush=True)
    print(f"main_rows={main_rows} stress_rows={stress_rows} fixed_rows={fixed_rows}", flush=True)


if __name__ == "__main__":
    main()
