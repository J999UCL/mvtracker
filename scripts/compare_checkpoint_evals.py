#!/usr/bin/env python3
"""Compare MVTracker forward parity and paired checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "kubric-multiview-v3-views0123-cached": {
        "label": "MV-Kubric / simulated",
        "point_type": "dynamic-static-mean",
        "metric_scales": {"mte_visible": 0.13},
    },
    "panoptic-multiview-views1_7_14_20-cached": {
        "label": "Panoptic / Dynamic3DGS",
        "point_type": "any",
    },
    "dex-ycb-multiview-duster0123-cached": {
        "label": "DexYCB / DUSt3R",
        "point_type": "dynamic-static-mean",
    },
}

METRICS = {
    "average_jaccard": {"label": "AJ", "higher_is_better": True, "margin": 1.0},
    "average_pts_within_thresh": {
        "label": "Delta-avg",
        "higher_is_better": True,
        "margin": 1.0,
    },
    "occlusion_accuracy": {
        "label": "Occlusion accuracy",
        "higher_is_better": True,
        "margin": 1.0,
    },
    "mte_visible": {"label": "MTE (cm)", "higher_is_better": False, "margin": 0.1},
}


def paired_bootstrap_ci(values: np.ndarray, *, resamples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired bootstrap requires a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, values.size, size=(resamples, values.size))
    sample_means = values[sample_indices].mean(axis=1)
    low, high = np.quantile(sample_means, [0.025, 0.975])
    return float(low), float(high)


def classify_non_regression(ci_low: float, ci_high: float, margin: float) -> str:
    if ci_high <= margin:
        return "non-regressed"
    if ci_low > margin:
        return "regressed"
    return "inconclusive"


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {pattern!r} below {root}, found {matches}")
    return matches[0]


def _dataset_csv(root: Path, dataset_name: str) -> Path:
    dataset_dir = root / f"eval_{dataset_name}"
    return _find_one(dataset_dir, "step-*_metrics.csv")


def _metric_column(frame: pd.DataFrame, metric: str, point_type: str) -> str:
    suffix = f"/model__{metric}__{point_type}"
    matches = [str(column) for column in frame.columns if str(column).endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one metric ending in {suffix!r}, found {matches}")
    return matches[0]


def _load_metric_frame(root: Path, dataset_name: str) -> pd.DataFrame:
    frame = pd.read_csv(_dataset_csv(root, dataset_name), index_col=0)
    if frame.empty or not frame.index.is_unique:
        raise RuntimeError(f"invalid per-sequence metrics for {dataset_name} below {root}")
    return frame.sort_index()


def _metric_scale(dataset: dict, metric_name: str) -> float:
    return dataset.get("metric_scales", {}).get(metric_name, 1.0)


def compare_regression(
    original_root: Path,
    step1500_root: Path,
    *,
    resamples: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    summary = []
    per_sequence = []
    for dataset_name, dataset in DATASETS.items():
        original = _load_metric_frame(original_root, dataset_name)
        candidate = _load_metric_frame(step1500_root, dataset_name)
        if not original.index.equals(candidate.index):
            raise RuntimeError(f"sequence indices differ for {dataset_name}")

        for metric_name, metric in METRICS.items():
            original_column = _metric_column(original, metric_name, dataset["point_type"])
            candidate_column = _metric_column(candidate, metric_name, dataset["point_type"])
            scale = _metric_scale(dataset, metric_name)
            original_values = original[original_column].to_numpy(dtype=np.float64) * scale
            candidate_values = candidate[candidate_column].to_numpy(dtype=np.float64) * scale
            if not np.isfinite(original_values).all() or not np.isfinite(candidate_values).all():
                raise RuntimeError(f"non-finite {metric_name} values for {dataset_name}")

            candidate_minus_original = candidate_values - original_values
            degradation = (
                -candidate_minus_original if metric["higher_is_better"] else candidate_minus_original
            )
            ci_low, ci_high = paired_bootstrap_ci(
                degradation,
                resamples=resamples,
                seed=seed,
            )
            decision = classify_non_regression(ci_low, ci_high, metric["margin"])
            improvement = (
                candidate_minus_original if metric["higher_is_better"] else -candidate_minus_original
            )
            summary.append(
                {
                    "dataset": dataset["label"],
                    "metric": metric["label"],
                    "sequences": int(original_values.size),
                    "original": float(original_values.mean()),
                    "step1500": float(candidate_values.mean()),
                    "step1500_minus_original": float(candidate_minus_original.mean()),
                    "median_degradation": float(np.median(degradation)),
                    "degradation_ci_low": ci_low,
                    "degradation_ci_high": ci_high,
                    "non_regression_margin": metric["margin"],
                    "improved_sequences": int((improvement > 0).sum()),
                    "unchanged_sequences": int((improvement == 0).sum()),
                    "regressed_sequences": int((improvement < 0).sum()),
                    "decision": decision,
                }
            )
            for sequence_index, original_value, candidate_value, delta in zip(
                original.index,
                original_values,
                candidate_values,
                candidate_minus_original,
                strict=True,
            ):
                per_sequence.append(
                    {
                        "dataset": dataset["label"],
                        "metric": metric["label"],
                        "sequence_index": sequence_index,
                        "original": original_value,
                        "step1500": candidate_value,
                        "step1500_minus_original": delta,
                    }
                )
    return summary, per_sequence


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.inexact):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def compare_forward_parity(upstream_root: Path, optimized_root: Path) -> dict:
    upstream_path = _find_one(upstream_root, "step-*_seq-*_tracks.npz")
    optimized_path = _find_one(optimized_root, "step-*_seq-*_tracks.npz")
    input_keys = (
        "gt_trajectories_2d",
        "gt_trajectories_3d",
        "gt_visibilities_per_view",
        "gt_visibilities_any_view",
        "query_points_2d",
        "query_points_3d",
        "track_upscaling_factor",
    )
    with np.load(upstream_path, allow_pickle=True) as upstream, np.load(
        optimized_path, allow_pickle=True
    ) as optimized:
        identical_inputs = {
            key: _arrays_equal(upstream[key], optimized[key]) for key in input_keys
        }
        upstream_tracks = upstream["pred_trajectories_3d"]
        optimized_tracks = optimized["pred_trajectories_3d"]
        if upstream_tracks.shape != optimized_tracks.shape:
            raise RuntimeError("forward predictions have different trajectory shapes")
        difference = np.abs(optimized_tracks.astype(np.float64) - upstream_tracks.astype(np.float64))
        finite = bool(np.isfinite(upstream_tracks).all() and np.isfinite(optimized_tracks).all())
        trajectory_close = bool(
            np.allclose(upstream_tracks, optimized_tracks, rtol=1e-3, atol=1e-3)
        )
        upstream_visibility = upstream["pred_visibilities_any_view"].astype(bool)
        optimized_visibility = optimized["pred_visibilities_any_view"].astype(bool)
        if upstream_visibility.shape != optimized_visibility.shape:
            raise RuntimeError("forward predictions have different visibility shapes")
        visibility_disagreement = float(
            np.not_equal(upstream_visibility, optimized_visibility).mean() * 100
        )

    dataset_name = next(iter(DATASETS))
    upstream_metrics = _load_metric_frame(upstream_root, dataset_name)
    optimized_metrics = _load_metric_frame(optimized_root, dataset_name)
    if upstream_metrics.shape[0] != 1 or optimized_metrics.shape[0] != 1:
        raise RuntimeError("forward parity roots must each contain exactly one evaluated scene")
    metric_differences = {}
    metric_checks = {}
    for metric_name, metric in METRICS.items():
        upstream_column = _metric_column(
            upstream_metrics, metric_name, DATASETS[dataset_name]["point_type"]
        )
        optimized_column = _metric_column(
            optimized_metrics, metric_name, DATASETS[dataset_name]["point_type"]
        )
        difference_value = _metric_scale(DATASETS[dataset_name], metric_name) * abs(
            float(optimized_metrics.iloc[0][optimized_column])
            - float(upstream_metrics.iloc[0][upstream_column])
        )
        tolerance = 0.01
        metric_differences[metric["label"]] = difference_value
        metric_checks[metric["label"]] = difference_value <= tolerance

    passed = (
        all(identical_inputs.values())
        and finite
        and trajectory_close
        and visibility_disagreement <= 0.01
        and all(metric_checks.values())
    )
    return {
        "passed": passed,
        "upstream_artifact": str(upstream_path),
        "optimized_artifact": str(optimized_path),
        "identical_inputs": identical_inputs,
        "finite_predictions": finite,
        "trajectory_allclose_rtol": 1e-3,
        "trajectory_allclose_atol": 1e-3,
        "trajectory_allclose": trajectory_close,
        "trajectory_mean_abs_difference": float(difference.mean()),
        "trajectory_p99_abs_difference": float(np.quantile(difference, 0.99)),
        "trajectory_max_abs_difference": float(difference.max()),
        "visibility_disagreement_percent": visibility_disagreement,
        "metric_absolute_differences": metric_differences,
    }


def _markdown_report(summary: list[dict], parity: dict | None) -> str:
    lines = ["# MVTracker clean-depth regression evaluation", ""]
    if parity is not None:
        lines.extend(
            [
                "## Forward parity gate",
                "",
                f"**Result:** {'PASS' if parity['passed'] else 'FAIL'}",
                "",
                "| Mean abs trajectory diff | P99 diff | Max diff | Visibility disagreement |",
                "|---:|---:|---:|---:|",
                (
                    f"| {parity['trajectory_mean_abs_difference']:.6g} "
                    f"| {parity['trajectory_p99_abs_difference']:.6g} "
                    f"| {parity['trajectory_max_abs_difference']:.6g} "
                    f"| {parity['visibility_disagreement_percent']:.6g}% |"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Checkpoint comparison",
            "",
            "| Benchmark / depth | Metric | Original GT | Step 1500 | Step 1500 - Original | Degradation 95% CI | Result |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['original']:.2f} "
            f"| {row['step1500']:.2f} | {row['step1500_minus_original']:+.2f} "
            f"| [{row['degradation_ci_low']:.2f}, {row['degradation_ci_high']:.2f}] "
            f"| {row['decision']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--step1500-root", type=Path)
    parser.add_argument("--parity-upstream-root", type=Path)
    parser.add_argument("--parity-optimized-root", type=Path)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.parity_upstream_root is None) != (args.parity_optimized_root is None):
        parser.error("both parity roots must be provided together")
    if args.parity_only and args.parity_upstream_root is None:
        parser.error("--parity-only requires both parity roots")
    if not args.parity_only and (args.original_root is None or args.step1500_root is None):
        parser.error("checkpoint comparison requires --original-root and --step1500-root")

    parity = None
    if args.parity_upstream_root is not None:
        parity = compare_forward_parity(args.parity_upstream_root, args.parity_optimized_root)
    if args.parity_only:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "forward_parity.json").write_text(
            json.dumps(parity, indent=2) + "\n",
            encoding="utf-8",
        )
        report = _markdown_report([], parity).split("## Checkpoint comparison", 1)[0]
        (args.output_dir / "forward_parity.md").write_text(report, encoding="utf-8")
        print(report, end="")
        return 0 if parity["passed"] else 2
    if parity is not None and not parity["passed"]:
        print(json.dumps(parity, indent=2))
        return 2

    summary, per_sequence = compare_regression(
        args.original_root,
        args.step1500_root,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summary).to_csv(args.output_dir / "regression_summary.csv", index=False)
    pd.DataFrame(per_sequence).to_csv(args.output_dir / "sequence_deltas.csv", index=False)
    report = {"forward_parity": parity, "regression": summary}
    (args.output_dir / "regression_summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "regression_report.md").write_text(
        _markdown_report(summary, parity),
        encoding="utf-8",
    )
    print(_markdown_report(summary, parity), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
