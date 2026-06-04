"""Held-out test-slice scoring for the toy 1D guidance benchmark.

This evaluator mirrors eval/train/evaluate.py but uses the complementary
time-step split.  It reads method artifacts only and never imports from method/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPERIMENT = "toy_1d_guidance_approximation"
SLICE = "test"


def discover_artifacts(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    artifacts: list[Path] = []
    root_result = path / "results.json"
    if root_result.is_file():
        artifacts.append(root_result)
    artifacts.extend(sorted(p for p in path.glob("*/results.json") if p != root_result))
    if not artifacts:
        raise FileNotFoundError(f"No results.json artifacts found under: {path}")
    return artifacts


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Artifact must contain a JSON object: {path}")
    return data


def selected_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    records = result.get("guidance", {}).get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Expected non-empty guidance.records in method artifact")
    ordered = sorted(records, key=lambda r: int(r.get("time_step", 0)))
    picked = [record for idx, record in enumerate(ordered) if idx % 2 == 1]
    if not picked:
        raise ValueError("Test slice selected no guidance records")
    return picked


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty metric list")
    return float(sum(values) / len(values))


def available_method_prefixes(records: list[dict[str, Any]]) -> list[str]:
    prefixes: set[str] = set()
    for record in records:
        for key, value in record.items():
            if not isinstance(value, (int, float)) or not key.endswith("_mae"):
                continue
            prefix = key[: -len("_mae")]
            if f"{prefix}_win_rate" in record:
                prefixes.add(prefix)
    ordered = [prefix for prefix in ("cfg", "reg") if prefix in prefixes]
    ordered.extend(sorted(prefixes - set(ordered)))
    return ordered


def method_name_for_prefix(prefix: str, artifact: Path, multi_artifact: bool) -> str:
    base = {"cfg": "vanilla_cfg", "reg": "reg_cfg"}.get(prefix, prefix)
    if not multi_artifact:
        return base
    run_name = artifact.parent.name
    if run_name in ("toy_1d", "outputs", "."):
        return base
    return f"{run_name}_{base}"


def metric_from_summary(summary: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def score_artifact(path: Path, multi_artifact: bool) -> dict[str, dict[str, float]]:
    result = load_json(path)
    records = selected_records(result)
    summary = result.get("guidance", {}).get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    prefixes = available_method_prefixes(records)
    if not prefixes:
        raise ValueError(f"No method metric prefixes found in guidance.records: {path}")

    cfg_mae = mean([float(record["cfg_mae"]) for record in records if isinstance(record.get("cfg_mae"), (int, float))])
    out: dict[str, dict[str, float]] = {}
    for prefix in prefixes:
        method = method_name_for_prefix(prefix, path, multi_artifact)
        guidance_mae = mean(
            [float(record[f"{prefix}_mae"]) for record in records if isinstance(record.get(f"{prefix}_mae"), (int, float))]
        )
        win_rate = mean(
            [
                float(record[f"{prefix}_win_rate"])
                for record in records
                if isinstance(record.get(f"{prefix}_win_rate"), (int, float))
            ]
        )
        metrics = {
            "guidance_mae": guidance_mae,
            "win_rate": win_rate,
        }
        if prefix != "cfg":
            metrics["relative_mae_reduction"] = float((cfg_mae - guidance_mae) / cfg_mae) if cfg_mae else 0.0
        overhead = metric_from_summary(
            summary,
            f"{prefix}_additional_vjp_per_guided_step",
            "additional_vjp_per_guided_step" if prefix == "reg" else "",
        )
        if overhead is None:
            overhead = 0.0 if prefix == "cfg" else 1.0
        metrics["additional_vjp_per_guided_step"] = overhead
        out[method] = metrics
    return out


def train_method_keys(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    with path.open() as f:
        data = json.load(f)
    exp = data.get("experiments", {}).get(EXPERIMENT, {})
    results = exp.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"Train scores missing experiments.{EXPERIMENT}.results: {path}")
    return [key for key in results if not key.startswith("_")]


def build_scores(input_path: Path, train_scores: Path) -> dict[str, Any]:
    artifacts = discover_artifacts(input_path)
    multi_artifact = len(artifacts) > 1
    results: dict[str, dict[str, float]] = {}
    for artifact in artifacts:
        for method, metrics in score_artifact(artifact, multi_artifact).items():
            if method in results:
                raise ValueError(f"Duplicate method key after artifact discovery: {method}")
            results[method] = metrics

    expected_methods = train_method_keys(train_scores)
    if expected_methods is not None:
        missing = [method for method in expected_methods if method not in results]
        if missing:
            raise ValueError(f"Test slice is missing methods scored by train slice: {missing}")
        results = {method: results[method] for method in expected_methods}

    return {
        "_metadata": {
            "slice": SLICE,
            "split": "interleaved_time_steps_odd_indices",
            "input": str(input_path),
            "train_scores": str(train_scores),
            "artifacts": [str(path) for path in artifacts],
        },
        "experiments": {
            EXPERIMENT: {
                "_metadata": {"slice": SLICE},
                "results": results,
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/toy_1d")
    parser.add_argument("--scores", default="scoring/scores_test.json")
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    args = parser.parse_args()

    scores = build_scores(Path(args.input), Path(args.train_scores))
    output = Path(args.scores)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
