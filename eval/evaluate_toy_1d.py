"""Evaluation for reproduced outputs.

This file reads artifacts produced by a method run. It does not import from
method/, so future scientist submissions can be evaluated by writing the same
`outputs/toy_1d/results.json` contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty metric list")
    return float(sum(values) / len(values))


def edge_slices(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda r: int(r.get("time_step", 0)))
    edge_count = max(1, len(ordered) // 2)
    return ordered[:edge_count], ordered[-edge_count:]


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


def method_name_for_prefix(prefix: str) -> str:
    return {"cfg": "vanilla_cfg", "reg": "reg_cfg"}.get(prefix, prefix)


def overhead_for_prefix(prefix: str, summary: dict[str, Any]) -> float:
    specific_key = f"{prefix}_additional_vjp_per_guided_step"
    if isinstance(summary.get(specific_key), (int, float)):
        return float(summary[specific_key])
    if prefix == "cfg":
        return 0.0
    if prefix == "reg" and isinstance(summary.get("additional_vjp_per_guided_step"), (int, float)):
        return float(summary["additional_vjp_per_guided_step"])
    return 1.0


def score_prefix(prefix: str, records: list[dict[str, Any]], cfg_mae: float, additional_vjp: float) -> dict[str, float]:
    early_records, late_records = edge_slices(records)
    guidance_mae = mean(
        [float(record[f"{prefix}_mae"]) for record in records if isinstance(record.get(f"{prefix}_mae"), (int, float))]
    )
    metrics = {
        "guidance_mae": guidance_mae,
        "win_rate": mean(
            [
                float(record[f"{prefix}_win_rate"])
                for record in records
                if isinstance(record.get(f"{prefix}_win_rate"), (int, float))
            ]
        ),
        "early_step_guidance_mae": mean(
            [
                float(record[f"{prefix}_mae"])
                for record in early_records
                if isinstance(record.get(f"{prefix}_mae"), (int, float))
            ]
        ),
        "late_step_guidance_mae": mean(
            [
                float(record[f"{prefix}_mae"])
                for record in late_records
                if isinstance(record.get(f"{prefix}_mae"), (int, float))
            ]
        ),
        "early_step_win_rate": mean(
            [
                float(record[f"{prefix}_win_rate"])
                for record in early_records
                if isinstance(record.get(f"{prefix}_win_rate"), (int, float))
            ]
        ),
        "late_step_win_rate": mean(
            [
                float(record[f"{prefix}_win_rate"])
                for record in late_records
                if isinstance(record.get(f"{prefix}_win_rate"), (int, float))
            ]
        ),
        "relative_mae_reduction": float((cfg_mae - guidance_mae) / cfg_mae) if cfg_mae else 0.0,
        "additional_vjp_per_guided_step": additional_vjp,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/toy_1d/results.json")
    parser.add_argument("--scores", default="scoring/scores.json")
    args = parser.parse_args()

    with Path(args.input).open() as f:
        result = json.load(f)

    guidance = result["guidance"]
    summary = guidance["summary"]
    records = guidance.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Expected non-empty guidance.records in method artifact")
    cfg_mae = mean(
        [float(record["cfg_mae"]) for record in records if isinstance(record.get("cfg_mae"), (int, float))]
    )
    prefixes = available_method_prefixes(records)
    if not prefixes:
        raise ValueError("No method metric prefixes found in guidance.records")
    scores = {
        "experiments": {
            "toy_1d_guidance_approximation": {
                "results": {
                    method_name_for_prefix(prefix): score_prefix(
                        prefix,
                        records,
                        cfg_mae,
                        overhead_for_prefix(prefix, summary),
                    )
                    for prefix in prefixes
                }
            }
        }
    }
    Path(args.scores).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.scores).open("w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
