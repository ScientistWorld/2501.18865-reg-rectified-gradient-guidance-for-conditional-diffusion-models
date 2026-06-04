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
    reg_overhead = float(summary.get("additional_vjp_per_guided_step", 1.0))
    scores = {
        "experiments": {
            "toy_1d_guidance_approximation": {
                "results": {
                    "vanilla_cfg": score_prefix("cfg", records, cfg_mae, 0.0),
                    "reg_cfg": score_prefix("reg", records, cfg_mae, reg_overhead),
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
