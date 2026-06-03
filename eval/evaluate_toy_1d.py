"""Evaluation for reproduced outputs.

This file reads artifacts produced by a method run. It does not import from
method/, so future scientist submissions can be evaluated by writing the same
`outputs/toy_1d/results.json` contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/toy_1d/results.json")
    parser.add_argument("--scores", default="scoring/scores.json")
    args = parser.parse_args()

    with Path(args.input).open() as f:
        result = json.load(f)

    summary = result["guidance"]["summary"]
    scores = {
        "experiments": {
            "toy_1d_guidance_approximation": {
                "results": {
                    "vanilla_cfg": {
                        "guidance_mae": summary["cfg_mae"],
                        "win_rate": summary["cfg_win_rate"],
                        "additional_vjp_per_guided_step": 0.0,
                    },
                    "reg_cfg": {
                        "guidance_mae": summary["reg_mae"],
                        "win_rate": summary["reg_win_rate"],
                        "relative_mae_reduction": summary["relative_mae_reduction"],
                        "additional_vjp_per_guided_step": summary["additional_vjp_per_guided_step"],
                    },
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
