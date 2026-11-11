#!/bin/bash
# Score every guidance rule under outputs/guidance_rules/ on the paper's own
# 1D problems (paper mixtures; our denoisers and the authors' released
# denoisers) and write scoring/scores.json.
#
# Uses eval/ only (never method/).  The evaluator runs each rule against the
# frozen denoisers through a counted oracle and recomputes every metric.
#
# Usage: bash scripts/evaluate.sh

set -euo pipefail

cd /home/user

mkdir -p scoring
python eval/train/evaluate.py --problems paper_mixture paper_checkpoint --slice-name paper --scores scoring/scores.json
