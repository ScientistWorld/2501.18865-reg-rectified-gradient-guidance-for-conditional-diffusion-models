#!/bin/bash
# Visible train-slice evaluator.
#
# Contract mirrors scripts/evaluate.sh: pass an optional method output artifact
# or output directory; scores are written under scoring/.

set -e

cd /home/user

mkdir -p scoring

INPUT="${1:-outputs/toy_1d}"
python eval/train/evaluate.py --input "$INPUT" --scores scoring/scores_train.json
