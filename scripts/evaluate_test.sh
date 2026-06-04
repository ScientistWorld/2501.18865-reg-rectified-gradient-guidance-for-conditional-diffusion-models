#!/bin/bash
# Held-out test-slice evaluator.
#
# Contract mirrors scripts/evaluate.sh: pass an optional method output artifact
# or output directory; scores are written under scoring/.  When train scores are
# present, the test slice scores exactly the same method entries.

set -e

cd /home/user

mkdir -p scoring

INPUT="${1:-outputs/toy_1d}"
TRAIN_SCORES="${TRAIN_SCORES:-scoring/scores_train.json}"
python eval/test/evaluate.py --input "$INPUT" --scores scoring/scores_test.json --train-scores "$TRAIN_SCORES"
