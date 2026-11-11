#!/bin/bash
# Visible training-slice evaluator: scores every guidance rule found under
# outputs/guidance_rules/ on all public problems (data/toy_1d_train/) and
# writes scoring/scores_train.json.  No arguments.

set -euo pipefail

cd /home/user

mkdir -p scoring
python eval/train/evaluate.py
