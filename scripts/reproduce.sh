#!/bin/bash
# Reproduce the paper's 1D result end-to-end: build (or reuse) the fixed
# problems, install the reference REG rule and the baselines, and score them
# on the paper problems, the visible slice and the hidden slice.

set -euo pipefail

cd /home/user

bash scripts/build_problems.sh
bash scripts/method.sh
bash scripts/baseline.sh
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
