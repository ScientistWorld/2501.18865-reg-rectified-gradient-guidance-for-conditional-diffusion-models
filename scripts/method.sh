#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python method/train_toy_1d.py "$@"
bash scripts/evaluate.sh outputs/toy_1d/results.json
