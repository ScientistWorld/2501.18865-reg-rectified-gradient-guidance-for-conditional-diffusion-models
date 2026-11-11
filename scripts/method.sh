#!/bin/bash
# Run the paper's proposed method (REG on classifier-free guidance).
#
# REG is a training-free, sampling-time guidance rule.  Installing it means
# placing its rule file where the evaluators discover methods:
#   outputs/guidance_rules/reg_cfg/guidance.py
# then scoring it on the paper's own 1D problems (scoring/scores.json).
#
# Usage: bash scripts/method.sh

set -euo pipefail

cd /home/user

mkdir -p outputs/guidance_rules/reg_cfg
cp method/reg_guidance.py outputs/guidance_rules/reg_cfg/guidance.py
bash scripts/evaluate.sh
