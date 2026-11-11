#!/bin/bash
# Install baseline guidance rules (vanilla CFG at scale 1, and no guidance).
#
# Usage:
#   bash scripts/baseline.sh            # install all baselines
#   bash scripts/baseline.sh <name>     # vanilla_cfg | no_guidance
#
# Baselines are scored like any other method by scripts/evaluate*.sh.

set -euo pipefail

cd /home/user

for name in ${1:-vanilla_cfg no_guidance}; do
    mkdir -p "outputs/guidance_rules/$name"
    cp "baseline/$name.py" "outputs/guidance_rules/$name/guidance.py"
done
