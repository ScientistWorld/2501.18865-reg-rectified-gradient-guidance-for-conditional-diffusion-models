#!/bin/bash
# Reproduce the reference end to end inside the compute container:
# build (or reuse) the fixed problems, install the reference REG rule and the
# baselines, and score them on the paper problems, the visible slice and the
# hidden slice.  The hardening check is `bash redteam/hardening_check/run_all.sh`.

set -euo pipefail

cd /home/user

bash scripts/reproduce.sh
python validate.py
