#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
# Runs inside the compute container (workspace at /home/user, no internet).

set -e

cd /home/user

bash scripts/method.sh          # installs outputs/guidance_rules/reg_cfg and scores the paper problems
bash scripts/baseline.sh        # installs vanilla_cfg and no_guidance
bash scripts/evaluate_train.sh  # visible slice -> scoring/scores_train.json
bash scripts/evaluate_test.sh   # hidden slice  -> scoring/scores_test.json
python validate.py --compare
