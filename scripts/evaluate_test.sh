#!/bin/bash
# Hidden test-slice evaluator: scores every guidance rule found under
# outputs/guidance_rules/ on the hidden problems (eval/test/assets/) and
# writes scoring/scores_test.json.  No arguments; never retrains anything.

set -euo pipefail

cd /home/user

mkdir -p scoring
python eval/test/evaluate.py
