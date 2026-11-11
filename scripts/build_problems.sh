#!/bin/bash
# (Re)build the fixed problem assets.  Both builders are resumable and skip
# finished files, so on a clone with the committed assets this is a no-op.
#   data/toy_1d_train/   public training problems (needs scripts/download.sh first)
#   eval/test/assets/    hidden test problems
# Building from scratch trains 24 small denoisers and estimates the foresight
# targets with 1,000 completions per state (GPU recommended, ~30 min).

set -euo pipefail

cd /home/user

python data/build_train_problems.py
python eval/test/build_hidden.py
