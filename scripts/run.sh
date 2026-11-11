#!/bin/bash
# Job script: build the fixed problems (resumable), then run the hardening
# check, which ends by regenerating the canonical reference/baseline scores.

set -euo pipefail

cd /home/user

nvidia-smi -L || true
python -c "import torch, os; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cpus', os.cpu_count())"
bash scripts/build_problems.sh
bash redteam/hardening_check/run_all.sh
python validate.py || true
