#!/bin/bash
set -euo pipefail
cd /home/user

# Save exactly the pre-audit artifacts, including ignored trained weights.
mkdir -p redteam/_state
for file in scoring/scores_train.json scoring/scores_test.json outputs/toy_1d/results.json; do
    cp "$file" "redteam/_state/$(basename "$(dirname "$file")")_$(basename "$file")"
done
restore() {
    cp redteam/_state/scoring_scores_train.json scoring/scores_train.json
    cp redteam/_state/scoring_scores_test.json scoring/scores_test.json
    cp redteam/_state/toy_1d_results.json outputs/toy_1d/results.json
}
trap restore EXIT

for name in \
    reference_seed0 reference_seed1 reference_seed2 \
    better_calibrated_seed0 better_calibrated_seed1 better_calibrated_seed2 \
    better_reward_scale_fix \
    weaker_remove_reg_seed0 weaker_remove_reg_seed1 weaker_remove_reg_seed2 \
    shortcut_zero_guidance \
    shortcut_score_claims shortcut_impossible_fields \
    shortcut_move_baseline shortcut_compute_claim shortcut_bruteforce_target \
    shortcut_replace_fixed_denoisers shortcut_grid_subset shortcut_time_subset \
    shortcut_empty_training_selection shortcut_filter_disk_methods \
    shortcut_underscore_omission shortcut_null_failure shortcut_empty_failure \
    shortcut_score_file_prewrite; do
    bash "redteam/$name/run.sh"
done
python redteam/summarize.py
