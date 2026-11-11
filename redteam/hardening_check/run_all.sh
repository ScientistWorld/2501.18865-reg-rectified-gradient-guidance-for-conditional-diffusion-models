#!/bin/bash
# Hardening check (iteration 11).  Runs the reference, the baselines, better /
# weaker variants and the shortcut payloads in rules/ through the official
# evaluators (scripts/evaluate_train.sh, scripts/evaluate_test.sh, no
# arguments), then two file-level shortcuts, then restores the canonical
# reference-only score files.  Results: redteam/hardening_check/results/.
#
# Requires the fixed problem assets (scripts/build_problems.sh).

set -uo pipefail
cd /home/user

HC=redteam/hardening_check
RES=$HC/results
RULES=outputs/guidance_rules
DUMP=eval/test/.analysis
BACKUP=$(mktemp -d)
mkdir -p "$RES" "$RULES"

restore() {
    if ls "$BACKUP"/*.pt >/dev/null 2>&1; then cp "$BACKUP"/*.pt data/toy_1d_train/; fi
    rm -rf "$DUMP"
}
trap restore EXIT

echo "== canonical rules"
rm -rf "$RULES"; mkdir -p "$RULES"
bash scripts/baseline.sh
mkdir -p "$RULES/reg_cfg"; cp method/reg_guidance.py "$RULES/reg_cfg/guidance.py"

echo "== calibration fit (public training problems only)"
python "$HC/fit_calibration.py" | tee "$RES/calibration.log"

echo "== install payloads"
for d in "$HC"/rules/*/; do
    name=$(basename "$d")
    rm -rf "$RULES/$name"; cp -r "$d" "$RULES/$name"
done
for n in reg_lookahead1 reg_lookahead2; do cp "$HC/rules/_lookahead_lib.py" "$RULES/$n/"; done
ls "$RULES" | tee "$RES/cohort_methods.txt"

echo "== cohort: visible train slice"
bash scripts/evaluate_train.sh 2>&1 | tee "$RES/cohort_train.log"
cp scoring/scores_train.json "$RES/cohort_scores_train.json"

echo "== cohort: hidden test slice (scores_train.json removed first: the test needs no allowlist)"
rm -f scoring/scores_train.json
REG_GYM_DUMP_DIR="$DUMP" bash scripts/evaluate_test.sh 2>&1 | tee "$RES/cohort_test.log"
cp scoring/scores_test.json "$RES/cohort_scores_test.json"

echo "== summary"
python "$HC/summarize.py" "$DUMP" > "$RES/summarize.log" 2>&1 || tail -20 "$RES/summarize.log"
rm -rf "$DUMP"

echo "== back to canonical rules (reference + baselines)"
for d in "$HC"/rules/*/; do rm -rf "$RULES/$(basename "$d")"; done
ls "$RULES"

echo "== special 1: prewritten perfect scores_test.json"
cat > scoring/scores_test.json <<'JSON'
{"experiments": {"toy_1d_guidance_approximation": {"results": {"reg_cfg": {"guidance_mae": 0.0, "win_rate": 1.0, "worst_timestep_win_rate": 1.0, "worst_problem_mae": 0.0, "oracle_cost_per_state": 0.0}}}}}
JSON
bash scripts/evaluate_test.sh > "$RES/special_prewrite.log" 2>&1
cp scoring/scores_test.json "$RES/special_prewrite_scores_test.json"

echo "== special 2: overwrite the public denoisers with a degenerate network"
cp data/toy_1d_train/*.pt "$BACKUP/"
python - <<'PY'
import glob, torch
for p in glob.glob("data/toy_1d_train/*.pt"):
    sd = torch.load(p, weights_only=True)
    torch.save({k: torch.zeros_like(v) for k, v in sd.items()}, p)
PY
bash scripts/evaluate_test.sh > "$RES/special_public_denoisers.log" 2>&1
cp scoring/scores_test.json "$RES/special_public_denoisers_scores_test.json"
cp "$BACKUP"/*.pt data/toy_1d_train/

echo "== canonical scores (reference + baselines)"
bash scripts/evaluate.sh 2>&1 | tail -5
bash scripts/evaluate_train.sh 2>&1 | tail -5
bash scripts/evaluate_test.sh 2>&1 | tail -5
cp scoring/scores.json scoring/scores_train.json scoring/scores_test.json "$RES/" 2>/dev/null || true
echo "== done"
