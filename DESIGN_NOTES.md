# Train/Test Evaluation Design Notes

## Checked

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The train script invokes `eval/train/evaluate.py` and writes `scoring/scores_train.json`; the test script invokes `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Checked that both slice evaluators read method artifacts from the `outputs/toy_1d/results.json` contract and do not import from `method/`.
- Grepped for train/test cross-imports, hardcoded `METHOD_NAME` defaults, paper-agreement metric names, and train/test score stubs.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.
- Ran `bash -n` on both train/test evaluator scripts.
- Parsed `scoring/reference.json` with Python because `jq` is not installed in this environment.

## Changed

- Added low-noise and high-noise edge metrics to both `eval/train/evaluate.py` and `eval/test/evaluate.py`:
  - `early_step_guidance_mae`
  - `late_step_guidance_mae`
  - `early_step_win_rate`
  - `late_step_win_rate`
- Kept the held-out test evaluator lightweight: it still only re-scores already-produced guidance records and does not rerun training, inference, or method code.
- Added the new metrics to `scoring/reference.json` with coefficients and descriptions.
- Added missing `description` fields for all pre-existing active metrics in `scoring/reference.json`.

## Runtime Concerns For Next Iteration

- The split is necessarily small because the current artifact records only five evaluated time steps. The added edge metrics improve triangulation, but each edge metric may be based on one held-out time step unless future method artifacts include more evaluated steps.
- The evaluator discovers candidate methods from metric prefixes ending in `_mae` plus matching `_win_rate`. A future method must emit both fields per time-step record to be scored.
- `additional_vjp_per_guided_step` falls back to `1.0` for non-CFG methods if a method-specific overhead summary is absent. The runner should confirm future method artifacts report this explicitly when overhead differs.
- Full `git status`/`git diff --stat` is currently affected by unrelated root `README.md` and `manifest.json` character-device entries. Path-limited git commands were used to avoid touching those unrelated files.

## Post-Run Assessment

- Job `bb65fe94-e5a` ran `scripts/evaluate_train.sh outputs/toy_1d` and `scripts/evaluate_test.sh outputs/toy_1d` successfully on CPU.
- Both split score files contain real scheduled-run values for `vanilla_cfg` and `reg_cfg`; there are no null metrics.
- Train/test are close but not byte-identical. REG primary win rate is 0.844961 on train and 0.848837 on test; REG relative MAE reduction is 0.492609 on train and 0.547741 on test.
- The split preserves the intended signal: REG beats vanilla CFG on benefit metrics in both slices while carrying the one-additional-VJP constraint cost, and the held-out slice scores different time steps.
- Plain post-design validation initially failed because `reference.json` carried a 2D Table 1 without-REG win fraction as if it were the paper's 1D baseline target. The paper reports 1D qualitatively and Table 1 quantitatively for 2D, so the baseline primary reference was removed and the REG Table 1 aggregate was corrected to 0.6708.
