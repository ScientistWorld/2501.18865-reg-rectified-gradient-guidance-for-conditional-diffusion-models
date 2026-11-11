# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-11-11] - core_claim (hardening, iteration 11)
- The evaluator now runs submitted guidance rules against frozen denoisers through a counted oracle (budget 8 units/state, per-state consistency check) and recomputes every metric against evaluator-held foresight targets; failures get worst values; every rule directory is scored.
- Restored the authors' released 1D setting (uncond mixture (-2,0), betas 1e-4..2e-2, w = 1, upstream MLP); an independent recomputation reproduces the paper's Figure 5 target only with it.
- 12 hidden test problems live only in eval/test/. Reference REG: paper problems MAE 0.264 / win 0.791 (CFG 0.519, no guidance 0.664); hidden MAE 0.413 / win 0.586 (CFG 0.511, no guidance 0.804).
- Hardening check (job 7b9ab485-dbb): a better variant ranks above the reference beyond paired noise, weaker variants and all shortcuts below. Details in HARDENING.md.

### [2026-06-04 19:42] - core_claim
- Repaired the canonical evaluator so `scoring/scores.json` now includes the same full metric family declared in `scoring/reference.json`: overall, early-step, late-step, relative improvement, and derivative-overhead metrics.
- Updated `scripts/run.sh` to regenerate the `outputs/toy_1d` artifact with `scripts/method.sh` before evaluating, so a fresh clone/job does not depend on ignored local outputs.
- Local lightweight validation on the existing artifact passed with `python validate.py --compare`.

### [2026-06-04 19:44] - core_claim
- Final verification removed method-specific instructions from the method-agnostic evaluation brief.
- The canonical evaluator now discovers candidate metric prefixes from `guidance.records`, matching the reusable train/test scoring interface.
- `python validate.py --compare` passed after the verification edits.

### [2026-06-03 00:00] - method_runs
- Continuation audit confirmed the existing code implements the paper's REG computation rather than a surrogate.
- Added a paper-style stochastic expected-reward target for the 1D optimal guidance estimate and submitted a regeneration job before restoring `core_claim`.

### [2026-06-03 00:00] - core_claim
- Job `d504d843-623` regenerated the 1D reproduction using 100 stochastic denoising completions per grid point for the expected-reward target.
- REG achieved guidance MAE 0.3720 versus vanilla CFG 0.7692, a 51.64% relative MAE reduction, with REG winning 84.65% of grid/time-step comparisons.
- The evaluation records the paper's added computation constraint: REG uses one additional VJP per guided step while vanilla CFG uses zero.
- `python validate.py --compare` passed schema, score, separation, and output-contract checks.

### [2026-06-03 00:00] - method_runs
- Implemented the paper's epsilon-prediction REG update for classifier-free guidance on the 1D synthetic DDPM setting.
- Trained conditional and unconditional denoisers and produced method artifacts under `outputs/toy_1d/`.

### [2026-06-03 00:00] - core_claim
- Evaluated vanilla CFG and REG against the joint-scaling guidance target by differentiating through the remaining deterministic denoising chain.
- REG achieved lower guidance MAE than vanilla CFG and won 61.40% of grid/time-step comparisons while recording the added one VJP per guided step constraint.
- `python validate.py --compare` passed schema, score, separation, and output-contract checks.

## Stop Justification
- Completed at milestone `core_claim`.
