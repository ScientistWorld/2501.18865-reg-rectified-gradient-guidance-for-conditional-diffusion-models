# Reproduction Milestones

**Current: core_claim**

## Progress Log

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
- Stopped at `core_claim`: the paper's REG computation runs end-to-end, is evaluated against a stochastic expected-reward guidance target in the 1D Gaussian-mixture setting, and the reusable environment validates successfully.
