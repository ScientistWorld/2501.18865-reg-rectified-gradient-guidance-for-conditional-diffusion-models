# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-03 00:00] - method_runs
- Continuation audit confirmed the existing code implements the paper's REG computation rather than a surrogate.
- Added a paper-style stochastic expected-reward target for the 1D optimal guidance estimate and submitted a regeneration job before restoring `core_claim`.

### [2026-06-03 00:00] - method_runs
- Implemented the paper's epsilon-prediction REG update for classifier-free guidance on the 1D synthetic DDPM setting.
- Trained conditional and unconditional denoisers and produced method artifacts under `outputs/toy_1d/`.

### [2026-06-03 00:00] - core_claim
- Evaluated vanilla CFG and REG against the joint-scaling guidance target by differentiating through the remaining deterministic denoising chain.
- REG achieved lower guidance MAE than vanilla CFG and won 61.40% of grid/time-step comparisons while recording the added one VJP per guided step constraint.
- `python validate.py --compare` passed schema, score, separation, and output-contract checks.

## Stop Justification
- Not stopped; stochastic target regeneration is pending before restoring `core_claim`.
