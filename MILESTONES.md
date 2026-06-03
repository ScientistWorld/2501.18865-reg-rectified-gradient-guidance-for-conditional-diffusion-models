# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-03 09:16] - method_runs
- Implemented the paper's 1D synthetic REG experiment with conditional and unconditional epsilon-prediction DDPM denoisers.
- Added evaluation artifacts and validated the workspace structure.

### [2026-06-03 09:35] - method_runs
- Tightened the guidance target to differentiate through the deterministic remaining denoising chain, matching the paper's low-dimensional foresight evaluation more closely.
- Submitting a compute job to rerun training and evaluation before claiming `core_claim`.

### [2026-06-03 10:05] - method_runs
- Audited the workspace as a continuation agent: method code implements CFG plus the REG Jacobian correction, and eval remains separated from `method/`.
- Found that stored scores were from the older one-step target and that the default 1D training length was below the paper's 200 epochs; changed the default to 200 epochs and will submit a fresh run.

### [2026-06-03 10:20] - core_claim
- Completed corrected CPU job `df24a0a7-242` with the paper's 200-epoch 1D synthetic setup and full deterministic denoising-chain target.
- REG CFG reduced guidance MAE from 1.2693 to 0.9484 and won 61.40% of grid/time comparisons against vanilla CFG while adding one VJP per guided step.
- `python validate.py --compare` passed after the run.

## Stop Justification
- Completed at milestone `core_claim`.
