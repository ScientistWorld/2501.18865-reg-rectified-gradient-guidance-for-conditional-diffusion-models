# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-03 09:16] - method_runs
- Implemented the paper's 1D synthetic REG experiment with conditional and unconditional epsilon-prediction DDPM denoisers.
- Added evaluation artifacts and validated the workspace structure.

### [2026-06-03 09:35] - method_runs
- Tightened the guidance target to differentiate through the deterministic remaining denoising chain, matching the paper's low-dimensional foresight evaluation more closely.
- Submitting a compute job to rerun training and evaluation before claiming `core_claim`.

## Stop Justification
- Not stopped. Next job should determine whether the corrected reproduction reaches `core_claim`.
