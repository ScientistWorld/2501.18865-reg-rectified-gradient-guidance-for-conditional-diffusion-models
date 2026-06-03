# Reproduction Milestones

**Current: core_claim**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-06-03 09:08] - none
- Read the paper and identified REG's core computation: a Jacobian-based correction to existing guidance.
- Implemented the 1D synthetic DDPM reproduction and evaluation pipeline.
- Prepared the first compute job to test whether the method runs and supports the low-dimensional core claim.

### [2026-06-03 09:14] - core_claim
- CPU job `026b8819-b0f` trained the conditional and unconditional 1D DDPM denoisers, ran vanilla CFG and REG guidance, and wrote `scoring/scores.json`.
- REG achieved lower optimal-guidance approximation error than vanilla CFG: MAE 2.142 vs 2.435.
- REG won 69.24% of evaluated guidance comparisons vs 30.76% for vanilla CFG, supporting the paper's central claim on the low-dimensional setting.
- `python validate.py --compare` passed inside the job.

## Stop Justification
- Stopping at `core_claim`: the reusable environment implements and evaluates the paper's actual REG computation on the paper's 1D Gaussian-mixture setup. Larger ImageNet and COCO reproductions remain future work because they require substantially more diffusion-model generation and evaluation.
