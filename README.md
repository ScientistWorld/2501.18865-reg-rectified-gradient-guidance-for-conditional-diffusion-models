# Overview

- **Paper ID:** 2501.18865
- **Title:** REG: Rectified Gradient Guidance for Conditional Diffusion Models
- **Domain:** Generative Modeling / Diffusion Sampling
- **TL;DR:** The paper derives diffusion guidance from joint distribution scaling and adds a Jacobian-based correction that improves existing guidance methods.

## Short Summary

Guidance improves conditional diffusion sampling, but common explanations based on marginal distribution scaling do not match the way guidance is implemented. The paper argues that joint denoising-chain scaling is the correct theoretical object and that practical guidance approximates an intractable optimal solution. REG improves that approximation by multiplying the guidance direction with a correction term computed from the denoiser's input gradient. The paper validates this on low-dimensional synthetic tasks and larger ImageNet/COCO generation benchmarks.

## Key Results

- In low-dimensional synthetic experiments, REG is more often closer to the optimal guidance target than vanilla CFG.
- On ImageNet 256x256 with DiT-XL/2 and vanilla CFG, the paper reports FID improving from 2.21 to 2.04 and Inception Score from 248.36 to 276.26.
- On COCO 5K with SD-v1-4 and vanilla CFG, the paper reports FID improving from 20.27 to 19.63 and CLIP score from 30.68% to 30.75%.

---

# Problem Definition

## Research Question

Conditional diffusion samplers often improve sample fidelity by modifying the denoising prediction at inference time using a conditioning signal. The research question is whether an inference-time guidance rule can better match the ideal conditional denoising direction while preserving diversity and keeping computational overhead bounded.

## Why It Matters

Better guidance improves conditional generation without retraining the base diffusion model. A reliable evaluation should distinguish guidance rules that genuinely approximate the desired denoising direction from rules that only look good on one metric by increasing compute or collapsing diversity.

## Success Criteria

A successful solution produces guidance directions closer to an optimal target on low-dimensional benchmarks and improves downstream conditional generation metrics when applied to pretrained image generators. It should also report the extra per-step computation needed to achieve the improvement.

---

# Paper's Method

## Key Contribution

The paper reframes diffusion guidance as scaling the joint denoising-chain distribution rather than independently scaling endpoint or per-time marginals. From that view it derives an optimal but intractable guidance target and proposes Rectified Gradient Guidance (REG), an inference-time correction for existing guidance methods.

## Approach

For classifier-free guidance, the usual guidance direction is the difference between conditional and unconditional denoiser predictions. REG multiplies that direction elementwise by a correction term derived from the Jacobian of the conditional denoiser output with respect to the current noisy sample. In epsilon-prediction DDPM form, the correction is `1 - sqrt(1 - alpha_bar_t) * d(sum epsilon_theta(x_t,t,y))/dx_t`. The same idea is adapted to other samplers and prediction parameterizations in the paper's larger image experiments.

## Main Claims

Primary claim: adding REG to existing guidance methods better approximates the optimal joint-scaling guidance direction and improves conditional generation quality.

Secondary claims: REG is method-agnostic across vanilla, cosine, linear, interval, and auto-guidance variants; it improves FID and Inception/CLIP scores in ImageNet and text-to-image settings; and its overhead is affordable because it adds one gradient-style calculation per guided step.

---

# Evaluation

## Metrics

- `guidance_mae`: mean absolute error between a candidate guidance direction and the optimal guidance direction on a fixed 1D grid. Lower is better.
- `win_rate`: fraction of grid/time-step cases where a candidate guidance direction has lower absolute error than the competing reference guidance direction. Higher is better.
- `relative_mae_reduction`: relative reduction in guidance error compared with the reference guidance direction. Higher is better.
- `additional_vjp_per_guided_step`: additional vector-Jacobian products required for each guided denoising step. Lower is better and captures the paper's overhead constraint in a machine-independent way.

## Baselines and Targets

The primary baseline is standard classifier-free guidance on the same trained 1D diffusion model. The paper's low-dimensional experiments report that the proposed enhanced guidance wins more often than standard guidance when compared to an optimal target; Table 1 reports win rates above 50% across all 2D time-step/class settings. The reproduction target is to show the same effect on the paper's 1D synthetic mixture setting.

## Evaluation Protocol

Run the method to train conditional and unconditional 1D denoisers on the fixed synthetic Gaussian mixtures, then evaluate guidance directions on the fixed grid `[-1, 2]` at time steps 4, 8, 12, 16, and 20. Evaluation reads `outputs/toy_1d/results.json` and writes `scoring/scores.json`. Future methods must use the same output contract so `scripts/evaluate.sh` can score them without importing their code.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-03 09:08] - none
- Read the paper and identified REG's core computation: a Jacobian-based correction to existing guidance.
- Implemented the 1D synthetic DDPM reproduction and evaluation pipeline.
- Prepared the first compute job to test whether the method runs and supports the low-dimensional core claim.

### [2026-06-03 09:14] - core_claim
- CPU job `026b8819-b0f` trained the conditional and unconditional 1D DDPM denoisers, ran vanilla CFG and REG guidance, and wrote `scoring/scores.json`.
- REG achieved lower optimal-guidance approximation error than vanilla CFG: MAE 2.142 vs 2.435.
- REG won 69.24% of evaluated guidance comparisons vs 30.76% for vanilla CFG, supporting the paper's central claim on the low-dimensional setting.
- `python validate.py --compare` passed inside the job.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
