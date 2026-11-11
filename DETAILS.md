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

Conditional diffusion samplers steer generation by replacing the conditional noise prediction with a *guided* prediction at every denoising step. The ideal guidance for a reward-tilted target distribution is known in closed form, but it requires foresight: it depends on how the rest of the denoising chain will unfold from the current state to the clean sample. Practical guidance rules must approximate it using only what is available at the current step.

The question is: **using only local information from the denoisers at sampling time, how closely can a guidance rule match the optimal foresight guidance, and at what extra computational cost?**

## Setting

A 1D DDPM (20 steps, linear betas from 1e-4 to 2e-2) with two frozen epsilon-prediction networks: one trained on a conditional data distribution q(x|y) and one trained on an unconditional distribution q(x). Both distributions are 1D Gaussian mixtures. The reward that tilts the conditional model is R0(x) = q(x|y) / q(x). The optimal guidance at a noisy state (x_t, t) is

    g*(x_t, t) = -sqrt(1 - alpha_bar_t) * d/dx_t log E_t(x_t),   E_t(x_t) = E[R0(x_0) | x_t],

where the expectation runs over the conditional model's own stochastic denoising chain from x_t down to x_0. A guidance rule outputs g(x_t, t); the guided prediction is eps_cond(x_t, t) + g(x_t, t).

The networks are deliberately capacity-limited (few steps, small MLPs), so the learned densities differ from the data mixtures. That is what makes the approximation problem non-trivial.

## Why It Matters

Guidance is a central component of modern conditional diffusion models. A rule that better approximates the optimal guidance, without retraining the generator and with bounded extra network evaluations, should transfer to better conditional generation in high-dimensional models, where the foresight computation is impossible.

## Success Criteria

A better rule produces guidance closer to g* than standard classifier-free guidance across noise levels and across different problems (mixtures and denoisers it has never seen), with as little extra denoiser computation as possible. Methods that look ahead along the chain are allowed, but they pay for every denoiser evaluation, and the per-state budget is capped.

---

# Paper's Method

## Key Contribution

The paper reframes diffusion guidance as scaling the joint denoising-chain distribution rather than independently scaling endpoint or per-time marginals. From that view it derives an optimal but intractable guidance target and proposes Rectified Gradient Guidance (REG), an inference-time correction for existing guidance methods. The target is g* = -sqrt(1-alpha_bar_t) grad log E_t, where E_t is the expected reward over the rest of the chain, so computing it needs foresight.

## Approach

For classifier-free guidance, the usual guidance direction is the difference between conditional and unconditional denoiser predictions. REG multiplies that direction elementwise by a correction term derived from the Jacobian of the conditional denoiser output with respect to the current noisy sample. In epsilon-prediction DDPM form:

    eps_REG = eps_c + w (eps_c - eps_u) * (1 - sqrt(1 - alpha_bar_t) * d(sum eps_c)/dx_t)

It comes from the chain rule through the Tweedie estimate x0_hat(x_t), with two simplifications: the reward gradient at x0_hat is replaced by the guidance direction at x_t, and the 1/sqrt(alpha_bar_t) factor is absorbed into w. The paper's larger image experiments adapt the same idea to other samplers and parameterizations.

## Reference implementation in this gym

`method/reg_guidance.py`, installed by `bash scripts/method.sh` as `outputs/guidance_rules/reg_cfg/guidance.py`. It uses w = 1, the default of the authors' 1D code, which matches the unpowered reward of the target. Cost: eps_c with its derivative (2 units) plus eps_u (1 unit), i.e. 3 units, one derivative evaluation more than CFG.

## Main Claims

Primary claim: adding REG to existing guidance methods better approximates the optimal joint-scaling guidance direction and improves conditional generation quality.

Secondary claims: REG is method-agnostic across vanilla, cosine, linear, interval, and auto-guidance variants; it improves FID and Inception/CLIP scores in ImageNet and text-to-image settings; and its overhead is affordable because it adds one gradient-style calculation per guided step.

## 1D setting as released by the authors

The paper's text gives the unconditional mixture as N(-1, 0.5²)/N(1, 0.5²) and betas 0.001–0.2. The released code and checkpoint (`1d-toy/`, which produced Figures 1 and 5) instead use:
- unconditional means (-2, 0);
- betas 1e-4–2e-2;
- guidance scale 1;
- a residual-GELU MLP with sinusoidal embeddings;
- batch size 64.

This gym follows the released version. An independent recomputation reproduces the Figure 5 target only with it.

---

# Evaluation

## What you submit

A guidance rule: a directory `outputs/guidance_rules/<method_name>/` containing `guidance.py` (plus any small files it reads from its own directory). Every directory under `outputs/guidance_rules/` is scored as a method, including directories without a valid rule, which get the worst scores. Nothing else you write is read: the evaluators never trust numbers, flags or claims produced by a method.

```python
def guidance(x, t, query, schedule):
    """
    x        float64 array (B,)  noisy states x_t
    t        int64 array (B,)    1-indexed time steps, 1..20 (t=1 is the last denoising step)
    query    query(model, xq, tq, deriv=False)
               model "cond" or "uncond"; xq float array (B,); tq int or int array (B,) in 1..20
               -> eps (B,)            or (eps, d eps/dx) (B,), (B,) when deriv=True
    schedule dict: "T" and float64 arrays "betas", "alphas", "alpha_bars",
             "posterior_variance" (index t-1)
    returns  g, float array (B,)    guided prediction = eps_cond(x_t, t) + g
    """
```

Rules for the rule:
- **The only access to the denoisers is `query`.** The rule runs in a separate process and does not hold the networks.
- **Cost:** each `query` call costs 1 unit per state, or 2 with `deriv=True`. The budget is **8 units per state**; exceeding it fails the method.
- **Per-state:** the output for a state must not depend on which other states share its batch. The evaluator scores every state twice, under two different random batchings; any difference beyond 1e-5 fails the method. Rules must be deterministic.
- **Failure:** a crash, timeout, wrong output shape, non-finite output, query with `t` outside 1..20, budget overrun or batching dependence all fail the method. A failed method receives the worst value of every metric.

The reference CFG direction at guidance scale 1 is `query("cond") - query("uncond")` (2 units).

## Metrics (experiment `toy_1d_guidance_approximation`)

All metrics are computed by the evaluator from the raw guidance values, against an optimal-target estimate it holds itself. That estimate uses 1,000 stochastic denoising completions per state.

| metric | direction | meaning |
|---|---|---|
| `guidance_mae` | lower | mean \|g - g*\| over all evaluation states; per-state error clipped at 4 |
| `win_rate` | higher | fraction of states where the rule is closer to g* than vanilla CFG (scale 1, computed by the evaluator); ties count 1/2 |
| `worst_timestep_win_rate` | higher | minimum over the 20 time steps of the per-step win rate |
| `worst_problem_mae` | lower | largest per-problem MAE (robustness across mixtures and denoisers) |
| `oracle_cost_per_state` | lower | mean denoiser units spent per state, as counted by the evaluator |

Ranking uses the coefficients in `scoring/reference.json`.

## Slices

- **Paper** (`bash scripts/evaluate.sh` → `scoring/scores.json`): the paper's 1D mixtures, with denoisers trained by the paper's recipe and with the authors' released denoisers.
- **Train / visible** (`bash scripts/evaluate_train.sh` → `scoring/scores_train.json`): the two paper problems plus three public paper-like mixtures. Denoisers, states and targets are in `data/toy_1d_train/`, so you can develop against them.
- **Test / hidden** (`bash scripts/evaluate_test.sh` → `scoring/scores_test.json`): hidden paper-like mixtures with their own denoisers and hidden evaluation states. It covers all 20 time steps, with x_t uniform in [-1, 2].

All three scripts take no arguments and score every rule on disk.

## Baselines

- `vanilla_cfg`: classifier-free guidance at scale 1 (2 units).
- `no_guidance`: g = 0 (0 units).

The comparison direction used by `win_rate` is fixed inside the evaluator and cannot be changed by a method.

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

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
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

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
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

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

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

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

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

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 24m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 35m | **GPU**: 1.0h
- **Jobs**: 6 total (6 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
