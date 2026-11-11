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
