# Paper's Method

## Key Contribution

The paper reframes diffusion guidance as scaling the joint denoising-chain distribution rather than independently scaling endpoint or per-time marginals. From that view it derives an optimal but intractable guidance target and proposes Rectified Gradient Guidance (REG), an inference-time correction for existing guidance methods.

## Approach

For classifier-free guidance, the usual guidance direction is the difference between conditional and unconditional denoiser predictions. REG multiplies that direction elementwise by a correction term derived from the Jacobian of the conditional denoiser output with respect to the current noisy sample. In epsilon-prediction DDPM form, the correction is `1 - sqrt(1 - alpha_bar_t) * d(sum epsilon_theta(x_t,t,y))/dx_t`. The same idea is adapted to other samplers and prediction parameterizations in the paper's larger image experiments.

## Main Claims

Primary claim: adding REG to existing guidance methods better approximates the optimal joint-scaling guidance direction and improves conditional generation quality.

Secondary claims: REG is method-agnostic across vanilla, cosine, linear, interval, and auto-guidance variants; it improves FID and Inception/CLIP scores in ImageNet and text-to-image settings; and its overhead is affordable because it adds one gradient-style calculation per guided step.
