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
