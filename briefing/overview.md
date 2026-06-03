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
