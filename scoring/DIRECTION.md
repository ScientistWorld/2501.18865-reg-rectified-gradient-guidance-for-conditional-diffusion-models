# Research Direction

## Research Problem

This benchmark is about improving inference-time guidance for conditional diffusion models. The central question is whether a guidance rule can better approximate the ideal conditional denoising direction without retraining or replacing the base generator.

## Core Contribution

Work should focus on denoising-time guidance corrections: how the conditional and unconditional denoiser predictions are transformed into the final guided update. Improvements should target closer approximation to the optimal guidance direction and should report their derivative or compute overhead.

## Approach Scope

Scientists may redesign the guidance correction, use different local derivative approximations, change how the correction is scheduled over time, or modify the guidance scale policy. The base problem remains conditional diffusion guidance on the provided denoiser outputs.

## Out of Bounds

- Replacing diffusion guidance with a non-diffusion generator or a direct sampler from the known mixture.
- Training a separate predictor to imitate the evaluation target instead of defining an inference-time guidance rule.
- Changing the benchmark distributions or evaluation grid for the primary score.
- Improving the score only by increasing denoiser capacity or training data.
