# Research Direction

## Research Problem

Improving inference-time guidance for conditional diffusion models: approximating the ideal (foresight) conditional guidance direction from local denoiser information, without retraining or replacing the generator.

## Core Contribution

Work should focus on how conditional and unconditional denoiser predictions, and their input derivatives, are turned into the guidance added at each denoising step. Improvements should bring the guidance closer to the optimal direction, consistently across noise levels and across unseen problems, at a small number of extra denoiser evaluations.

## Approach Scope

Scientists may:
- redesign the guidance correction;
- use different local derivative or Jacobian approximations;
- evaluate the denoisers at other points or noise levels, such as cheap lookahead or predicted clean samples, with every evaluation counted;
- schedule the correction over time;
- calibrate scale policies on the public training problems.

## Out of Bounds

- Brute-force foresight (running many denoising completions per state); the budget forbids it.
- Training a predictor on many synthetic problems to imitate the evaluation target, or hand-coding the data-generating mixtures, instead of defining a guidance rule from denoiser outputs.
- Reading hidden test assets, modifying the fixed denoisers, problem files or evaluators.
- Pooling information across evaluation states or across calls.
