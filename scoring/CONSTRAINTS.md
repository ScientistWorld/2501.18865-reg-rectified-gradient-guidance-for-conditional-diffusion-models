# Scientist Constraints

## Data

Use only the fixed 1D Gaussian mixture distributions encoded in `data/toy_1d.py` for the primary benchmark. Do not alter mixture means, variances, weights, evaluation grid, or evaluated time steps when reporting `toy_1d_guidance_approximation`.

## Model Scale

Use the same 1D epsilon-prediction DDPM setting for benchmark comparison: 20 diffusion steps, the released 1D-code beta schedule from `0.0001` to `0.02`, MLP denoisers with three 128-unit hidden layers, and the same training sample count unless explicitly reporting a separate scaled experiment. Larger denoisers can improve approximation by capacity alone and are not comparable for the primary score.

## Evaluation

`scripts/evaluate.sh` must read produced artifacts and must not regenerate the method's guidance internally. The reported `additional_vjp_per_guided_step` must count derivative evaluations used by the candidate guidance rule.

## Compliance Checklist

- [ ] Primary benchmark uses the unchanged distributions in `data/toy_1d.py`
- [ ] Primary benchmark uses the unchanged 20-step 1D DDPM schedule and MLP scale
- [ ] Evaluation artifact is `outputs/toy_1d/results.json`
- [ ] `additional_vjp_per_guided_step` reflects actual extra derivative work
