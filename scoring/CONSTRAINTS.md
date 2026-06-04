# Scientist Constraints

## Data

Use only the fixed 1D Gaussian mixture distributions encoded in `data/toy_1d.py` for the primary benchmark. Do not alter mixture means, variances, weights, evaluation grid, or evaluated time steps when reporting `toy_1d_guidance_approximation`.

## Benchmark Scope

Report the primary score only on the fixed low-dimensional benchmark encoded by the provided data and output contract. Do not change the number of denoising steps, evaluated states, training sample budget used to produce the artifact, or the output schema when comparing against `scores.json`; report any scaled variant as a separate experiment. Improvements should come from a better guidance rule rather than from changing the benchmark problem or increasing generator capacity.

## Evaluation

`scripts/evaluate.sh` must read produced artifacts and must not regenerate the method's guidance internally. The reported `additional_vjp_per_guided_step` must count derivative evaluations used by the candidate guidance rule.

## Compliance Checklist

- [ ] Primary benchmark uses the unchanged distributions in `data/toy_1d.py`
- [ ] Primary benchmark uses the unchanged low-dimensional benchmark schedule and output schema
- [ ] Evaluation artifact is `outputs/toy_1d/results.json`
- [ ] `additional_vjp_per_guided_step` reflects actual extra derivative work
