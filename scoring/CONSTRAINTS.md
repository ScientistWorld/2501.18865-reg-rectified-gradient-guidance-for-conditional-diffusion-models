# Scientist Constraints

## Data and problem

The problem is fixed:
- the DDPM schedule;
- the frozen denoisers;
- the evaluation states;
- the reward q(x|y)/q(x);
- the optimal-target estimates.

The public training problems live in `data/toy_1d_train/`, and the hidden test problems in `eval/test/` (not available to methods). Do not modify the problem assets or the evaluators. Do not read anything under `eval/test/`.

## What a method is

A method is a guidance rule (`outputs/guidance_rules/<name>/guidance.py`) that computes guidance for each state from denoiser queries made through the provided `query` function, using the public schedule. It must:
- access the hidden-problem denoisers only through `query`. Loading networks, data mixtures or targets from disk to compute the answer is not allowed;
- be a deterministic per-state function: no pooling of information across the states of a batch or across calls;
- stay within 8 oracle units per state.

Constants or small learned components may be fitted on the public training problems. They must then generalize to the hidden problems through `query` alone. Hand-coding the public data mixtures, or private searches with copies of the public denoisers, do not transfer to the hidden problems and are out of scope.

## Evaluation

`scripts/evaluate.sh`, `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` take no arguments, score every rule directory on disk, and recompute every metric. Cost is measured, not declared.

## Compliance Checklist

- [ ] Rule lives in `outputs/guidance_rules/<name>/guidance.py` and uses only `query` and `schedule` to reach the denoisers
- [ ] Output is a deterministic function of each individual state
- [ ] At most 8 oracle units per state
- [ ] No reads of `eval/test/`, no edits to `data/toy_1d_train/` or the evaluators
