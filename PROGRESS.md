# Progress

## What Works

A hardened 1D reproduction of REG (Rectified Gradient Guidance), following the paper's Section 5.1 / Appendix E.1 as released by the authors (`1d-toy/`):
- 20-step DDPM, betas from 1e-4 to 2e-2;
- separate conditional and unconditional epsilon MLPs;
- q(x|y) = ½N(0.5,0.25²) + ½N(1.5,0.25²) and q(x) = ½N(-2,0.5²) + ½N(0,0.5²);
- reward q(x|y)/q(x);
- the optimal foresight guidance estimated from stochastic denoising completions.

The pieces:
- **Problems:** `data/toy_1d.py` (public machinery), `data/toy_1d_train/` (public problems, including the authors' released checkpoint), `eval/test/` (12 hidden problems, builder and assets).
- **Method:** `method/reg_guidance.py`, a guidance rule installed by `scripts/method.sh`.
- **Baselines:** `baseline/vanilla_cfg.py` and `baseline/no_guidance.py`.
- **Evaluators:** `eval/train/`, `eval/test/`. Rules run in a subprocess against a counted denoiser oracle; every metric is recomputed against evaluator-held targets.

## Results (reference REG, w = 1)

| Slice | REG MAE | REG win vs CFG | CFG MAE | no-guidance MAE |
|---|---:|---:|---:|---:|
| Paper problems (`scores.json`) | 0.264 | 0.791 | 0.519 | 0.664 |
| Visible train (`scores_train.json`) | 0.339 | 0.709 | 0.530 | 0.676 |
| Hidden test (`scores_test.json`) | 0.413 | 0.586 | 0.511 | 0.804 |

REG costs 3 oracle units per state (CFG: 2). On the hidden test, a REG variant that evaluates the CFG direction at the Tweedie estimate improves MAE by 0.068 (95% CI −0.103 to −0.037). See `HARDENING.md`.

## Remaining

The 2D example (Table 1) and the ImageNet/text-to-image experiments are not reproduced.

## Deviations from Paper (text)

These follow the authors' released 1D code rather than the paper text:
- unconditional means (-2, 0);
- betas from 1e-4 to 2e-2;
- guidance scale 1, batch 64, residual-GELU MLP.

An independent recomputation reproduces the paper's Figure 5 target only with this version. The target uses 1,000 completions per state instead of 100, to reduce estimator noise.
