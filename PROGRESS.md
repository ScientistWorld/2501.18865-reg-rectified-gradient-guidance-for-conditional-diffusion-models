# Progress

## What Works

Implemented a self-contained 1D synthetic reproduction of REG for classifier-free guidance. The method trains conditional and unconditional epsilon-prediction DDPM denoisers on the paper's Gaussian mixture setup, computes vanilla CFG and REG guidance adjustments, and compares both with the deterministic optimal joint-scaling guidance target.

## Results

Pending compute job. `scripts/run.sh` trains the 1D models, evaluates guidance approximation, writes `scoring/scores.json`, and runs `python validate.py --compare`.

## Remaining

Run the submitted job and inspect whether REG achieves lower guidance MAE and a win rate above 50%. If successful, update milestone to `core_claim`; if not, adjust training duration/hyperparameters and resubmit.

## Issues

The full ImageNet and COCO experiments require large pretrained diffusion models and many generated images. The first reproduction targets the paper's low-dimensional synthetic claim because it implements the same REG computation and is feasible within the initial job.

## Deviations from Paper

The paper's 1D experiment evaluates the optimal expected reward by sampling 100 denoising-chain completions at each observed `x_t`. This reproduction uses the deterministic-sampler optimal target from the paper's derivation, `grad log R0(x0_hat)`, to keep evaluation affordable while preserving the REG computation in Eq. 21. The full image-generation tables are documented in `reference.json` as future zero-weight targets but are not part of the first submitted run.
