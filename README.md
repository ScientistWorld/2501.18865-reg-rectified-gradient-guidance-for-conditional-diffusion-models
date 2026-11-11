# REG: Rectified Gradient Guidance for Conditional Diffusion Models

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

<!-- hardening-note -->
> [!NOTE]
> **Red-teamed and hardened.** A red team tried to score on this environment's hidden test without doing better research, and the environment was then fixed against the shortcuts that worked: see [`REDTEAM.md`](REDTEAM.md) and [`HARDENING.md`](HARDENING.md). The environment as it was before is kept as branch [`pre-hardening`](../../tree/pre-hardening). The runs listed below were made on that earlier version.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2501.18865"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2501.18865-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2501.18865](https://arxiv.org/abs/2501.18865) |
|:---|:---|
| **Domain** | Generative Modeling / Diffusion Sampling |
| **TL;DR** | The paper derives diffusion guidance from joint distribution scaling and adds a Jacobian-based correction that improves existing guidance methods. |

## Summary

Guidance improves conditional diffusion sampling, but common explanations based on marginal distribution scaling do not match the way guidance is implemented. The paper argues that joint denoising-chain scaling is the correct theoretical object and that practical guidance approximates an intractable optimal solution. REG improves that approximation by multiplying the guidance direction with a correction term computed from the denoiser's input gradient. The paper validates this on low-dimensional synthetic tasks and larger ImageNet/COCO generation benchmarks.

## Problem

Conditional diffusion samplers often improve sample fidelity by modifying the denoising prediction at inference time using a conditioning signal. The research question is whether an inference-time guidance rule can better match the ideal conditional denoising direction while preserving diversity and keeping computational overhead bounded.

## The paper's method

The paper reframes diffusion guidance as scaling the joint denoising-chain distribution rather than independently scaling endpoint or per-time marginals. From that view it derives an optimal but intractable guidance target and proposes Rectified Gradient Guidance (REG), an inference-time correction for existing guidance methods.

## Key results of the paper

- In low-dimensional synthetic experiments, REG is more often closer to the optimal guidance target than vanilla CFG.
- On ImageNet 256x256 with DiT-XL/2 and vanilla CFG, the paper reports FID improving from 2.21 to 2.04 and Inception Score from 248.36 to 276.26.
- On COCO 5K with SD-v1-4 and vanilla CFG, the paper reports FID improving from 20.27 to 19.63 and CLIP score from 30.68% to 30.75%.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.5-xhigh` | +116.7 |
| `gpt-5.6-terra-xhigh` | +93.4 |
| `gpt-5.6-sol-xhigh` | +77.9 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2501.18865 --mode innovate --agent claude-code
sw evaluate 2501.18865
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

