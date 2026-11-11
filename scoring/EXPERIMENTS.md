# Experiment Overview

## toy_1d_guidance_approximation

Establishes whether a sampling-time guidance rule gets closer to the optimal foresight guidance of a conditional diffusion model than standard classifier-free guidance, and what it costs in extra denoiser evaluations.

The setting is a small 1D diffusion problem where the optimal guidance can be estimated accurately offline but not by the rule itself. The rule only sees the denoisers through counted queries, on problems whose data mixtures and denoisers it has never seen.

- **Benefit:** closeness to the optimal guidance, overall (`guidance_mae`) and as a head-to-head comparison with classifier-free guidance (`win_rate`).
- **Robustness:** the worst noise level (`worst_timestep_win_rate`) and the worst problem (`worst_problem_mae`). A rule cannot score well by working only at some noise levels or on some mixtures.
- **Constraint:** denoiser evaluations per guided state (`oracle_cost_per_state`). The per-state budget is capped, so foresight by brute force is not possible.

The visible slice includes the paper's own 1D problem, twice: with denoisers trained by the paper's recipe and with the authors' released denoisers. It also includes three public paper-like problems. The hidden slice contains twelve hidden paper-like problems.
