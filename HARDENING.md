# Hardening report (iteration 11)

## Summary

The old hidden test ranked what the method wrote: per-record MAE and win-rate numbers and a self-declared overhead. It also used a setup that departed from the paper's own 1D experiment. In that setup the optimal target was close to zero, so returning constant zero beat the reference by a wide margin. Both problems are fixed.

1. **The evaluator recomputes everything.**
   - A method now submits a guidance rule (`outputs/guidance_rules/<name>/guidance.py`), not numbers.
   - `eval/` runs each rule in an isolated subprocess (`python -I`). The rule reaches the frozen denoisers only through a counted `query` oracle, with a budget of 8 units per state.
   - The evaluator holds the optimal-target estimates, the evaluation states and the vanilla-CFG comparison direction. It computes every metric from the raw guidance values.
   - Each state is scored under two different random batchings, so rules that pool information across states fail.
   - Any failure (crash, timeout, NaN, wrong shape, t outside 1..20, budget overrun, batching dependence, missing file) gives the worst value of every metric.
   - Every rule directory on disk is scored; there is no allowlist and no underscore filtering.
2. **Hidden problems.** Twelve hidden paper-like problems live only in `eval/test/`:
   - mixtures drawn with a seed that exists only there;
   - their own denoisers, trained with the paper's recipe;
   - hidden states, all 20 steps, x_t uniform in [-1, 2];
   - targets from 1,000 stochastic completions per state.
   
   `eval/test/build_hidden.py` is the only code that can regenerate them. The test never retrains; it scores a rule in a few seconds on CPU.
3. **Paper's own 1D setting restored.** This follows the authors' released code and checkpoint (`1d-toy/`), which produced Figures 1 and 5:
   - betas from 1e-4 to 2e-2 (the old gym: 0.001–0.2);
   - guidance scale 1, the scale of the target's reward (old gym: 1.5);
   - the residual-GELU MLP with scale-25 sinusoidal input embedding, batch 64, posterior-variance sampler;
   - **unconditional mixture ½N(-2, 0.5²) + ½N(0, 0.5²)**, taken from the released `.pkl` config.
   
   The paper's text says N(-1,0.5²)/N(1,0.5²). An independent pure-Python recomputation of the target on the authors' checkpoint reproduces Figure 5 at t=8 only with (-2, 0): x = 0, 1, 1.5 give -0.60, -1.68, -0.57 against the figure's ≈ -0.65, -1.5, -0.7. With (-1, 1) they give -0.35, -0.14, -0.18.

   Job 1 was run with (-1, 1) and is kept in `redteam/hardening_check/results_run1_text_mixture/`. There zero guidance was nearly as good as REG on the paper problems (MAE 0.339 vs 0.314) and better than vanilla CFG on the test (0.430 vs 0.453).

Everything below is from job `7b9ab485-dbb` with the final code. The cohort ran through `bash scripts/evaluate_test.sh` with no arguments, after deleting `scores_train.json`. Full tables are in `redteam/hardening_check/results/SUMMARY.md`; payloads are in `redteam/hardening_check/rules/`. Rerun with `bash redteam/hardening_check/run_all.sh`.

**Rank used in this report.** Rank = Σ |coef| × directed improvement over `reg_cfg` / σ. σ is the population standard deviation of each metric over the 11 honest rules. Coefficients are those in `scoring/reference.json`.

## Do better results now score higher?

Yes. Every rule is deterministic and the denoisers and targets are frozen, so reruns of the reference give bit-identical scores: the cohort and canonical runs match exactly. The remaining noise comes from which problems and states are sampled. I measured it with a paired cluster bootstrap over the 12 hidden problems and their states.

| Variant (hidden test) | MAE | win | worst-t win | worst-problem MAE | cost | ΔMAE vs ref [95% CI] | Δwin [95% CI] | problems improved | rank |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| **Better:** `reg_tweedie_point`: REG Jacobian, CFG direction evaluated at x0_hat | 0.345 | 0.598 | 0.422 | 0.661 | 4 | **−0.068 [−0.103, −0.037]** | +0.012 [−0.010, +0.036] | 11/12 | **+0.64** |
| **Better:** `reg_exact_jacobian`: keeps the 1/sqrt(ᾱ) factor | 0.409 | 0.608 | 0.435 | 0.794 | 3 | −0.004 [−0.011, +0.003] | **+0.022 [+0.014, +0.031]** | 8/12 | **+0.31** |
| `reg_taylor_shift` | 0.400 | 0.583 | 0.411 | 0.688 | 4 | −0.013 [−0.040, +0.014] | −0.003 [−0.024, +0.020] | 7/12 | +0.07 |
| **Reference** `reg_cfg` (REG, w=1) | 0.413 | 0.586 | 0.383 | 0.801 | 3 | 0 | 0 | — | 0 |
| `reg_calibrated_per_t` (20 scales fitted on train) | 0.409 | 0.501 | 0.344 | 0.765 | 3 | −0.003 [−0.037, +0.031] | −0.085 [−0.108, −0.060] | 6/12 | −0.72 |
| `reg_calibrated` (one scale fitted on train, 0.81) | 0.425 | 0.499 | 0.340 | 0.802 | 3 | +0.012 [−0.016, +0.040] | −0.087 [−0.109, −0.062] | 5/12 | −0.90 |
| **Weaker:** `vanilla_cfg` (REG removed) | 0.511 | 0.500 | 0.500 | 0.727 | 2 | +0.098 [+0.010, +0.188] | −0.086 [−0.184, +0.015] | 3/12 | −0.97 |
| `reg_lookahead1` (5 units) | 0.432 | 0.548 | 0.074 | 0.794 | 5 | +0.019 [+0.001, +0.035] | −0.038 | 2/12 | −1.31 |
| `reg_lookahead2` (7 units) | 0.470 | 0.534 | 0.062 | 0.816 | 7 | +0.057 [+0.029, +0.084] | −0.052 | 1/12 | −2.10 |
| **Weaker:** `reg_flipped_correction` | 0.748 | 0.267 | 0.166 | 1.190 | 3 | +0.335 | −0.319 | 1/12 | −5.94 |
| **Weaker:** `no_guidance` | 0.804 | 0.264 | 0.080 | 1.320 | 0 | +0.391 [+0.312, +0.474] | −0.322 | 0/12 | −6.12 |

**Before (REDTEAM.md, old test).**

| Variant | Old test rank (R10) | MAE |
|---|---:|---:|
| Reference | 0 | 0.375 |
| Calibrated REG (multipliers 0.19–0.26) | +2.09 to +2.42 | 0.10–0.12 |
| Scale-one fix | +1.05 | |
| Remove REG | −6.2 to −6.5 | |

Calibration's old "gain" came from the mis-specified target being near zero. With the paper's setting, the fitted scale is 0.81 and no longer helps. The scale-one fix is now part of the reference (restored from the paper's code).

Train shows the same ordering: Tweedie point +0.35, per-t calibration +0.09, then the reference, vanilla −2.42, zero −4.40.

## Major shortcuts: fix, and score before and after

"Before" figures are from REDTEAM.md: test MAE / win / overhead, with the R10 rank against the old reference. "After" figures are from the hidden test with the final code; the rank uses the cohort σ above.

| Shortcut | Before | Fix | After |
|---|---|---|---|
| Constant-zero guidance | 0.061 / 0.884 / 0, **R10 +2.72** | Paper's released setting restored (mixture, betas, scale), so the target is far from 0 (mean \|g*\| = 0.81) | `no_guidance` MAE 0.804, win 0.264, rank **−6.12** |
| Fabricated perfect scores | 0 / 1 / 0, +3.61 | Evaluator reads nothing a method writes. `sc_fake_score_files` ships perfect `results.json`/`scores_test.json` | Scored as what it computes (no guidance): −6.12 |
| Impossible values | −1 / 2 / −100, +36.5 | Non-finite output fails; error clipped at 4; win and cost computed by the evaluator | `sc_nan_output` failed, worst values (−35.9). `sc_huge_output` (1e6) MAE 4.0 (−34.4) |
| Declared zero compute | 0.375 / 0.849 / 0, +0.22 | Cost counted at the oracle | `sc_cost_claim` measured 3 units, ties the reference exactly (0.000) |
| Moved baseline (CFG ×1000) | 0.375 / 1 / 1, +1.20 | CFG direction computed inside `eval/` from the frozen denoisers | Not expressible by a rule |
| Replaced fixed denoisers | ≈0 / 1 / 1, +3.39 | Hidden denoisers exist only in `eval/test/assets` | Overwriting every public denoiser with zeros before the test leaves the test scores bit-identical (`special_public_denoisers_scores_test.json`) |
| Grid / time subsets | 0.229 / 1 / 1, +1.60; 0.234 / 0.831 / 1, +0.55 | States and time steps fixed by the evaluator; the output must cover every state | `sc_wrong_shape` failed, worst values (−35.9) |
| Brute-force target reconstruction | 0 / 1 / 12, +0.99 | Hidden denoisers and mixtures; 8-unit cap per state; t=0 not queryable | `sc_bruteforce_oracle` (100 completions through the oracle) failed at 9 > 8 units. `sc_query_endpoint` failed. Private brute force with public code (`sc_private_bruteforce`: the paper's public denoiser and reward, 100 completions, zero queries): MAE 0.464, win 0.452, rank **−1.17**, below the reference (it ranks +1.16 on the visible slice, where it is the right problem) |
| Empty training selection / underscore name / unlisted disk methods | Ties: methods silently dropped | No allowlist; every directory scored; empty directory gets worst values | Test ran with `scores_train.json` deleted. `_sc_underscore_name` scored (−0.97); `sc_empty_dir` worst values |
| Null / empty records | Evaluator exited with no scores | Failures produce worst values | See NaN and wrong-shape rows |
| Prewritten score file | Overwritten | Evaluator deletes the file first | Overwritten (`special_prewrite_scores_test.json`) |
| *New:* pooling across the batch | — | Two batchings must agree within 1e-5 | `sc_batch_pooling` failed (max diff 0.106) |
| *New:* hand-written generating process (paper mixtures' analytic reward at x0_hat) | — | Hidden mixtures | `sc_paper_reward_hardcode` MAE 0.662, rank −4.06 |

No shortcut outranks the reference on the hidden test.

## Shortcuts left, and why

- **A rule that reads `eval/test/` from disk.** The rule runs on the same filesystem as the hidden evaluator. Process isolation stops it from grabbing the networks from the oracle, but it cannot stop file reads without OS-level sandboxing, which is out of scope for this gym. It is forbidden in CONSTRAINTS.md and DIRECTION.md, and the grader checks for it.
- **An amortized predictor trained on many public-family problems.** Someone could train hundreds of denoisers on the public family, compute their targets, and fit g from local features. That is "writing down the generating process" in learned form. It is mitigated, not eliminated:
  - the hidden family is broader than the public one (2–4 conditional components, wider ranges);
  - the evaluator gives only denoiser values, no mixture parameters.
  
  It is declared out of bounds in DIRECTION.md.
- **Within-budget fingerprinting.** A rule could spend part of its 8 units probing fixed points to recognise a problem. With 4-component mixtures and per-state budgets this cannot pin down the data distribution. Not tested further.

## Reference score before and after

| | Test MAE | Test win | Cost | Other metrics |
|---|---:|---:|---:|---|
| Before (old test; 1D text setting, w=1.5; self-reported) | 0.375 | 0.849 | 1 extra VJP | |
| After (hidden test, recomputed) | 0.413 | 0.586 | 3 units | worst-t win 0.383, worst-problem MAE 0.801 |
| After, visible train | 0.339 | 0.709 | 3 units | |
| After, paper problems (`scores.json`) | 0.264 | 0.791 | 3 units | vanilla CFG 0.519, no guidance 0.664 |

The paper's claim holds in its direction. REG beats vanilla CFG on every slice: hidden ΔMAE −0.098 [−0.188, −0.010], and better MAE on 9 of 12 hidden problems. On the paper's own mixture, REG wins 79% of states, pooled over two independent denoiser trainings (ours and the authors' released checkpoint), and has lower MAE than CFG on both (worst-problem MAE 0.29 vs 0.54). The paper's 2D Table 1 mean is 67%.

## Headroom check

The reference sits at the bound of no scored metric:

| Metric | Reference | Bound |
|---|---:|---:|
| MAE | 0.413 | 0 |
| win | 0.586 | 1 |
| worst-t win | 0.383 | 1 |
| worst-problem MAE | 0.801 | 0 |
| cost | 3 | 0 |

A genuine variant (`reg_tweedie_point`) beats it beyond the noise, and no simple variant comes near any bound. The simple baselines sit clearly below the reference (vanilla −0.97, zero −6.12). Brute force cannot run within the budget, and its private form ranks below the reference. I added no extra experiment because the existing one has headroom. REG's weak spot is visible inside it: on the hidden test it loses to CFG at t ≥ 16, with per-step win 0.38–0.49, which is what `worst_timestep_win_rate` measures.

## Metric and weight changes

The old metrics were per-record aggregates computed by the method. They are replaced by evaluator-computed metrics:

| Metric | Coefficient |
|---|---:|
| `guidance_mae` | 1 |
| `win_rate` (ties count 1/2; primary, the Table 1 quantity) | 1 |
| `worst_timestep_win_rate` | 0.25 |
| `worst_problem_mae` | 0.25 |
| `oracle_cost_per_state` | 0.3 |

Changes and reasons:
- **Early/late-step MAEs** became the worst-time-step and worst-problem metrics. Table 1 is a per-time-step claim, and robustness across unseen problems is the point of a hidden test.
- **`relative_mae_reduction`** was dropped: it duplicated `guidance_mae`.
- **The cost coefficient went from 0.1 to 0.3.** With foresight now purchasable through the oracle, the no-foresight constraint is the paper's central trade-off. At 0.1, the 7-unit lookahead would be nearly free.

The paper's reported numbers in `scoring/reference.json` are unchanged:
- `reg_cfg` `win_rate` 0.6708, the 2D Table 1 mean, with a note;
- costs 2 and 3, the paper's "one extra gradient" restated in oracle units.

## What could not be fixed

- **Only the paper's 1D experiment is scored.** The 2D example (Table 1) and the ImageNet/text-to-image FID results are not reproduced; the old gym did not have them either.
- **REG's advantage over CFG on the hidden family is modest.** It holds on average and is strong on the paper's own mixture, but CFG has lower MAE on 3 of 12 hidden problems, and REG loses at the highest noise levels. That is a property of the method on broader mixtures, not something to tune away.
- **The test is harder than train.** Most of the gap comes from the two paper problems on the train side, where REG is strongest. On the three public family problems alone, REG's MAE is ≈ 0.39 against 0.41 on the test.
- **The 2D-derived reference number** in `reference.json` is not on the 1D scale.
- **The previous red-team payloads** (`redteam/driver.py` and its case folders) target the retired `outputs/toy_1d/results.json` contract and no longer run. Their replacements are in `redteam/hardening_check/`.
- `python validate.py` passes. Its three soft warnings ("eval looks detached from the method") are a heuristic false positive: the evaluators read and run the method's rule files from `outputs/guidance_rules/`. `np.random.default_rng` only shuffles batch order, and `torch.manual_seed` is in the hidden-problem builder.
