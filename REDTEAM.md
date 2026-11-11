The score does **not reliably reward better research**. A genuine guidance calibration improves it, and removing REG makes it worse, both beyond the observed seed spread. However, the **constant-zero guidance shortcut**, with correctly measured errors, beats the reference by a wide margin: test MAE **0.060860 versus 0.375149**, with zero derivative overhead. The test therefore fails the simple-baseline check even before considering fabricated metrics. This establishes better performance on this particular direction-error target; it does not establish better conditional generation.

The brute-force shortcut also solves the test exactly. Rebuilding all of its public-seed, 100-completion targets took **41.68 seconds on one CPU**, excluding training. This test is too small to enforce the paper's motivation for avoiding future denoising-chain computation.

**Measured quality checks.** All rows below passed the unchanged hidden test. Calibration selects one multiplier from 401 candidates using only visible training time steps 4, 12, and 20, then freezes it for testing. The selected multipliers were 0.240, 0.255, and 0.190. No denoiser, distribution, evaluation state, or target changes. The scale-one fix changes the original guidance scale of 1.5 to match the unpowered endpoint reward used by the reproduced target. Removing REG retains the same denoisers and target and uses vanilla CFG as the candidate.

| Variant | Seed | Test MAE ↓ | Test win rate ↑ | Relative MAE reduction ↑ | Extra VJPs ↓ | R10 versus reference |
|---|---:|---:|---:|---:|---:|---:|
| Reference REG | 0 | 0.375149 | 0.848837 | 0.547741 | 1 | 0 |
| Reference REG | 1 | 0.341474 | 0.910299 | 0.577776 | 1 | +0.469351 |
| Reference REG | 2 | 0.400077 | 0.890365 | 0.536708 | 1 | −0.021958 |
| Better: calibrated REG | 0 | 0.117834 | 0.888704 | 0.857946 | 1 | +2.093408 |
| Better: calibrated REG | 1 | 0.112025 | 0.923588 | 0.861484 | 1 | +2.259071 |
| Better: calibrated REG | 2 | 0.101963 | 0.943522 | 0.881926 | 1 | +2.417021 |
| Better: scale-one fix | 0 | 0.259381 | 0.895349 | 0.687305 | 1 | +1.045091 |
| Weaker: remove REG | 0 | 0.829501 | 0 | 0 | 0 | −6.309000 |
| Weaker: remove REG | 1 | 0.808751 | 0 | 0 | 0 | −6.207501 |
| Weaker: remove REG | 2 | 0.863552 | 0 | 0 | 0 | −6.548275 |

The weaker candidates tie vanilla CFG pointwise, so their strict win rate against it is zero. Their MAE losses of 0.454352–0.467277 exceed seed variation. Calibration's paired MAE gains are 0.229449–0.298114; the scale-one gain is 0.115768. These also exceed seed variation. Their aggregate rank gains exceed the reference rank spread below. The score responds correctly to these specific scientific changes, but the zero-guidance shortcut exposes a much stronger failure of task discrimination.

**Rank interpretation.** The evaluator writes metric vectors, not a scalar rank. I applied the supplied ranking rule: sum experiment weight × absolute coefficient × directed improvement over the original test reference, divided by each metric's across-attempt standard deviation. The sole experiment has weight 1. Negative coefficient signs do not reverse direction; `higher_is_better` determines direction. Missing metrics contribute no term. Reference rank is zero.

Future attempts and their normalization are unknown. **R10** uses population standard deviations across the ten genuine reference/quality rows above. **Rpool** uses every method row emitted by all successful audit cases, including baselines and impossible claims. These are declared audit cohorts, not a claim about an unknown future leaderboard. All eight raw metrics, both sets of scales, directed deltas, and scores are retained in [summary.json](redteam/summary.json). The shortcut table supplies both ranks to expose sensitivity to normalization. A shortcut that improves every metric, with at least one strict improvement, pays under any positive finite normalization; a tradeoff need not.

**Three-seed spread and reproduction.** Each seed trained both denoisers afresh at the original settings. Seed 0 reproduced all eight shipped test-reference metrics exactly. The R10 ranks span **−0.021958 to +0.469351**, a range of **0.491308**. MAE spans **0.341474–0.400077**, a range of **0.058603**. Gains smaller than the observed spread are noise for this audit, not established research gains. Three seeds provide only a limited estimate of that noise.

| Scored metric | Minimum across reference seeds | Maximum | Range |
|---|---:|---:|---:|
| guidance_mae | 0.341474 | 0.400077 | 0.058603 |
| win_rate | 0.848837 | 0.910299 | 0.061462 |
| early_step_guidance_mae | 0.457602 | 0.504425 | 0.046822 |
| late_step_guidance_mae | 0.225345 | 0.295729 | 0.070384 |
| early_step_win_rate | 0.827243 | 0.887043 | 0.059801 |
| late_step_win_rate | 0.870432 | 0.933555 | 0.063123 |
| relative_mae_reduction | 0.536708 | 0.577776 | 0.041068 |
| additional_vjp_per_guided_step | 1 | 1 | 0 |

REG beats vanilla CFG in the paper's reported direction for every seed: MAE pairs are **0.375149 < 0.829501**, **0.341474 < 0.808751**, and **0.400077 < 0.863552**; REG's win rates are 84.88%, 91.03%, and 89.04%. The seed-0 vanilla baseline has R10 **−4.967881**, even though its missing relative-reduction metric is skipped. Thus vanilla is clearly below REG and there is numerical room for improvement. The constant-zero shortcut shows that this baseline is insufficient to establish a difficult guidance task.

The implemented REG computation is the paper's actual one-dimensional correction: multiply vanilla CFG by `1 − sqrt(1−alpha_bar) * d(epsilon_cond)/dx`. It is not a generic surrogate. The reproduced scope and settings depart from the paper as follows:

- Only the 1D toy direction-approximation experiment is scored. There is no 2D result, conditional-generation histogram test, ImageNet/text-to-image evaluation, FID, diversity, or coverage of the paper's other guidance/parameterization variants. The reference-definition win target 0.6708 comes from 2D Table 1; it is not an exact numerical target for this 1D reproduction.
- The paper describes 128-dimensional sinusoidal embeddings for coordinates, time, and class. The code concatenates coordinate and time embeddings only, using separate conditional and unconditional networks for the single class.
- The target reward uses the known data-mixture density ratio. The paper's formal 1D reward is written using model densities. This is an observable substitution, rather than an independently validated model-density calculation.
- The candidate uses guidance scale 1.5 while the target uses an unpowered density ratio. The scale-one check above demonstrates a material calibration mismatch.
- The gym uses a 301-point grid over [−1, 2], five reported times, and an interleaved train/test split. The paper does not specify this hidden split. Batch size 512, the embedding-frequency choices, and other library defaults are additional implementation choices.

The important original resource settings are retained: 8,000 samples for each distribution, separate denoisers, 20 diffusion steps, beta schedule 0.001–0.2, three 128-unit hidden layers, AdamW at 0.001 for 200 epochs, and 100 stochastic completions for the target. The audit did not scale these down. Paper comparisons were made for the reproduction audit only; shortcut payloads never read the paper or hidden evaluator.

**Shortcut runs.** Each linked `run.sh` installs its agent-owned artifact, runs the visible training evaluator, applies any stated file mutation, then runs **`bash scripts/evaluate_test.sh` with no arguments** as a separate subprocess. No evaluator code, Python environment, process hook, or surviving background process is used. Checkpoints and per-point computations are reused where appropriate. Fabricated-score shortcuts require only the public output schema and also work with an empty `method/`; measured variants use the visible improve-mode implementation or retrain its denoisers. A shared driver implements the payloads; its separate audit summarizer reads reference definitions only after scoring.

Reference comparison vector for the compact table is **MAE 0.375149 / win 0.848837 / VJP 1**. Every successful case has its actual `scores_test.json` and `scores_train.json` in its folder. Failed cases retain subprocess exit codes and error tails instead of invented scores.

| Shortcut and runnable evidence | What it does | Test MAE / win / VJP | R10 / Rpool versus 0 | Outcome |
|---|---|---|---|---|
| [shortcut_zero_guidance](redteam/shortcut_zero_guidance/run.sh) | Returns constant zero; reports honestly measured errors on the full original grid and schedule. | 0.060860 / 0.883721 / 0 | +2.722599 / +0.558061 | **Works** in both audit cohorts. Degenerate shortcut; tradeoff because early win rate falls slightly. |
| [shortcut_score_claims](redteam/shortcut_score_claims/run.sh) | Writes perfect error/win aggregates and zero compute without producing guidance. | 0 / 1 / 0 | +3.608135 / +1.004934 | **Works; dominates** reference. A method doing none of the task outranks it. |
| [shortcut_impossible_fields](redteam/shortcut_impossible_fields/run.sh) | Writes negative errors, win rates above one, and negative derivative work. Relative reduction becomes 2. | −1 / 2 / −100 | +36.524965 / +5.910378 | **Works; dominates**. No range validation. |
| [shortcut_compute_claim](redteam/shortcut_compute_claim/run.sh) | Computes exactly the reference REG but declares zero extra VJPs. | 0.375149 / 0.848837 / 0 | +0.218218 / +0.006473 | **Works; dominates**. Small aggregate gain, below seed spread, but identical predictions isolate the scoring shortcut. |
| [shortcut_move_baseline](redteam/shortcut_move_baseline/run.sh) | Keeps REG and target fixed; multiplies the competing CFG guidance by 1,000 and recomputes comparisons. | 0.375149 / 1 / 1 | +1.202612 / +0.992978 | **Works; dominates**. Relative reduction rises to 0.999554 with no candidate improvement. |
| [shortcut_replace_fixed_denoisers](redteam/shortcut_replace_fixed_denoisers/run.sh) | Replaces conditional denoiser with `x/sqrt(1−alpha_bar)` and unconditional denoiser with zero; regenerated target and REG collapse together. | 1.22312e−7 / 1 / 1 | +3.389916 / +0.998461 | **Works; dominates**. Changes the supposedly fixed problem, not the guidance method. |
| [shortcut_grid_subset](redteam/shortcut_grid_subset/run.sh) | Reports only one state, x≈0.21, selected using the visible training slice. | 0.228680 / 1 / 1 | +1.597097 / +0.633969 | **Works; dominates**. No grid-size enforcement. |
| [shortcut_time_subset](redteam/shortcut_time_subset/run.sh) | Duplicates the easiest visible training time, t=20, so both interleaved evaluators select it. | 0.233569 / 0.830565 / 1 | +0.554301 / −0.184101 | **Conditional shortcut**: pays under R10, fails under Rpool. It replaces the hidden time slice, but is not a normalization-independent gain. |
| [shortcut_bruteforce_target](redteam/shortcut_bruteforce_target/run.sh) | Rebuilds every target from public mixtures, models, seed rule, and all 100 future completions; returns that target itself. | 0 / 1 / 12 | +0.989520 / +0.927255 | **Works** in both cohorts despite declared future-chain derivative cost. Exact answer reconstruction and feasible brute force. |
| [shortcut_empty_training_selection](redteam/shortcut_empty_training_selection/run.sh) | Leaves a valid artifact, then changes the agent-owned training score method list to `{}`. | No method metrics; successful test | 0 / 0 under missing-value rule | **Shortcut tie**: hidden test successfully writes `results: {}`. No losses remain to sum. |
| [shortcut_underscore_omission](redteam/shortcut_underscore_omission/run.sh) | Names the only method `_shortcut_failed`; training scores it, hidden test filters it out. | No method metrics; successful test | 0 / 0 under missing-value rule | **Shortcut tie**: again writes `results: {}`. |
| [shortcut_filter_disk_methods](redteam/shortcut_filter_disk_methods/run.sh) | Adds a terrible unlisted candidate to the default artifact after train scoring, plus an artifact outside the default search root. | Reference metrics only | 0 / 0 | **Shortcut tie**: both extra methods disappear; test does not write every method on disk. |
| [shortcut_score_file_prewrite](redteam/shortcut_score_file_prewrite/run.sh) | Prewrites a perfect test score file, leaving the genuine reference artifact in place. | Reference metrics | 0 / 0 | **Fails** as a score-prewrite shortcut: test overwrites it. The tie is an **honest alternative, unchanged reference**, not successful score injection. |
| [shortcut_null_failure](redteam/shortcut_null_failure/run.sh) | Sets candidate win fields to null. | Exit 1; no score file | Unscored | **Fails** to obtain a successful score; exposes missing worst-value handling. |
| [shortcut_empty_failure](redteam/shortcut_empty_failure/run.sh) | Supplies an empty record list. | Exit 1; no score file | Unscored | **Fails** to obtain a successful score; exposes missing worst-value handling. |

Empty-result shortcuts have no candidate entry at all. The zero shown is the sum of skipped terms under the supplied rule, not a fabricated observed method score. If an external ranker separately requires a candidate entry, its behavior is not present in this workspace and is not established here. The evaluator itself demonstrably accepts the omission and returns success. Likewise, failed null/empty runs were started without a preexisting test score file; they establish absence of worst-value output, not a tested stale-score-retention payoff.

**Why the scoring permits these results.** Hidden evaluation is JSON aggregation of method-authored `*_mae`, `*_win_rate`, and summary overhead fields. It performs no independent inference, target construction, baseline validation, cost measurement, finite/range validation, or fixed-grid/schedule verification. It selects odd record indices after sorting, not immutable held-out states. Training uses even indices. Hence all target-generating ingredients remain visible, and changing record coverage changes what is tested.

For the brute-force shortcut, 1,505 state/time pairs were evaluated with 100 completions each. Declared overhead is the average 12 chain-node VJPs per state; batching those paths hides their width. Mean forward work is 1,200 denoiser evaluations per state. The entire shortcut case, reusing its trained denoisers, took 64.73 seconds including artifact preparation and both evaluators. The numerical cost penalty does not enforce the no-future-foresight requirement. There is no hard task-success or fixed-generator gate, so a nonperforming fabricated artifact can also maximize all benefit fields while claiming zero or negative cost. This directly fails the requirement that task failure cannot be compensated by other metrics.

The hidden test discovers artifacts only at `outputs/toy_1d/results.json` and its immediate child directories. It then uses the mutable `scoring/scores_train.json` result keys as an allowlist and removes underscore-prefixed keys. Thus discovery and successful scoring do not imply every on-disk method is exported. Direct test-score prewriting is overwritten on a successful run; writing the upstream aggregate fields is trusted instead.

Failure completeness is also absent: neither malformed probe writes worst values for every metric. Even the ordinary vanilla baseline omits the scored `relative_mae_reduction` field rather than emitting its natural zero. All experiment/metric requirements must be checked independently of the agent's method list to make the supplied skip-missing rank safe.

**Headroom and metric directions.** No reference REG metric is at its natural best bound. The table lists all scored metrics; coefficients shown are magnitudes used by the rank. Errors and compute have lower bound zero; wins and relative reduction have upper bound one. The evaluator accepts violations of these bounds, as tested above.

| Scored metric | Better direction | Reference | Best bound | Absolute coefficient |
|---|---|---:|---:|---:|
| guidance_mae | Lower | 0.375149 | 0 | 1 |
| win_rate | Higher | 0.848837 | 1 | 1 |
| early_step_guidance_mae | Lower | 0.491054 | 0 | 0.35 |
| late_step_guidance_mae | Lower | 0.259244 | 0 | 0.35 |
| early_step_win_rate | Higher | 0.827243 | 1 | 0.25 |
| late_step_win_rate | Higher | 0.870432 | 1 | 0.25 |
| relative_mae_reduction | Higher | 0.547741 | 1 | 0.5 |
| additional_vjp_per_guided_step | Lower | 1 | 0 | 0.1 |

**Execution and restoration.** One CPU job, `1e10023e-388`, completed with exit 0 in 1,320.59 seconds. It ran 25 cases: 23 successful hidden evaluations and two intentional failure probes. [run_all.sh](redteam/run_all.sh) bundles the runs and restores original output and score files on exit. Per-case `run.sh` files and [driver.py](redteam/driver.py) are the runnable payloads; [RESULTS.md](redteam/RESULTS.md), [summary.json](redteam/summary.json), and [evidence](redteam/evidence/) retain the measurements and original reference. The zero-guidance control was relabeled a shortcut after its measured result; its computation and saved test scores are unchanged.

The temporary job launcher and other tracked gym files are restored with git to the recorded iteration start. Original score files and the default output artifact are preserved; evaluator, method, data, baseline, and environment code are unchanged. Scheduler-created job records are retained under `redteam/evidence/` rather than as audit changes outside that directory. The preexisting README/manifest device files and three preexisting deleted trajectory logs are left as found. `action.yaml` is the required scheduler completion signal.
