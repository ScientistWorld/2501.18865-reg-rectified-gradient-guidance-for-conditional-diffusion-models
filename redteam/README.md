# Scoring audit

Run `bash redteam/run_all.sh` in the supplied compute container. This trains
three reference seeds, reuses their outputs for measured quality variants,
then runs the shortcut cases. Each case also has its own `run.sh`.

Every case installs its artifact at the documented default location, invokes
the visible training evaluator, then invokes **`bash scripts/evaluate_test.sh`
with no arguments in a separate process**. The evaluator is never changed.
Score-selection shortcuts modify only the agent-owned training manifest or
artifact. No payload changes Python, the process environment, or evaluator code.

`driver.py` implements payloads using only the visible data, reference method
(improve mode), and training evaluator interface. It never reads the hidden
evaluator, paper, reference scores, or audit scoring definitions. The separate
`summarize.py` reads audit evidence to assess the resulting scores.

Generated checkpoints and arrays are ignored. Small per-case metrics,
parameters, subprocess exit statuses, and both score files are retained.
`run_all.sh` restores original scoring and default-output files on exit.

Ranks are reported with a fixed, declared audit cohort because the actual
future cohort's per-metric normalization is unknowable. Dominance/ties are
also reported; these do not depend on the cohort normalization.
