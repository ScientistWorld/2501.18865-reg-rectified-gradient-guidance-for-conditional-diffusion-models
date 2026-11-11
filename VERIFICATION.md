> Superseded (iteration 11): this report predates the hardening; see HARDENING.md for the current verification.

# Verification Report

## Initial Assessment

The workspace claimed `core_claim` and contained a valid 1D synthetic guidance reproduction artifact with matching `scoring/scores.json`, train/test score files, and a passing validator. The claimed milestone was justified by the existing results: the proposed guidance rule substantially reduced guidance error versus the reference rule while recording the extra derivative-work constraint.

## Issues Found

- Minor: the latest method defaults and progress notes described an unrun regeneration, while the committed scores still reflected the previously validated artifact.
- Minor: `scoring/CONSTRAINTS.md` described benchmark scope with unnecessary implementation-specific detail for a high-level scientist-facing file.

## Fixes Applied

- Restored the method defaults and stochastic target implementation to match the scored artifact and documented completed run.
- Removed stale pending-regeneration notes from `PROGRESS.md` and `MILESTONES.md`.
- Reworded `scoring/CONSTRAINTS.md` to focus on fixed benchmark conditions, output contract, and overhead accounting.
- Re-ran `scripts/evaluate.sh`, `scripts/evaluate_train.sh`, `scripts/evaluate_test.sh`, and `python validate.py --compare`; all passed.

## Final Judgment

The workspace is ready for gym use. Improve-mode agents can run the packaged method and compare against the existing scores, while from-scratch agents can use the data and evaluation contract without importing method code. The scoring captures both the guidance-quality benefit and the derivative-overhead constraint, and there are no scored wall-clock metrics or author-deviation metrics.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: high
- Key limitation: core claim is demonstrated on the paper-style 1D synthetic benchmark only, not the full image-generation benchmark suite.
