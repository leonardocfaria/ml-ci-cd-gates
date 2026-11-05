# ml-ci-cd-gates Implementation Plan

## Brainstorm (answered from SPEC + repo evidence)
- **What is it?** The smallest credible "model CI": every PR retrains, is compared with a committed champion on a frozen eval set, and is blocked on regression or a broken data contract, with a report artifact.
- **Data:** sklearn's bundled breast-cancer dataset (real, BSD, no download), split once and committed. The eval set is frozen by its SHA-256 in the champion file.
- **Champion = a committed JSON file** (metrics, params, schema, eval hash). No registry is needed for the template; promotion is `--promote` on main.
- **Gates:** per-metric tolerance (AUC, accuracy, F1, log-loss) plus a schema diff (columns, int/float dtype, labels) checked before training.
- **Report:** `report.md` + `report.json`, uploaded as a CI artifact even when the gate fails, and appended to `$GITHUB_STEP_SUMMARY`.
- **Non-goals:** significance testing, automatic commits from CI, PR comment bots.

## Tasks (TDD)
- [x] 1. gate passes on unchanged params; report written.
- [x] 2. deliberately bad changes (over-regularised, weak features) fail; a harmless change passes.
- [x] 3. schema changes (drop/rename column, new label) fail before training; a value change does not; eval-set edits fail.
- [x] 4. promote only on pass; CLI exit codes; CI workflow with artifact; README.
