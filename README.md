# ml-ci-cd-gates

A template for CI/CD of an ML model. On every pull request, CI:
- **trains** the candidate model;
- **compares** it with the current **champion** on a **fixed eval set**;
- **fails the build** on a metric regression or a **data-schema change**;
- publishes an **evaluation report** as a build artifact and in the job summary.

## Setup
```bash
cd ml/ml-ci-cd-gates
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python -m pytest -q             # 16 tests, ~6 s: includes deliberately bad changes that must fail
python -m gates.gate            # what CI runs: exit 0 = pass, 1 = fail; writes report/report.{md,json}
python -m gates.gate --promote  # after merge to main: candidate becomes the champion (commit champion/champion.json)
```
A PR changes `params.json`, the training code, or `data/train.csv`. The gate decides whether it may merge.

## Architecture
```
PR ─► CI (.github/workflows/ml-ci-cd-gates.yml)
        1. pytest
        2. python -m gates.gate:
             schema(data/train.csv) == champion.schema ?   ── no ─► FAIL "data contract broken" (no training)
             sha256(data/eval.csv) == champion.eval_sha256 ? ── no ─► FAIL (metrics would not be comparable)
             train candidate (params.json) ─► metrics on the fixed eval set
             for each metric: candidate within tolerance of champion ?   ── no ─► FAIL
        3. upload report/ as artifact "model-eval-report" (always, also on failure) + job summary
main ─► gates.gate --promote rewrites champion/champion.json (metrics, params, schema, eval hash)
```
| file | role |
|---|---|
| `data/train.csv`, `data/eval.csv` | sklearn's bundled breast-cancer data (BSD), split 70/30 once with seed 0. The eval set is frozen and guarded by its hash |
| `params.json` | the candidate's hyper-parameters (what a PR changes) |
| `champion/champion.json` | the champion's metrics, params, data schema and eval-set hash: the contract |
| `gates/gate.py` | schema inference and diff, training (StandardScaler + LogisticRegression), the gates, the report |

Gates (allowed regression vs champion):

| metric | tolerance |
|---|---|
| ROC AUC | −0.005 |
| accuracy | −0.01 |
| F1 | −0.01 |
| log-loss | +0.02 |

The schema check looks at:
- column names (order does not matter: training selects by name);
- dtype (int vs float);
- the label set.

Small value changes inside a column are allowed. A renamed, dropped or added column, a dtype flip, or a new label fails the gate.

## Results
The current champion is `C=1.0`, all 30 features: ROC AUC 0.9956, accuracy 0.9591, F1 0.9671, log-loss 0.0846, trained in 0.015 s.

| PR (change to `params.json` / data) | gate | why |
|---|---|---|
| none | **pass** | identical metrics |
| `C=0.8` | **pass** | AUC −0.0004, accuracy +0.006: within tolerance |
| `C=1e-4` (over-regularised) | **fail** | accuracy 0.637, AUC 0.965, log-loss 0.598 |
| keep 2 weak features | **fail** | AUC 0.498 (chance), accuracy 0.626 |
| drop / rename a training column | **fail** | "missing column …", model not trained |
| a third label appears | **fail** | "labels [0, 1] → [0, 1, 2]" |
| eval set edited | **fail** | eval hash differs |

Each row is a test in `tests/test_gate.py`. The tests run the gate on a copy of the template with the change applied, and also check the CLI exit codes 0 and 1.

## Trust boundary
- **Champion and eval set.** On a pull request, CI reads `champion/champion.json` **from the base branch** (`--champion`). The champion pins the eval set's SHA-256, so editing either file in the PR cannot lower the bar. A test shows that a weakened in-PR champion is ignored. The only exception is a bootstrap PR that introduces the template: there is no base champion yet, and CI warns.
- **The gate code itself.** Tolerances live in `gates/gate.py`, so a PR could still edit them. Protect `gates/` and `champion/` with CODEOWNERS and a required review; branch protection is outside a template's reach.
- **Bad data.** Non-numeric, empty or NaN cells, and unknown feature names, fail the gate with a readable message, and the report is still written and uploaded.

## Not done (time box)
Automatic champion promotion committed by CI (needs `contents: write`; the `--promote` step is manual or a separate main-branch job), statistical significance of metric deltas (e.g. a bootstrap CI on the eval set), model artifact storage/registry, PR comment bot.
