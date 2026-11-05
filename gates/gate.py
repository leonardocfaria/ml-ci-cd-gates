"""Train the candidate, compare with the champion on the fixed eval set, check the data schema, write a report.

python -m gates.gate                       exit 0 = pass, 1 = regression or schema change
python -m gates.gate --promote             (on main) make the candidate the new champion
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
TARGET = "target"
# metric -> (higher is better?, allowed regression vs champion)
GATES = {"roc_auc": (True, 0.005), "accuracy": (True, 0.01), "f1": (True, 0.01), "log_loss": (False, 0.02)}


def read_csv(path: Path):
    with path.open() as f:
        rows = list(csv.reader(f))
    header, body = rows[0], np.array(rows[1:], dtype=float)
    return header, body


def schema(header, body) -> dict:
    """Columns in order with dtype (int if every value is integral, else float) and the label set."""
    cols = [{"name": h, "dtype": "int" if np.all(body[:, j] == np.round(body[:, j])) else "float"} for j, h in enumerate(header)]
    labels = sorted(set(body[:, header.index(TARGET)].astype(int).tolist())) if TARGET in header else []
    return {"columns": cols, "labels": labels}


def schema_diff(expected: dict, actual: dict) -> list[str]:
    exp, act = {c["name"]: c["dtype"] for c in expected["columns"]}, {c["name"]: c["dtype"] for c in actual["columns"]}
    out = [f"missing column {c}" for c in exp if c not in act] + [f"new column {c}" for c in act if c not in exp]
    out += [f"dtype of {c}: {exp[c]} -> {act[c]}" for c in exp if c in act and exp[c] != act[c]]
    if expected["labels"] != actual["labels"]:
        out.append(f"labels {expected['labels']} -> {actual['labels']}")
    return out


def train_and_eval(params: dict, train_path: Path, eval_path: Path) -> tuple[dict, dict]:
    th, tb = read_csv(train_path)
    eh, eb = read_csv(eval_path)
    feats = [h for h in th if h != TARGET and (params.get("features", "all") == "all" or h in params["features"])]
    X = tb[:, [th.index(f) for f in feats]]
    y = tb[:, th.index(TARGET)]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], max_iter=params.get("max_iter", 1000)))
    t = time.perf_counter()
    model.fit(X, y)
    train_s = time.perf_counter() - t
    Xe, ye = eb[:, [eh.index(f) for f in feats]], eb[:, eh.index(TARGET)]
    p = model.predict_proba(Xe)[:, 1]
    metrics = {"roc_auc": roc_auc_score(ye, p), "accuracy": accuracy_score(ye, p > 0.5), "f1": f1_score(ye, p > 0.5),
               "log_loss": log_loss(ye, p)}
    return {k: round(float(v), 6) for k, v in metrics.items()}, {"n_train": len(y), "n_eval": len(ye),
                                                                  "n_features": len(feats), "train_seconds": round(train_s, 3)}


def gate(candidate: dict, champion: dict) -> list[dict]:
    rows = []
    for m, (higher, tol) in GATES.items():
        delta = candidate[m] - champion[m]
        ok = delta >= -tol if higher else delta <= tol
        rows.append({"metric": m, "champion": champion[m], "candidate": candidate[m], "delta": round(delta, 6),
                     "allowed": f"{'-' if higher else '+'}{tol}", "pass": ok})
    return rows


def run(root: Path = ROOT, out: Path | None = None, promote: bool = False) -> dict:
    out = out or root / "report"
    out.mkdir(parents=True, exist_ok=True)
    params = json.loads((root / "params.json").read_text())
    champion = json.loads((root / "champion" / "champion.json").read_text())
    data_schema = schema(*read_csv(root / "data" / "train.csv"))
    eval_hash = hashlib.sha256((root / "data" / "eval.csv").read_bytes()).hexdigest()
    problems = schema_diff(champion["schema"], data_schema)
    if eval_hash != champion["eval_sha256"]:
        problems.append("the fixed eval set changed (eval.csv hash differs): metrics would not be comparable")
    metrics, info, gates = None, {}, []
    if not problems:  # only train when the data contract holds
        metrics, info = train_and_eval(params, root / "data" / "train.csv", root / "data" / "eval.csv")
        gates = gate(metrics, champion["metrics"])
    passed = not problems and all(g["pass"] for g in gates)
    report = {"passed": passed, "schema_problems": problems, "gates": gates, "candidate_metrics": metrics,
              "champion_metrics": champion["metrics"], "params": params, **info}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    md = render(report)
    (out / "report.md").write_text(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md)
    if promote and passed:
        (root / "champion" / "champion.json").write_text(json.dumps(
            {"metrics": metrics, "params": params, "schema": data_schema, "eval_sha256": eval_hash}, indent=2) + "\n")
    return report


def render(r: dict) -> str:
    lines = [f"## Model quality gate: {'PASSED ✅' if r['passed'] else 'FAILED ❌'}", ""]
    if r["schema_problems"]:
        lines += ["**Data contract broken (model not trained):**", ""] + [f"- {p}" for p in r["schema_problems"]] + [""]
    if r["gates"]:
        lines += ["| metric | champion | candidate | delta | allowed | result |", "|---|---|---|---|---|---|"]
        lines += [f"| {g['metric']} | {g['champion']:.4f} | {g['candidate']:.4f} | {g['delta']:+.4f} | {g['allowed']} | "
                  f"{'pass' if g['pass'] else '**FAIL**'} |" for g in r["gates"]]
        lines += ["", f"params: `{json.dumps(r['params'])}`; {r['n_features']} features, {r['n_train']} train / "
                      f"{r['n_eval']} eval rows, trained in {r['train_seconds']} s."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--promote", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    rep = run(ROOT, a.out, a.promote)
    print(render(rep))
    sys.exit(0 if rep["passed"] else 1)
