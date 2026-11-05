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
    """(header, float matrix). Raises ValueError with a readable message on non-numeric, empty or NaN cells."""
    with path.open() as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"{path.name}: no data rows")
    header = rows[0]
    try:
        body = np.array(rows[1:], dtype=float)
    except ValueError as e:
        raise ValueError(f"{path.name}: non-numeric or ragged data ({e})") from None
    bad = [header[j] for j in np.where(np.isnan(body).any(0))[0]]
    if bad:
        raise ValueError(f"{path.name}: missing values in {bad}")
    return header, body


def schema(header, body) -> dict:
    """Columns with dtype (int if every value is integral, else float) and the label set. Column order is not
    part of the contract (training selects columns by name). shortcut: the int/float rule is a heuristic that a
    tiny sample of an integral-looking float column could flip; declare dtypes explicitly for real data."""
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
    wanted = params.get("features", "all")
    unknown = [] if wanted == "all" else sorted(set(wanted) - set(th))
    if unknown:
        raise ValueError(f"params.json lists unknown features {unknown}")
    feats = [h for h in th if h != TARGET and (wanted == "all" or h in wanted)]
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
        delta = round(candidate[m] - champion[m], 6)  # metrics are 6-dp rounded: no float noise at the boundary
        ok = delta >= -tol if higher else delta <= tol
        rows.append({"metric": m, "champion": champion[m], "candidate": candidate[m], "delta": round(delta, 6),
                     "allowed": f"{'-' if higher else '+'}{tol}", "pass": ok})
    return rows


def run(root: Path = ROOT, out: Path | None = None, promote: bool = False, champion_path: Path | None = None) -> dict:
    """champion_path: the champion to compare with. CI passes the copy from the PR's *base* branch, so a PR
    cannot weaken the bar (or swap the eval set, whose hash the champion pins) by editing champion.json."""
    out = out or root / "report"
    out.mkdir(parents=True, exist_ok=True)
    params = json.loads((root / "params.json").read_text())
    champion = json.loads((champion_path or root / "champion" / "champion.json").read_text())
    eval_hash = hashlib.sha256((root / "data" / "eval.csv").read_bytes()).hexdigest()
    metrics, info, gates, data_schema = None, {}, [], None
    try:
        data_schema = schema(*read_csv(root / "data" / "train.csv"))
        problems = schema_diff(champion["schema"], data_schema)
    except ValueError as e:
        problems = [str(e)]
    if eval_hash != champion["eval_sha256"]:
        problems.append("the fixed eval set changed (eval.csv hash differs): metrics would not be comparable")
    if not problems:  # only train when the data contract holds
        try:
            metrics, info = train_and_eval(params, root / "data" / "train.csv", root / "data" / "eval.csv")
            gates = gate(metrics, champion["metrics"])
        except ValueError as e:
            problems.append(str(e))
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
    p.add_argument("--champion", type=Path, help="champion.json to compare with (CI: the base branch's copy)")
    a = p.parse_args()
    rep = run(ROOT, a.out, a.promote, a.champion)
    print(render(rep))
    sys.exit(0 if rep["passed"] else 1)
