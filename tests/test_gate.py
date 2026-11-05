import json
import shutil
import subprocess
import sys

import pytest

from gates.gate import ROOT, run


@pytest.fixture(autouse=True)
def no_job_summary(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)  # test reports must not land in the CI summary


@pytest.fixture
def repo(tmp_path):
    """A copy of the template: data, params and champion, which each test mutates like a PR would."""
    for d in ("data", "champion", "gates"):
        shutil.copytree(ROOT / d, tmp_path / d)
    shutil.copy(ROOT / "params.json", tmp_path / "params.json")
    return tmp_path


def set_params(repo, **kw):
    p = json.loads((repo / "params.json").read_text())
    (repo / "params.json").write_text(json.dumps({**p, **kw}))


def test_unchanged_candidate_passes_and_writes_report(repo):
    r = run(repo)
    assert r["passed"] and all(g["pass"] for g in r["gates"])
    assert (repo / "report" / "report.md").read_text().startswith("## Model quality gate: PASSED")
    assert json.loads((repo / "report" / "report.json").read_text())["passed"] is True


@pytest.mark.parametrize("bad", [
    {"features": ["mean_fractal_dimension", "texture_error"]},  # a PR that drops the useful features
    {"C": 1e-4},  # a PR that over-regularises
])
def test_deliberately_bad_change_fails_the_gate(repo, bad):
    set_params(repo, **bad)
    r = run(repo)
    assert not r["passed"]
    assert any(not g["pass"] for g in r["gates"])
    assert "**FAIL**" in (repo / "report" / "report.md").read_text()


def test_small_harmless_change_passes(repo):
    set_params(repo, C=0.8)
    assert run(repo)["passed"]


def _rewrite_train(repo, fn):
    lines = (repo / "data" / "train.csv").read_text().splitlines()
    (repo / "data" / "train.csv").write_text("\n".join(fn(lines)) + "\n")


@pytest.mark.parametrize("change, expect", [
    (lambda ls: [",".join(l.split(",")[1:]) for l in ls], "missing column mean_radius"),
    (lambda ls: [ls[0].replace("mean_texture", "texture_mean")] + ls[1:], "missing column mean_texture"),
    (lambda ls: [ls[0]] + [l.rsplit(",", 1)[0] + ",2" if i % 50 == 0 else l for i, l in enumerate(ls[1:])], "labels [0, 1] -> [0, 1, 2]"),
    (lambda ls: [ls[0]] + [l.replace(l.split(",")[0], "17.5", 1) if i == 0 else l for i, l in enumerate(ls[1:])], None),
])
def test_data_schema_change_fails_before_training(repo, change, expect):
    _rewrite_train(repo, change)
    r = run(repo)
    if expect is None:  # a plain value change in a float column is not a schema change
        assert r["passed"]
        return
    assert not r["passed"] and r["gates"] == [] and any(expect in p for p in r["schema_problems"])
    assert "Data contract broken" in (repo / "report" / "report.md").read_text()


def test_changing_the_fixed_eval_set_is_rejected(repo):
    lines = (repo / "data" / "eval.csv").read_text().splitlines()
    (repo / "data" / "eval.csv").write_text("\n".join(lines[:-20]) + "\n")  # e.g. drop hard examples
    r = run(repo)
    assert not r["passed"] and any("eval set changed" in p for p in r["schema_problems"])


def test_pr_cannot_lower_the_bar_by_editing_the_champion(repo, tmp_path_factory):
    base_champion = tmp_path_factory.mktemp("base") / "champion.json"
    shutil.copy(repo / "champion" / "champion.json", base_champion)  # what CI extracts from the base branch
    weak = json.loads((repo / "champion" / "champion.json").read_text())
    weak["metrics"] = {"roc_auc": 0.0, "accuracy": 0.0, "f1": 0.0, "log_loss": 99.0}
    (repo / "champion" / "champion.json").write_text(json.dumps(weak))
    set_params(repo, C=1e-4)
    assert run(repo)["passed"]  # comparing with the PR's own edited champion would let it through...
    assert not run(repo, champion_path=base_champion)["passed"]  # ...which is why CI uses the base copy


@pytest.mark.parametrize("cell", ["abc", "", "nan"])
def test_malformed_data_fails_cleanly_with_a_report(repo, cell):
    _rewrite_train(repo, lambda ls: [ls[0], cell + ls[1][ls[1].index(","):]] + ls[2:])
    r = run(repo)
    assert not r["passed"] and r["schema_problems"]
    assert "Data contract broken" in (repo / "report" / "report.md").read_text()


def test_unknown_feature_name_fails_cleanly(repo):
    set_params(repo, features=["mean_radius", "typo_feature"])
    r = run(repo)
    assert not r["passed"] and any("typo_feature" in p for p in r["schema_problems"])


def test_promote_only_when_passing(repo):
    before = (repo / "champion" / "champion.json").read_text()
    set_params(repo, C=1e-4)
    run(repo, promote=True)
    assert (repo / "champion" / "champion.json").read_text() == before
    set_params(repo, C=2.0)
    assert run(repo, promote=True)["passed"]
    assert json.loads((repo / "champion" / "champion.json").read_text())["params"]["C"] == 2.0


def test_cli_exit_code(repo):
    ok = subprocess.run([sys.executable, "-m", "gates.gate"], cwd=repo, capture_output=True)
    set_params(repo, C=1e-4)
    bad = subprocess.run([sys.executable, "-m", "gates.gate"], cwd=repo, capture_output=True)
    assert ok.returncode == 0 and bad.returncode == 1
