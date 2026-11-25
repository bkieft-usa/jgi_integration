"""CLI parsing, stage selection, run discovery, and hash-consistent run records."""
import json
import os

import pytest
import yaml

from tests.explorer_fixtures import make_run
from tools import runs
from tools.cli import build_parser, resolve_analysis_dir


def test_parser_commands():
    p = build_parser()
    a = p.parse_args(["run", "--config-dir", "cfg", "--stop-after", "group_features", "--no-explorer"])
    assert a.command == "run" and a.stop_after == "group_features" and a.no_explorer
    a = p.parse_args(["explore", "--bundle", "b", "--port", "9000", "--host", "0.0.0.0"])
    assert a.port == 9000 and a.host == "0.0.0.0"
    a = p.parse_args(["build-explorer", "--project-dir", "P", "-d", "aaaaaaaa", "-a", "bbbbbbbb", "--n-permutations", "10"])
    assert a.data_processing_hash == "aaaaaaaa" and a.n_permutations == 10
    with pytest.raises(SystemExit):
        p.parse_args(["run", "--config-dir", "c", "--start-at", "not_a_stage"])


def test_stage_selection():
    pl = pytest.importorskip("tools.pipeline")
    names = [n for n, _ in pl.select_stages("integrate", "enrich")]
    assert names == ["integrate", "select_features", "annotate", "group_features", "enrich"]
    assert "build_explorer" not in [n for n, _ in pl.select_stages(skip=["build_explorer"])]
    with pytest.raises(ValueError):
        pl.select_stages("enrich", "link")
    assert pl.STAGE_NAMES == runs.STAGE_NAMES


def test_resolve_analysis_dir_and_list_runs(tmp_path):
    a = make_run(str(tmp_path))
    ns = build_parser().parse_args(["build-explorer", "--project-dir", os.path.join(tmp_path, "Proj"),
                                    "-d", "aaaaaaaa", "-a", "bbbbbbbb"])
    assert resolve_analysis_dir(ns) == a
    df = runs.list_runs(str(tmp_path))
    assert list(df["analysis_hash"]) == ["bbbbbbbb"] and not df["configs_ok"].iloc[0]


def test_config_status_detects_mismatched_saved_config(tmp_path):
    proj = tmp_path / "Proj"
    (proj / "configs").mkdir(parents=True)
    good = {"analysis": {"replicate_pairing": "paired", "feature_grouping": {"method": "hdbscan"}}}
    dp = {"datasets": {"tx": {"dataset_dir": "transcriptomics"}}}
    a_hash, d_hash = runs.config_hash(good), runs.config_hash(dp)
    base = proj / "configs" / f"Dataset_Processing--{d_hash}_Analysis--{a_hash}"
    for kind, cfg in (("project", {"project": {}}), ("data_processing", dp), ("analysis", good)):
        (proj / "configs" / f"{base.name}_{kind}_config.yml").write_text(yaml.safe_dump(cfg))
    assert runs.run_config_status(str(proj), d_hash, a_hash)[1]
    # a config saved under the wrong name (the bug seen in real runs) is detected
    wrong = proj / "configs" / f"Dataset_Processing--{d_hash}_Analysis--deadbeef_analysis_config.yml"
    wrong.write_text(yaml.safe_dump(good))
    for kind in ("project", "data_processing"):
        (proj / "configs" / f"Dataset_Processing--{d_hash}_Analysis--deadbeef_{kind}_config.yml").write_text(
            yaml.safe_dump({"project": {}} if kind == "project" else dp))
    assert not runs.run_config_status(str(proj), d_hash, "deadbeef")[1]


def test_run_record_writes_hashed_snapshot_not_edited_file(tmp_path):
    objs = pytest.importorskip("tools.objects")
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "project.yml").write_text(yaml.safe_dump({
        "user_settings": {"project_name": "P"},
        "project": {"dataset_list": ["tx"], "PI_name": "x", "proposal_ID": 1, "results_path": str(tmp_path / "out"),
                    "raw_data_path": str(tmp_path), "link_table": str(tmp_path / "lt.csv")}}))
    (cfg / "data_processing.yml").write_text(yaml.safe_dump({"datasets": {"tx": {"dataset_dir": "transcriptomics"}}}))
    (cfg / "analysis.yml").write_text(yaml.safe_dump({"analysis": {"replicate_pairing": "paired"}}))
    project = objs.Project(config_dir=str(cfg))
    assert project.study_variables == []          # variable_list is optional
    # the user edits analysis.yml while the run is going
    (cfg / "analysis.yml").write_text(yaml.safe_dump({"analysis": {"replicate_pairing": "unpaired"}}))
    manifest = project.save_run_record(stage_timings={"link": 1.0}, argv=["x"])
    saved = yaml.safe_load(open(manifest["configs"]["analysis"]))
    assert runs.config_hash(saved) == project.analysis_hash
    with open(os.path.join(project.analysis_dir, "run_manifest.json")) as fh:
        assert json.load(fh)["stage_timings_sec"] == {"link": 1.0}
    # resuming by hash restores the snapshot
    resumed = objs.Project(config_dir=str(tmp_path / "out" / "P" / "configs"),
                           data_processing_hash=project.data_processing_hash, analysis_hash=project.analysis_hash)
    assert resumed.analysis_hash == project.analysis_hash


def test_explore_host_default(monkeypatch):
    from tools import explorer_launch as el
    assert build_parser().parse_args(["explore", "--bundle", "b"]).host is None  # resolved at launch
    monkeypatch.delenv("JGI_EXPLORER_HOST", raising=False)
    monkeypatch.setattr(el, "in_docker", lambda: True)
    assert el.default_host() == "0.0.0.0"
    monkeypatch.setattr(el, "in_docker", lambda: False)
    assert el.default_host() == "127.0.0.1"
    monkeypatch.setenv("JGI_EXPLORER_HOST", "10.0.0.5")
    assert el.default_host() == "10.0.0.5"
    assert el.browser_url("0.0.0.0", 8050) == "http://localhost:8050"


def test_start_and_stop_explorer_in_background(tmp_path, monkeypatch):
    pytest.importorskip("dash")
    import socket
    import urllib.request
    from tests.explorer_fixtures import run_info
    from tools import explorer_launch as el
    from tools.explorer_bundle.build import build_explorer_bundle

    a = make_run(str(tmp_path / "run"), "network_modules")
    build_explorer_bundle(a, explorer_config={"n_permutations": 19}, run_info=run_info("network_modules"), force=True)
    monkeypatch.setattr(el, "_STATE_DIR", str(tmp_path / "state"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = el.start_explorer(a, port=port, host="127.0.0.1", timeout=120)
    try:
        assert url == f"http://localhost:{port}"
        assert urllib.request.urlopen(url, timeout=10).status == 200
        # starting again replaces the running server rather than failing on the busy port
        el.start_explorer(a, port=port, host="127.0.0.1", timeout=120)
    finally:
        assert el.stop_explorer(port)
    assert not el._port_open(port)
    assert not el.stop_explorer(port)
