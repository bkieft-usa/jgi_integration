"""Lightweight run discovery utilities (no heavy imports, safe for the CLI and the explorer)."""
from __future__ import annotations

from typing import Dict

STAGE_NAMES = [
    "link", "filter", "normalize", "devariance", "replicability", "scale", "integrate",
    "select_features", "annotate", "group_features", "enrich", "build_explorer",
]


def config_hash(config: dict, length: int = 8) -> str:
    """Same hashing as ConfigManager._generate_config_hash."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:length]


def infer_grouping_method(analysis_dir: str):
    """Best-effort grouping method from a run's outputs (used when configs are missing/inconsistent)."""
    import os
    import pandas as pd
    if os.path.isdir(os.path.join(analysis_dir, "wgcna_results")):
        return "wgcna"
    edges = os.path.join(analysis_dir, "feature_network_edge_table.csv")
    if os.path.exists(edges) and os.path.getsize(edges) > 64:
        return "network_modules"
    nodes = os.path.join(analysis_dir, "feature_network_node_table.csv")
    if os.path.exists(nodes):
        groups = pd.read_csv(nodes, usecols=lambda c: c == "group")
        if "group" in groups and (groups["group"].astype(str) == "noise").any():
            return "hdbscan"
    return None


def run_config_status(project_dir: str, data_processing_hash: str, analysis_hash: str):
    """``(configs, ok)``; ``ok`` is False when a saved config does not hash to the run's name."""
    cfg = read_run_configs(project_dir, data_processing_hash, analysis_hash)
    ok = len(cfg) == 3
    if "analysis" in cfg and config_hash(cfg["analysis"]) != analysis_hash:
        ok = False
    if "data_processing" in cfg and config_hash(cfg["data_processing"]) != data_processing_hash:
        ok = False
    return cfg, ok


def read_run_configs(project_dir: str, data_processing_hash: str, analysis_hash: str) -> Dict[str, dict]:
    """Load the persistent hash-tagged configs of a run without instantiating a Project."""
    import os
    import yaml
    base = f"Dataset_Processing--{data_processing_hash}_Analysis--{analysis_hash}"
    out = {}
    for kind in ("project", "data_processing", "analysis"):
        path = os.path.join(project_dir, "configs", f"{base}_{kind}_config.yml")
        if os.path.exists(path):
            with open(path) as fh:
                out[kind] = yaml.safe_load(fh) or {}
    return out


def list_runs(results_dir: str):
    """Every completed or partial run under ``results_dir`` (one row per analysis directory)."""
    import glob
    import os
    from datetime import datetime
    import pandas as pd
    rows = []
    for analysis_dir in sorted(glob.glob(os.path.join(results_dir, "*", "Dataset_Processing--*", "Analysis--*"))):
        proc_dir = os.path.dirname(analysis_dir)
        project_dir = os.path.dirname(proc_dir)
        dp = os.path.basename(proc_dir).split("--", 1)[1]
        an = os.path.basename(analysis_dir).split("--", 1)[1]
        cfg, ok = run_config_status(project_dir, dp, an)
        grouping = ((cfg.get("analysis") or {}).get("analysis") or {}).get("feature_grouping", {}).get("method")
        if not ok:
            grouping = infer_grouping_method(analysis_dir)
        rows.append({
            "project": os.path.basename(project_dir),
            "data_processing_hash": dp,
            "analysis_hash": an,
            "grouping_method": grouping,
            "configs_ok": ok,
            "has_groups": os.path.exists(os.path.join(analysis_dir, "feature_network_node_table.csv")),
            "has_enrichment": os.path.exists(os.path.join(analysis_dir, "enrichment", "enrichment_long.csv")),
            "has_explorer": os.path.exists(os.path.join(analysis_dir, "explorer", "manifest.json")),
            "modified": datetime.fromtimestamp(os.path.getmtime(analysis_dir)).strftime("%Y-%m-%d %H:%M"),
            "analysis_dir": analysis_dir,
        })
    return pd.DataFrame(rows)
