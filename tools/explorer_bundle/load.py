"""Read a run's CSV outputs into clean in-memory tables for bundle building.

Only files on disk are used (no Project/Analysis objects), so the bundle can be built
for any completed run directory, including on a machine other than the one that ran it.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from tools import metadata as md

log = logging.getLogger(__name__)

DATASET_DIR_DEFAULTS = {"tx": "transcriptomics", "mx": "metabolomics", "px": "proteomics"}


def _read_record_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")])


def _read_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, low_memory=False)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


@dataclass
class RunData:
    analysis_dir: str
    dataset_names: List[str]
    pairing: str
    z_all: pd.DataFrame              # integrated z-scores, all features x columns
    z_selected: pd.DataFrame         # features used for grouping x columns
    metadata: pd.DataFrame           # aligned to matrix columns
    sample_metadata: pd.DataFrame    # one row per sample (unique_group), for per-dataset log data
    log_data: Dict[str, pd.DataFrame]  # dataset -> log-scale features x samples (prefixed IDs)
    annotations: pd.DataFrame        # feature_annotation_table (feature_id column)
    groups: pd.Series                # feature_id -> group label (selected/grouped features)
    edges: pd.DataFrame              # source, target, weight (may be empty)
    structure_dir: Optional[str] = None
    sources: Dict[str, float] = field(default_factory=dict)


def source_files(analysis_dir: str) -> List[str]:
    names = ["integrated_data.csv", "integrated_data_selected.csv", "integrated_metadata.csv",
             "feature_annotation_table.csv", "feature_network_node_table.csv", "feature_network_edge_table.csv"]
    return [os.path.join(analysis_dir, n) for n in names]


def infer_dataset_names(z: pd.DataFrame) -> List[str]:
    return sorted({str(i).split("_", 1)[0] for i in z.index})


def load_run(analysis_dir: str, dataset_names: Optional[List[str]] = None,
             datasets_config: Optional[dict] = None, pairing: Optional[str] = None) -> RunData:
    a = analysis_dir
    need = ["integrated_data_selected.csv", "integrated_metadata.csv", "feature_annotation_table.csv",
            "feature_network_node_table.csv"]
    missing = [n for n in need if not os.path.exists(os.path.join(a, n))]
    if missing:
        raise FileNotFoundError(f"Run is incomplete; missing {missing} in {a}. Run the pipeline through "
                                "the 'group_features' stage first.")
    z_sel = _read_matrix(os.path.join(a, "integrated_data_selected.csv"))
    z_all_path = os.path.join(a, "integrated_data.csv")
    z_all = _read_matrix(z_all_path) if os.path.exists(z_all_path) else z_sel
    dataset_names = list(dataset_names or infer_dataset_names(z_all))

    raw_meta = _read_record_table(os.path.join(a, "integrated_metadata.csv"))
    if pairing is None:
        pairing = "paired" if set(z_all.columns) <= set(raw_meta.get("unique_group", pd.Series()).astype(str)) \
            else "unpaired"
    metadata = md.align_metadata(raw_meta, z_all.columns, pairing)
    sample_meta = md.align_metadata(raw_meta, None, "paired")

    ann = _read_record_table(os.path.join(a, "feature_annotation_table.csv"))
    if "feature_id" not in ann.columns:
        ann = ann.rename(columns={ann.columns[0]: "feature_id"})
    ann["feature_id"] = ann["feature_id"].astype(str)

    nodes = pd.read_csv(os.path.join(a, "feature_network_node_table.csv"), usecols=lambda c: c in ("node_id", "group"),
                        low_memory=False)
    groups = nodes.set_index(nodes["node_id"].astype(str))["group"].astype(str)
    groups.index.name = "feature_id"

    edge_path = os.path.join(a, "feature_network_edge_table.csv")
    edges = pd.DataFrame(columns=["source", "target", "weight"])
    if os.path.exists(edge_path):
        e = _read_record_table(edge_path)
        if {"source", "target"} <= set(e.columns) and len(e):
            edges = e[["source", "target"] + (["weight"] if "weight" in e.columns else [])].copy()
            if "weight" not in edges:
                edges["weight"] = 1.0
            key = edges[["source", "target"]].astype(str).apply(lambda r: tuple(sorted(r)), axis=1)
            edges = edges.loc[~key.duplicated()].reset_index(drop=True)

    proc_dir = os.path.dirname(a)
    log_data = {}
    datasets_config = datasets_config or {}
    for ds in dataset_names:
        ds_dir = (datasets_config.get(ds) or {}).get("dataset_dir") or DATASET_DIR_DEFAULTS.get(ds, ds)
        path = os.path.join(proc_dir, ds_dir, "replicate_filtered_data.csv")
        if not os.path.exists(path):
            log.warning(f"No log-scale data for '{ds}' at {path}; feature-level tests will skip it.")
            continue
        df = _read_matrix(path)
        df.index = [i if i.startswith(f"{ds}_") else f"{ds}_{i}" for i in df.index]
        log_data[ds] = df

    structure_dir = os.path.join(a, "grouping_structure")
    sources = {p: os.path.getmtime(p) for p in source_files(a) if os.path.exists(p)}
    return RunData(a, dataset_names, pairing, z_all, z_sel, metadata, sample_meta, log_data, ann, groups, edges,
                   structure_dir if os.path.isdir(structure_dir) else None, sources)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as fh:
        return json.load(fh)
