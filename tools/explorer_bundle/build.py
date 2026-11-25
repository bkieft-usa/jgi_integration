"""Build the explorer bundle: everything the Dash explorer needs, precomputed.

The bundle (``<analysis_dir>/explorer/``) is self-contained: copy the folder anywhere and
open it with ``python -m tools.cli explore --bundle <folder>``.

Statistics (see the explorer's Methods page for the full description):
* features, group eigengenes and term eigengenes vs metadata: moderated linear model
  (F test across levels, t test for every pair of levels, or slope), BH per comparison;
* term vs group: one-sided hypergeometric test, BH per annotation type;
* overall agreement per annotation type: same-group share of co-annotated feature pairs vs
  a label-permutation null;
* unannotated-feature discovery: neighbour voting (hypergeometric) on the feature network.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tools import annotation_types as annot
from tools import metadata as md
from tools.explorer_bundle import agreement as agr
from tools.explorer_bundle import embed, structure
from tools.explorer_bundle import network as net
from tools.explorer_bundle import stats as st
from tools.explorer_bundle.load import RunData, load_run, read_json, source_files

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2
BUNDLE_DIRNAME = "explorer"
P_VALUE_COLUMNS = {"p", "padj", "best_padj"}

DEFAULT_CONFIG = {
    "n_permutations": 1000,       # label permutations for the agreement null distribution
    "min_term_size": 2,           # terms with fewer grouped features are not tested for enrichment
    "min_group_size": 2,          # groups with fewer features are not tested for enrichment
    "min_set_size_tests": 3,      # terms need this many features for a term eigengene (metadata tests)
    "fdr": 0.05,
    "exclude_global_maps": True,  # drop KEGG overview maps (map011xx/012xx) from enrichment
    "variable_types": {},         # {variable: categorical|continuous} overrides
    "network_k": 5,               # neighbours per feature in the derived network (non-network methods)
    "max_votes_per_feature": 20,  # best discovery terms kept per feature and annotation type
    "seed": 0,
}

DISPLAY_NAME_COLUMNS = ["{ds}_display_name", "{ds}_Compound_Name", "{ds}_KOfam", "{ds}_COG24_FUNCTION",
                        "{ds}_product", "{ds}_description"]

ASSOC_SCHEMA = pa.schema([
    ("entity_type", pa.string()), ("entity_id", pa.string()), ("variable", pa.string()),
    ("contrast", pa.string()), ("test", pa.string()), ("statistic", pa.float32()), ("effect", pa.float32()),
    ("p", pa.float64()), ("padj", pa.float64()), ("n", pa.int32()), ("unit", pa.string()), ("dtype", pa.string()),
])


def bundle_dir_for(analysis_dir: str) -> str:
    return os.path.join(analysis_dir, BUNDLE_DIRNAME)


def bundle_is_current(analysis_dir: str) -> bool:
    manifest = read_json(os.path.join(bundle_dir_for(analysis_dir), "manifest.json"))
    if not manifest or manifest.get("schema_version") != SCHEMA_VERSION:
        return False
    recorded = manifest.get("sources", {})
    for path in source_files(analysis_dir):
        if os.path.exists(path):
            name = os.path.basename(path)
            if name not in recorded or os.path.getmtime(path) > recorded[name] + 1e-6:
                return False
    return True


def _write(df: pd.DataFrame, out_dir: str, name: str, index: bool = False) -> None:
    df = df.copy()
    for c in df.columns:
        # p-values keep float64: float32 underflows to 0 below ~1e-45, common for strong effects
        if df[c].dtype == "float64" and c not in P_VALUE_COLUMNS:
            df[c] = df[c].astype("float32")
        elif df[c].dtype == object:
            df[c] = df[c].astype(str).where(df[c].notna(), None)
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=index)


class _AssocWriter:
    """Streams association tables to one parquet file, one row group per chunk, sorted so that
    DuckDB can skip row groups when the app asks for one (entity type, variable, comparison)."""

    def __init__(self, path: str):
        self.path = path
        self.writer = pq.ParquetWriter(path, ASSOC_SCHEMA)
        self.rows = 0
        self.tests: Dict[str, int] = {}

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.copy()
        if "dtype" not in df:
            df["dtype"] = None
        df = df.sort_values(["contrast", "p"], kind="stable")
        for t, n in df["test"].value_counts().items():
            self.tests[t] = self.tests.get(t, 0) + int(n)
        self.rows += len(df)
        self.writer.write_table(pa.Table.from_pandas(df[ASSOC_SCHEMA.names], schema=ASSOC_SCHEMA,
                                                     preserve_index=False), row_group_size=100_000)

    def close(self) -> None:
        self.writer.close()


def _display_names(ann: pd.DataFrame, dataset_names: List[str]) -> pd.Series:
    out = pd.Series("", index=ann["feature_id"].to_numpy(), dtype=object)
    for ds in dataset_names:
        is_ds = ann["feature_id"].str.startswith(f"{ds}_").to_numpy()
        for pattern in DISPLAY_NAME_COLUMNS:
            col = pattern.format(ds=ds)
            if col not in ann.columns:
                continue
            vals = ann[col].astype(str).to_numpy()
            fill = is_ds & (out.to_numpy() == "") & ~pd.Series(vals).str.lower().isin(
                list(annot.MISSING_TOKENS)).to_numpy()
            out.iloc[np.where(fill)[0]] = [v.split(";;")[0][:120] for v in vals[fill]]
    return out


def _group_scores(z: pd.DataFrame, groups: pd.Series):
    """Eigengenes, variance explained, coherence and per-feature membership (kME)."""
    eig, info, kme = {}, {}, pd.Series(np.nan, index=groups.index, dtype=float)
    for g, members in groups.groupby(groups).groups.items():
        if md.is_noise_label(g):
            continue
        members = [m for m in members if m in z.index]
        if not members:
            continue
        Z = z.loc[members].to_numpy(dtype=float)
        e, ve = st.eigengene(Z)
        eig[g] = e
        kme.loc[members] = st.row_correlation(Z, e)
        Zc = np.nan_to_num(Z - Z.mean(axis=1, keepdims=True))
        norms = np.linalg.norm(Zc, axis=1, keepdims=True)
        U = np.divide(Zc, norms, out=np.zeros_like(Zc), where=norms > 0)
        m = len(members)
        s = U.sum(axis=0)
        coherence = float((s @ s - m) / (m * (m - 1))) if m > 1 else np.nan
        info[g] = {"eigengene_var_explained": ve, "coherence": coherence}
    E = pd.DataFrame(eig, index=z.columns).T
    return E, pd.DataFrame(info).T, kme


def _term_eigengenes(z_all: pd.DataFrame, long_ann: pd.DataFrame, min_size: int) -> pd.DataFrame:
    """Eigengene of each term's features (rows ``ann_type::term_id``), the same summary used
    for data-driven groups, so groups and terms are tested identically."""
    sub = long_ann[long_ann["feature_id"].isin(z_all.index)].drop_duplicates(["ann_type", "term_id", "feature_id"])
    sizes = sub.groupby(["ann_type", "term_id"])["feature_id"].transform("size")
    sub = sub[sizes >= min_size]
    if sub.empty:
        return pd.DataFrame(columns=z_all.columns)
    Z = np.nan_to_num(z_all.to_numpy(dtype=float))
    row_of = pd.Series(np.arange(len(z_all)), index=z_all.index)
    key = (sub["ann_type"] + "::" + sub["term_id"]).to_numpy()
    rows = row_of.reindex(sub["feature_id"]).to_numpy()
    order = np.argsort(key, kind="stable")
    key, rows = key[order], rows[order]
    bounds = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    names, scores = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        names.append(key[a])
        scores.append(st.eigengene(Z[rows[a:b]])[0])
    return pd.DataFrame(np.vstack(scores), index=names, columns=z_all.columns)


def _bh_by(df: pd.DataFrame, keys) -> pd.Series:
    return df.groupby(keys)["p"].transform(lambda p: st.bh_adjust(p.to_numpy()))


def _write_associations(writer: _AssocWriter, run: RunData, E: pd.DataFrame, T: pd.DataFrame, variables, types,
                        nesting_samples, nesting_columns) -> pd.DataFrame:
    """Features (per-sample log data), group eigengenes and term eigengenes vs every variable.
    One variable at a time to bound memory. Returns the group table (for group summaries)."""
    for var in variables:
        parts = []
        for ds, df in run.log_data.items():
            meta = run.sample_metadata.reindex(df.columns).dropna(how="all")
            tab = st.associate_matrix(df[meta.index], meta, [var], types, nesting_samples, "feature")
            parts.append(tab.assign(dtype=ds))
        if parts:
            tab = pd.concat(parts, ignore_index=True)
            if not tab.empty:
                tab["padj"] = _bh_by(tab, "contrast")  # one BH family per comparison, across data types
                writer.write(tab)
    group_tab = st.associate_matrix(E, run.metadata, variables, types, nesting_columns, "group")
    writer.write(group_tab)
    for var in variables:
        tab = st.associate_matrix(T, run.metadata, [var], types, nesting_columns, "term")
        if not tab.empty:  # BH per annotation type and comparison
            tab["padj"] = _bh_by(tab, [tab["entity_id"].str.split("::").str[0], "contrast"])
            writer.write(tab)
    return group_tab


def build_explorer_bundle(
    analysis_dir: str,
    explorer_config: Optional[dict] = None,
    dataset_names: Optional[List[str]] = None,
    requested_variables: Optional[List[str]] = None,
    datasets_config: Optional[dict] = None,
    run_info: Optional[dict] = None,
    force: bool = False,
) -> str:
    """Compute and write ``<analysis_dir>/explorer/``. Returns the bundle path."""
    out = bundle_dir_for(analysis_dir)
    if not force and bundle_is_current(analysis_dir):
        log.info(f"Explorer bundle is current: {out}")
        return out
    t_start = time.time()
    cfg = {**DEFAULT_CONFIG, **(explorer_config or {})}
    run_info = dict(run_info or {})
    os.makedirs(out, exist_ok=True)
    timings = {}

    def tick(label, t0):
        timings[label] = round(time.time() - t0, 1)
        log.info(f"[explorer] {label}: {timings[label]}s")

    t0 = time.time()
    run = load_run(analysis_dir, dataset_names, datasets_config, run_info.get("replicate_pairing"))
    run_info.setdefault("replicate_pairing", run.pairing)
    groups = run.groups[run.groups.index.isin(run.z_all.index)]
    tick("load", t0)

    # ---- metadata ---------------------------------------------------------------------
    variables = md.infer_metadata_variables(run.sample_metadata, run.dataset_names, requested_variables)
    types = md.infer_variable_types(run.sample_metadata, variables, cfg["variable_types"])
    aliases = md.identical_variables(run.sample_metadata, variables)  # same partition: test once
    tested = [v for v in variables if v not in aliases]
    nesting_samples = md.detect_nesting(run.sample_metadata, tested)
    nesting_columns = md.detect_nesting(run.metadata, tested)

    # ---- annotations --------------------------------------------------------------------
    t0 = time.time()
    ann_types = annot.detect_annotation_types(run.annotations, run.dataset_names, datasets_config)
    ann = run.annotations[run.annotations["feature_id"].isin(run.z_all.index)]
    long_ann = annot.explode_all(ann, ann_types)
    term_names = long_ann.drop_duplicates(["ann_type", "term_id"]).set_index(["ann_type", "term_id"])["term_name"]
    tick("annotations", t0)

    # ---- group and term eigengenes, sample embedding ------------------------------------
    t0 = time.time()
    E, ginfo, kme = _group_scores(run.z_selected, groups)
    T = _term_eigengenes(run.z_all, long_ann, cfg["min_set_size_tests"])
    tick("eigengenes", t0)
    t0 = time.time()
    sample_coords, explained = embed.sample_embedding(run.z_selected, seed=cfg["seed"])
    tick("sample embedding", t0)

    # ---- associations -------------------------------------------------------------------
    t0 = time.time()
    writer = _AssocWriter(os.path.join(out, "associations.parquet"))
    try:
        group_assoc = _write_associations(writer, run, E, T, tested, types, nesting_samples, nesting_columns)
    finally:
        writer.close()
    tick("associations", t0)
    descriptive_only = run.pairing != "paired"

    # ---- enrichment and agreement -------------------------------------------------------
    t0 = time.time()
    at_keys = [a.key for a in ann_types]
    enrich = agr.enrichment_table(long_ann, groups, at_keys, cfg["min_term_size"], cfg["min_group_size"],
                                  cfg["exclude_global_maps"])
    enrich["term_name"] = [term_names.get((a, t), "") for a, t in zip(enrich["ann_type"], enrich["term_id"])]
    agree = agr.agreement(long_ann, groups, at_keys, n_permutations=int(cfg["n_permutations"]),
                          min_term_size=cfg["min_term_size"], exclude_global_maps=cfg["exclude_global_maps"],
                          seed=cfg["seed"])
    tick("enrichment and agreement", t0)

    # ---- feature network, layout, discovery votes ----------------------------------------
    t0 = time.time()
    network_nodes = pd.Index(groups.index.astype(str))
    if len(run.edges):
        network_kind = "real"
        edges = run.edges.astype({"source": str, "target": str})
        edges = edges[edges["source"].isin(network_nodes) & edges["target"].isin(network_nodes)].reset_index(drop=True)
    else:
        network_kind = "knn"
        edges = net.knn_edges(run.z_selected.reindex(network_nodes).dropna(how="all"), k=int(cfg["network_k"]))
    xy = net.layout(network_nodes, edges, seed=cfg["seed"])
    tick("network layout", t0)
    t0 = time.time()
    dtype_of = pd.Series(network_nodes.str.split("_").str[0], index=network_nodes)
    votes = net.neighbor_votes(edges, network_nodes, long_ann, ann_types, dtype_of,
                               max_terms_per_feature=int(cfg["max_votes_per_feature"]))
    tick("discovery votes", t0)

    # ---- feature table ---------------------------------------------------------------------
    feats = pd.DataFrame(index=pd.Index(run.z_all.index.astype(str), name="feature_id"))
    feats["dtype"] = feats.index.str.split("_").str[0]
    feats["display_name"] = _display_names(run.annotations, run.dataset_names).reindex(feats.index).fillna("")
    feats["group"] = groups.reindex(feats.index)
    feats["is_selected"] = feats.index.isin(run.z_selected.index)
    feats["is_noise"] = feats["group"].map(lambda g: md.is_noise_label(g) if isinstance(g, str) else False)
    n_terms = long_ann.groupby("feature_id").size()
    feats["n_terms"] = n_terms.reindex(feats.index).fillna(0).astype(int)
    feats["is_annotated"] = feats["n_terms"] > 0
    feats["kME"] = kme.reindex(feats.index)
    feats[["x", "y"]] = xy.reindex(feats.index)
    degree = pd.concat([edges["source"], edges["target"]]).value_counts() if len(edges) else pd.Series(dtype=int)
    feats["degree"] = degree.reindex(feats.index).fillna(0).astype(int).where(feats.index.isin(network_nodes))
    log_all = pd.concat(run.log_data.values()) if run.log_data else pd.DataFrame()
    if not log_all.empty:
        feats["mean_log_abundance"] = log_all.mean(axis=1).reindex(feats.index)
        feats["sd_log_abundance"] = log_all.std(axis=1).reindex(feats.index)
    feats = feats.reset_index()

    # ---- group table ---------------------------------------------------------------------
    gsize = groups.value_counts()
    gt = pd.DataFrame({"group": gsize.index.astype(str), "size": gsize.to_numpy()})
    dcounts = feats.dropna(subset=["group"]).groupby(["group", "dtype"]).size().unstack(fill_value=0)
    for ds in dcounts.columns:
        gt[f"n_{ds}"] = gt["group"].map(dcounts[ds]).fillna(0).astype(int).to_numpy()
    ann_frac = feats.dropna(subset=["group"]).groupby("group")["is_annotated"].mean()
    gt["frac_annotated"] = gt["group"].map(ann_frac).to_numpy()
    gt["is_noise"] = gt["group"].map(md.is_noise_label)
    gt = gt.merge(ginfo.rename_axis("group").reset_index(), on="group", how="left")
    if not group_assoc.empty:
        best = group_assoc.sort_values("padj").drop_duplicates("entity_id").set_index("entity_id")
        gt["best_variable"] = gt["group"].map(best["variable"])
        gt["best_contrast"] = gt["group"].map(best["contrast"])
        gt["best_padj"] = gt["group"].map(best["padj"])
    sig = enrich[enrich["padj"] < cfg["fdr"]]
    gt["n_enriched_terms"] = gt["group"].map(sig.groupby("group").size()).fillna(0).astype(int)
    top = sig.sort_values("padj").drop_duplicates("group").set_index("group")
    gt["top_term"] = gt["group"].map(top["ann_type"] + ": " + top["term_name"].where(top["term_name"] != "", top["term_id"]))
    gt = gt.sort_values("size", ascending=False).reset_index(drop=True)

    # ---- term table ------------------------------------------------------------------------
    tt = long_ann.groupby(["ann_type", "term_id"]).agg(term_name=("term_name", "first"),
                                                      n_features=("feature_id", "nunique")).reset_index()
    lg = long_ann.assign(group=groups.reindex(long_ann["feature_id"]).to_numpy(),
                         dtype=long_ann["feature_id"].str.split("_").str[0])
    tt = tt.merge(lg.dropna(subset=["group"]).groupby(["ann_type", "term_id"])["feature_id"].nunique()
                  .rename("n_grouped").reset_index(), on=["ann_type", "term_id"], how="left")
    dt = lg.groupby(["ann_type", "term_id", "dtype"]).size().unstack(fill_value=0)
    for ds in dt.columns:
        tt[f"n_{ds}"] = pd.MultiIndex.from_frame(tt[["ann_type", "term_id"]]).map(dt[ds].to_dict()).fillna(0).astype(int)
    if not agree["terms"].empty:
        tt = tt.merge(agree["terms"].drop(columns=["m"]), on=["ann_type", "term_id"], how="left")
    tt["n_grouped"] = tt["n_grouped"].fillna(0).astype(int)
    label_of = {a.key: a.label for a in ann_types}
    tt["ann_label"] = tt["ann_type"].map(label_of)
    tt["entity_id"] = tt["ann_type"] + "::" + tt["term_id"]

    # ---- how groups were formed (non-network methods) ------------------------------------------
    t0 = time.time()
    method = run_info.get("grouping_method")
    centroids = run.z_selected.reindex(groups.index).groupby(groups).mean()
    native = None
    try:
        if method == "hierarchical_clustering":
            native = structure.hierarchical_dendrogram(run.structure_dir, groups, centroids)
        elif method == "hdbscan":
            native = structure.hdbscan_tree(run.structure_dir, groups, run.z_selected, run_info.get("grouping_params"))
    except Exception as exc:
        log.warning(f"Clustering tree view unavailable: {exc}")
    sim = structure.sample_similarity(run.z_selected)
    tick("structure", t0)

    # ---- write -------------------------------------------------------------------------------
    t0 = time.time()
    samples = run.metadata.copy()
    samples.index = samples.index.astype(str)
    samples = samples.drop(columns=[c for c in samples.columns
                                    if any(c == d or str(c).startswith(f"{d}_") for d in run.dataset_names)])
    samples = samples.join(sample_coords).rename_axis("sample_id").reset_index()
    _write(samples, out, "samples")
    _write(sim.rename_axis("sample_id").reset_index(), out, "sample_similarity")
    _write(feats, out, "features")
    _write(run.z_all.astype("float32").rename_axis("feature_id").reset_index(), out, "abundance_z")
    if not log_all.empty:
        _write(log_all.astype("float32").rename_axis("feature_id").reset_index(), out, "abundance_log")
    smeta = run.sample_metadata.drop(columns=[c for c in run.sample_metadata.columns
                                              if any(c == d or str(c).startswith(f"{d}_") for d in run.dataset_names)])
    _write(smeta.rename_axis("sample_id").reset_index(drop=False), out, "sample_metadata")
    _write(long_ann, out, "annotations_long")
    _write(gt, out, "groups")
    _write(E.rename_axis("group").reset_index(), out, "group_eigengenes")
    _write(T.rename_axis("entity_id").reset_index(), out, "term_eigengenes")
    _write(tt, out, "terms")
    _write(enrich, out, "enrichment")
    _write(agree["types"], out, "agreement")
    _write(edges, out, "feature_edges")
    _write(votes, out, "discovery_votes")
    with open(os.path.join(out, "native_structure.json"), "w") as fh:
        json.dump(native, fh)
    for stale in ("candidates", "group_network", "term_scores", "agreement_types", "agreement_terms",
                  "agreement_groups", "agreement_null"):  # files from schema 1
        path = os.path.join(out, f"{stale}.parquet")
        if os.path.exists(path):
            os.remove(path)
    tick("write", t0)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_dir": os.path.abspath(analysis_dir),
        **run_info,
        "grouping_params": run_info.get("grouping_params", {}),
        "dataset_names": run.dataset_names,
        "n_samples": int(run.z_all.shape[1]),
        "sample_ids": [str(c) for c in run.z_all.columns],
        "n_features": int(run.z_all.shape[0]),
        "n_selected": int(run.z_selected.shape[0]),
        "n_grouped": int(len(groups)),
        "n_groups": int((~gt["is_noise"]).sum()),
        "n_noise": int(gt.loc[gt["is_noise"], "size"].sum()),
        "variables": variables,
        "variable_aliases": aliases,
        "variable_types": types,
        "nesting": {v: u for v, u in nesting_samples.items() if u},
        "descriptive_only": descriptive_only,
        "sample_pca_explained": explained,
        "annotation_types": annot.ann_types_summary(ann_types),
        "network": {"kind": network_kind, "n_nodes": int(len(network_nodes)), "n_edges": int(len(edges)),
                    "k": int(cfg["network_k"]) if network_kind == "knn" else None},
        "native_structure": (native or {}).get("kind"),
        "n_tests": writer.tests,
        "config": cfg,
        "timings_sec": timings,
        "build_seconds": round(time.time() - t_start, 1),
        "sources": {os.path.basename(p): t for p, t in run.sources.items()},
    }
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    log.info(f"[explorer] bundle written to {out} in {manifest['build_seconds']}s")
    return out
