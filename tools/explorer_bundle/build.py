"""Build the explorer bundle: everything the Dash explorer needs, precomputed.

The bundle (``<analysis_dir>/explorer/``) is self-contained: copy the folder anywhere and
open it with ``python -m tools.cli explore --bundle <folder>``.
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

from tools import annotation_types as annot
from tools import metadata as md
from tools.explorer_bundle import agreement as agr
from tools.explorer_bundle import embed, structure
from tools.explorer_bundle import stats as st
from tools.explorer_bundle.load import RunData, load_run, read_json, source_files

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
BUNDLE_DIRNAME = "explorer"
P_VALUE_COLUMNS = {"p", "padj", "perm_p", "perm_padj", "best_padj"}

DEFAULT_CONFIG = {
    "n_permutations": 1000,       # label permutations for agreement null distributions
    "min_term_size": 2,           # terms with fewer grouped features are not tested
    "min_group_size": 2,          # groups with fewer features are not tested for enrichment
    "min_set_size_tests": 3,      # minimum set size for term score / CAMERA tests
    "fdr": 0.05,
    "exclude_global_maps": True,  # drop KEGG overview maps (map011xx/012xx) from enrichment
    "variable_types": {},         # {variable: categorical|continuous} overrides
    "eigengene_network": {"min_abs_r": 0.7, "top_k": 5},
    "seed": 0,
}

DISPLAY_NAME_COLUMNS = ["{ds}_display_name", "{ds}_Compound_Name", "{ds}_KOfam", "{ds}_COG24_FUNCTION",
                        "{ds}_product", "{ds}_description"]


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
        r = st.row_correlation(Z, e)
        kme.loc[members] = r
        Zc = np.nan_to_num(Z - Z.mean(axis=1, keepdims=True))
        norms = np.linalg.norm(Zc, axis=1, keepdims=True)
        U = np.divide(Zc, norms, out=np.zeros_like(Zc), where=norms > 0)
        m = len(members)
        s = U.sum(axis=0)
        coherence = float((s @ s - m) / (m * (m - 1))) if m > 1 else np.nan
        info[g] = {"eigengene_var_explained": ve, "coherence": coherence}
    E = pd.DataFrame(eig, index=z.columns).T
    return E, pd.DataFrame(info).T, kme


def _term_scores(z_all: pd.DataFrame, long_ann: pd.DataFrame, min_size: int) -> pd.DataFrame:
    """Mean z-profile of each term's features (rows ``ann_type::term_id``)."""
    sub = long_ann[long_ann["feature_id"].isin(z_all.index)]
    sizes = sub.groupby(["ann_type", "term_id"])["feature_id"].transform("size")
    sub = sub[sizes >= min_size]
    if sub.empty:
        return pd.DataFrame(columns=z_all.columns)
    key = sub["ann_type"] + "::" + sub["term_id"]
    codes, uniq = pd.factorize(key)
    import scipy.sparse as sp
    fi = z_all.index.get_indexer(sub["feature_id"])
    A = sp.csr_matrix((np.ones(len(sub)), (codes, fi)), shape=(len(uniq), len(z_all)))
    counts = np.asarray(A.sum(axis=1)).ravel()
    scores = (A @ np.nan_to_num(z_all.to_numpy(dtype=float))) / counts[:, None]
    return pd.DataFrame(scores, index=uniq, columns=z_all.columns)


def _feature_associations(run: RunData, variables, types, nesting):
    frames, fits = [], {}
    for ds, df in run.log_data.items():
        meta = run.sample_metadata.reindex(df.columns).dropna(how="all")
        tab, f = st.associate_matrix(df[meta.index], meta, variables, types, nesting, "feature", keep_fits=True)
        tab["dtype"] = ds
        frames.append(tab)
        fits[ds] = f
    if not frames:
        return pd.DataFrame(columns=st.TABLE_COLUMNS), fits
    tab = pd.concat(frames, ignore_index=True)
    # One BH family per (variable, contrast) across all data types
    tab["padj"] = tab.groupby(["variable", "contrast"])["p"].transform(lambda p: st.bh_adjust(p.to_numpy()))
    return tab, fits


def _camera_terms(fits: Dict[str, dict], long_ann: pd.DataFrame, min_size: int) -> pd.DataFrame:
    """CAMERA for every term x (variable, directional contrast)."""
    rows = []
    variables = sorted({v for f in fits.values() for v in f})
    for var in variables:
        parts = [(ds, f[var]) for ds, f in fits.items() if var in f]
        if not parts:
            continue
        ids = np.concatenate([np.asarray(ids) for _, (_, ids) in parts])
        blocks = np.concatenate([np.full(len(ids_), i) for i, (_, (_, ids_)) in enumerate(parts)])
        resid_shapes = {fit.residuals.shape[1] for _, (fit, _) in parts}
        same_samples = len(resid_shapes) == 1 and len({fit.n_obs for _, (fit, _) in parts}) == 1
        resid = np.vstack([fit.residuals for _, (fit, _) in parts]) if same_samples else None
        index = pd.Index(ids)
        sets_by_type = {}
        sub = long_ann[long_ann["feature_id"].isin(index)]
        for (at, term), grp in sub.groupby(["ann_type", "term_id"]):
            if len(grp) >= min_size:
                sets_by_type.setdefault(at, {})[term] = index.get_indexer(grp["feature_id"])
        first = parts[0][1][0]
        contrast_names = [c for c in first.contrasts if all(c in fit.contrasts for _, (fit, _) in parts)]
        if first.var_type == "categorical" and len(first.levels) == 2:
            contrast_names = contrast_names[:1]  # binary: the second contrast mirrors the first
        for cname in contrast_names:
            z = np.concatenate([st.t_to_z(st.moderated_t(fit, fit.contrasts[cname])[1], fit.df_total)
                                for _, (fit, _) in parts])
            ok = np.isfinite(z)
            label = cname if parts[0][1][0].var_type == "continuous" else f"{cname} vs rest"
            if resid is None:
                # data types measured on different samples: correlation estimated within each type only
                cam_resid, cam_blocks = _block_residuals(parts), blocks
            else:
                cam_resid, cam_blocks = resid, None
            for at, sets in sets_by_type.items():
                res = st.camera(np.where(ok, z, 0.0), cam_resid, sets, blocks=cam_blocks, min_size=min_size)
                if res.empty:
                    continue
                res["padj"] = st.bh_adjust(res["p"].to_numpy())
                rows.append(pd.DataFrame({
                    "entity_type": "term", "entity_id": at + "::" + res["set"], "variable": var,
                    "contrast": label, "test": "camera", "statistic": res["statistic"], "effect": res["mean_z"],
                    "p": res["p"], "padj": res["padj"], "n": res["n"], "unit": parts[0][1][0].unit or "",
                    "aux": res["correlation"],
                }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _block_residuals(parts):
    """Pad per-block residual matrices to a common width (zeros) for within-block correlation."""
    width = max(fit.residuals.shape[1] for _, (fit, _) in parts)
    return np.vstack([np.pad(fit.residuals, ((0, 0), (0, width - fit.residuals.shape[1]))) for _, (fit, _) in parts])


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
    cfg["eigengene_network"] = {**DEFAULT_CONFIG["eigengene_network"], **(cfg.get("eigengene_network") or {})}
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
    nesting_samples = md.detect_nesting(run.sample_metadata, variables)
    nesting_columns = md.detect_nesting(run.metadata, variables)

    # ---- annotations --------------------------------------------------------------------
    t0 = time.time()
    ann_types = annot.detect_annotation_types(run.annotations, run.dataset_names, datasets_config)
    ann = run.annotations[run.annotations["feature_id"].isin(run.z_all.index)]
    long_ann = annot.explode_all(ann, ann_types)
    term_names = long_ann.drop_duplicates(["ann_type", "term_id"]).set_index(["ann_type", "term_id"])["term_name"]
    tick("annotations", t0)

    # ---- group scores and embeddings --------------------------------------------------
    t0 = time.time()
    E, ginfo, kme = _group_scores(run.z_selected, groups)
    tick("group eigengenes", t0)
    t0 = time.time()
    sample_coords, explained = embed.sample_embedding(run.z_selected, seed=cfg["seed"])
    feat_xy = embed.feature_embedding(run.z_selected.reindex(groups.index).dropna(how="all"), seed=cfg["seed"])
    tick("embeddings", t0)

    # ---- associations -----------------------------------------------------------------
    t0 = time.time()
    feat_assoc, fits = _feature_associations(run, variables, types, nesting_samples)
    tick("feature associations", t0)
    t0 = time.time()
    group_assoc, _ = st.associate_matrix(E, run.metadata, variables, types, nesting_columns, "group")
    T = _term_scores(run.z_all, long_ann, cfg["min_set_size_tests"])
    term_assoc, _ = st.associate_matrix(T, run.metadata, variables, types, nesting_columns, "term")
    if not term_assoc.empty:  # BH per annotation type, not across all types at once
        at_of = term_assoc["entity_id"].str.split("::").str[0]
        term_assoc["padj"] = term_assoc.groupby([at_of, "variable", "contrast"])["p"].transform(
            lambda p: st.bh_adjust(p.to_numpy()))
    tick("group/term associations", t0)
    t0 = time.time()
    camera_tab = _camera_terms(fits, long_ann, cfg["min_set_size_tests"])
    tick("CAMERA", t0)
    assoc = pd.concat([feat_assoc, group_assoc, term_assoc, camera_tab], ignore_index=True)
    for c in ("dtype", "aux"):
        if c not in assoc:
            assoc[c] = np.nan
    descriptive_only = run.pairing != "paired"

    # ---- enrichment and agreement -------------------------------------------------------
    t0 = time.time()
    at_keys = [a.key for a in ann_types]
    enrich = agr.enrichment_table(long_ann, groups, at_keys, cfg["min_term_size"], cfg["min_group_size"],
                                  cfg["exclude_global_maps"])
    enrich["term_name"] = [term_names.get((a, t), "") for a, t in zip(enrich["ann_type"], enrich["term_id"])]
    tick("enrichment", t0)
    t0 = time.time()
    agree = agr.agreement(long_ann, groups, at_keys, enrich, n_permutations=int(cfg["n_permutations"]),
                          min_term_size=cfg["min_term_size"], alpha=cfg["fdr"],
                          exclude_global_maps=cfg["exclude_global_maps"], seed=cfg["seed"])
    tick("agreement", t0)

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
    feats[["x", "y"]] = feat_xy.reindex(feats.index)
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
    gpos = embed.group_layout(E.loc[[g for g in E.index if not md.is_noise_label(g)]], seed=cfg["seed"])
    gt = gt.merge(gpos.rename_axis("group").reset_index(), on="group", how="left")
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

    # ---- structure ---------------------------------------------------------------------------
    t0 = time.time()
    real_E = E.loc[[g for g in E.index if not md.is_noise_label(g)]]
    gnet = structure.eigengene_network(real_E, **cfg["eigengene_network"])
    method = run_info.get("grouping_method")
    centroids = run.z_selected.reindex(groups.index).groupby(groups).mean()
    native = None
    try:
        if method == "hierarchical_clustering":
            native = structure.hierarchical_dendrogram(run.structure_dir, groups, centroids)
        elif method == "hdbscan":
            native = structure.hdbscan_tree(run.structure_dir, groups, run.z_selected, run_info.get("grouping_params"))
        elif method == "network_modules" and len(run.edges):
            native = {"kind": "network", "approximate": False,
                      "note": "Correlation network used to define the modules (edges = correlations above the cutoff)."}
    except Exception as exc:
        log.warning(f"Native structure view unavailable: {exc}")
    sim = structure.sample_similarity(run.z_selected)
    tick("structure", t0)

    # ---- candidates (guilt by association) ---------------------------------------------------
    best_term = sig.sort_values("padj").groupby("group").head(3)
    best_term = best_term.assign(label=best_term["ann_type"] + ": " +
                                 best_term["term_name"].where(best_term["term_name"] != "", best_term["term_id"]) +
                                 " (padj " + best_term["padj"].map(lambda v: f"{v:.1e}") + ")")
    group_terms = best_term.groupby("group")["label"].agg(" | ".join)
    group_strength = -np.log10(sig.groupby("group")["padj"].min().clip(lower=1e-300))
    cand = feats[(~feats["is_annotated"]) & feats["group"].isin(group_terms.index)].copy()
    cand["group_evidence"] = cand["group"].map(group_strength)
    cand["score"] = cand["kME"].clip(lower=0) * cand["group_evidence"]
    cand["enriched_terms"] = cand["group"].map(group_terms)
    cand = cand[["feature_id", "dtype", "display_name", "group", "kME", "group_evidence", "score", "enriched_terms"]] \
        .sort_values("score", ascending=False)

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
    _write(T.rename_axis("entity_id").reset_index(), out, "term_scores")
    _write(tt, out, "terms")
    _write(enrich, out, "enrichment")
    _write(assoc, out, "associations")
    for k, v in agree.items():
        if not v.empty:
            _write(v, out, f"agreement_{k}")
    _write(gnet, out, "group_network")
    if len(run.edges):
        _write(run.edges.astype({"source": str, "target": str}), out, "feature_edges")
    _write(cand, out, "candidates")
    with open(os.path.join(out, "native_structure.json"), "w") as fh:
        json.dump(native, fh)
    tick("write", t0)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_dir": os.path.abspath(analysis_dir),
        **run_info,
        "grouping_params": run_info.get("grouping_params", {}),
        "dataset_names": run.dataset_names,
        "n_samples": int(run.z_all.shape[1]),
        "n_features": int(run.z_all.shape[0]),
        "n_selected": int(run.z_selected.shape[0]),
        "n_grouped": int(len(groups)),
        "n_groups": int((~gt["is_noise"]).sum()),
        "n_noise": int(gt.loc[gt["is_noise"], "size"].sum()),
        "variables": variables,
        "variable_types": types,
        "nesting": {v: u for v, u in nesting_samples.items() if u},
        "descriptive_only": descriptive_only,
        "sample_pca_explained": explained,
        "annotation_types": annot.ann_types_summary(ann_types),
        "native_structure": (native or {}).get("kind"),
        "config": cfg,
        "timings_sec": timings,
        "build_seconds": round(time.time() - t_start, 1),
        "sources": {os.path.basename(p): t for p, t in run.sources.items()},
    }
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    log.info(f"[explorer] bundle written to {out} in {manifest['build_seconds']}s")
    return out
