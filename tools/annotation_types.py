"""Registry of knowledge-driven annotation types found in ``feature_annotation_table``.

The full annotation scope is always used: KEGG pathway/module/reaction and ModelSEED
pathways (cross-omic, derived by ``annotate_integrated_features``) plus every
per-dataset annotation column that defines shareable terms (KO, COG category/pathway/
function, KEGG BRITE, EC, ClassyFire and NPClassifier classes, ...).

Each type is parsed into a long table ``feature_id, ann_type, term_id, term_name`` with
normalized IDs, which is used by both the static enrichment stage (``enrich_all``) and
the explorer bundle, so the two always agree.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

import pandas as pd

MISSING_TOKENS = {"", "unassigned", "nan", "none", "na", "n/a", "-"}

# Cross-omic columns added by annotate_integrated_features
DERIVED_TYPES = [
    ("kegg_pathway", "KEGG pathway", "kegg_pathway_name"),
    ("kegg_module", "KEGG module", "kegg_module_name"),
    ("kegg_reaction", "KEGG reaction", "kegg_reaction_name"),
]
MODELSEED_COL = "modelseed_pathway"

# Raw per-dataset term columns that duplicate a derived type or are not term-like
SKIP_RAW = {"KEGG_Module"}          # duplicated by the cross-omic kegg_module
LABEL_FROM_NAME = {"KEGG_Class"}     # its _ACC holds module IDs; the name holds the class path
BRITE = {"KEGG_BRITE"}

ID_LIKE_RE = re.compile(
    r"(smiles|inchi|formula|usi|display_name|compound_name|(^|_)name$|(^|_)id$|_acc$|mz|rt$|retention)",
    re.IGNORECASE,
)
KNOWN_LABELS = {
    "KOfam": "KEGG ortholog (KO)",
    "COG24_PATHWAY": "COG pathway",
    "COG24_CATEGORY": "COG category",
    "COG24_FUNCTION": "COG function",
    "KEGG_BRITE": "KEGG BRITE",
    "KEGG_Class": "KEGG module class",
    "superclass": "ClassyFire superclass",
    "class": "ClassyFire class",
    "subclass": "ClassyFire subclass",
    "npclassifier_pathway": "NPClassifier pathway",
    "npclassifier_superclass": "NPClassifier superclass",
    "npclassifier_class": "NPClassifier class",
    "modelseed_rxn": "ModelSEED reaction",
}
MODELSEED_RE = re.compile(r"^(\S+)\s+\((.*)\)$")


@dataclass
class AnnType:
    key: str                  # ann_type identifier, e.g. 'kegg_module', 'tx_COG24_PATHWAY'
    label: str                # human-readable name
    id_col: str               # column holding term IDs (or labels)
    name_col: Optional[str]   # parallel column holding term names, aligned by position
    sep: str                  # delimiter between multiple terms in one cell
    kind: str                 # 'id_name' | 'modelseed' | 'brite' | 'label'
    dtype_scope: Optional[str]  # dataset prefix the type applies to, or None (cross-omic)

    def to_dict(self) -> dict:
        return asdict(self)


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip().lower() in MISSING_TOKENS


def _shared_terms(series: pd.Series, sep: str) -> bool:
    """True when the column's terms are shared across features (a grouping, not an identifier)."""
    vals = series[~series.map(_is_missing)].astype(str)
    if len(vals) < 4:
        return False
    terms = vals.str.split(re.escape(sep)).explode().str.strip()
    return terms.nunique() <= 0.5 * len(vals)


def _ds_label(ds: str, raw: str) -> str:
    return f"{KNOWN_LABELS.get(raw, raw.replace('_', ' '))} ({ds})"


def detect_annotation_types(
    annotation_df: pd.DataFrame,
    dataset_names: Iterable[str],
    datasets_config: Optional[dict] = None,
) -> List[AnnType]:
    """Find every annotation type present in ``annotation_df`` (the full scope)."""
    cols = set(annotation_df.columns)
    out: List[AnnType] = []

    for key, label, name_col in DERIVED_TYPES:
        if key in cols:
            out.append(AnnType(key, label, key, name_col if name_col in cols else None, ";", "id_name", None))
    if MODELSEED_COL in cols:
        out.append(AnnType(MODELSEED_COL, "ModelSEED pathway", MODELSEED_COL, None, ";", "modelseed", None))

    datasets_config = datasets_config or {}
    for ds in dataset_names:
        prefix = f"{ds}_"
        ds_cols = [c for c in annotation_df.columns if str(c).startswith(prefix)]
        used = set()
        # ID/name pairs: <ds>_<X>_ACC + <ds>_<X>
        for c in ds_cols:
            if not c.endswith("_ACC"):
                continue
            raw = c[len(prefix):-4]
            name_col = f"{prefix}{raw}"
            used.update({c, name_col})
            if raw in SKIP_RAW:
                continue
            if raw in LABEL_FROM_NAME and name_col in cols:
                out.append(AnnType(name_col, _ds_label(ds, raw), name_col, None, ";;", "label", ds))
            elif raw in BRITE:
                out.append(AnnType(name_col, _ds_label(ds, raw), c, name_col if name_col in cols else None,
                                   ";;", "brite", ds))
            else:
                out.append(AnnType(name_col, _ds_label(ds, raw), c, name_col if name_col in cols else None,
                                   ";;", "id_name", ds))
        # EC numbers from the dataset config
        ec_col = ((datasets_config.get(ds) or {}).get("annotation_columns") or {}).get("ec_number_col")
        if ec_col and f"{prefix}{ec_col}" in cols:
            c = f"{prefix}{ec_col}"
            used.add(c)
            out.append(AnnType(c, f"EC number ({ds})", c, None, ";;", "label", ds))
        # Remaining shareable categorical label columns (ClassyFire, NPClassifier, modelseed_rxn...)
        for c in ds_cols:
            if c in used:
                continue
            raw = c[len(prefix):]
            s = annotation_df[c]
            present = s[~s.map(_is_missing)]
            if pd.api.types.is_numeric_dtype(s) or pd.to_numeric(present, errors="coerce").notna().all():
                continue  # numeric metrics (e.g. anvi'o homogeneity indices) are not terms
            if ID_LIKE_RE.search(raw) and raw not in KNOWN_LABELS:
                continue
            sep = ";" if raw == "modelseed_rxn" else ";;"
            if raw not in KNOWN_LABELS and not _shared_terms(s, sep):
                continue
            out.append(AnnType(c, _ds_label(ds, raw), c, None, sep, "label", ds))
    return out


def _split(cell, sep: str) -> List[str]:
    if _is_missing(cell):
        return []
    return [p.strip() for p in str(cell).split(sep)]


def normalize_term_id(ann: AnnType, term_id: str) -> str:
    if ann.key == "kegg_pathway" and term_id.startswith("path:"):
        return term_id[5:]
    return term_id


def explode_annotation_type(annotation_df: pd.DataFrame, ann: AnnType, id_col: str = "feature_id") -> pd.DataFrame:
    """Long table ``feature_id, ann_type, term_id, term_name`` for one annotation type."""
    fids = annotation_df[id_col].astype(str).to_numpy()
    ids = annotation_df[ann.id_col].to_numpy()
    names = annotation_df[ann.name_col].to_numpy() if ann.name_col else [None] * len(ids)
    rows = []
    for fid, cell, ncell in zip(fids, ids, names):
        terms = _split(cell, ann.sep)
        if not terms:
            continue
        nlist = _split(ncell, ann.sep) if ncell is not None else []
        aligned = len(nlist) == len(terms)
        for i, t in enumerate(terms):
            if _is_missing(t):
                continue
            name = nlist[i] if aligned else ""
            if ann.kind == "modelseed":
                if t.startswith("NOPATHWAY_"):
                    continue
                m = MODELSEED_RE.match(t)
                t, name = (m.group(1), m.group(2)) if m else (t, "")
                name = re.sub(r"<[^>]+>", "", name)  # strip HTML tags such as <i>a</i>
            elif ann.kind == "brite":
                name = name.split(">>>")[0].strip() if name else ""
            elif ann.kind == "label":
                name = t
            if _is_missing(name):
                name = ""
            rows.append((fid, ann.key, normalize_term_id(ann, t), name))
    out = pd.DataFrame(rows, columns=["feature_id", "ann_type", "term_id", "term_name"])
    out = out.drop_duplicates(["feature_id", "term_id"])
    # One canonical name per term (first non-empty)
    if not out.empty:
        canon = out[out["term_name"] != ""].groupby("term_id")["term_name"].first()
        out["term_name"] = out["term_id"].map(canon).fillna("")
    return out.reset_index(drop=True)


def explode_all(annotation_df: pd.DataFrame, ann_types: List[AnnType], id_col: str = "feature_id") -> pd.DataFrame:
    frames = [explode_annotation_type(annotation_df, a, id_col=id_col) for a in ann_types]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["feature_id", "ann_type", "term_id", "term_name"])
    return pd.concat(frames, ignore_index=True)


def to_wide_term_columns(long_df: pd.DataFrame, ann: AnnType, feature_ids: Iterable[str],
                         missing_token: str = "Unassigned") -> pd.DataFrame:
    """Per-feature ``<key>`` (';'-joined term IDs) and ``<key>_name`` columns in the
    convention expected by ``helpers.prepare_universe``/``get_term_names``."""
    sub = long_df[long_df["ann_type"] == ann.key]
    ids = sub.groupby("feature_id")["term_id"].agg(";".join)
    names = sub.assign(term_name=sub["term_name"].str.replace(";", ",", regex=False)) \
        .groupby("feature_id")["term_name"].agg(";".join)
    idx = pd.Index(list(feature_ids), name="feature_id")
    out = pd.DataFrame(index=idx)
    out[ann.key] = ids.reindex(idx).fillna(missing_token)
    if ann.kind != "label":
        out[f"{ann.key}_name"] = names.reindex(idx).fillna(missing_token)
    return out


def ann_types_summary(ann_types: List[AnnType]) -> List[Dict]:
    return [a.to_dict() for a in ann_types]
