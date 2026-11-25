"""Shared helpers for sample metadata: which columns are study variables, their types,
and nesting (pseudo-replication) structure.

Used by the pipeline (plots) and by the explorer bundle (statistics), so both agree on
what "all available metadata variables" means.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ID_COLUMNS = {"unique_group", "sample_id"}
CONTINUOUS_MIN_LEVELS = 6


def _is_dataset_column(col: str, dataset_names: Iterable[str]) -> bool:
    return any(col == ds or col.startswith(f"{ds}_") for ds in dataset_names)


def infer_metadata_variables(
    metadata: pd.DataFrame,
    dataset_names: Iterable[str],
    requested: Optional[List[str]] = None,
) -> List[str]:
    """Return the study variables to use.

    ``requested`` (e.g. ``user_settings.variable_list``) wins when non-empty. Otherwise
    every metadata column is used except ID columns: the per-dataset raw sample-name
    columns (``tx``, ``mx_hilic_pos``...), ``unique_group``, unnamed index columns,
    constant columns, and columns whose values are unique per row.
    """
    if requested:
        missing = [v for v in requested if v not in metadata.columns]
        if missing:
            raise ValueError(f"Requested metadata variables not found in metadata: {missing}")
        return list(requested)
    dataset_names = list(dataset_names)
    out = []
    n = len(metadata)
    for col in metadata.columns:
        col = str(col)
        if col in ID_COLUMNS or col.startswith("Unnamed") or _is_dataset_column(col, dataset_names):
            continue
        nun = metadata[col].nunique(dropna=True)
        if nun < 2:
            continue
        if n > 2 and nun == n and not pd.api.types.is_numeric_dtype(metadata[col]):
            continue  # per-sample identifier
        out.append(col)
    return out


def infer_variable_types(
    metadata: pd.DataFrame,
    variables: Iterable[str],
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """``{variable: 'categorical' | 'continuous'}``.

    Numeric columns with at least ``CONTINUOUS_MIN_LEVELS`` distinct values are
    continuous; everything else is categorical. ``overrides`` takes precedence.
    """
    overrides = overrides or {}
    types = {}
    for v in variables:
        if v in overrides:
            if overrides[v] not in ("categorical", "continuous"):
                raise ValueError(f"Variable type override for '{v}' must be categorical|continuous")
            types[v] = overrides[v]
            continue
        s = metadata[v]
        numeric = pd.to_numeric(s, errors="coerce")
        is_num = numeric.notna().sum() == s.notna().sum() and s.notna().any()
        types[v] = "continuous" if is_num and numeric.nunique() >= CONTINUOUS_MIN_LEVELS else "categorical"
    return types


def identical_variables(metadata: pd.DataFrame, variables: Iterable[str]) -> Dict[str, str]:
    """Variables that split the samples into exactly the same groups as an earlier variable
    (e.g. ``group`` = ``organism`` + ``lineage`` label when lineage is nested in organism).

    Testing both would duplicate every result, so each duplicate maps to the first variable
    with that partition: ``{duplicate: kept}``.
    """
    aliases: Dict[str, str] = {}
    kept: list = []
    for v in variables:
        s = metadata[v]
        codes = pd.factorize(s.astype(str).where(s.notna(), None))[0]
        match = None
        for k in kept:
            t = metadata[k]
            kcodes = pd.factorize(t.astype(str).where(t.notna(), None))[0]
            both = pd.DataFrame({"a": codes, "b": kcodes})
            if (both.groupby("a")["b"].nunique() == 1).all() and (both.groupby("b")["a"].nunique() == 1).all():
                match = k
                break
        if match:
            aliases[v] = match
        else:
            kept.append(v)
    return aliases


def detect_nesting(metadata: pd.DataFrame, variables: Iterable[str]) -> Dict[str, Optional[str]]:
    """For each variable, the experimental unit it should be tested at.

    Variable ``A`` is *nested over* variable ``B`` when every level of ``B`` maps to a
    single level of ``A`` and ``B`` has more levels than ``A`` (e.g. ``organism`` within
    ``lineage``). Replicate samples within one ``B`` level are then not independent
    observations of ``A``; ``A`` must be tested on ``B``-level means (n = number of ``B``
    levels). Among candidate ``B`` variables, the one with the most levels that still
    has replication (fewer levels than samples) is chosen.

    Returns ``{A: B or None}``.
    """
    variables = list(variables)
    n = len(metadata)
    out: Dict[str, Optional[str]] = {}
    for a in variables:
        best, best_levels = None, 0
        a_levels = metadata[a].nunique(dropna=True)
        for b in variables:
            if b == a:
                continue
            sub = metadata[[a, b]].dropna()
            b_levels = sub[b].nunique()
            if not (a_levels < b_levels < n):
                continue
            if (sub.groupby(b)[a].nunique() <= 1).all() and b_levels > best_levels:
                best, best_levels = b, b_levels
        out[a] = best
    return out


def is_noise_label(label) -> bool:
    """Grouping labels that mean 'not assigned to a real group'."""
    return str(label).lower() in {"noise", "grey", "gray", "-1", "unassigned", "nan", "none"}


def safe_numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def align_metadata(metadata: pd.DataFrame, columns: Iterable[str], pairing: str) -> pd.DataFrame:
    """Align the sample-level link table to the integrated matrix columns.

    ``paired`` runs have one column per sample (``unique_group``); ``unpaired`` runs have
    one column per condition (``group``), so metadata is collapsed to conditions and
    columns that vary within a condition become NaN. Returns a frame indexed by the
    matrix column names (in sorted order) with a ``unique_group`` column.
    """
    metadata = metadata.reset_index(drop=True).copy()
    metadata = metadata.drop(columns=[c for c in metadata.columns if str(c).startswith("Unnamed:")])
    key = "unique_group" if pairing == "paired" else "group"
    if key not in metadata.columns:
        raise ValueError(f"Integrated metadata is missing required column '{key}'.")
    keys = set(map(str, columns)) if columns is not None else set(metadata[key].dropna().astype(str))
    metadata[key] = metadata[key].astype(str)
    rows = []
    for k, subset in metadata[metadata[key].isin(keys)].groupby(key, sort=True):
        row = {}
        for column in metadata.columns:
            values = subset[column].dropna().unique()
            row[column] = values[0] if len(values) == 1 else np.nan
        row[key] = k
        rows.append(row)
    aligned = pd.DataFrame(rows)
    if rows:
        aligned = aligned.set_index(key, drop=False)
    else:
        aligned.index = pd.Index([])
    if key == "group":
        aligned["unique_group"] = aligned["group"]
    aligned.index.name = None
    return aligned
