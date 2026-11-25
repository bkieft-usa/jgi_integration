"""Read-only access to an explorer bundle (parquet files written by tools.explorer_bundle)."""
from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

from tools.explorer_bundle import network as net
from tools.explorer_bundle.stats import PAIR_SEP

SUPPORTED_SCHEMA = 2


def _read(bundle_dir: str, name: str, index: Optional[str] = None) -> pd.DataFrame:
    path = os.path.join(bundle_dir, f"{name}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if index and index in df.columns:
        df = df.set_index(index)
    return df


def contrast_label(contrast: str) -> str:
    return "all levels (F test)" if contrast == "omnibus" else contrast


class Bundle:
    """All bundle tables in memory, except associations, which are queried with DuckDB."""

    def __init__(self, bundle_dir: str):
        self.dir = os.path.abspath(bundle_dir)
        with open(os.path.join(self.dir, "manifest.json")) as fh:
            self.manifest: Dict = json.load(fh)
        if self.manifest.get("schema_version") != SUPPORTED_SCHEMA:
            raise ValueError(f"Bundle schema {self.manifest.get('schema_version')} is not supported by this explorer "
                             f"(expects {SUPPORTED_SCHEMA}); rebuild it with `python -m tools.cli build-explorer --force`.")
        d = self.dir
        self.samples = _read(d, "samples", "sample_id")
        self.sample_meta = _read(d, "sample_metadata", "sample_id")
        self.similarity = _read(d, "sample_similarity", "sample_id")
        self.features = _read(d, "features", "feature_id")
        self.z = _read(d, "abundance_z", "feature_id")
        self.logab = _read(d, "abundance_log", "feature_id")
        self.ann = _read(d, "annotations_long")
        self.terms = _read(d, "terms", "entity_id")
        self.groups = _read(d, "groups", "group")
        self.eig = _read(d, "group_eigengenes", "group")
        self.term_eig = _read(d, "term_eigengenes", "entity_id")
        self.enrich = _read(d, "enrichment")
        self.agree = _read(d, "agreement")
        self.feature_edges = _read(d, "feature_edges")
        self.votes = _read(d, "discovery_votes")
        native_path = os.path.join(d, "native_structure.json")
        self.native = json.load(open(native_path)) if os.path.exists(native_path) else None
        self._assoc_path = os.path.join(d, "associations.parquet")
        self._db = duckdb.connect(database=":memory:")
        self._db.execute(f"CREATE VIEW assoc AS SELECT * FROM read_parquet('{self._assoc_path}')")
        self._lock = threading.Lock()

        self.ann_types: List[Dict] = self.manifest.get("annotation_types", [])
        self.ann_label = {a["key"]: a["label"] for a in self.ann_types}
        self.variables: List[str] = self.manifest.get("variables", [])
        self.aliases: Dict[str, str] = self.manifest.get("variable_aliases", {})
        self.var_types: Dict[str, str] = self.manifest.get("variable_types", {})
        self.nesting: Dict[str, str] = self.manifest.get("nesting", {})
        self.dtypes: List[str] = sorted(self.features["dtype"].dropna().unique().tolist())
        self.fdr = float(self.manifest.get("config", {}).get("fdr", 0.05))
        self._ann_by_feature = self.ann.groupby("feature_id") if not self.ann.empty else None
        self._members = self.features.dropna(subset=["group"]).groupby("group").groups
        self._search = self._build_search_index()
        net_info = self.manifest.get("network", {})
        self.network_kind: str = net_info.get("kind", "knn")
        self.network_nodes = self.features.index[self.features["x"].notna()]
        self.adj = net.adjacency(self.feature_edges)
        e = self.feature_edges
        # every edge joins two different data types (e.g. correlation computed tx-vs-mx only)
        self.bipartite = bool(len(e)) and bool((e["source"].str.split("_").str[0] !=
                                                 e["target"].str.split("_").str[0]).all())
        self._adj_by_feature = self.adj.groupby("feature_id") if len(self.adj) else None
        self.vote_bg = net.vote_background(self.ann, self.network_nodes)

    # ---- identity ------------------------------------------------------------------------
    @property
    def title(self) -> str:
        m = self.manifest
        return f"{m.get('project_name', 'Project')}  ·  Dataset_Processing--{m.get('data_processing_hash')}  ·  " \
               f"Analysis--{m.get('analysis_hash')}"

    def term_label(self, entity_id: str, max_len: int = 70) -> str:
        if entity_id not in self.terms.index:
            return entity_id
        r = self.terms.loc[entity_id]
        name = r["term_name"] if isinstance(r["term_name"], str) and r["term_name"] and r["term_name"] != r["term_id"] \
            else ""
        text = f"{r['term_id']} ({name})" if name else str(r["term_id"])
        return text if len(text) <= max_len else text[: max_len - 1] + "…"

    def feature_label(self, fid: str) -> str:
        if fid in self.features.index:
            name = self.features.at[fid, "display_name"]
            if isinstance(name, str) and name:
                return f"{fid} ({name})"
        return fid

    # ---- search ----------------------------------------------------------------------------
    def _build_search_index(self) -> pd.DataFrame:
        parts = []
        f = self.features
        parts.append(pd.DataFrame({
            "value": "feature:" + f.index, "kind": "feature",
            "label": [f"Feature · {fid}" + (f" · {n}" if isinstance(n, str) and n else "")
                      for fid, n in zip(f.index, f["display_name"])],
        }))
        t = self.terms
        if not t.empty:
            parts.append(pd.DataFrame({
                "value": "term:" + t.index, "kind": "term",
                "label": [f"{self.ann_label.get(a, a)} · {tid}" + (f" · {n}" if isinstance(n, str) and n and n != tid
                                                                  else "") + f"  ({nf} features)"
                          for a, tid, n, nf in zip(t["ann_type"], t["term_id"], t["term_name"], t["n_features"])],
            }))
        g = self.groups
        parts.append(pd.DataFrame({
            "value": "group:" + g.index, "kind": "group",
            "label": [f"Group · {gid} ({s} features)" + (f" · {tt}" if isinstance(tt, str) else "")
                      for gid, s, tt in zip(g.index, g["size"], g.get("top_term", pd.Series(index=g.index)))],
        }))
        idx = pd.concat(parts, ignore_index=True)
        idx["hay"] = idx["label"].str.lower()
        idx["rank"] = idx["kind"].map({"group": 0, "term": 1, "feature": 2})
        return idx

    def search(self, query: str, kinds: Optional[List[str]] = None, limit: int = 60) -> List[Dict]:
        if not query or len(query.strip()) < 2:
            return []
        q = query.strip().lower()
        idx = self._search if not kinds else self._search[self._search["kind"].isin(kinds)]
        mask = None
        for tok in q.split():  # every word must match, in any order
            m = idx["hay"].str.contains(tok, regex=False)
            mask = m if mask is None else mask & m
        hits = idx[mask]
        exact_id = hits["hay"].str.split(" · ").str[1].str.startswith(q)
        phrase = hits["hay"].str.contains(q, regex=False)
        hits = hits.assign(k1=~exact_id, k2=~phrase).sort_values(["k1", "k2", "rank", "label"]).head(limit)
        return [{"label": l, "value": v} for l, v in zip(hits["label"], hits["value"])]

    # ---- members & annotations ----------------------------------------------------------------
    def group_members(self, group: str) -> pd.DataFrame:
        ids = self._members.get(group, [])
        return self.features.loc[list(ids)].rename_axis("feature_id")

    def feature_annotations(self, fid: str) -> pd.DataFrame:
        if self._ann_by_feature is None or fid not in self._ann_by_feature.groups:
            return pd.DataFrame(columns=["ann_type", "term_id", "term_name"])
        return self._ann_by_feature.get_group(fid)[["ann_type", "term_id", "term_name"]]

    def term_members(self, entity_id: str) -> pd.DataFrame:
        at, tid = entity_id.split("::", 1)
        ids = self.ann.loc[(self.ann["ann_type"] == at) & (self.ann["term_id"] == tid), "feature_id"].unique()
        return self.features.reindex(ids).dropna(how="all").rename_axis("feature_id")

    def enrichment_for_group(self, group: str, ann_type: Optional[str] = None) -> pd.DataFrame:
        e = self.enrich[self.enrich["group"] == group]
        return e[e["ann_type"] == ann_type] if ann_type else e

    def enrichment_for_term(self, entity_id: str) -> pd.DataFrame:
        at, tid = entity_id.split("::", 1)
        return self.enrich[(self.enrich["ann_type"] == at) & (self.enrich["term_id"] == tid)]

    # ---- associations (DuckDB) ----------------------------------------------------------------
    def _query(self, sql: str, params: list) -> pd.DataFrame:
        with self._lock:
            return self._db.execute(sql, params).fetchdf()

    def canonical(self, variable: str) -> str:
        """Variables with the same partition as another are tested once, under the first name."""
        return self.aliases.get(variable, variable)

    def variable_options(self) -> List[Dict]:
        return [{"label": f"{v} (same grouping as {self.aliases[v]})" if v in self.aliases else v, "value": v}
                for v in self.variables]

    @lru_cache(maxsize=64)
    def contrasts(self, entity_type: str, variable: str) -> List[str]:
        df = self._query("SELECT DISTINCT contrast FROM assoc WHERE entity_type = ? AND variable = ? ORDER BY contrast",
                         [entity_type, self.canonical(variable)])
        return df["contrast"].tolist()

    @lru_cache(maxsize=64)
    def comparison_levels(self, entity_type: str, variable: str) -> List[str]:
        """Levels that appear in pairwise comparisons, in the variable's natural order."""
        levels = set()
        for c in self.contrasts(entity_type, variable):
            if PAIR_SEP in c:
                levels.update(c.split(PAIR_SEP, 1))
        try:
            return sorted(levels, key=lambda v: float(v))
        except ValueError:
            return sorted(levels)

    def resolve_pair(self, entity_type: str, variable: str, a: str, b: str):
        """``(stored contrast, sign)``: comparisons are stored once per pair; ``sign = -1`` when
        the request is the reverse of the stored direction."""
        cs = set(self.contrasts(entity_type, variable))
        if f"{a}{PAIR_SEP}{b}" in cs:
            return f"{a}{PAIR_SEP}{b}", 1
        if f"{b}{PAIR_SEP}{a}" in cs:
            return f"{b}{PAIR_SEP}{a}", -1
        return None, 1

    def associations_for(self, entity_type: str, entity_id: str) -> pd.DataFrame:
        df = self._query("SELECT * FROM assoc WHERE entity_type = ? AND entity_id = ? ORDER BY p",
                         [entity_type, entity_id])
        df["contrast"] = df["contrast"].map(contrast_label)
        return df

    def association_table(self, entity_type: str, variable: str, contrast: str, sign: int = 1,
                          test: Optional[str] = None, ann_type: Optional[str] = None,
                          dtype: Optional[str] = None) -> pd.DataFrame:
        sql = "SELECT * FROM assoc WHERE entity_type = ? AND variable = ? AND contrast = ?"
        params: list = [entity_type, self.canonical(variable), contrast]
        if test:
            sql += " AND test = ?"
            params.append(test)
        if ann_type:
            sql += " AND starts_with(entity_id, ?)"
            params.append(ann_type + "::")
        if dtype:
            sql += " AND dtype = ?"
            params.append(dtype)
        df = self._query(sql, params)
        if sign == -1 and len(df):
            df[["effect", "statistic"]] = -df[["effect", "statistic"]]
            a, b = contrast.split(PAIR_SEP, 1)
            df["contrast"] = f"{b}{PAIR_SEP}{a}"
        return df

    # ---- network ---------------------------------------------------------------------------
    def neighbors(self, fid: str) -> pd.DataFrame:
        """Direct network neighbours of a feature: ``neighbor, weight``."""
        if self._adj_by_feature is None or fid not in self._adj_by_feature.groups:
            return pd.DataFrame(columns=["neighbor", "weight"])
        return self._adj_by_feature.get_group(fid)[["neighbor", "weight"]]

    def neighbors_of_set(self, ids) -> pd.DataFrame:
        """Edges from any feature in ``ids`` to features outside it: ``feature_id, neighbor, weight``."""
        ids = set(ids)
        a = self.adj
        return a[a["feature_id"].isin(ids) & ~a["neighbor"].isin(ids)]

    # ---- metadata helpers ------------------------------------------------------------------
    def levels(self, variable: str, frame: Optional[pd.DataFrame] = None) -> List:
        s = (frame if frame is not None else self.samples)[variable].dropna()
        try:
            return sorted(s.unique(), key=lambda v: float(v))
        except (TypeError, ValueError):
            return sorted(s.unique(), key=str)

    def correlated_features(self, fid: str, candidates: List[str], top: int = 15) -> pd.Series:
        if fid not in self.z.index:
            return pd.Series(dtype=float)
        cand = [c for c in candidates if c in self.z.index and c != fid]
        if not cand:
            return pd.Series(dtype=float)
        A = self.z.loc[cand].to_numpy(dtype=float)
        b = self.z.loc[fid].to_numpy(dtype=float)
        Ac = A - A.mean(axis=1, keepdims=True)
        bc = b - b.mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            r = Ac @ bc / np.sqrt((Ac ** 2).sum(axis=1) * (bc ** 2).sum())
        return pd.Series(r, index=cand).dropna().sort_values(ascending=False).head(top)
