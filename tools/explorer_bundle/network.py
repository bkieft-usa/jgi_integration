"""The feature network behind the Data-Driven Groupings view and unannotated-feature discovery.

* For network-based grouping methods (e.g. ``network_modules``) the network is the run's own
  correlation network: the edges that defined the modules.
* For other methods (hierarchical clustering, HDBSCAN, ...) there are no edges, so each
  grouped feature is linked to its ``k`` most correlated grouped features (a k-nearest-
  neighbour graph on Pearson correlation of integrated z-scores). This is derived for display
  and discovery; it is not how those groups were formed.

Neighbour voting ("guilt by association", Gillis & Pavlidis 2012): for a feature, count how
many of its annotated network neighbours carry each term and test that count with the same
one-sided hypergeometric test used for group enrichment (background: network features
annotated with that annotation type).
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom

from tools.explorer_bundle.stats import bh_adjust

EDGE_COLUMNS = ["source", "target", "weight"]


# ---- network ------------------------------------------------------------------------------

def knn_edges(z: pd.DataFrame, k: int = 5, block: int = 2000) -> pd.DataFrame:
    """Undirected k-nearest-neighbour edges by Pearson correlation of the rows of ``z``.

    An edge joins two features when either is among the other's ``k`` most correlated
    (positive correlation only). ``weight`` is the correlation.
    """
    ids = z.index.astype(str).to_numpy()
    X = np.nan_to_num(z.to_numpy(dtype=np.float32))
    X = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)
    n = len(ids)
    k = min(k, n - 1)
    if k < 1:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    src, dst, w = [], [], []
    for start in range(0, n, block):
        R = X[start:start + block] @ X.T
        rows = np.arange(R.shape[0])
        R[rows, start + rows] = -np.inf                       # no self-edges
        nn = np.argpartition(-R, k, axis=1)[:, :k]
        r = R[rows[:, None], nn]
        keep = r > 0
        src.append(np.repeat(start + rows, k)[keep.ravel()])
        dst.append(nn.ravel()[keep.ravel()])
        w.append(r.ravel()[keep.ravel()])
    src, dst, w = np.concatenate(src), np.concatenate(dst), np.concatenate(w)
    a, b = np.minimum(src, dst), np.maximum(src, dst)
    df = pd.DataFrame({"a": a, "b": b, "weight": w}).drop_duplicates(["a", "b"])
    return pd.DataFrame({"source": ids[df["a"]], "target": ids[df["b"]], "weight": df["weight"].to_numpy()})


def layout(nodes: Iterable[str], edges: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """2-D positions for every node: a force-directed layout (Fruchterman-Reingold) of each
    connected component, components packed in rows from largest to smallest."""
    import igraph as ig

    nodes = pd.Index(pd.unique(np.asarray(list(nodes), dtype=object)).astype(str))
    if len(nodes) == 0:
        return pd.DataFrame(columns=["x", "y"])
    e = edges[edges["source"].isin(nodes) & edges["target"].isin(nodes)] if len(edges) else edges
    g = ig.Graph(n=len(nodes), edges=list(zip(nodes.get_indexer(e["source"]), nodes.get_indexer(e["target"])))
                 if len(e) else [])
    weights = np.abs(e["weight"].to_numpy(dtype=float)) if len(e) else None
    random.seed(seed)  # igraph draws from Python's RNG: fixed seed -> reproducible layout
    comps = sorted(g.connected_components(), key=len, reverse=True)
    pos = np.zeros((len(nodes), 2))
    placed = []
    for comp in comps:
        if len(comp) == 1:
            xy = np.zeros((1, 2))
        elif len(comp) == 2:
            xy = np.array([[-0.5, 0.0], [0.5, 0.0]])
        else:
            sub = g.induced_subgraph(comp)
            ew = None
            if weights is not None and sub.ecount():
                eid = g.get_eids([(comp[a], comp[b]) for a, b in sub.get_edgelist()])
                ew = weights[eid]
            xy = np.asarray(sub.layout_fruchterman_reingold(weights=ew, niter=500, grid=len(comp) > 1000).coords)
            xy = xy - np.median(xy, axis=0)
            r = np.linalg.norm(xy, axis=1)
            # robust scale: a few far-out nodes must not shrink a dense component to a dot
            span = float(np.quantile(r, 0.9)) or float(r.max()) or 1.0
            xy = xy / span * np.sqrt(len(comp))          # area proportional to component size
            xy = np.clip(xy, -3 * np.sqrt(len(comp)), 3 * np.sqrt(len(comp)))
            xy = _spread(xy, seed=seed)
        placed.append((comp, xy))
    # shelf packing: fill rows left to right, row width ~ sqrt(total area)
    total = sum(max(np.ptp(xy[:, 0]), 1) * max(np.ptp(xy[:, 1]), 1) for _, xy in placed)
    row_width = max(np.sqrt(total) * 1.6, 4.0)
    x_cursor, y_cursor, row_h, gap = 0.0, 0.0, 0.0, 1.5
    for comp, xy in placed:
        w = max(np.ptp(xy[:, 0]), 1.0)
        h = max(np.ptp(xy[:, 1]), 1.0)
        if x_cursor > 0 and x_cursor + w > row_width:
            x_cursor, y_cursor, row_h = 0.0, y_cursor - row_h - gap, 0.0
        pos[comp] = xy - xy.min(axis=0) + np.array([x_cursor, y_cursor - h])
        x_cursor += w + gap
        row_h = max(row_h, h)
    return pd.DataFrame(pos, index=nodes, columns=["x", "y"])


def _spread(xy: np.ndarray, min_dist: float = 0.9, iterations: int = 150, seed: int = 0) -> np.ndarray:
    """Push apart nodes closer than ``min_dist``. Features with identical neighbours (e.g. many
    genes linked to the same few metabolites) get identical force-directed positions; this turns
    such stacks into visible clouds without changing the overall picture."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    xy = xy + rng.normal(scale=1e-3, size=xy.shape)
    for _ in range(iterations):
        pairs = cKDTree(xy).query_pairs(min_dist, output_type="ndarray")
        if len(pairs) == 0:
            break
        d = xy[pairs[:, 0]] - xy[pairs[:, 1]]
        dist = np.linalg.norm(d, axis=1, keepdims=True)
        push = d / np.maximum(dist, 1e-9) * (min_dist - dist) / 2
        delta = np.zeros_like(xy)
        np.add.at(delta, pairs[:, 0], push)
        np.add.at(delta, pairs[:, 1], -push)
        xy = xy + np.clip(delta, -min_dist, min_dist)
    return xy


def adjacency(edges: pd.DataFrame) -> pd.DataFrame:
    """Both directions of every edge: ``feature_id, neighbor, weight``."""
    if edges is None or edges.empty:
        return pd.DataFrame(columns=["feature_id", "neighbor", "weight"])
    fwd = edges.rename(columns={"source": "feature_id", "target": "neighbor"})
    rev = edges.rename(columns={"target": "feature_id", "source": "neighbor"})
    return pd.concat([fwd, rev], ignore_index=True)[["feature_id", "neighbor", "weight"]]


# ---- neighbour voting ---------------------------------------------------------------------

def vote_background(long_ann: pd.DataFrame, network_nodes: Iterable[str]) -> Dict[str, dict]:
    """Per annotation type: ``N`` = network features annotated with the type, ``K`` = per-term
    counts among them, and ``members`` = the annotated feature set."""
    nodes = set(map(str, network_nodes))
    sub = long_ann[long_ann["feature_id"].isin(nodes)]
    out = {}
    for at, grp in sub.groupby("ann_type"):
        members = set(grp["feature_id"])
        out[at] = {"N": len(members), "K": grp.groupby("term_id")["feature_id"].nunique(), "members": members}
    return out


def votes_from_neighbors(neighbors: Iterable[str], long_ann: pd.DataFrame, background: Dict[str, dict],
                         ann_types: Optional[List[str]] = None) -> pd.DataFrame:
    """Neighbour-voting test for one feature, given its neighbours.

    For each annotation type, ``n`` = neighbours annotated with that type; for each term among
    them, ``k`` = neighbours carrying it. ``p`` = P(X >= k), X ~ Hypergeometric(N, K, n).
    Returns ``ann_type, term_id, term_name, k, n, K, N, share, p`` (no FDR: one feature).
    """
    nb = set(map(str, neighbors))
    sub = long_ann[long_ann["feature_id"].isin(nb)]
    if ann_types is not None:
        sub = sub[sub["ann_type"].isin(ann_types)]
    rows = []
    for at, grp in sub.groupby("ann_type"):
        bg = background.get(at)
        if not bg:
            continue
        n = len(nb & bg["members"])
        k = grp.groupby("term_id").agg(k=("feature_id", "nunique"), term_name=("term_name", "first")).reset_index()
        k["ann_type"], k["n"], k["N"] = at, n, bg["N"]
        k["K"] = k["term_id"].map(bg["K"]).fillna(k["k"]).astype(int)
        rows.append(k)
    cols = ["ann_type", "term_id", "term_name", "k", "n", "K", "N", "share", "p"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.concat(rows, ignore_index=True)
    df["share"] = df["k"] / df["n"].clip(lower=1)
    df["p"] = hypergeom.sf(df["k"] - 1, df["N"], df["K"], df["n"])
    return df[cols].sort_values("p").reset_index(drop=True)


def neighbor_votes(edges: pd.DataFrame, network_nodes: pd.Index, long_ann: pd.DataFrame, ann_types,
                   feature_dtype: pd.Series, max_terms_per_feature: int = 20) -> pd.DataFrame:
    """Neighbour-voting test for every network feature that lacks a given annotation type.

    ``ann_types``: objects with ``key`` and ``dtype_scope`` (a type limited to one data type
    is only predicted for features of that data type). BH is applied within each annotation
    type over all (feature, term) pairs tested (terms carried by at least one neighbour).
    Only the ``max_terms_per_feature`` best terms per feature and type are kept.
    Returns ``feature_id, ann_type, term_id, k, n, K, N, share, p, padj``.
    """
    nodes = pd.Index(network_nodes.astype(str))
    cols = ["feature_id", "ann_type", "term_id", "k", "n", "K", "N", "share", "p", "padj"]
    if len(edges) == 0 or len(nodes) == 0:
        return pd.DataFrame(columns=cols)
    e = edges[edges["source"].isin(nodes) & edges["target"].isin(nodes)]
    i, j = nodes.get_indexer(e["source"]), nodes.get_indexer(e["target"])
    adj = sp.csr_matrix((np.ones(2 * len(e)), (np.concatenate([i, j]), np.concatenate([j, i]))),
                        shape=(len(nodes), len(nodes)))
    adj.data[:] = 1.0
    dtype = feature_dtype.reindex(nodes).to_numpy()
    frames = []
    for a in ann_types:
        sub = long_ann[(long_ann["ann_type"] == a.key) & long_ann["feature_id"].isin(nodes)]
        if sub.empty:
            continue
        terms = pd.Index(sub["term_id"].unique())
        M = sp.csr_matrix((np.ones(len(sub)), (nodes.get_indexer(sub["feature_id"]), terms.get_indexer(sub["term_id"]))),
                          shape=(len(nodes), len(terms)))
        M.data[:] = 1.0
        annotated = np.asarray(M.sum(axis=1)).ravel() > 0
        N = int(annotated.sum())
        K = np.asarray(M.sum(axis=0)).ravel()
        cand = ~annotated & ((dtype == a.dtype_scope) if a.dtype_scope else np.ones(len(nodes), bool))
        if not cand.any() or N < 2:
            continue
        ci = np.where(cand)[0]
        C = (adj[ci] @ M).tocoo()                        # k per (candidate, term)
        if C.nnz == 0:
            continue
        n = np.asarray(adj[ci] @ annotated.astype(float)).ravel()
        df = pd.DataFrame({"row": C.row, "t": C.col, "k": C.data.astype(int)})
        df["n"] = n[df["row"]].astype(int)
        df["K"] = K[df["t"]].astype(int)
        df["N"] = N
        df["p"] = hypergeom.sf(df["k"] - 1, N, df["K"], df["n"])
        df["padj"] = bh_adjust(df["p"].to_numpy())
        df["share"] = df["k"] / df["n"].clip(lower=1)
        df["feature_id"] = nodes[ci[df["row"]]]
        df["term_id"] = terms[df["t"]]
        df["ann_type"] = a.key
        df = df.sort_values("p").groupby("feature_id", sort=False).head(max_terms_per_feature)
        frames.append(df[cols])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
