"""How the groups were formed, for methods that are not networks.

* ``hierarchical_dendrogram`` - the clustering's own linkage, truncated at the group
  level (faithful). Falls back to a dendrogram of group centroids (approximation,
  flagged) when the run predates saving the linkage.
* ``hdbscan_tree`` - the HDBSCAN condensed cluster tree (faithful); recomputed and
  verified against the saved labels when the run predates saving it.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree

from tools.metadata import is_noise_label

log = logging.getLogger(__name__)


def hierarchical_dendrogram(structure_dir: Optional[str], groups: pd.Series,
                            centroids: pd.DataFrame) -> Optional[Dict]:
    """Group-level dendrogram (linkage among groups) for plotting.

    The faithful version truncates the saved feature linkage at the group level: every
    group is a subtree, so the merges above them form a linkage among groups.
    """
    path = os.path.join(structure_dir or "", "hierarchical_linkage.csv")
    if structure_dir and os.path.exists(path):
        Z = pd.read_csv(path).to_numpy(dtype=float)
        leaf_ids = pd.read_csv(os.path.join(structure_dir, "hierarchical_linkage_leaf_ids.csv"))["feature_id"].astype(str)
        meta = json.load(open(os.path.join(structure_dir, "hierarchical_linkage.json")))
        n = len(leaf_ids)
        g = groups.reindex(leaf_ids).astype(str).to_numpy()
        k = len(pd.unique(g))
        # Group subtree roots: nodes whose leaves all share one group, with a parent that does not.
        _, nodes = to_tree(Z, rd=True)
        node_group = {}
        for i in range(n):
            node_group[i] = g[i]
        for j, (a, b) in enumerate(Z[:, :2].astype(int)):
            ga, gb = node_group.get(a), node_group.get(b)
            node_group[n + j] = ga if (ga is not None and ga == gb) else None
        # Truncated dendrogram with exactly one leaf per group
        d = dendrogram(Z, truncate_mode="lastp", p=k, no_plot=True)
        leaf_labels = []
        for leaf in d["leaves"]:
            if node_group.get(leaf) is not None:
                leaf_labels.append(node_group[leaf])
            else:  # tie at the cut height: label with the majority group
                members = nodes[leaf].pre_order()
                leaf_labels.append(pd.Series(g[members]).mode().iloc[0])
        return {"kind": "dendrogram", "approximate": False,
                "note": f"Clustering linkage ({meta.get('linkage_method')} linkage, {meta.get('distance_metric')} "
                        f"distance) truncated at the group level; cut height {meta.get('height_cutoff')}.",
                "cut_height": meta.get("height_cutoff"),
                "icoord": d["icoord"], "dcoord": d["dcoord"], "leaf_groups": leaf_labels}
    # Approximation: linkage of group centroid profiles
    real = centroids.loc[[c for c in centroids.index if not is_noise_label(c)]]
    if len(real) < 3:
        return None
    Zc = linkage(np.nan_to_num(real.to_numpy(dtype=float)), method="average", metric="correlation")
    d = dendrogram(Zc, no_plot=True, labels=real.index.tolist())
    return {"kind": "dendrogram", "approximate": True,
            "note": "Approximation: average-linkage dendrogram of group centroid profiles (correlation distance). "
                    "The run did not save the clustering's own linkage; re-run grouping to get the exact tree.",
            "cut_height": None, "icoord": d["icoord"], "dcoord": d["dcoord"], "leaf_groups": d["ivl"]}


def _condensed_to_clusters(tree: pd.DataFrame, n_points: int, selected, groups_by_label: Dict[int, str]) -> pd.DataFrame:
    clusters = tree[tree["child_size"] > 1].copy()
    root = n_points
    rows = [{"node": root, "parent": "", "lambda_birth": 0.0, "size": n_points}]
    for r in clusters.itertuples():
        rows.append({"node": int(r.child), "parent": int(r.parent), "lambda_birth": float(r.lambda_val),
                     "size": int(r.child_size)})
    df = pd.DataFrame(rows)
    sel = {int(c): groups_by_label.get(i, f"group_{i + 1}") for i, c in enumerate(selected)}
    df["selected"] = df["node"].isin(sel)
    df["group"] = df["node"].map(sel).fillna("")
    # stability = sum over points/sub-clusters leaving the cluster of (lambda_leave - lambda_birth) * size
    births = df.set_index("node")["lambda_birth"]
    leave = tree.assign(b=tree["parent"].map(births))
    leave["contrib"] = (leave["lambda_val"] - leave["b"]).clip(lower=0) * leave["child_size"]
    df["stability"] = df["node"].map(leave.groupby("parent")["contrib"].sum()).fillna(0.0)
    return df


def hdbscan_tree(structure_dir: Optional[str], groups: pd.Series, z_selected: pd.DataFrame,
                 params: Optional[dict]) -> Optional[Dict]:
    """HDBSCAN condensed cluster tree (clusters only) with selected clusters marked."""
    path = os.path.join(structure_dir or "", "hdbscan_condensed_tree.csv")
    if structure_dir and os.path.exists(path):
        tree = pd.read_csv(path)
        leaf_ids = pd.read_csv(os.path.join(structure_dir, "hdbscan_condensed_tree_leaf_ids.csv"))["feature_id"]
        meta = json.load(open(os.path.join(structure_dir, "hdbscan_condensed_tree.json")))
        selected = meta.get("selected_clusters", [])
        df = _condensed_to_clusters(tree, len(leaf_ids), selected, {})
        return {"kind": "hdbscan_tree", "approximate": False, "note": "HDBSCAN condensed cluster tree.",
                "clusters": df.to_dict(orient="list")}
    if not params:
        return None
    try:
        import hdbscan
        from scipy.spatial.distance import pdist, squareform
    except ImportError:
        return None
    # Same rows and order as the pipeline's fit (HDBSCAN tie-breaking depends on row order)
    X = z_selected[z_selected.index.isin(groups.index)]
    X = X.loc[~X.isna().all(axis=1)].fillna(0)
    if len(X) > 15000:
        log.info("Skipping HDBSCAN tree recomputation (more than 15000 features).")
        return None
    metric = params.get("metric", "euclidean")
    data, hmetric = (squareform(pdist(X.values, metric="correlation")), "precomputed") if metric == "correlation" \
        else (X.values, metric)
    cl = hdbscan.HDBSCAN(metric=hmetric, min_cluster_size=params.get("min_cluster_size", 5),
                         min_samples=params.get("min_samples", 3))
    labels = cl.fit_predict(data)
    saved = groups.reindex(X.index).astype(str).to_numpy()
    from sklearn.metrics import adjusted_rand_score
    if adjusted_rand_score(saved, labels) < 0.999:
        log.warning("Recomputed HDBSCAN partition differs from the saved groups; condensed tree omitted.")
        return None
    # recomputed label -> saved group name (identical partitions, possibly renumbered)
    names = pd.crosstab(labels, saved).idxmax(axis=1).to_dict()
    df = _condensed_to_clusters(cl.condensed_tree_.to_pandas(), len(X), cl.condensed_tree_._select_clusters(),
                                {i: names.get(i, f"group_{i + 1}") for i in range(labels.max() + 1)})
    return {"kind": "hdbscan_tree", "approximate": False,
            "note": "HDBSCAN condensed cluster tree (recomputed from the saved data and parameters; labels verified "
                    "identical to the run).", "clusters": df.to_dict(orient="list")}


def sample_similarity(z: pd.DataFrame) -> pd.DataFrame:
    """Sample x sample Pearson correlation, rows/columns in average-linkage leaf order."""
    X = np.nan_to_num(z.to_numpy(dtype=float)).T
    R = np.corrcoef(X)
    if len(R) > 2:
        order = dendrogram(linkage(X, method="average", metric="correlation"), no_plot=True)["leaves"]
    else:
        order = list(range(len(R)))
    cols = z.columns.to_numpy()[order]
    return pd.DataFrame(R[np.ix_(order, order)], index=cols, columns=cols)
