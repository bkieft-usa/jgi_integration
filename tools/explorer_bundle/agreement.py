"""Knowledge-driven vs data-driven groups: enrichment and agreement statistics.

* ``enrichment_table`` - one-sided hypergeometric test of every annotation term in every
  data-driven group (BH within each annotation type). Background = grouped features with
  at least one annotation of that type; noise/grey features stay in the background but
  their pseudo-group is not tested. Same semantics as ``helpers.compute_enrichment_matrix``
  with ``background='annotated'``.
* ``agreement`` - how well groups recover the terms of one annotation type overall: the
  share of co-annotated feature pairs that fall in the same group, compared with the same
  share after randomly shuffling group labels (sizes kept). Per term, descriptive only:
  the largest share of the term in one group, and how many groups it spans.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom

from tools.explorer_bundle.stats import bh_adjust
from tools.metadata import is_noise_label

KEGG_GLOBAL_MAP_RE = re.compile(r"^(?:path:)?map01[12]\d\d$")


def _filtered_pairs(long_ann: pd.DataFrame, ann_type: str, groups: pd.Series, min_term_size: int,
                    exclude_global_maps: bool) -> pd.DataFrame:
    sub = long_ann.loc[long_ann["ann_type"] == ann_type, ["feature_id", "term_id"]]
    sub = sub[sub["feature_id"].isin(groups.index)]
    if exclude_global_maps and ann_type == "kegg_pathway":
        sub = sub[~sub["term_id"].str.match(KEGG_GLOBAL_MAP_RE)]
    size = sub.groupby("term_id")["feature_id"].nunique()
    sub = sub[sub["term_id"].isin(size[size >= min_term_size].index)]
    return sub.assign(group=groups.reindex(sub["feature_id"]).to_numpy())


def enrichment_table(long_ann: pd.DataFrame, groups: pd.Series, ann_types: List[str], min_term_size: int = 2,
                     min_group_size: int = 2, exclude_global_maps: bool = True) -> pd.DataFrame:
    """Long table ``ann_type, term_id, group, k, K, n, N, fold, p, padj``."""
    group_size = groups.value_counts()
    testable_groups = set(g for g, s in group_size.items() if s >= min_group_size and not is_noise_label(g))
    frames = []
    for at in ann_types:
        pairs = _filtered_pairs(long_ann, at, groups, min_term_size, exclude_global_maps)
        if pairs.empty:
            continue
        universe = pairs["feature_id"].unique()
        N = len(universe)
        n_by_group = pd.Series(groups.reindex(universe).to_numpy()).value_counts()
        K_by_term = pairs.groupby("term_id")["feature_id"].nunique()
        tested = pairs[pairs["group"].isin(testable_groups)]
        if tested.empty:
            continue
        k = tested.groupby(["term_id", "group"])["feature_id"].nunique().rename("k").reset_index()
        k["K"] = k["term_id"].map(K_by_term).to_numpy()
        k["n"] = k["group"].map(n_by_group).to_numpy()
        k["N"] = N
        k["p"] = hypergeom.sf(k["k"] - 1, N, k["K"], k["n"])
        k["fold"] = k["k"] / (k["K"] * k["n"] / N)
        k["padj"] = bh_adjust(k["p"].to_numpy())
        k.insert(0, "ann_type", at)
        frames.append(k)
    cols = ["ann_type", "term_id", "group", "k", "K", "n", "N", "fold", "p", "padj"]
    return pd.concat(frames, ignore_index=True)[cols] if frames else pd.DataFrame(columns=cols)


def agreement(long_ann: pd.DataFrame, groups: pd.Series, ann_types: List[str],
              n_permutations: int = 1000, min_term_size: int = 2,
              exclude_global_maps: bool = True, seed: int = 0) -> Dict[str, pd.DataFrame]:
    """Agreement between data-driven groups and each annotation type.

    Only non-noise grouped features with at least one (size-filtered) term are used. The
    null distribution permutes group labels among those features, preserving group sizes
    and each feature's annotations. Returns ``{"types": ..., "terms": ...}``.
    """
    rng = np.random.default_rng(seed)
    real = groups[~groups.map(is_noise_label)]
    type_rows, term_frames = [], []
    for at in ann_types:
        pairs = _filtered_pairs(long_ann, at, real, min_term_size, exclude_global_maps)
        if pairs.empty:
            continue
        feats = pd.Index(pairs["feature_id"].unique())
        terms = pd.Index(pairs["term_id"].unique())
        if len(feats) < 4 or len(terms) < 2:
            continue
        glabels = real.reindex(feats).to_numpy()
        gcodes, gnames = pd.factorize(glabels)
        fi = feats.get_indexer(pairs["feature_id"])
        ti = terms.get_indexer(pairs["term_id"])
        A = sp.csr_matrix((np.ones(len(pairs)), (ti, fi)), shape=(len(terms), len(feats)))
        A.data[:] = 1.0
        m = np.asarray(A.sum(axis=1)).ravel()
        ok = m >= 2
        pair_total = m * (m - 1) / 2

        def counts(codes) -> sp.csr_matrix:
            G = sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(gnames)))
            return (A @ G).tocsr()

        def same_group_share(Cs: sp.csr_matrix) -> float:
            """Mean over terms of the share of the term's feature pairs that share a group."""
            same = Cs.copy()
            same.data = same.data * (same.data - 1) / 2
            same_t = np.asarray(same.sum(axis=1)).ravel()
            return float(np.mean(same_t[ok] / pair_total[ok])) if ok.any() else np.nan

        Cs = counts(gcodes)
        obs = same_group_share(Cs)
        null = np.array([same_group_share(counts(rng.permutation(gcodes))) for _ in range(n_permutations)])
        null_mean = float(null.mean()) if n_permutations else np.nan
        type_rows.append({
            "ann_type": at, "n_features": len(feats), "n_terms": len(terms), "n_groups": len(gnames),
            "observed": obs, "expected": null_mean,
            "null_sd": float(null.std(ddof=1)) if n_permutations > 1 else np.nan,
            "ratio": obs / null_mean if null_mean and null_mean > 0 else np.nan,
            "p": float((1 + (null >= obs).sum()) / (1 + n_permutations)),  # Phipson & Smyth 2010
        })
        term_frames.append(pd.DataFrame({
            "ann_type": at, "term_id": terms, "m": m.astype(int),
            "concentration": Cs.max(axis=1).toarray().ravel() / m,
            "top_group": gnames[np.asarray(Cs.argmax(axis=1)).ravel()],
            "n_groups": Cs.getnnz(axis=1),
        }))
    return {
        "types": pd.DataFrame(type_rows),
        "terms": pd.concat(term_frames, ignore_index=True) if term_frames else pd.DataFrame(),
    }
