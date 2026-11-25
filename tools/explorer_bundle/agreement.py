"""Knowledge-driven vs data-driven groups: enrichment and agreement statistics.

* ``enrichment_table`` - one-sided hypergeometric test of every annotation term in every
  data-driven group (BH within each annotation type). Background = grouped features with
  at least one annotation of that type; noise/grey features stay in the background but
  their pseudo-group is not tested. Same semantics as ``helpers.compute_enrichment_matrix``
  with ``background='annotated'``.
* ``agreement`` - how well groups recover annotation terms overall (AMI/ARI, term
  co-membership against a size-preserving label-permutation null), per term
  (concentration with permutation p) and per group (purity, entropy).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

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


def _gini_sparse(nonzero: np.ndarray, n_total: int) -> float:
    """Gini coefficient of a count vector given its nonzero entries and full length."""
    x = np.sort(np.asarray(nonzero, dtype=float))
    if n_total < 2 or x.sum() == 0:
        return 0.0
    ranks = np.arange(n_total - len(x) + 1, n_total + 1)  # zeros occupy the lowest ranks
    return float((2 * ranks - n_total - 1) @ x / (n_total * x.sum()))


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0 or len(x) < 2:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def agreement(long_ann: pd.DataFrame, groups: pd.Series, ann_types: List[str], enrichment: pd.DataFrame,
              n_permutations: int = 1000, min_term_size: int = 2, alpha: float = 0.05,
              exclude_global_maps: bool = True, seed: int = 0) -> Dict[str, pd.DataFrame]:
    """Agreement between data-driven groups and each annotation type.

    Only non-noise grouped features with at least one (size-filtered) term are used.
    The null distribution permutes group labels among those features, preserving group
    sizes and each feature's annotations.
    """
    rng = np.random.default_rng(seed)
    real = groups[~groups.map(is_noise_label)]
    type_rows, term_frames, group_frames, null_rows = [], [], [], []
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

        def summarize(Cs: sp.csr_matrix):
            """(term co-membership score, per-term concentration) from sparse term x group counts."""
            same = Cs.copy()
            same.data = same.data * (same.data - 1) / 2
            same_t = np.asarray(same.sum(axis=1)).ravel()
            score = float(np.mean(same_t[ok] / pair_total[ok])) if ok.any() else np.nan
            conc = Cs.max(axis=1).toarray().ravel() / m
            return score, conc

        Cs = counts(gcodes)
        obs, conc_obs = summarize(Cs)
        null = np.empty(n_permutations)
        exceed = np.zeros(len(terms))
        for i in range(n_permutations):
            null[i], conc_p = summarize(counts(rng.permutation(gcodes)))
            exceed += conc_p >= conc_obs - 1e-12
        sd = null.std(ddof=1) if n_permutations > 1 else np.nan
        # primary (most specific) term per feature for partition comparison
        prim = pairs.assign(size=pairs["term_id"].map(pd.Series(m, index=terms))) \
            .sort_values(["size", "term_id"]).drop_duplicates("feature_id").set_index("feature_id")["term_id"]
        prim = prim.reindex(feats)
        type_rows.append({
            "ann_type": at, "n_features": len(feats), "n_terms": len(terms), "n_groups": len(gnames),
            "ami": adjusted_mutual_info_score(prim, glabels), "ari": adjusted_rand_score(prim, glabels),
            "comembership": obs, "null_mean": float(null.mean()), "null_sd": float(sd),
            "z": float((obs - null.mean()) / sd) if sd and sd > 0 else np.nan,
            "p": float((1 + (null >= obs).sum()) / (1 + n_permutations)),
        })
        null_rows.append(pd.DataFrame({"ann_type": at, "null_comembership": null.astype("float32")}))

        perm_p = (1 + exceed) / (1 + n_permutations)
        multi = m >= 2
        n_g = len(gnames)
        tf = pd.DataFrame({
            "ann_type": at, "term_id": terms, "m": m.astype(int),
            "concentration": conc_obs, "top_group": gnames[np.asarray(Cs.argmax(axis=1)).ravel()],
            "n_groups": Cs.getnnz(axis=1),
            "gini": [_gini_sparse(Cs.data[Cs.indptr[i]:Cs.indptr[i + 1]], n_g) for i in range(len(terms))],
            "perm_p": np.where(multi, perm_p, np.nan),
        })
        tf["perm_padj"] = bh_adjust(tf["perm_p"].to_numpy())
        term_frames.append(tf)

        Cc = Cs.tocsc()
        n_annot = np.asarray(Cc.sum(axis=0)).ravel()  # term-memberships per group (multi-label)
        members = pd.Series(glabels).value_counts().reindex(gnames).to_numpy()
        ent = np.zeros(n_g)
        for j in range(n_g):
            x = Cc.data[Cc.indptr[j]:Cc.indptr[j + 1]]
            if x.sum() > 0:
                q = x / x.sum()
                ent[j] = float(-(q * np.log2(q)).sum())
        purity = Cc.max(axis=0).toarray().ravel() / members
        top_term_idx = np.asarray(Cc.argmax(axis=0)).ravel()
        sig = enrichment[(enrichment["ann_type"] == at) & (enrichment["padj"] < alpha)].groupby("group").size()
        group_frames.append(pd.DataFrame({
            "group": gnames, "ann_type": at, "n_annotated": members,
            "purity": purity, "top_term": terms[top_term_idx], "entropy_bits": ent,
            "n_significant_terms": sig.reindex(gnames).fillna(0).astype(int).to_numpy(),
        }))
    empty = pd.DataFrame()
    return {
        "types": pd.DataFrame(type_rows),
        "terms": pd.concat(term_frames, ignore_index=True) if term_frames else empty,
        "groups": pd.concat(group_frames, ignore_index=True) if group_frames else empty,
        "null": pd.concat(null_rows, ignore_index=True) if null_rows else empty,
    }
