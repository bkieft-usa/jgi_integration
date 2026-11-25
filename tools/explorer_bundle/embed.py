"""Low-dimensional embeddings for samples and features."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)


def sample_embedding(z: pd.DataFrame, seed: int = 0) -> tuple:
    """PCA (PC1-5) and UMAP of samples (columns of ``z``). Returns ``(coords, explained)``."""
    X = np.nan_to_num(z.to_numpy(dtype=float)).T
    n = X.shape[0]
    k = max(1, min(5, n - 1))
    pca = PCA(n_components=k, random_state=seed)
    pcs = pca.fit_transform(X)
    coords = pd.DataFrame(pcs, index=z.columns, columns=[f"PC{i + 1}" for i in range(k)])
    if n >= 6:
        coords[["UMAP1", "UMAP2"]] = _umap(pcs if k >= 2 else X, n_neighbors=min(15, n - 1), metric="euclidean",
                                           seed=seed)
    return coords, [float(v) for v in pca.explained_variance_ratio_]


def feature_embedding(z: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """2-D UMAP of feature profiles (correlation metric, matching correlation-based grouping)."""
    X = np.nan_to_num(z.to_numpy(dtype=float))
    xy = _umap(X, n_neighbors=15, metric="correlation", seed=seed)
    return pd.DataFrame(xy, index=z.index, columns=["x", "y"])


def _umap(X: np.ndarray, n_neighbors: int, metric: str, seed: int, min_dist: float = 0.1) -> np.ndarray:
    try:
        import umap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=seed).fit_transform(X)
    except Exception as exc:  # fall back to PCA so the bundle still builds
        log.warning(f"UMAP failed ({exc}); using PCA coordinates instead.")
        return PCA(n_components=2, random_state=seed).fit_transform(X)


def group_layout(eigengenes: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """2-D positions for groups: UMAP of eigengene profiles (correlation metric), so groups
    with similar abundance patterns sit together on the group network."""
    X = np.nan_to_num(eigengenes.to_numpy(dtype=float))
    n = len(X)
    if n == 0:
        return pd.DataFrame(columns=["x", "y"])
    if n < 15:
        k = min(2, n, X.shape[1])
        xy = PCA(n_components=k, random_state=seed).fit_transform(X) if n > 1 else np.zeros((1, 2))
        if xy.shape[1] < 2:
            xy = np.column_stack([xy, np.zeros(n)])
    else:
        xy = _umap(X, n_neighbors=min(15, n - 1), metric="correlation", seed=seed, min_dist=0.4)
    return pd.DataFrame(xy, index=eigengenes.index, columns=["x", "y"])
