"""Association statistics for the explorer bundle.

* Linear models with empirical-Bayes variance moderation (Smyth 2004, the limma method),
  vectorized over features. Categorical variables get a moderated F omnibus test plus a
  moderated t for each level vs the mean of the other levels; continuous variables get a
  moderated t on the slope.
* CAMERA (Wu & Smyth 2012) competitive set test: is a set of features more associated
  than the rest, with a variance-inflation factor for inter-feature correlation.
* Benjamini-Hochberg FDR.

Everything here is pure numpy/scipy so it is easy to unit test on synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import special, stats


# ---- multiple testing ----------------------------------------------------------------

def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaNs are left as NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


# ---- empirical Bayes -----------------------------------------------------------------

def _trigamma(x):
    return special.polygamma(1, x)


def _trigamma_inverse(y: float) -> float:
    """Solve trigamma(x) = y (limma's Newton iteration)."""
    if y > 1e7:
        return 1.0 / np.sqrt(y)
    if y < 1e-6:
        return 1.0 / y
    x = 0.5 + 1.0 / y
    for _ in range(50):
        tri = _trigamma(x)
        dif = tri * (1 - tri / y) / special.polygamma(2, x)
        x += dif
        if -dif / x < 1e-8:
            break
    return x


def squeeze_var(s2: np.ndarray, df: float) -> tuple:
    """Fit the scaled inverse-chi-square prior to residual variances.

    Returns ``(s2_post, d0, s0_2)``. ``d0`` is the prior degrees of freedom (``inf``
    when the variances are no more dispersed than expected by chance).
    """
    s2 = np.asarray(s2, dtype=float)
    ok = np.isfinite(s2) & (s2 > 0)
    if ok.sum() < 3 or df <= 0:
        return s2.copy(), 0.0, np.nan
    z = np.log(s2[ok])
    e = z - special.digamma(df / 2) + np.log(df / 2)
    emean = e.mean()
    evar = e.var(ddof=1) - _trigamma(df / 2)
    if evar > 0:
        d0 = 2 * _trigamma_inverse(evar)
        s0_2 = np.exp(emean + special.digamma(d0 / 2) - np.log(d0 / 2))
        post = (d0 * s0_2 + df * s2) / (d0 + df)
    else:
        d0 = np.inf
        s0_2 = np.exp(emean)
        post = np.full_like(s2, s0_2)
    post = np.where(ok, post, np.nan)
    return post, d0, s0_2


# ---- model fitting -------------------------------------------------------------------

@dataclass
class ModelFit:
    """Result of fitting one variable's design to every row of a matrix."""
    variable: str
    var_type: str
    levels: List[str]
    coef: np.ndarray            # rows x p
    s2_post: np.ndarray         # rows
    df_total: float             # residual df + prior df
    xtx_inv: np.ndarray         # p x p
    residuals: np.ndarray       # rows x n (for CAMERA correlation)
    n_obs: int
    df_resid: int
    unit: Optional[str] = None  # experimental unit when aggregated for nesting
    contrasts: Dict[str, np.ndarray] = field(default_factory=dict)


def _design(values: pd.Series, var_type: str):
    if var_type == "continuous":
        x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        X = np.column_stack([np.ones_like(x), x - np.nanmean(x)])
        return X, ["intercept", "slope"]
    cats = pd.Categorical(values.astype(str))
    levels = [str(c) for c in cats.categories]
    X = np.zeros((len(values), len(levels)))
    X[np.arange(len(values)), cats.codes] = 1.0
    return X, levels


def fit_variable(Y: np.ndarray, values: pd.Series, variable: str, var_type: str,
                 unit: Optional[str] = None) -> Optional[ModelFit]:
    """Fit ``Y`` (rows x samples, complete rows only) against one metadata variable.

    Categorical variables use a cell-means design (one coefficient per level).
    Returns None when the design has no residual degrees of freedom.
    """
    X, levels = _design(values, var_type)
    n, p = X.shape
    df_resid = n - np.linalg.matrix_rank(X)
    if df_resid < 1 or (var_type == "categorical" and len(levels) < 2):
        return None
    xtx_inv = np.linalg.pinv(X.T @ X)
    coef = Y @ X @ xtx_inv           # rows x p
    resid = Y - coef @ X.T
    s2 = (resid ** 2).sum(axis=1) / df_resid
    s2_post, d0, _ = squeeze_var(s2, df_resid)
    df_total = df_resid + (d0 if np.isfinite(d0) else 1e6)
    fit = ModelFit(variable, var_type, levels, coef, s2_post, df_total, xtx_inv, resid, n, df_resid, unit)
    if var_type == "continuous":
        fit.contrasts["slope"] = np.array([0.0, 1.0])
    else:
        k = len(levels)
        for j, lv in enumerate(levels):
            c = np.full(k, -1.0 / (k - 1))
            c[j] = 1.0
            fit.contrasts[lv] = c
    return fit


def moderated_t(fit: ModelFit, contrast: np.ndarray):
    """``(effect, t, p)`` arrays for one contrast vector."""
    effect = fit.coef @ contrast
    se = np.sqrt(fit.s2_post * float(contrast @ fit.xtx_inv @ contrast))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = effect / se
    p = 2 * stats.t.sf(np.abs(t), fit.df_total)
    return effect, t, p


def moderated_f(fit: ModelFit):
    """Omnibus moderated F across all levels of a categorical variable: ``(F, p)``."""
    k = len(fit.levels)
    C = np.zeros((k - 1, k))
    C[:, 0] = -1.0
    C[np.arange(k - 1), np.arange(1, k)] = 1.0
    est = fit.coef @ C.T                                  # rows x (k-1)
    M = np.linalg.pinv(C @ fit.xtx_inv @ C.T)
    quad = np.einsum("ij,jk,ik->i", est, M, est)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = quad / ((k - 1) * fit.s2_post)
    p = stats.f.sf(F, k - 1, fit.df_total)
    return F, p


def aggregate_to_unit(Y: np.ndarray, meta: pd.DataFrame, variable: str, unit: str):
    """Average replicate columns within each level of ``unit`` (pseudo-replication fix)."""
    keys = meta[unit].astype(str).to_numpy()
    uniq = pd.unique(keys)
    Yu = np.column_stack([Y[:, keys == u].mean(axis=1) for u in uniq])
    values = meta.groupby(meta[unit].astype(str))[variable].first().reindex(uniq)
    return Yu, values


def associate_matrix(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    variables: Sequence[str],
    var_types: Dict[str, str],
    nesting: Dict[str, Optional[str]],
    entity_type: str,
    keep_fits: bool = False,
):
    """Moderated association tests of every row of ``data`` (rows x samples) with each variable.

    Returns ``(table, fits)``; ``table`` has columns
    ``entity_type, entity_id, variable, contrast, test, statistic, effect, p, padj, n, unit``.
    BH is applied within each (variable, contrast). Rows with missing values in the
    samples used for a variable are skipped for that variable.
    """
    data = data.loc[:, [c for c in data.columns if c in metadata.index]]
    meta = metadata.loc[data.columns]
    frames, fits = [], {}
    for var in variables:
        if var not in meta.columns:
            continue
        has = meta[var].notna().to_numpy()
        sub_meta = meta.loc[has]
        Y = data.loc[:, has].to_numpy(dtype=float)
        complete = ~np.isnan(Y).any(axis=1)
        if complete.sum() < 3:
            continue
        Y = Y[complete]
        ids = data.index[complete].astype(str)
        unit = nesting.get(var)
        if unit and unit in sub_meta.columns and sub_meta[unit].notna().all():
            Y, values = aggregate_to_unit(Y, sub_meta, var, unit)
        else:
            unit, values = None, sub_meta[var]
        fit = fit_variable(Y, values, var, var_types.get(var, "categorical"), unit=unit)
        if fit is None:
            continue
        if keep_fits:
            fits[var] = (fit, ids)
        if fit.var_type == "categorical" and len(fit.levels) > 2:
            F, p = moderated_f(fit)
            frames.append(_frame(entity_type, ids, var, "omnibus", "moderated_F", F, np.full(len(ids), np.nan), p,
                                 fit.n_obs, unit))
        for name, c in fit.contrasts.items():
            eff, t, p = moderated_t(fit, c)
            label = name if fit.var_type == "continuous" else f"{name} vs rest"
            frames.append(_frame(entity_type, ids, var, label, "moderated_t", t, eff, p, fit.n_obs, unit))
            if fit.var_type == "categorical" and len(fit.levels) == 2:
                break  # the two level-vs-rest contrasts of a binary variable are mirror images
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TABLE_COLUMNS)
    return table, fits


TABLE_COLUMNS = ["entity_type", "entity_id", "variable", "contrast", "test", "statistic", "effect", "p", "padj",
                 "n", "unit"]


def _frame(entity_type, ids, var, contrast, test, statistic, effect, p, n, unit):
    return pd.DataFrame({
        "entity_type": entity_type,
        "entity_id": np.asarray(ids),
        "variable": var,
        "contrast": contrast,
        "test": test,
        "statistic": np.asarray(statistic, dtype=float),
        "effect": np.asarray(effect, dtype=float),
        "p": np.asarray(p, dtype=float),
        "padj": bh_adjust(np.asarray(p, dtype=float)),
        "n": int(n),
        "unit": unit or "",
    })


# ---- CAMERA --------------------------------------------------------------------------

def t_to_z(t: np.ndarray, df: float) -> np.ndarray:
    """Convert t statistics to z-scores with the same tail probability."""
    t = np.asarray(t, dtype=float)
    lower = stats.t.logcdf(-np.abs(t), df)
    z = -stats.norm.ppf(np.exp(lower))
    z = np.where(np.isfinite(z), z, np.sign(t) * 37.0)
    return np.sign(t) * np.abs(z)


def camera(
    z: np.ndarray,
    residuals: Optional[np.ndarray],
    sets: Dict[str, np.ndarray],
    blocks: Optional[np.ndarray] = None,
    min_size: int = 3,
) -> pd.DataFrame:
    """CAMERA competitive test of each set against all other features.

    ``z``: per-feature statistic (z-scored moderated t) for one contrast.
    ``residuals``: features x samples residual matrix used to estimate the mean
    inter-feature correlation within each set (variance inflation factor); when
    ``blocks`` is given (feature -> block id, e.g. data type measured on different
    samples), correlation is only estimated within blocks.
    ``sets``: name -> integer indices into ``z``.
    Returns ``set, n, mean_z, correlation, statistic, p, direction``.
    """
    z = np.asarray(z, dtype=float)
    G = len(z)
    if residuals is not None:
        norms = np.sqrt((residuals ** 2).sum(axis=1, keepdims=True))
        U = np.divide(residuals, norms, out=np.zeros_like(residuals), where=norms > 0)
    rows = []
    var_all = z.var(ddof=1)
    for name, idx in sets.items():
        idx = np.unique(np.asarray(idx, dtype=int))
        m = len(idx)
        if m < min_size or G - m < 3:
            continue
        if residuals is None:
            rho = 0.0
        elif blocks is None:
            vif_raw = m * float((U[idx].mean(axis=0) ** 2).sum())
            rho = (vif_raw - 1) / (m - 1)
        else:
            within = 0.0
            for b in np.unique(blocks[idx]):
                bi = idx[blocks[idx] == b]
                mb = len(bi)
                if mb > 1:
                    s = U[bi].sum(axis=0)
                    within += float(s @ s) - mb   # sum of pairwise cosines within the block
            rho = within / (m * (m - 1))
        rho = max(rho, 0.0)                       # negative estimates would be anti-conservative
        vif = 1 + (m - 1) * rho
        mean_set = z[idx].mean()
        delta = G / (G - m) * (mean_set - z.mean())
        var_pooled = ((G - 1) * var_all - delta ** 2 * m * (G - m) / G) / (G - 2)
        stat = delta / np.sqrt(var_pooled * (vif / m + 1 / (G - m)))
        p = 2 * stats.t.sf(abs(stat), G - 2)
        rows.append((name, m, mean_set, rho, stat, p, "up" if stat > 0 else "down"))
    return pd.DataFrame(rows, columns=["set", "n", "mean_z", "correlation", "statistic", "p", "direction"])


# ---- set scores ----------------------------------------------------------------------

def eigengene(Z: np.ndarray) -> tuple:
    """First principal component of a set's (features x samples) z-scores.

    Sign-aligned with the mean profile and scaled to unit variance. Returns
    ``(eigengene, variance_explained)``.
    """
    Z = np.nan_to_num(np.asarray(Z, dtype=float))
    if Z.shape[0] == 1:
        v = Z[0] - Z[0].mean()
        sd = v.std()
        return (v / sd if sd > 0 else v), 1.0
    Zc = Z - Z.mean(axis=1, keepdims=True)
    try:
        _, s, vt = np.linalg.svd(Zc, full_matrices=False)
    except np.linalg.LinAlgError:
        return Zc.mean(axis=0), np.nan
    pc = vt[0]
    if np.corrcoef(pc, Zc.mean(axis=0))[0, 1] < 0:
        pc = -pc
    sd = pc.std()
    pc = (pc - pc.mean()) / sd if sd > 0 else pc
    var_exp = float(s[0] ** 2 / (s ** 2).sum()) if (s ** 2).sum() > 0 else np.nan
    return pc, var_exp


def row_correlation(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row of ``A`` with vector ``b``."""
    A = np.nan_to_num(np.asarray(A, dtype=float))
    Ac = A - A.mean(axis=1, keepdims=True)
    bc = b - b.mean()
    denom = np.sqrt((Ac ** 2).sum(axis=1) * (bc ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, Ac @ bc / denom, np.nan)
