"""Association statistics for the explorer bundle.

One model for everything that is tested against metadata (features, group eigengenes,
term eigengenes): a linear model with empirical-Bayes variance moderation (Smyth 2004, the
limma method), vectorized over rows.

* Categorical variable: moderated F test across all levels ("any difference"), plus a
  moderated t test for every pair of levels.
* Continuous variable: moderated t test of the slope.
* Benjamini-Hochberg FDR within each comparison.

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
    fit = ModelFit(variable, var_type, levels, coef, s2_post, df_total, xtx_inv, n, df_resid, unit)
    if var_type == "continuous":
        fit.contrasts["slope"] = np.array([0.0, 1.0])
    else:  # every pair of levels; effect = mean(first) - mean(second)
        k = len(levels)
        for i in range(k):
            for j in range(i + 1, k):
                c = np.zeros(k)
                c[i], c[j] = 1.0, -1.0
                fit.contrasts[pair_label(levels[i], levels[j])] = c
    return fit


PAIR_SEP = " vs "


def pair_label(a: str, b: str) -> str:
    return f"{a}{PAIR_SEP}{b}"


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
):
    """Moderated association tests of every row of ``data`` (rows x samples) with each variable.

    Returns a table with columns
    ``entity_type, entity_id, variable, contrast, test, statistic, effect, p, padj, n, unit``.
    BH is applied within each (variable, contrast). Rows with missing values in the
    samples used for a variable are skipped for that variable.
    """
    data = data.loc[:, [c for c in data.columns if c in metadata.index]]
    meta = metadata.loc[data.columns]
    frames = []
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
        if fit.var_type == "categorical" and len(fit.levels) > 2:
            F, p = moderated_f(fit)
            frames.append(_frame(entity_type, ids, var, "omnibus", "moderated_F", F, np.full(len(ids), np.nan), p,
                                 fit.n_obs, unit))
        for name, c in fit.contrasts.items():
            eff, t, p = moderated_t(fit, c)
            frames.append(_frame(entity_type, ids, var, name, "moderated_t", t, eff, p, fit.n_obs, unit))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TABLE_COLUMNS)


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
