import logging
import re
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)
if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    fmt = "\033[47m%(levelname)s - %(message)s\033[0m"
    handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

DATA_TYPES = ("tx", "mx", "px")
INPUT_STATES = ("raw", "prenormalized")
PAIRINGS = ("paired", "unpaired")
SAMPLE_NORMALIZATIONS = ("auto", "median_of_ratios", "pqn", "median", "tic", "none")
_DEFAULT_NORMALIZATION = {"tx": "median_of_ratios", "mx": "pqn", "px": "pqn"}
_POLARITY_RE = re.compile(r"_(positive|negative)$")


def _check_choice(value: Any, choices: Tuple[str, ...], name: str) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got '{value}'.")


def inspect_input(data: pd.DataFrame, data_type: str, input_state: str) -> Dict[str, Any]:
    """Validate the declared input_state and classify the value scale.

    ``detected_scale`` is ``linear`` (counts / intensities), ``log`` (log-transformed,
    e.g. voom log-CPM) or ``scaled`` (per-feature centered and unit variance).
    Raw input must be non-negative and linear; a contradiction raises ValueError.
    """

    _LOG_MAX_VALUE = 40.0
    _LOG_MAX_SKEW = 3.0

    _check_choice(data_type, DATA_TYPES, "data_type")
    _check_choice(input_state, INPUT_STATES, "input_state")

    X = data.apply(pd.to_numeric, errors="coerce").astype(float)
    values = X.to_numpy().ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Input matrix contains no finite values.")

    frac_negative = float((finite < 0).mean())
    integer_like = bool(np.all(np.isclose(finite, np.round(finite))))
    vmax = float(finite.max())
    skew = float(stats.skew(finite)) if finite.std() > 0 else 0.0
    skew = 0.0 if np.isnan(skew) else skew
    row_mean = X.mean(axis=1)
    row_sd = X.std(axis=1)
    feature_centered = bool(abs(row_mean).median() < 0.05 and abs(row_sd.median() - 1.0) < 0.1)

    if feature_centered:
        scale = "scaled"
    elif frac_negative > 0:
        scale = "log"
    elif integer_like:
        scale = "linear"
    elif vmax <= _LOG_MAX_VALUE and skew < _LOG_MAX_SKEW:
        scale = "log"
    else:
        scale = "linear"

    report = {
        "data_type": data_type,
        "input_state": input_state,
        "n_features": int(X.shape[0]),
        "n_samples": int(X.shape[1]),
        "frac_nan": float(1 - finite.size / values.size),
        "frac_negative": frac_negative,
        "frac_zero": float((finite == 0).mean()),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "max": vmax,
        "skew": skew,
        "integer_like": integer_like,
        "feature_centered": feature_centered,
        "detected_scale": scale,
    }

    if input_state == "raw":
        if frac_negative > 0:
            raise ValueError(
                f"{data_type}: input_state is 'raw' but {frac_negative:.1%} of values are negative. "
                "Set input_state: prenormalized if the data are already normalized/transformed."
            )
        if scale != "linear":
            raise ValueError(
                f"{data_type}: input_state is 'raw' but the values look {scale}-scaled "
                f"(max={vmax:.3g}, skew={skew:.2f}). Set input_state: prenormalized if appropriate."
            )
        if data_type == "tx" and not integer_like:
            log.warning("tx: raw counts are not integers (e.g. expected counts); proceeding.")
    else:
        log.info(
            f"{data_type}: prenormalized input detected as '{scale}' scale "
            f"(negative={frac_negative:.1%}, zero={report['frac_zero']:.1%}, "
            f"min={report['min']:.3g}, median={report['median']:.3g}, max={vmax:.3g}, skew={skew:.2f})."
        )
    return report


def filter_features(
    data: pd.DataFrame,
    input_state: str,
    presence_min_percent: Optional[float] = None,
    magnitude_min_mean: Optional[float] = None,
    dataset_name: str = "",
) -> pd.DataFrame:
    """Keep features detected (>0) in >= presence_min_percent% of samples and with mean raw value > magnitude_min_mean.

    Either threshold may be None to skip it. Never applied to prenormalized input.
    """
    if input_state == "prenormalized":
        if presence_min_percent is not None or magnitude_min_mean is not None:
            log.info(f"{dataset_name}: prenormalized input; skipping raw presence/magnitude filters.")
        return data
    keep = pd.Series(True, index=data.index)
    if presence_min_percent is not None:
        keep &= ((data > 0).mean(axis=1) * 100) >= presence_min_percent
    if magnitude_min_mean is not None:
        keep &= data.mean(axis=1) > magnitude_min_mean
    out = data.loc[keep.to_numpy()]
    log.info(f"{dataset_name}: raw filters kept {out.shape[0]} of {data.shape[0]} features.")
    return out


def _center_factors(sf: pd.Series) -> pd.Series:
    bad = ~np.isfinite(sf.to_numpy(dtype=float)) | (sf.to_numpy(dtype=float) <= 0)
    if bad.any():
        raise ValueError(f"Could not estimate a valid size factor for samples: {list(sf.index[bad])}")
    return sf / np.exp(np.log(sf).mean())


def median_of_ratios_size_factors(counts: pd.DataFrame, min_features: int = 10) -> pd.Series:
    """DESeq2-style size factors from features positive in all samples (poscounts-style fallback)."""
    X = counts.where(counts > 0)
    all_pos = X.notna().all(axis=1)
    if all_pos.sum() >= min_features:
        use = X.loc[all_pos]
    else:
        log.warning("Few features are positive in all samples; using positive-only geometric means.")
        use = X.loc[X.notna().mean(axis=1) >= 0.5]
        if use.shape[0] < min_features:
            use = X.loc[X.notna().any(axis=1)]
    logx = np.log(use)
    ratios = logx.sub(logx.mean(axis=1), axis=0)
    return _center_factors(np.exp(ratios.median(axis=0)))


def pqn_size_factors(data: pd.DataFrame, min_detect_fraction: float = 0.5) -> pd.Series:
    """Probabilistic quotient normalization factors; zeros are treated as missing."""
    X = data.where(data > 0)
    use = X.loc[X.notna().mean(axis=1) >= min_detect_fraction]
    if use.empty:
        use = X.loc[X.notna().any(axis=1)]
    quotients = use.div(use.median(axis=1), axis=0)
    return _center_factors(quotients.median(axis=0))


def median_size_factors(data: pd.DataFrame) -> pd.Series:
    return _center_factors(data.where(data > 0).median(axis=0))


def tic_size_factors(data: pd.DataFrame) -> pd.Series:
    return _center_factors(data.where(data > 0).sum(axis=0))


_SIZE_FACTOR_FUNCS = {
    "median_of_ratios": median_of_ratios_size_factors,
    "pqn": pqn_size_factors,
    "median": median_size_factors,
    "tic": tic_size_factors,
}


def _feature_groups(index: pd.Index, data_type: str) -> Dict[str, np.ndarray]:
    """Boolean masks per normalization group; mx features are grouped by polarity suffix."""
    labels = np.array(["all"] * len(index), dtype=object)
    if data_type == "mx":
        for i, name in enumerate(index.astype(str)):
            m = _POLARITY_RE.search(name)
            if m:
                labels[i] = m.group(1)
    return {g: labels == g for g in pd.unique(labels)}


def normalize_samples(
    data: pd.DataFrame, method: str = "auto", data_type: str = "tx"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide each sample by its size factor. Returns (normalized, size_factors[samples x groups])."""
    _check_choice(method, SAMPLE_NORMALIZATIONS, "sample_normalization")
    if method == "auto":
        method = _DEFAULT_NORMALIZATION[data_type]
    X = data.astype(float)
    if method == "none":
        return X.copy(), pd.DataFrame({"all": 1.0}, index=X.columns)
    values = X.to_numpy().copy()
    factors = {}
    for name, mask in _feature_groups(X.index, data_type).items():
        sf = _SIZE_FACTOR_FUNCS[method](X.loc[mask])
        factors[name] = sf
        values[mask, :] = values[mask, :] / sf.to_numpy()[None, :]
        log.info(f"Sample normalization '{method}' ({name}): size factors range "
                 f"{sf.min():.3g}-{sf.max():.3g}.")
    return pd.DataFrame(values, index=X.index, columns=X.columns), pd.DataFrame(factors)


def impute_feature_minimum(data: pd.DataFrame) -> pd.DataFrame:
    """Treat zeros/NaN as missing and fill with each feature's minimum observed value.

    Features with no observed value are dropped.
    """
    X = data.where(data > 0)
    mins = X.min(axis=1)
    keep = mins.notna()
    X = X.loc[keep]
    arr = X.to_numpy()
    arr = np.where(np.isnan(arr), mins[keep].to_numpy()[:, None], arr)
    return pd.DataFrame(arr, index=X.index, columns=X.columns)


def log_transform(data: pd.DataFrame, data_type: str, pseudocount: float = 1.0) -> pd.DataFrame:
    """tx: log2(x + pseudocount) on size-factor-normalized counts; mx/px: log2(x)."""
    if data_type == "tx":
        return np.log2(data + pseudocount)
    if not (data > 0).all().all():
        raise ValueError(f"{data_type}: log2 requires strictly positive values; impute first.")
    return np.log2(data)


def normalize_and_transform(
    data: pd.DataFrame,
    data_type: str,
    input_state: str,
    scale: str,
    sample_normalization: str = "auto",
    recenter_samples: bool = False,
    pseudocount: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """Sample normalization, imputation (mx/px) and log transform.

    Returns dict with ``normalized`` (linear scale, zeros kept as NaN for mx/px),
    ``detected`` (boolean observed mask) and ``log`` (analysis-ready log matrix).
    """
    X = data.apply(pd.to_numeric, errors="coerce").astype(float)
    if input_state == "raw":
        normalized, _ = normalize_samples(X, sample_normalization, data_type)
    else:
        if sample_normalization not in ("auto", "none"):
            log.info("Sample normalization is ignored for prenormalized input.")
        normalized = X

    if scale in ("log", "scaled"):
        detected = normalized.notna()
        logm = normalized
    elif data_type == "tx":
        detected = normalized > 0
        logm = log_transform(normalized, "tx", pseudocount)
    else:
        missing_form = normalized.where(normalized > 0)
        imputed = impute_feature_minimum(missing_form)
        normalized = missing_form.loc[imputed.index]
        detected = normalized.notna()
        logm = log_transform(imputed, data_type)

    medians = logm.median(axis=0)
    spread = float(medians.max() - medians.min())
    if recenter_samples:
        logm = logm.sub(medians - medians.mean(), axis=1)
        log.info(f"Recentered sample medians (spread was {spread:.2f}).")
    elif input_state == "prenormalized" and spread > 1.0:
        log.warning(f"Per-sample median spread is {spread:.2f} on the log scale; "
                    "consider recenter_samples: true.")
    return {"normalized": normalized, "detected": detected, "log": logm}


def variance_score(data: pd.DataFrame) -> pd.Series:
    """Log variance minus the median log variance of features with similar mean level."""
    lv = np.log(data.var(axis=1, ddof=1).fillna(0.0).clip(lower=1e-12))
    n_bins = int(min(20, max(1, len(lv) // 50)))
    bins = pd.qcut(data.mean(axis=1).rank(method="first"), n_bins, labels=False)
    return lv - lv.groupby(bins).transform("median")


def filter_low_variance(
    data: pd.DataFrame,
    method: Optional[str] = "percent",
    percent: Optional[float] = None,
    mean_adjusted: bool = True,
    scale: str = "log",
    dataset_name: str = "",
) -> pd.DataFrame:
    """Remove the lowest-variance ``percent``% of features on the log-scale matrix."""
    if method in (None, "none") or percent is None:
        return data
    if method != "percent":
        raise ValueError("devariancing method must be 'percent' or 'none'.")
    if scale == "scaled":
        log.warning(f"{dataset_name}: data are already feature-scaled; variance filter skipped.")
        return data
    score = variance_score(data) if mean_adjusted else data.var(axis=1, ddof=1).fillna(0.0)
    n_remove = int(np.floor(len(score) * percent / 100.0))
    if n_remove == 0:
        return data
    drop = score.sort_values(kind="mergesort").index[:n_remove]
    out = data.loc[~data.index.isin(drop)]
    log.info(f"{dataset_name}: variance filter removed {n_remove} of {data.shape[0]} features.")
    return out


def _groups_to_samples(columns: pd.Index, sample_to_group: Dict[str, Any]) -> Dict[Any, list]:
    groups: Dict[Any, list] = {}
    for s in columns:
        g = sample_to_group.get(s)
        if g is not None and not pd.isna(g):
            groups.setdefault(g, []).append(s)
    return groups


def filter_unreliable_features(
    data: pd.DataFrame,
    detected: pd.DataFrame,
    sample_to_group: Dict[str, Any],
    sd_threshold: Optional[float] = 1.0,
    majority_fraction: float = 0.5,
    min_replicates: int = 2,
    dataset_name: str = "",
) -> pd.DataFrame:
    """Drop features whose within-group log-scale SD exceeds ``sd_threshold`` in more than
    ``majority_fraction`` of the groups in which they are detected in >= ``min_replicates`` samples."""
    if sd_threshold is None:
        return data
    groups = {g: s for g, s in _groups_to_samples(data.columns, sample_to_group).items()
              if len(s) >= min_replicates}
    if not groups:
        log.warning(f"{dataset_name}: no groups with >= {min_replicates} replicates; replicate filter skipped.")
        return data
    det = detected.reindex(index=data.index, columns=data.columns).fillna(True).astype(bool)
    n_eval = pd.Series(0, index=data.index)
    n_flag = pd.Series(0, index=data.index)
    for samples in groups.values():
        sd = data[samples].std(axis=1, ddof=1)
        evaluable = det[samples].sum(axis=1) >= min_replicates
        n_eval += evaluable.astype(int)
        n_flag += (evaluable & (sd > sd_threshold)).astype(int)
    drop = (n_flag / n_eval.where(n_eval > 0)) > majority_fraction
    out = data.loc[~drop.to_numpy()]
    log.info(f"{dataset_name}: replicate filter removed {int(drop.sum())} of {data.shape[0]} features.")
    return out


def zscore_samples(data: pd.DataFrame, robust: bool = False) -> pd.DataFrame:
    """Per-feature z-score across columns. ``robust`` uses median/MAD with the scale floored at 0.5*SD."""
    X = data.astype(float)
    sd = X.std(axis=1, ddof=1)
    if robust:
        center = X.median(axis=1)
        mad = 1.4826 * X.sub(center, axis=0).abs().median(axis=1)
        denom = pd.concat([mad, 0.5 * sd], axis=1).max(axis=1)
    else:
        center = X.mean(axis=1)
        denom = sd
    out = X.sub(center, axis=0).div(denom.where(denom > 0), axis=0)
    return out.fillna(0.0)


def condition_profiles(
    data: pd.DataFrame, sample_to_group: Dict[str, Any], robust: bool = False
) -> pd.DataFrame:
    """Group medians on the log scale, then per-feature z-scored across conditions."""
    groups = _groups_to_samples(data.columns, sample_to_group)
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 conditions to build condition profiles; found {len(groups)}.")
    profiles = pd.DataFrame({str(g): data[s].median(axis=1) for g, s in sorted(groups.items(), key=lambda kv: str(kv[0]))})
    return zscore_samples(profiles, robust=robust)


def represent(
    data: pd.DataFrame,
    pairing: str,
    sample_to_group: Optional[Dict[str, Any]] = None,
    robust: bool = False,
) -> pd.DataFrame:
    """paired -> per-sample z-scores; unpaired -> per-condition z-scored medians."""
    _check_choice(pairing, PAIRINGS, "replicate_pairing")
    if pairing == "paired":
        return zscore_samples(data, robust=robust)
    if sample_to_group is None:
        raise ValueError("sample_to_group is required for unpaired representation.")
    return condition_profiles(data, sample_to_group, robust=robust)


def select_top_variable(blocks: Dict[str, pd.DataFrame], max_features: int) -> Dict[str, list]:
    """Pick the most variable features (mean-adjusted log variance) per dataset within a shared budget.

    Smaller datasets are served first; unused quota is passed on to larger datasets.
    """
    scores = {name: variance_score(df).sort_values(ascending=False) for name, df in blocks.items()}
    order = sorted(scores, key=lambda n: len(scores[n]))
    remaining = int(max_features)
    chosen: Dict[str, list] = {}
    for i, name in enumerate(order):
        take = min(len(scores[name]), remaining // (len(order) - i))
        chosen[name] = list(scores[name].index[:take])
        remaining -= take
    return chosen


def block_scale(data: pd.DataFrame, prefixes: Tuple[str, ...]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Weight each dataset block by sqrt(mean_k / k) so every block carries equal total variance."""
    sizes = {p: int(data.index.astype(str).str.startswith(p).sum()) for p in prefixes}
    sizes = {p: k for p, k in sizes.items() if k > 0}
    if not sizes:
        return data, {}
    mean_k = float(np.mean(list(sizes.values())))
    weights = {p: float(np.sqrt(mean_k / k)) for p, k in sizes.items()}
    out = data.copy()
    for p, w in weights.items():
        out.loc[out.index.astype(str).str.startswith(p)] *= w
    return out, weights


def run_filter(data: pd.DataFrame, input_state: str, params: Dict[str, Any], dataset_name: str = "") -> pd.DataFrame:
    p = params.get("filtering", {}) or {}
    log.info(f"Running filter stage for dataset: {dataset_name} with input state: {input_state}")
    log.info(f"Starting dataset: {data.shape[0]} features and {data.shape[1]} samples")
    filtered_data = filter_features(
        data, input_state,
        presence_min_percent=p.get("presence_min_percent"),
        magnitude_min_mean=p.get("magnitude_min_mean"),
        dataset_name=dataset_name,
    )
    log.info(f"Filtered dataset: {filtered_data.shape[0]} features and {filtered_data.shape[1]} samples")
    return filtered_data


def run_normalize(
    data: pd.DataFrame, data_type: str, input_state: str, scale: str, params: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    return normalize_and_transform(
        data, data_type, input_state, scale,
        sample_normalization=params.get("sample_normalization", "auto"),
        recenter_samples=bool(params.get("recenter_samples", False)),
        pseudocount=float(params.get("pseudocount", 1.0)),
    )


def run_devariance(data: pd.DataFrame, scale: str, params: Dict[str, Any], dataset_name: str = "") -> pd.DataFrame:
    step = params.get("devariancing", {}) or {}
    p = step.get("params", {}) or {}
    log.info(f"Running devariance stage for dataset: {dataset_name} with scale: {scale}")
    log.info(f"Starting dataset: {data.shape[0]} features and {data.shape[1]} samples")
    devarianced_data = filter_low_variance(
        data, method=step.get("method", "none"), percent=p.get("value"),
        mean_adjusted=bool(p.get("mean_adjusted", True)), scale=scale, dataset_name=dataset_name,
    )
    log.info(f"Devarianced dataset: {devarianced_data.shape[0]} features and {devarianced_data.shape[1]} samples")
    return devarianced_data


def run_replicate_filter(
    data: pd.DataFrame, detected: pd.DataFrame, sample_to_group: Dict[str, Any],
    params: Dict[str, Any], dataset_name: str = "",
) -> pd.DataFrame:
    step = params.get("replicate_handling", {}) or {}
    if step.get("method", "none") in (None, "none"):
        return data
    if step["method"] != "sd":
        raise ValueError("replicate_handling method must be 'sd' or 'none'.")
    p = step.get("params", {}) or {}
    log.info(f"Running replicate filter stage for dataset: {dataset_name}")
    log.info(f"Starting dataset: {data.shape[0]} features and {data.shape[1]} samples")
    replicate_filtered_data = filter_unreliable_features(
        data, detected, sample_to_group,
        sd_threshold=p.get("sd_threshold", 1.0),
        majority_fraction=float(p.get("majority_fraction", 0.5)),
        min_replicates=int(p.get("min_replicates", 2)),
        dataset_name=dataset_name,
    )
    log.info(f"Replicate filtered dataset: {replicate_filtered_data.shape[0]} features and {replicate_filtered_data.shape[1]} samples")
    return replicate_filtered_data


def preprocess_dataset(
    data: pd.DataFrame,
    data_type: str,
    input_state: str,
    params: Dict[str, Any],
    sample_to_group: Dict[str, Any],
    dataset_name: str = "",
) -> Dict[str, Any]:
    """Run stages 1-6 and return every intermediate matrix plus the input report."""
    report = inspect_input(data, data_type, input_state)
    scale = report["detected_scale"]
    filtered = run_filter(data, input_state, params, dataset_name)
    result = run_normalize(filtered, data_type, input_state, scale, params)
    devarianced = run_devariance(result["log"], scale, params, dataset_name)
    replicate_filtered = run_replicate_filter(devarianced, result["detected"], sample_to_group, params, dataset_name)
    return {
        "report": report,
        "filtered": filtered,
        "normalized": result["normalized"],
        "detected": result["detected"],
        "log": result["log"],
        "devarianced": devarianced,
        "replicate_filtered": replicate_filtered,
    }
