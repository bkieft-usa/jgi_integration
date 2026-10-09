import re

import numpy as np
import pandas as pd
import pytest

from tools import preprocessing as prep

GROUPS = ["A", "B", "C", "D"]
REPS = 3
SAMPLES = [f"{g}_{r}" for g in GROUPS for r in range(REPS)]
SAMPLE_TO_GROUP = {s: s.split("_")[0] for s in SAMPLES}
COND_EFFECT = {"A": -1.5, "B": -0.5, "C": 0.5, "D": 1.5}
N_SIGNAL = 30
N_NULL = {"tx": 270, "mx": 170, "px": 120}


def _beta(rng, n_total):
    beta = np.zeros(n_total)
    beta[:N_SIGNAL] = rng.choice([-1.0, 1.0], N_SIGNAL)
    return beta


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(7)
    n = len(SAMPLES)
    latent = np.array([COND_EFFECT[SAMPLE_TO_GROUP[s]] for s in SAMPLES]) + rng.normal(0, 0.05, n)
    depth = np.exp(rng.normal(0, 0.4, n))  # shared biomass artifact across data types
    true_sf = {t: depth * np.exp(rng.normal(0, 0.1, n)) for t in ("tx", "mx", "px")}

    out = {"latent": latent, "true_sf": true_sf}
    n_tx = N_SIGNAL + N_NULL["tx"]
    base = rng.lognormal(6, 1, n_tx)
    mu = base[:, None] * 2 ** (_beta(rng, n_tx)[:, None] * latent[None, :]) * true_sf["tx"][None, :]
    out["tx"] = pd.DataFrame(rng.poisson(mu), index=[f"tx_g{i}" for i in range(n_tx)], columns=SAMPLES)

    for t, polarity in (("mx", True), ("px", False)):
        n_f = N_SIGNAL + N_NULL[t]
        base = rng.lognormal(12, 1, n_f)
        vals = (base[:, None] * 2 ** (_beta(rng, n_f)[:, None] * latent[None, :])
                * true_sf[t][None, :] * np.exp(rng.normal(0, 0.15, (n_f, n))))
        vals[vals < 2e4] = 0.0  # detection limit
        names = [f"{t}_f{i}" + (("_positive" if i % 2 else "_negative") if polarity else "") for i in range(n_f)]
        out[t] = pd.DataFrame(vals, index=names, columns=SAMPLES)
    return out


def _centered_log(x):
    x = np.log(np.asarray(x, dtype=float))
    return x - x.mean()


def _feature_number(name):
    return int(re.search(r"_[fg](\d+)", name).group(1))


def _mean_null_cross_corr(rep_a, rep_b, n_signal=N_SIGNAL):
    a = rep_a.iloc[n_signal:].to_numpy()
    b = rep_b.iloc[n_signal:].to_numpy()
    a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True).clip(1e-9)
    b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True).clip(1e-9)
    return float(np.abs(a @ b.T / a.shape[1]).mean())


BASE_PARAMS = {
    "filtering": {"presence_min_percent": 50, "magnitude_min_mean": 1},
    "replicate_handling": {"method": "sd", "params": {"sd_threshold": 1.0, "majority_fraction": 0.5, "min_replicates": 2}},
}


def _voom(counts):
    libsize = counts.sum(axis=0)
    return np.log2((counts + 0.5) / (libsize + 1) * 1e6)


def _run(synthetic, types, tx_state, params=None):
    params = params or BASE_PARAMS
    results = {}
    for t in types:
        data = synthetic[t]
        state = "raw"
        if t == "tx" and tx_state == "prenormalized":
            data, state = _voom(synthetic["tx"]), "prenormalized"
        results[t] = prep.preprocess_dataset(data, t, state, params, SAMPLE_TO_GROUP, dataset_name=t)
    return results


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_size_factors_recovered(synthetic):
    for t in ("tx", "mx", "px"):
        sf = prep.median_of_ratios_size_factors(synthetic[t]) if t == "tx" else prep.pqn_size_factors(synthetic[t])
        assert np.corrcoef(np.log(sf.to_numpy()), _centered_log(synthetic["true_sf"][t]))[0, 1] > 0.98


def test_median_of_ratios_poscounts_fallback():
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(rng.poisson(0.7, (200, 8)) * 20, columns=list("abcdefgh"))
    counts.iloc[:, 0] *= 3
    sf = prep.median_of_ratios_size_factors(counts)
    assert np.isclose(np.exp(np.log(sf).mean()), 1.0)
    assert sf["a"] > sf.drop("a").max()


def test_polarity_groups_and_per_polarity_normalization():
    rng = np.random.default_rng(1)
    base = rng.lognormal(10, 1, (120, 6))
    depth_pos = np.array([1, 1, 1, 2, 2, 2.0])
    depth_neg = np.array([1, 2, 1, 2, 1, 2.0])
    df = pd.DataFrame(np.vstack([base[:60] * depth_pos, base[60:] * depth_neg]),
                      index=[f"mx_{i}_positive" for i in range(60)] + [f"mx_{i}_negative" for i in range(60)])
    norm, factors = prep.normalize_samples(df, "pqn", "mx")
    assert set(factors.columns) == {"positive", "negative"}
    pos = norm.iloc[:60].median(axis=0)
    neg = norm.iloc[60:].median(axis=0)
    assert pos.std() / pos.mean() < 0.15 and neg.std() / neg.mean() < 0.15


def test_raw_filters_semantics():
    df = pd.DataFrame([[0, 0, 0, 10], [5, 5, 0, 0], [100, 100, 100, 100], [1, 1, 1, 1]],
                      index=["a", "b", "c", "d"], columns=list("wxyz"))
    out = prep.filter_features(df, "raw", presence_min_percent=50, magnitude_min_mean=1)
    assert list(out.index) == ["b", "c"]  # d has mean exactly 1 (not > 1); a is present in only 25%
    assert list(prep.filter_features(df, "raw", presence_min_percent=50).index) == ["b", "c", "d"]
    assert prep.filter_features(df, "raw").shape == df.shape
    assert prep.filter_features(df, "prenormalized", presence_min_percent=99, magnitude_min_mean=1e9).shape == df.shape


def test_impute_feature_minimum():
    df = pd.DataFrame([[0, 4.0, 2.0], [3.0, 0, 0], [0, 0, 0]], index=list("abc"))
    out = prep.impute_feature_minimum(df)
    assert list(out.index) == ["a", "b"]
    assert out.loc["a"].tolist() == [2.0, 4.0, 2.0]
    assert out.loc["b"].tolist() == [3.0, 3.0, 3.0]


def test_variance_filter_removes_lowest():
    rng = np.random.default_rng(2)
    flat = pd.DataFrame(rng.normal(10, 0.01, (25, 10)), index=[f"flat{i}" for i in range(25)])
    var = pd.DataFrame(rng.normal(10, 2.0, (75, 10)), index=[f"var{i}" for i in range(75)])
    out = prep.filter_low_variance(pd.concat([flat, var]), "percent", 25, mean_adjusted=False)
    assert out.shape[0] == 75 and not any(i.startswith("flat") for i in out.index)
    assert prep.filter_low_variance(pd.concat([flat, var]), "none", 25).shape[0] == 100


def test_replicate_filter_is_condition_aware():
    rng = np.random.default_rng(3)
    cols = SAMPLES
    data = pd.DataFrame(rng.normal(8, 0.1, (3, 12)), index=["stable", "noisy_everywhere", "noisy_one_group"], columns=cols)
    data.loc["noisy_everywhere"] = rng.normal(8, 3.0, 12)
    data.loc["noisy_one_group", ["A_0", "A_1", "A_2"]] = [4.0, 8.0, 12.0]
    detected = pd.DataFrame(True, index=data.index, columns=cols)
    out = prep.filter_unreliable_features(data, detected, SAMPLE_TO_GROUP, sd_threshold=1.0)
    assert list(out.index) == ["stable", "noisy_one_group"]
    # a group in which the feature is detected in fewer than min_replicates samples is not evaluated
    detected.loc["noisy_everywhere", :] = False
    out = prep.filter_unreliable_features(data, detected, SAMPLE_TO_GROUP, sd_threshold=1.0)
    assert "noisy_everywhere" in out.index


def test_detect_scales():
    rng = np.random.default_rng(4)
    counts = pd.DataFrame(rng.poisson(rng.lognormal(5, 1, (300, 1)), (300, 10)))
    assert prep.inspect_input(counts, "tx", "raw")["detected_scale"] == "linear"
    heights = pd.DataFrame(rng.lognormal(11, 1, (200, 10)))
    assert prep.inspect_input(heights, "mx", "raw")["detected_scale"] == "linear"
    assert prep.inspect_input(np.log2(heights), "mx", "prenormalized")["detected_scale"] == "log"
    assert prep.inspect_input(np.log2(counts + 1), "tx", "prenormalized")["detected_scale"] == "log"
    assert prep.inspect_input(_voom(counts), "tx", "prenormalized")["detected_scale"] == "log"
    assert prep.inspect_input(counts, "tx", "prenormalized")["detected_scale"] == "linear"
    z = prep.zscore_samples(np.log2(heights))
    assert prep.inspect_input(z, "mx", "prenormalized")["detected_scale"] == "scaled"


def test_raw_validation_errors():
    rng = np.random.default_rng(5)
    counts = pd.DataFrame(rng.poisson(50, (100, 6)))
    with pytest.raises(ValueError):
        prep.inspect_input(counts - 60, "tx", "raw")
    with pytest.raises(ValueError):
        prep.inspect_input(np.log2(counts + 1.5), "tx", "raw")


def test_robust_zscore_and_block_scale():
    rng = np.random.default_rng(6)
    data = pd.DataFrame(rng.normal(0, 1, (20, 12)), index=[f"tx_{i}" for i in range(20)])
    z = prep.zscore_samples(data)
    assert np.allclose(z.mean(axis=1), 0) and np.allclose(z.std(axis=1, ddof=1), 1)
    assert np.isfinite(prep.zscore_samples(data, robust=True)).all().all()
    both = pd.concat([z, prep.zscore_samples(pd.DataFrame(rng.normal(0, 1, (80, 12)), index=[f"mx_{i}" for i in range(80)]))])
    scaled, weights = prep.block_scale(both, ("tx_", "mx_"))
    var_tx = (scaled.loc[scaled.index.str.startswith("tx_")] ** 2).sum().sum()
    var_mx = (scaled.loc[scaled.index.str.startswith("mx_")] ** 2).sum().sum()
    assert np.isclose(var_tx, var_mx)
    assert weights["tx_"] > weights["mx_"]


# ---------------------------------------------------------------------------
# Scenario tests: (a) tx+mx+px raw, (b) tx+mx raw, (c) voom tx + mx raw; paired and unpaired
# ---------------------------------------------------------------------------

SCENARIOS = {
    "a_tx_mx_px_raw": (("tx", "mx", "px"), "raw"),
    "b_tx_mx_raw": (("tx", "mx"), "raw"),
    "c_tx_voom_mx_raw": (("tx", "mx"), "prenormalized"),
}


@pytest.mark.parametrize("pairing", prep.PAIRINGS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenarios(synthetic, scenario, pairing):
    types, tx_state = SCENARIOS[scenario]
    res = _run(synthetic, types, tx_state)

    # mx/px: no zeros or missing values after normalization + imputation + log
    for t in types:
        if t != "tx":
            assert np.isfinite(res[t]["log"].to_numpy()).all()
            assert res[t]["detected"].to_numpy().mean() < 1.0  # zeros were detected as missing

    if tx_state == "prenormalized":
        assert res["tx"]["report"]["detected_scale"] == "log"
        pd.testing.assert_frame_equal(res["tx"]["log"], _voom(synthetic["tx"]).loc[res["tx"]["log"].index])
        assert res["tx"]["filtered"].shape == synthetic["tx"].shape  # raw filters never run
    assert res["mx"]["report"]["detected_scale"] == "linear"

    reps = {t: prep.represent(res[t]["replicate_filtered"], pairing, SAMPLE_TO_GROUP) for t in types}
    expected_cols = sorted(SAMPLES) if pairing == "paired" else sorted(GROUPS)
    for t in types:
        assert sorted(reps[t].columns) == expected_cols
        assert np.allclose(reps[t].mean(axis=1), 0, atol=1e-8)

    # planted shared signal is recoverable in every data type
    target = (np.array([COND_EFFECT[SAMPLE_TO_GROUP[s]] for s in reps["tx"].columns]) if pairing == "paired"
              else np.array([COND_EFFECT[g] for g in reps["tx"].columns]))
    for t in types:
        ids = [i for i in reps[t].index if _feature_number(i) < N_SIGNAL]
        assert len(ids) >= N_SIGNAL * 0.8
        cors = [abs(np.corrcoef(reps[t].loc[i].to_numpy(), target)[0, 1]) for i in ids]
        assert np.median(cors) > 0.9


def test_depth_artifact_not_inflating_cross_datatype_correlation(synthetic):
    def null_corr(normalization):
        params = {"filtering": {"presence_min_percent": 50}, "sample_normalization": normalization}
        res = _run(synthetic, ("tx", "mx"), "raw", params)
        reps = {t: prep.zscore_samples(res[t]["replicate_filtered"]) for t in ("tx", "mx")}
        tx = reps["tx"].loc[[i for i in reps["tx"].index if _feature_number(i) >= N_SIGNAL]]
        mx = reps["mx"].loc[[i for i in reps["mx"].index if _feature_number(i) >= N_SIGNAL]]
        return _mean_null_cross_corr(tx, mx, 0)

    normalized, unnormalized = null_corr("auto"), null_corr("none")
    assert normalized < 0.35
    assert normalized < unnormalized - 0.1
