"""Explorer bundle: statistics, annotation parsing, enrichment/agreement, and end-to-end builds."""
import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from tests.explorer_fixtures import make_run, run_info
from tools import annotation_types as annot
from tools import metadata as md
from tools.explorer_bundle import agreement as agr
from tools.explorer_bundle import stats as st
from tools.explorer_bundle.build import build_explorer_bundle, bundle_is_current


# ---- statistics --------------------------------------------------------------------------

def _null_design(rng, n_feat=2000, groups=4, reps=5):
    samples = [f"s{i}" for i in range(groups * reps)]
    meta = pd.DataFrame({"trt": np.repeat([f"t{i}" for i in range(groups)], reps)}, index=samples)
    Y = rng.normal(size=(n_feat, len(samples))) * rng.gamma(2, 0.5, size=(n_feat, 1))
    return pd.DataFrame(Y, index=[f"f{i}" for i in range(n_feat)], columns=samples), meta


def test_moderated_tests_are_calibrated_under_null():
    df, meta = _null_design(np.random.default_rng(1))
    tab = st.associate_matrix(df, meta, ["trt"], {"trt": "categorical"}, {}, "feature")
    for test in ("moderated_F", "moderated_t"):
        p = tab.loc[tab["test"] == test, "p"].to_numpy()
        assert sps.kstest(p, "uniform").pvalue > 0.001, test
        assert abs((p < 0.05).mean() - 0.05) < 0.02


def test_planted_effect_detected_and_bh_controls_fdr():
    rng = np.random.default_rng(2)
    df, meta = _null_design(rng)
    sd = df.iloc[:100].std(axis=1).to_numpy()[:, None]
    df.iloc[:100, meta["trt"].to_numpy() == "t0"] += 3 * sd   # a 3-SD shift for each planted feature
    tab = st.associate_matrix(df, meta, ["trt"], {"trt": "categorical"}, {}, "feature")
    planted = [f"f{i}" for i in range(100)]
    for contrast, power in (("omnibus", 0.8), ("t0 vs t1", 0.7)):  # a pair has 5 vs 5 samples: less power
        t0 = tab[(tab["contrast"] == contrast)].set_index("entity_id")
        hits = t0["padj"] < 0.05
        assert hits.loc[planted].mean() > power, contrast
        false = hits.drop(planted).sum()
        assert false / max(hits.sum(), 1) < 0.1, contrast


def test_nesting_aggregation_prevents_pseudoreplication():
    rng = np.random.default_rng(3)
    orgs = [f"o{i}" for i in range(6)]
    samples = [f"{o}-{r}" for o in orgs for r in range(4)]
    meta = pd.DataFrame({"organism": [s.split("-")[0] for s in samples]}, index=samples)
    meta["lineage"] = meta["organism"].map({o: "A" if i < 3 else "B" for i, o in enumerate(orgs)})
    org_eff = rng.normal(0, 1, (3000, 6))
    Y = org_eff[:, [orgs.index(o) for o in meta["organism"]]] + rng.normal(0, 0.5, (3000, len(samples)))
    df = pd.DataFrame(Y, columns=samples)
    nesting = md.detect_nesting(meta, ["organism", "lineage"])
    assert nesting == {"organism": None, "lineage": "organism"}
    types = {"organism": "categorical", "lineage": "categorical"}
    nested = st.associate_matrix(df, meta, ["lineage"], types, nesting, "feature")
    naive = st.associate_matrix(df, meta, ["lineage"], types, {}, "feature")
    assert (nested["p"] < 0.05).mean() < 0.08          # calibrated at the organism level
    assert (naive["p"] < 0.05).mean() > 0.2            # pseudo-replication inflates
    assert (nested["n"] == 6).all() and (nested["unit"] == "organism").all()


def test_continuous_variable_slope():
    rng = np.random.default_rng(4)
    x = np.tile(np.arange(10, dtype=float), 3)
    meta = pd.DataFrame({"dose": x}, index=[f"s{i}" for i in range(len(x))])
    Y = rng.normal(size=(500, len(x)))
    Y[:50] += 0.5 * x
    tab = st.associate_matrix(pd.DataFrame(Y, columns=meta.index), meta, ["dose"], {"dose": "continuous"}, {},
                                 "feature")
    assert set(tab["test"]) == {"moderated_t"}
    assert (tab.iloc[:50]["padj"] < 0.05).all()
    assert np.allclose(tab.iloc[:50]["effect"].mean(), 0.5, atol=0.1)


def test_pairwise_contrasts_cover_every_pair_with_mean_difference_effects():
    df, meta = _null_design(np.random.default_rng(5), n_feat=200, groups=4)
    tab = st.associate_matrix(df, meta, ["trt"], {"trt": "categorical"}, {}, "feature")
    pairs = sorted(set(tab.loc[tab["test"] == "moderated_t", "contrast"]))
    assert pairs == ["t0 vs t1", "t0 vs t2", "t0 vs t3", "t1 vs t2", "t1 vs t3", "t2 vs t3"]
    assert set(tab.loc[tab["test"] == "moderated_F", "contrast"]) == {"omnibus"}
    t = tab[tab["contrast"] == "t1 vs t3"].set_index("entity_id")
    means = df.T.groupby(meta["trt"]).mean().T
    assert np.allclose(t["effect"], (means["t1"] - means["t3"]).loc[t.index], atol=1e-8)
    # binary variable: one comparison, no F test (it would equal t squared)
    two = meta.assign(trt=np.where(meta["trt"].isin(["t0", "t1"]), "a", "b"))
    tab2 = st.associate_matrix(df, two, ["trt"], {"trt": "categorical"}, {}, "feature")
    assert set(tab2["contrast"]) == {"a vs b"} and set(tab2["test"]) == {"moderated_t"}


def test_identical_variables_are_tested_once():
    meta = pd.DataFrame({"organism": list("aabbcc"), "lineage": list("XXXXYY"),
                         "group": ["a-X", "a-X", "b-X", "b-X", "c-Y", "c-Y"], "rep": [1, 2] * 3})
    assert md.identical_variables(meta, ["organism", "lineage", "group", "rep"]) == {"group": "organism"}


def test_knn_network_layout_and_neighbor_votes():
    from tools.explorer_bundle import network as net
    rng = np.random.default_rng(7)
    base = rng.normal(size=(2, 12))
    z = pd.DataFrame(np.vstack([base[0] + rng.normal(0, 0.1, (10, 12)), base[1] + rng.normal(0, 0.1, (10, 12))]),
                     index=[f"tx_a{i}" for i in range(10)] + [f"tx_b{i}" for i in range(10)])
    e = net.knn_edges(z, k=3)
    assert len(e) and not (e["source"] == e["target"]).any()
    same = e["source"].str[3] == e["target"].str[3]
    assert same.mean() > 0.95                                   # neighbours come from the same cluster
    xy = net.layout(z.index, e)
    assert np.isfinite(xy.to_numpy()).all() and len(xy) == 20
    # cluster a carries term T1 (except tx_a0, unannotated); cluster b carries T2
    ann = pd.DataFrame({"feature_id": [f"tx_a{i}" for i in range(1, 10)] + [f"tx_b{i}" for i in range(10)],
                        "ann_type": "kegg_module", "term_name": ""})
    ann["term_id"] = np.where(ann["feature_id"].str.startswith("tx_a"), "T1", "T2")
    types = [annot.AnnType("kegg_module", "KEGG module", "kegg_module", None, ";", "id_name", None)]
    votes = net.neighbor_votes(e, z.index, ann, types, pd.Series("tx", index=z.index))
    v = votes.set_index(["feature_id", "term_id"])
    assert set(votes["feature_id"]) == {"tx_a0"}               # only the unannotated feature is a candidate
    assert v.loc[("tx_a0", "T1"), "share"] == 1.0
    one = net.votes_from_neighbors(net.adjacency(e).query("feature_id == 'tx_a0'")["neighbor"], ann,
                                   net.vote_background(ann, z.index))
    assert one.iloc[0]["term_id"] == "T1" and np.isclose(one.iloc[0]["p"], v.loc[("tx_a0", "T1"), "p"])


def test_bh_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = np.random.default_rng(6).uniform(size=200) ** 3
    assert np.allclose(st.bh_adjust(p), multipletests(p, method="fdr_bh")[1])


# ---- annotations & metadata ---------------------------------------------------------------

def test_annotation_registry_parsing():
    df = pd.DataFrame({
        "feature_id": ["tx_a", "tx_b", "mx_c", "mx_d", "tx_e"],
        "kegg_pathway": ["path:map00010;path:map01100", "Unassigned", "path:map00010", "Unassigned", "Unassigned"],
        "kegg_pathway_name": ["Glycolysis;Metabolic pathways", "Unassigned", "Glycolysis", "Unassigned", "Unassigned"],
        "modelseed_pathway": ["rn1 (Pathway <i>one</i>);NOPATHWAY_rxn9", "Unassigned", "rn1 (Pathway one)", "Unassigned",
                              "Unassigned"],
        "tx_KEGG_BRITE_ACC": ["ko001;;ko002", "ko001", "Unassigned", "Unassigned", "ko001"],
        "tx_KEGG_BRITE": ["Photo>>>A>>>B;;Chaperones>>>C", "Photo>>>D", "Unassigned", "Unassigned", "Photo>>>E"],
        "tx_AAI_avg": ["0.9", "0.8", "Unassigned", "Unassigned", "0.7"],
        "mx_superclass": ["Unassigned", "Unassigned", "Lipids", "Lipids", "Unassigned"],
        "mx_smiles": ["Unassigned", "Unassigned", "CCO", "CCC", "Unassigned"],
    })
    types = {a.key: a for a in annot.detect_annotation_types(df, ["tx", "mx"])}
    assert set(types) == {"kegg_pathway", "modelseed_pathway", "tx_KEGG_BRITE", "mx_superclass"}
    long = annot.explode_all(df, list(types.values()))
    kp = long[long["ann_type"] == "kegg_pathway"]
    assert set(kp["term_id"]) == {"map00010", "map01100"}           # 'path:' prefix stripped
    ms = long[long["ann_type"] == "modelseed_pathway"]
    assert set(ms["term_id"]) == {"rn1"} and set(ms["term_name"]) == {"Pathway one"}  # NOPATHWAY dropped, tags removed
    br = long[long["ann_type"] == "tx_KEGG_BRITE"].set_index(["feature_id", "term_id"])["term_name"]
    assert br[("tx_a", "ko001")] == "Photo" and br[("tx_a", "ko002")] == "Chaperones"


def test_metadata_variable_inference():
    meta = pd.DataFrame({"tx": list("abcd"), "mx_hilic_pos": list("efgh"), "organism": ["x", "x", "y", "y"],
                         "group": ["g1", "g1", "g2", "g2"], "replicate": [1, 2, 1, 2], "unique_group": list("ABCD"),
                         "constant": [1, 1, 1, 1]})
    assert md.infer_metadata_variables(meta, ["tx", "mx"]) == ["organism", "group", "replicate"]
    assert md.infer_metadata_variables(meta, ["tx", "mx"], requested=["group"]) == ["group"]
    types = md.infer_variable_types(pd.DataFrame({"t": range(10), "c": list("ab") * 5}), ["t", "c"])
    assert types == {"t": "continuous", "c": "categorical"}


def test_enrichment_matches_pipeline_helper():
    """The bundle's vectorized test equals helpers.compute_enrichment_matrix (annotated background)."""
    hlp = pytest.importorskip("tools.helpers")
    rng = np.random.default_rng(7)
    feats = [f"tx_{i}" for i in range(300)]
    groups = pd.Series(rng.choice([f"group_{i}" for i in range(6)], 300), index=feats)
    long = pd.DataFrame([(f, "kegg_module", f"M{rng.integers(0, 15):02d}", "") for f in feats if rng.uniform() < 0.6],
                        columns=["feature_id", "ann_type", "term_id", "term_name"]).drop_duplicates(["feature_id", "term_id"])
    ours = agr.enrichment_table(long, groups, ["kegg_module"], min_term_size=1, min_group_size=1,
                                exclude_global_maps=False)
    ex = long.set_index("feature_id")[["term_id"]].rename(columns={"term_id": "pathway"})
    ex["group"] = groups.reindex(ex.index).to_numpy()
    unannot = pd.DataFrame({"pathway": "Unassigned", "group": groups.drop(ex.index.unique()).to_numpy()},
                           index=groups.drop(ex.index.unique()).index)
    overlap, fold, pval, padj = hlp.compute_enrichment_matrix(pd.concat([ex, unannot]), background="annotated")
    for r in ours.itertuples():
        assert r.k == overlap.at[r.term_id, r.group]
        assert np.isclose(r.p, pval.at[r.term_id, r.group])
        assert np.isclose(r.padj, padj.at[r.term_id, r.group])


# ---- end-to-end builds ----------------------------------------------------------------------

@pytest.fixture(scope="module", params=["hierarchical_clustering", "network_modules", "hdbscan"])
def built(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    a = make_run(str(root), request.param)
    out = build_explorer_bundle(a, explorer_config={"n_permutations": 99}, run_info=run_info(request.param),
                                force=True)
    return request.param, a, out


def _pq(out, name):
    return pd.read_parquet(os.path.join(out, f"{name}.parquet"))


def test_bundle_files_and_no_unnamed_columns(built):
    method, a, out = built
    m = json.load(open(os.path.join(out, "manifest.json")))
    assert m["grouping_method"] == method and m["schema_version"] == 2
    for name in ("samples", "features", "abundance_z", "abundance_log", "annotations_long", "groups", "terms",
                 "enrichment", "associations", "group_eigengenes", "term_eigengenes", "agreement", "feature_edges",
                 "discovery_votes"):
        df = _pq(out, name)
        assert not any(str(c).startswith("Unnamed") for c in df.columns), name
    assert bundle_is_current(a)
    assert m["nesting"] == {"lineage": "organism"}
    assert set(m["variables"]) == {"organism", "lineage", "group", "replicate"}
    assert m["variable_aliases"] == {"group": "organism"}         # same partition: tested once
    assert "group" not in set(_pq(out, "associations")["variable"])


def test_bundle_recovers_planted_signals(built):
    method, a, out = built
    assoc = _pq(out, "associations")
    org = assoc[assoc["variable"] == "organism"]
    assert len(set(org.loc[(org["entity_type"] == "feature") & (org["test"] == "moderated_t"), "contrast"])) == 15
    feat = org[(org["entity_type"] == "feature") & (org["contrast"] == "omnibus")].set_index("entity_id")
    planted = [f for f in feat.index if f.startswith("tx_g0") or f.startswith("mx_0")]
    assert (feat.loc[planted, "padj"] < 0.05).mean() > 0.9
    grp = org[(org["entity_type"] == "group") & (org["contrast"] == "omnibus")].set_index("entity_id")
    assert grp.loc["group_1", "padj"] < 0.05
    term = org[(org["entity_type"] == "term") & (org["entity_id"] == "kegg_module::M00001") &
               (org["contrast"] == "omnibus")]
    assert term["padj"].iloc[0] < 0.05                          # term eigengene follows group_1
    enr = _pq(out, "enrichment")
    m1 = enr[(enr["ann_type"] == "kegg_module") & (enr["term_id"] == "M00001")].set_index("group")
    assert m1.loc["group_1", "padj"] < 0.05
    rnd = enr[(enr["ann_type"] == "kegg_module") & (enr["term_id"] == "M00099")]
    assert (rnd["padj"] > 0.05).all()
    # p-values keep float64 precision (float32 would underflow strong effects to 0)
    assert assoc["p"].dtype == np.float64


def test_bundle_network_and_method_specific_structure(built):
    method, a, out = built
    m = json.load(open(os.path.join(out, "manifest.json")))
    native = json.load(open(os.path.join(out, "native_structure.json")))
    groups = _pq(out, "groups").set_index("group")
    feats = _pq(out, "features").set_index("feature_id")
    enr = _pq(out, "enrichment")
    grouped = feats.dropna(subset=["group"])
    assert grouped[["x", "y"]].notna().all().all()               # every grouped feature is placed
    assert feats.loc[feats["group"].isna(), "x"].isna().all()
    e = _pq(out, "feature_edges")
    pairs = {tuple(sorted(p)) for p in zip(e["source"], e["target"])}
    assert len(pairs) == len(e)                                  # undirected, no duplicates
    if method == "network_modules":
        assert m["network"]["kind"] == "real" and native is None
    else:
        assert m["network"]["kind"] == "knn" and len(e) >= len(grouped)
    if method == "hierarchical_clustering":
        assert native["kind"] == "dendrogram" and native["approximate"] is False
        assert sorted(native["leaf_groups"]) == sorted(groups.index)
    if method == "hdbscan":
        assert groups.loc["noise", "is_noise"]
        assert "noise" not in set(enr["group"])                  # noise is never tested as a group
        assert int(m["n_noise"]) == 25


def test_bundle_rebuild_skipped_when_current(built, caplog):
    method, a, out = built
    before = os.path.getmtime(os.path.join(out, "manifest.json"))
    build_explorer_bundle(a, run_info=run_info(method))
    assert os.path.getmtime(os.path.join(out, "manifest.json")) == before
