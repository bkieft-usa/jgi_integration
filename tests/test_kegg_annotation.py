import pandas as pd
import pytest

from tools import helpers as hlp


def test_parse_kegg_gene_ids_handles_prefixes_and_delimiters():
    # ';;' is always split (row-collapse separator), plus the configured delimiter
    assert hlp._parse_kegg_gene_ids("EC:7.1.2.2;;EC:7.2.2.1", ";;", "ec") == ["7.1.2.2", "7.2.2.1"]
    assert hlp._parse_kegg_gene_ids("K07498;;K18320", ",", "ko") == ["K07498", "K18320"]
    assert hlp._parse_kegg_gene_ids("ko:K00001,ko:K00002", ",", "ko") == ["K00001", "K00002"]
    # partial ECs, sentinels and junk are dropped; duplicates removed
    assert hlp._parse_kegg_gene_ids("EC:7.3.2.7;;EC:7.3.-.-;;EC:7.3.2.7", ";;", "ec") == ["7.3.2.7"]
    assert hlp._parse_kegg_gene_ids("Unassigned", ",", "ko") == []
    assert hlp._parse_kegg_gene_ids(None, ",", "ec") == []
    assert hlp._parse_kegg_gene_ids("EC:2.7.7.n1", ";;", "ec") == ["2.7.7.n1"]


def test_to_kegg_map_pathway_normalizes_namespaces():
    assert hlp._to_kegg_map_pathway("path:ko00860") == "path:map00860"
    assert hlp._to_kegg_map_pathway("path:ec00860") == "path:map00860"
    assert hlp._to_kegg_map_pathway("map00860") == "path:map00860"
    assert hlp._to_kegg_map_pathway("garbage") is None
    assert hlp._dedupe_map_pathways(
        ["path:map00860", "path:ko00860", "path:map01100"]
    ) == ["path:map00860", "path:map01100"]


def _mock_kegg(monkeypatch, link_tables, names, inchikey_to_cpd):
    """Patch all KEGG/MW network calls; returns {(target, prefix): ids requested}."""
    requested = {}

    def fake_link_lookup(ids, prefix, target="pathway", delay=0.2, batch_size=10, cache_path=None):
        requested[(target, prefix)] = list(ids)
        table = link_tables.get(target, {}).get(prefix, {})
        return {i: table.get(i, []) for i in ids}

    monkeypatch.setattr(hlp, "build_kegg_link_lookup", fake_link_lookup)
    monkeypatch.setattr(hlp, "build_inchikey_kegg_lookup",
                        lambda keys, **kw: {k: inchikey_to_cpd.get(k) for k in keys})
    monkeypatch.setattr(hlp, "fetch_kegg_list_cached",
                        lambda db, cache_path=None, **kw: names.get(db, {}))
    return requested


LINKS = {
    "pathway": {
        "ec:": {"6.6.1.1": ["path:map00860", "path:ec00860"]},
        "ko:": {
            "K03403": ["path:map00860", "path:ko00860", "path:ko01100"],
            "K07498": [],
            "K18320": ["path:ko00010"],
        },
        "cpd:": {"C00022": ["path:map00860"], "C00002": ["path:map00230"]},
    },
    "module": {
        "ko:": {"K03403": ["md:M00121"], "K18320": ["md:M00001"]},
        "cpd:": {"C00022": ["md:M00001", "md:M00002"], "C00002": ["md:M00049"]},
    },
    "reaction": {
        "ec:": {"6.6.1.1": ["rn:R03877"]},
        "ko:": {"K03403": ["rn:R03877"], "K18320": ["rn:R00200"]},
        "cpd:": {"C00022": ["rn:R00200"], "C00002": ["rn:R00086"]},
    },
}
NAMES = {
    "pathway": {"map00860": "Porphyrin metabolism", "map01100": "Metabolic pathways",
                "map00010": "Glycolysis / Gluconeogenesis", "map00230": "Purine metabolism"},
    "module": {"M00001": "Glycolysis (Embden-Meyerhof pathway)", "M00002": "Glycolysis, core module",
               "M00121": "Heme biosynthesis"},
    "reaction": {"R03877": "magnesium chelatase", "R00200": "pyruvate kinase"},
}


def test_kegg_annotation_assigns_pathway_module_and_reaction_levels(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "tx_ECnum": ["EC:6.6.1.1", None, "EC:1.1.1.-", None, None],
            "tx_KOfam_ACC": ["K03403", "K07498;;K18320", None, None, None],
            "mx_InChiKey": [None, None, None, "PYRUVATE-KEY", "ATP-KEY"],
        },
        index=["tx_g1", "tx_g2", "tx_g3", "mx_pyr", "mx_atp"],
    )
    requested = _mock_kegg(monkeypatch, LINKS, NAMES,
                           {"PYRUVATE-KEY": "C00022", "ATP-KEY": "C00002"})

    out = hlp.add_kegg_pathway_annotations(
        df,
        cache_dir=tmp_path,
        inchikey_col="mx_InChiKey",
        ec_cols={"tx_ECnum": ";;"},
        ko_cols={"tx_KOfam_ACC": ","},
    )

    # Clean IDs are sent to KEGG (no 'EC:' prefix, no partial ECs, ';;' split)
    assert requested[("pathway", "ec:")] == ["6.6.1.1"]
    assert requested[("pathway", "ko:")] == ["K03403", "K07498", "K18320"]

    # --- pathway level: both data types, one map##### namespace, global maps kept
    assert out.loc["tx_g1", "kegg_pathway"] == "path:map00860;path:map01100"
    assert out.loc["tx_g2", "kegg_pathway"] == "path:map00010"
    assert pd.isna(out.loc["tx_g3", "kegg_pathway"])
    assert out.loc["mx_pyr", "kegg_pathway"] == "path:map00860"
    assert out.loc["tx_g1", "kegg_pathway_name"] == "Porphyrin metabolism;Metabolic pathways"
    # currency compounds still get pathways
    assert out.loc["mx_atp", "kegg_pathway"] == "path:map00230"

    # --- module level: bare M##### IDs, names aligned by position
    assert out.loc["tx_g1", "kegg_module"] == "M00121"
    assert out.loc["tx_g2", "kegg_module"] == "M00001"
    assert out.loc["mx_pyr", "kegg_module"] == "M00001;M00002"
    assert out.loc["mx_pyr", "kegg_module_name"] == "Glycolysis (Embden-Meyerhof pathway);Glycolysis, core module"

    # --- reaction level: EC + KO deduplicated, shared with the metabolite
    assert out.loc["tx_g1", "kegg_reaction"] == "R03877"
    assert out.loc["tx_g2", "kegg_reaction"] == "R00200"
    assert out.loc["mx_pyr", "kegg_reaction"] == "R00200"
    assert out.loc["mx_pyr", "kegg_reaction_name"] == "pyruvate kinase"

    # currency compounds (ATP) are not linked at module/reaction level
    assert "C00002" not in requested.get(("module", "cpd:"), [])
    assert "C00002" not in requested.get(("reaction", "cpd:"), [])
    assert pd.isna(out.loc["mx_atp", "kegg_module"])
    assert pd.isna(out.loc["mx_atp", "kegg_reaction"])


def test_prepare_universe_can_exclude_global_maps():
    node_table = pd.DataFrame(
        {
            "group": ["g1", "g1", "g2", "g2"],
            "kegg_pathway": [
                "path:map01100;path:map00010",
                "path:map01100;path:map00010",
                "path:map01110;path:map00230",
                "path:map01240;path:map00230",
            ],
        },
        index=["tx_a", "mx_b", "tx_c", "mx_d"],
    )
    pe = {"min_features_per_pathway": 1, "min_features_per_group": 1, "exclude_global_maps": True}
    kept = set(hlp.prepare_universe(node_table, "kegg_pathway", "group", pe)["pathway"])
    assert kept == {"path:map00010", "path:map00230"}

    pe["exclude_global_maps"] = False
    kept = set(hlp.prepare_universe(node_table, "kegg_pathway", "group", pe)["pathway"])
    assert "path:map01100" in kept


def test_prepare_universe_bipartite_only_filters_pathways_and_groups():
    node_table = pd.DataFrame(
        {
            # g1 spans tx+mx; g2 is tx-only; g3 spans tx+mx but mx member is unannotated
            "group": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "kegg_module": ["M1", "M1", "M2", "M1", "M2", None],
        },
        index=["tx_a", "mx_b", "tx_c", "tx_d", "tx_e", "mx_f"],
    )
    pe = {"min_features_per_pathway": 1, "min_features_per_group": 1, "bipartite_only": True}
    out = hlp.prepare_universe(node_table, "kegg_module", "group", pe)
    assert set(out["group"]) == {"g1", "g3"}
    assert set(out.loc[out["pathway"] != "Unassigned", "pathway"]) == {"M1"}

    pe["bipartite_only"] = False
    out = hlp.prepare_universe(node_table, "kegg_module", "group", pe)
    assert set(out["group"]) == {"g1", "g2", "g3"}
    assert set(out.loc[out["pathway"] != "Unassigned", "pathway"]) == {"M1", "M2"}


def test_get_trend_values_returns_category_share_of_summed_abundance():
    abundance = pd.DataFrame(
        {"s1": [1.0, 3.0, 100.0], "s2": [3.0, 1.0, 300.0], "s3": [0.0, 0.0, 600.0], "s4": [0.0, 0.0, 0.0]},
        index=["tx_a", "tx_b", "mx_c"],
    )
    meta = pd.DataFrame(
        {"cond": ["A", "A", "B", "B"], "order": [2, 2, 1, 1]}, index=["s1", "s2", "s3", "s4"]
    )
    mapping = pd.DataFrame({"pathway": ["P1", "P1", "P1"]}, index=["tx_a", "tx_b", "mx_c"])

    out = hlp.get_trend_values(abundance, mapping, "pathway", meta, "order", "cond", agg="mean")
    # tx: per-sample sums 4,4,0,0 -> means A=4, B=0 -> shares A=1, B=0
    # mx: per-sample sums 100,300,600,0 -> means A=200, B=300 -> shares A=0.4, B=0.6
    # averaged over data types: A=0.7, B=0.3; columns ordered by `order` (B first)
    assert list(out.columns) == ["B", "A"]
    assert out.loc["P1", "A"] == pytest.approx(0.7)
    assert out.loc["P1", "B"] == pytest.approx(0.3)


def test_log_annotation_coverage_reports_fraction_per_data_type():
    df = pd.DataFrame(
        {
            "kegg_module": ["M00001", "Unassigned", None, "M00002;M00003"],
            "modelseed_pathway": ["NOPATHWAY_rxn1", "rn00010 (Glycolysis)", None, None],
        },
        index=["tx_a", "tx_b", "mx_c", "mx_d"],
    )
    table = hlp.log_annotation_coverage(df, cols=["kegg_module", "modelseed_pathway"])
    assert table.loc["kegg_module", "tx (n=2)"] == "1 (50.0%)"
    assert table.loc["kegg_module", "mx (n=2)"] == "1 (50.0%)"
    # NOPATHWAY_ placeholders do not count as annotated
    assert table.loc["modelseed_pathway", "tx (n=2)"] == "1 (50.0%)"
    assert table.loc["modelseed_pathway", "mx (n=2)"] == "0 (0.0%)"


def test_term_labels_use_parallel_name_column():
    node_table = pd.DataFrame(
        {
            "kegg_module": ["M00001;M00002", "M00002", "Unassigned"],
            "kegg_module_name": ["Glycolysis (Embden-Meyerhof pathway);Glycolysis, core module",
                                 "Glycolysis, core module", "Unassigned"],
            "modelseed_pathway": ["rn00010 (Glycolysis)", None, None],
        },
        index=["tx_a", "mx_b", "tx_c"],
    )
    labels = hlp.get_term_labels(node_table, "kegg_module", ["M00001", "M00002", "M99999"], max_name_len=12)
    assert labels == {"M00001": "M00001 (Glycolysis…)", "M00002": "M00002 (Glycolysis,…)", "M99999": "M99999"}
    # ModelSEED IDs already carry names and are left unchanged
    assert hlp.get_term_labels(node_table, "modelseed_pathway", ["rn00010 (Glycolysis)"]) == {
        "rn00010 (Glycolysis)": "rn00010 (Glycolysis)"
    }
