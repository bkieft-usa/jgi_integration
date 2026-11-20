from types import SimpleNamespace

import pandas as pd

from tools import helpers as hlp
from tools.objects import Analysis


def test_multichromatography_mx_loading_and_linking(tmp_path):
    samples = {
        ("HILIC", "negative"): ("hilic_neg_sample", 10),
        ("HILIC", "positive"): ("hilic_pos_sample", 20),
        ("C18", "negative"): ("c18_neg_sample", 30),
        ("C18", "positive"): ("c18_pos_sample", 40),
    }
    for (chrom, polarity), (sample, value) in samples.items():
        chrom_dir = tmp_path / chrom
        chrom_dir.mkdir(exist_ok=True)
        pd.DataFrame({
            "Scan": [7],
            "row m/z": [100.0],
            "row retention time": [5.0],
            f"{sample}.mzML Peak height": [value],
        }).to_csv(
            chrom_dir / f"study_{polarity}_peak-height-filtered-3x-exctrl.csv",
            index=False,
        )

    raw_data = hlp.get_mx_data(
        input_dir=str(tmp_path),
        output_dir="",
        output_filename="raw_data.csv",
        chromatography=["HILIC", "C18"],
        polarity=["POS", "NEG"],
    )
    expected_ids = {
        "mx_7_hilic_neg",
        "mx_7_hilic_pos",
        "mx_7_c18_neg",
        "mx_7_c18_pos",
    }
    assert set(raw_data["CompoundID"]) == expected_ids

    link_table = pd.DataFrame({
        "unique_group": ["shared_sample"],
        **{
            f"mx_{chrom.lower()}_{'pos' if polarity == 'positive' else 'neg'}": [sample]
            for (chrom, polarity), (sample, _) in samples.items()
        },
    })
    dataset = SimpleNamespace(
        dataset_name="mx",
        raw_data=raw_data,
        output_dir="",
        _linked_data_filename="linked_data.csv",
        chromatography=["HILIC", "C18"],
        polarity=["POS", "NEG"],
    )
    linked = hlp.link_data_across_datasets(
        [dataset], link_table, overlap_only=False
    )["mx"]

    assert set(linked.index) == expected_ids
    assert linked.columns.tolist() == ["shared_sample"]
    for (chrom, polarity), (_, value) in samples.items():
        suffix = "pos" if polarity == "positive" else "neg"
        assert linked.loc[f"mx_7_{chrom.lower()}_{suffix}", "shared_sample"] == value


def test_mx_loading_accepts_single_string_config_values(tmp_path):
    chrom_dir = tmp_path / "HILIC"
    chrom_dir.mkdir()
    pd.DataFrame({
        "Scan": [7],
        "row m/z": [100.0],
        "row retention time": [5.0],
        "sample.mzML Peak height": [10],
    }).to_csv(
        chrom_dir / "study_negative_peak-height-filtered-3x-exctrl.csv",
        index=False,
    )

    raw_data = hlp.get_mx_data(
        input_dir=str(tmp_path),
        output_dir="",
        output_filename="raw_data.csv",
        chromatography="HILIC",
        polarity="NEG",
    )

    assert raw_data["CompoundID"].tolist() == ["mx_7"]


def test_multichromatography_skips_missing_link_table_columns(tmp_path):
    chrom_dir = tmp_path / "HILIC"
    chrom_dir.mkdir()
    pd.DataFrame({
        "Scan": [7],
        "row m/z": [100.0],
        "row retention time": [5.0],
        "sample.mzML Peak height": [10],
    }).to_csv(
        chrom_dir / "study_negative_peak-height-filtered-3x-exctrl.csv",
        index=False,
    )
    pd.DataFrame({
        "Scan": [8],
        "row m/z": [101.0],
        "row retention time": [6.0],
        "unlinked_sample.mzML Peak height": [20],
    }).to_csv(
        chrom_dir / "study_positive_peak-height-filtered-3x-exctrl.csv",
        index=False,
    )
    raw_data = hlp.get_mx_data(
        input_dir=str(tmp_path),
        output_dir="",
        output_filename="raw_data.csv",
        chromatography=["HILIC"],
        polarity=["POS", "NEG"],
    )
    dataset = SimpleNamespace(
        dataset_name="mx",
        raw_data=raw_data,
        output_dir="",
        _linked_data_filename="linked_data.csv",
        chromatography=["HILIC"],
        polarity=["POS", "NEG"],
    )
    linked = hlp.link_data_across_datasets(
        [dataset],
        pd.DataFrame({
            "unique_group": ["shared_sample"],
            "mx_hilic_neg": ["sample"],
        }),
        overlap_only=False,
    )["mx"]

    assert linked.index.tolist() == ["mx_7_hilic_neg"]
    assert linked.columns.tolist() == ["shared_sample"]
    assert linked.loc["mx_7_hilic_neg", "shared_sample"] == 10


def test_multichromatography_metadata_loads_all_configured_folders(tmp_path):
    for chrom, sample in (("HILIC", "hilic_sample"), ("C18", "c18_sample")):
        chrom_dir = tmp_path / chrom
        chrom_dir.mkdir()
        pd.DataFrame({
            "filename": [f"{sample}.mzML"],
            "ATTRIBUTE_sampletype": ["sample"],
        }).to_csv(chrom_dir / "study_metadata.tab", sep="\t", index=False)

    metadata = hlp.load_mx_metadata(
        input_dir=str(tmp_path),
        chromatography=["HILIC", "C18"],
        output_dir="",
    )

    assert set(metadata["file"]) == {"hilic_sample", "c18_sample"}
    assert metadata["full_sample_metadata"].tolist() == ["sample", "sample"]


def test_multichromatography_annotation_ids_include_chromatography_and_polarity(
    tmp_path, monkeypatch
):
    compound_dir = tmp_path / "HILIC" / "library-results"
    compound_dir.mkdir(parents=True)
    pd.DataFrame({
        "#Scan#": [7],
        "InChIKey": ["TEST-KEY"],
        "Compound_Name": ["test compound"],
    }).to_csv(compound_dir / "negative_library-results.tsv", sep="\t", index=False)
    monkeypatch.setattr(hlp, "_build_inchikey_to_cpd", lambda _: {})
    monkeypatch.setattr(hlp, "_build_inchikey_prefix_to_cpd", lambda _: {})
    monkeypatch.setattr(hlp, "_build_fuzzy_name_candidates", lambda _: ([], []))
    monkeypatch.setattr(hlp, "resolve_modelseed_ids", lambda *args: None)

    annotation = hlp.generate_mx_annotation_table(
        raw_data=pd.DataFrame({"CompoundID": ["mx_7_hilic_neg"]}),
        dataset_raw_dir=str(tmp_path),
        polarity=["POS", "NEG"],
        chromatography=["HILIC"],
        output_dir=str(tmp_path / "output"),
        output_filename="annotation_table.csv",
        inchikey_col="InChIKey",
    )

    assert annotation.index.tolist() == ["mx_7_hilic_neg"]
    assert annotation.loc["mx_7_hilic_neg", "InChiKey"] == "TEST-KEY"


def _analysis_for_metadata(pairing, link_table, data_columns):
    analysis = Analysis.__new__(Analysis)
    analysis._integrated_metadata_filename = "integrated_metadata.csv"
    analysis._integrated_data_filename = "integrated_data.csv"
    analysis._integration_mode = (
        "sample_resolution" if pairing == "paired" else "condition_resolution"
    )
    analysis._cache = {
        "integrated_metadata": link_table.copy(),
        "integrated_data": pd.DataFrame([[1] * len(data_columns)], columns=data_columns),
    }
    return analysis


def test_paired_metadata_aligns_by_unique_group():
    link_table = pd.DataFrame({
        "unique_group": ["sample_1", "sample_2"],
        "group": ["control", "treated"],
        "timepoint": ["day0", "day7"],
    })
    analysis = _analysis_for_metadata("paired", link_table, ["sample_2"])

    result = analysis._metadata_for_integrated_data()

    assert result.index.tolist() == ["sample_2"]
    assert result.index.name is None
    assert result.loc["sample_2", "group"] == "treated"
    assert result.loc["sample_2", "unique_group"] == "sample_2"
    pca_rows = pd.DataFrame({"unique_group": ["sample_2"]})
    assert pca_rows.merge(result, on="unique_group").shape[0] == 1


def test_unpaired_metadata_aggregates_by_group_and_aligns_conditions():
    link_table = pd.DataFrame({
        "unique_group": ["sample_1", "sample_2", "sample_3"],
        "group": ["control", "control", "treated"],
        "timepoint": ["day0", "day7", "day7"],
    })
    analysis = _analysis_for_metadata("unpaired", link_table, ["control", "treated"])

    result = analysis._metadata_for_integrated_data()

    assert result.index.tolist() == ["control", "treated"]
    assert pd.isna(result.loc["control", "timepoint"])
    assert result.loc["treated", "group"] == "treated"
    assert result.loc["treated", "unique_group"] == "treated"
    assert result.loc["treated", "timepoint"] == "day7"