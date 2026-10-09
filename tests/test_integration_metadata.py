import pandas as pd

from tools.objects import Analysis


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