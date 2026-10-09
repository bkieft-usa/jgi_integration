# Analysis Configuration Reference

## Overview

This document describes the options currently read from `analysis.yml`. The active configuration controls replicate representation, feature selection, feature grouping, and pathway enrichment. The workflow creates a separate analysis output directory for each analysis configuration, while reusing the data-processing results.

## Table of Contents
- [Configuration](#configuration)
- [Replicate Pairing](#replicate-pairing)
- [Feature Selection](#feature-selection)
- [Feature Grouping](#feature-grouping)
- [Pathway Enrichment](#pathway-enrichment)
- [Configuration Example](#configuration-example)

## Configuration

Analysis options are nested beneath the `analysis` key in `analysis.yml`. `replicate_pairing` is read directly from this block; `feature_selection`, `feature_grouping`, and `pathway_enrichment` are also read directly from it. These keys are not nested under an `analysis_parameters` key.

Changing analysis configuration creates an analysis hash and output directory under the current data-processing directory. Prior results are kept in their existing directories. Use the saved configuration associated with a run to reproduce its settings.

## Replicate Pairing

Configuration path: `analysis.replicate_pairing` (required; accepted values are `paired` and `unpaired`).

| Value | Integrated matrix columns | Representation |
|-------|---------------------------|----------------|
| `paired` | Shared sample names | Per-feature z-scores across samples. |
| `unpaired` | Shared `group` labels | Per-condition median abundance on the log scale, then per-feature z-scores across conditions. |

For paired data, samples are aligned across data types using the link table. For unpaired data, replicate measurements are summarized within each group. The selected representation is used when integrating the data matrices.

## Feature Selection

Configuration path: `analysis.feature_selection`.

The supported methods are `variance` and `feature_list`. A missing method passes all integrated features through. Other method names, including `glm`, `kruskalwallis`, `lasso`, `random_forest`, and `mutual_info`, are not implemented by the current feature-selection function.

| Key under `feature_selection` | Description |
|-------------------------------|-------------|
| `method` | `variance` selects features using mean-adjusted variance from each dataset's replicate-filtered log-scale data. `feature_list` selects IDs from a text file. |
| `params.top_n` | Maximum total feature budget for variance selection across datasets. The budget is distributed among datasets, ranking features within each dataset. |
| `params.max_features` | Hard upper bound applied to the selection budget. The current example sets it equal to `top_n`. |
| `params.block_scale` | When true, scales selected data-type blocks so each contributes equal total variance. Used by variance selection. |
| `params.feature_list_file` | For `feature_list`, path to a text file containing one feature ID per line. It must be readable from the workflow's current working directory. |

The integrated matrix is z-scored, so variance selection uses each dataset's `replicate_filtered_data` on the log scale rather than ranking the z-scored matrix. The example configuration uses `top_n: 12000`, `max_features: 12000`, and `block_scale: true`.

## Feature Grouping

Configuration path: `analysis.feature_grouping`.

`method` selects the grouping backend. The active implementation supports `network_modules`, `hierarchical_clustering`, `hdbscan`, `nmf`, `leiden_knn`, and `wgcna`. Method-specific options are placed under `feature_grouping.params`.

| Method | Parameters used by the current implementation |
|--------|-----------------------------------------------|
| `network_modules` | `corr_method`, `corr_cutoff`, `keep_negative`, `block_size`, `cores`, `corr_mode`, `submodule_mode`, `network_layout`, `show_network_plot`. Correlation methods include `pearson`, `spearman`, `cosine`, `centered_cosine`, `bicor`, `dcor`, and `sparse_partial`. `corr_mode` accepts `bipartite`, `full`, or a dataset prefix. |
| `hierarchical_clustering` | `distance_metric`, `linkage_method`, `height_cutoff`. |
| `hdbscan` | `metric`, `min_cluster_size`, `min_samples`. |
| `nmf` | `k_min`, `k_max`, `n_runs`, `max_iter`. |
| `leiden_knn` | `n_neighbors`, `resolution_min`, `resolution_max`, `resolution_steps`. |
| `wgcna` | `power` or `power_range`, `r2_threshold`, `signed`, `min_module_size`, `merge_cut_height`, `deep_split`. |

For `network_modules`, feature correlations are calculated before Louvain/Leiden submodule grouping. Other grouping methods do not require that correlation step. `show_network_plot` controls inline network display for the network grouping method.

## Pathway Enrichment

Configuration path: `analysis.pathway_enrichment`. The notebook calls `analysis.compare_groups_to_pathways()` after feature grouping. This compares feature groups with pathway annotations in the annotation table using hypergeometric enrichment and multiple-testing correction.

| Key | Description |
|-----|-------------|
| `pathway_col` | Annotation-table column containing pathway labels, such as `modelseed_pathway` or `kegg_pathway`. |
| `exclude_nopathway` | Excludes pathway labels beginning with `NOPATHWAY_` when true. |
| `bipartite_only` | When true, retains annotated pathways represented in every included data type. |
| `min_features_per_pathway` | Minimum annotated feature count for a pathway to be tested. |
| `min_features_per_group` | Minimum feature count for a group to be retained. |
| `alpha` | Significance level passed to the multiple-testing routine. Corrected p-value matrices are returned; entries are not filtered by this value. The implementation uses Benjamini-Hochberg when `fdr_method` is not supplied. |
| `fdr_method` | Optional multiple-testing correction method passed to `statsmodels.stats.multitest.multipletests`; omitted in the example, so the code uses `fdr_bh`. |
| `top_n` | Number of pathways shown when `selected_pathways` is not specified. |
| `selected_pathways` | Optional explicit pathway list; when present it takes precedence over ranking and `top_n`. |
| `rank_by` | `summed_significance` ranks pathways by adjusted significance; other values currently rank by feature count. |
| `trend_sort_by` | Metadata column used to order the trend-track categories. |
| `trend_collapse_by` | Metadata column used to pool samples for the trend track; a false/null value leaves sample columns uncollapsed. |
| `trend_agg` | Aggregation used for pooled trend values, such as `mean` or `median`. |
| `trend_cmap` | Colormap used for pathway/group trend tracks. |

The current example also contains `show_only_sig`, `enrichment_value`, and `trend_dispersion`. These keys are not currently read by the pathway-enrichment implementation and therefore do not change its results.

## Configuration Example

This example follows the current schema. It omits method-specific grouping parameters not needed for the selected `hierarchical_clustering` backend.

```yaml
analysis:
  replicate_pairing: paired
  feature_grouping:
    method: hierarchical_clustering
    params:
      distance_metric: correlation
      linkage_method: weighted
      height_cutoff: 0.5
  feature_selection:
    method: variance
    params:
      top_n: 12000
      max_features: 12000
      block_scale: true
  pathway_enrichment:
    pathway_col: modelseed_pathway
    exclude_nopathway: true
    bipartite_only: true
    min_features_per_pathway: 2
    min_features_per_group: 2
    alpha: 0.05
    top_n: 75
    rank_by: n_features
    trend_sort_by: timepoint
    trend_collapse_by: timepoint
    trend_agg: median
```
