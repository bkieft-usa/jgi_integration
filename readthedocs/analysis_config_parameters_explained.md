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

Configuration path: `analysis.pathway_enrichment`. The `enrich` stage (`analysis.enrich_all()`) runs after feature grouping. It tests every data-driven group against **every annotation type** in the feature annotation table: KEGG pathway, module and reaction; ModelSEED pathway; and each dataset's term columns (KO, COG category/pathway/function, KEGG BRITE, EC, ClassyFire and NPClassifier classes, ModelSEED reactions). The test is a one-sided hypergeometric test with Benjamini–Hochberg correction within each annotation type. Results for each type go to `enrichment/<annotation_type>/`.

| Key | Description |
|-----|-------------|
| `background` | `annotated` (default, recommended): the universe for each annotation type is the grouped features with at least one annotation of that type, and group sizes count annotated members only. Unannotated features carry no information about a term; including them makes enrichment depend on how well each group happens to be annotated. `all`: every grouped feature. |
| `exclude_global_maps` | Excludes KEGG global/overview maps (`map011xx`/`map012xx`, e.g. `map01100` Metabolic pathways) when true. They contain most metabolic features and would otherwise dominate. Only affects `kegg_pathway`. |
| `bipartite_only` | When true, cross-omic annotation types (KEGG, ModelSEED) keep only pathways and groups with features from every data type. Single-data-type annotation types (e.g. COG for transcripts, compound classes for metabolites) ignore this setting, because it would remove all of their terms. |
| `min_features_per_pathway` | Minimum annotated feature count for a term to be tested. |
| `min_features_per_group` | Minimum feature count for a group to be tested. |
| `alpha` | FDR threshold. Heatmap rows show the groups with at least one significant term among the displayed terms. |
| `fdr_method` | Optional correction method for `statsmodels.stats.multitest.multipletests` (default `fdr_bh`). |
| `top_n` | Number of terms shown in each heatmap when `selected_pathways` is not given. Every term is tested and saved regardless. |
| `selected_pathways` | Optional `{annotation_type: [term IDs]}` mapping that fixes the heatmap columns for that type. A plain list applies to every type. |
| `rank_by` | How terms are chosen for display: `feature_count` (default), `summed_significance` or `mean_abs_value`. Any other value is an error. |
| `trend_sort_by` | Metadata column used to order the trend-track categories. |
| `trend_collapse_by` | Metadata column used to pool samples for the trend track; null leaves one column per sample. The trend track shows, for each group or term, the share of its members' summed normalized abundance in each category (rows sum to 1; data types weighted equally). |
| `trend_agg` | How replicates' summed abundances are pooled within a category: `mean` or `median`. |
| `trend_cmap` | Sequential colormap for trend tracks (default `Purples`). |
| `label_max_chars` | Maximum length of term names in heatmap labels (default 60). |

HDBSCAN `noise` and WGCNA `grey` features stay in the background but are never tested as a group.

The interactive explorer computes its own enrichment table with less strict display filters (every group and term above the explorer's minimum sizes; no `bipartite_only`). The test and background are identical, so p-values match whenever the filters match. Explorer settings live in an optional, unhashed `explorer.yml` (see `explorer.md`).

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
    background: annotated
    exclude_global_maps: true
    bipartite_only: true
    min_features_per_pathway: 2
    min_features_per_group: 2
    alpha: 0.05
    top_n: 75
    rank_by: feature_count
    trend_sort_by: timepoint
    trend_collapse_by: timepoint
    trend_agg: median
```
