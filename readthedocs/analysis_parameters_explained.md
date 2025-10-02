# Analysis Configuration Reference  

---  

## Overview

This document describes the practical effect of each option in the **analysis** section of the integration workflow configuration file (`/input_data/config/project_config.yml`), including the analysis tag, feature selection, network analysis, and MOFA modeling. Each section lists available methods, their options, and default values, and at the end there is an example of a full analysis workflow.

---  

## Table of Contents
1. [Tagging](#tagging)  
2. [Feature Selection](#feature-selection)  
   - [Shared options](#feature-selection-shared-options)  
   - [Methods](#feature-selection-methods)  
3. [Feature Correlation](#feature-correlation)  
4. [Network Analysis](#network-analysis)  
5. [MOFA Modeling](#mofa-modeling)  
6. [Configuration Example](#configuration-example)  

---

## Tagging <a id="tagging"></a>

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `analysis.data_analysis_tag` | string | `"0"` | Tag used to create a sub‑folder `Analysis--<TAG>` under the data‑processing output. Helps keep runs with different settings separate. **Note**: if the folder already exists and overwriting is disabled the workflow aborts. |

---  

## Feature Selection <a id="feature-selection"></a>

Configuration path: `analysis.analysis_parameters.feature_selection`

### Shared options <a id="feature-selection-shared-options"></a>

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `selected_method` | string | `"variance"` | Method used to pick informative features. Allowed values: `variance`, `glm`, `kruskalwallis`, `feature_list`, `lasso`, `random_forest`, `mutual_info`, `none`. See table below for details. |
| `max_features` | integer | `5000` | Upper bound on the number of features kept after selection (must be ≤ 10 000 for performance). |

### Methods <a id="feature-selection-methods"></a>

Below each method’s *method‑specific* parameters are shown. All method blocks are optional **unless** that method is chosen via `selected_method`.

| Method | Config block | Parameter | Type | Default | Description |
|--------|----------------------------|-----------|------|---------|-------------|
| **variance** | `variance` | `top_n` | integer > 0 | `5000` | Number of highest‑variance features to retain. |
| **glm** | `glm` | `metadata_category` | string | – | Column in `user_settings.variable_list` used for group comparison. |
| | | `metadata_category_reference` | string | – | Reference group within the chosen metadata column. |
| | | `significance_level` | float ∈ (0, 1] | `0.05` | FDR‑adjusted p‑value cutoff. |
| | | `log2fc_cutoff` | float > 0 | `0.5` | Minimum absolute log₂ fold‑change. |
| **kruskalwallis** | `kruskalwallis` | `metadata_category` | string | – | Same meaning as for *glm*. |
| | | `significance_level` | float ∈ (0, 1] | `0.05` | FDR‑adjusted p‑value cutoff. |
| | | `log2fc_cutoff` | float > 0 | `0.5` | Minimum absolute log₂ fold‑change. |
| **feature_list** | `feature_list` | `feature_list_file` | string (path) | – | File containing one feature ID per line. Must reside in the analysis output directory (`…/Analysis--<TAG>/`). |
| **lasso** | `lasso` | `metadata_category` | string | – | Continuous target variable for Lasso regression. |
| **random_forest** | `random_forest` | `metadata_category` | string | – | Target variable (continuous → regression, categorical → classification). |
| **mutual_info** | `mutual_info` | `metadata_category` | string | – | Target variable (continuous or categorical). |
| **none** | — | — | — | — | No selection; all features passed through (subject to `max_features`). |

---  

## Feature Correlation <a id="feature-correlation"></a>  

Configuration path: `analysis.analysis_parameters.correlation`

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `corr_method` | string | `"pearson"` | Correlation metric. Options: `pearson`, `spearman`, `cosine`, `centered_cosine`, `bicor`. |
| `corr_cutoff` | float ∈ [0, 1] | `0.5` | Minimum absolute correlation value for an edge to be kept. |
| `keep_negative` | boolean | `false` | If `true`, include negative correlations whose absolute value ≥ `corr_cutoff`. |
| `block_size` | integer > 0 | `500` | Number of features processed per block (affects memory & speed). |
| `n_jobs` | integer | `1` | Parallel workers for block‑wise calculation (`-1` → use all cores). |
| `only_bipartite` | boolean | `true` | Restrict calculations to cross‑type feature pairs (e.g., transcript ↔ metabolite). |

---  

## Network Analysis <a id="network-analysis"></a>  

Configuration path: `analysis.analysis_parameters.networking`

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `network_mode` | string | `"bipartite"` | Edge topology. Options: `bipartite` (cross‑type only) or `full` (all pairwise). `full` is not currently recommended due to computational constraints. |
| `submodule_mode` | string | `"louvain"` | Sub‑module extraction method. Options: `none`, `subgraphs`, `louvain`, `leiden`, `wgcna`. |
| `wgcna_params` *(required only if `submodule_mode: wgcna`)* | mapping | {5, 10, 0.25} | `beta`, `min_module_size`, `distance_cutoff`. |
| `interactive_plot` | boolean | `true` | Render interactive network in the notebook (`true`) or generate static files (`false`). |
| `interactive_layout` | string | `spring` | Layout algorithm for interactive view (`spring`, `fr`, `force`, `bipartite`, `pydot`, `random`, `circular`, `kamada_kawai`). Ignored if `interactive_plot` is `false`. |

---  

## MOFA Modeling <a id="mofa-modeling"></a>  

Configuration path: `analysis.analysis_parameters.mofa`

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `num_mofa_factors` | integer > 0 | `5` | Number of latent factors to learn. |
| `num_mofa_iterations` | integer > 0 | `1000` | Training iterations. |
| `seed_for_training` | integer > 0 | `555` | Random seed to guarantee reproducibility. |

---  

## Configuration Example <a id="configuration-example"></a>

Below is a **minimal, valid** `analysis` block that exercises each major option.  
Only parameters that differ from defaults are shown; omitted fields assume their defaults.

```yaml
analysis:
  data_analysis_tag: KW-FS
  analysis_parameters:
    feature_selection:
      selected_method: kruskalwallis
      max_features: 5000
      ...
      kruskalwallis:
        metadata_category: temperature
        significance_level: 0.01
        log2fc_cutoff: 0.5
      ...
    correlation:
      corr_method: pearson
      corr_cutoff: 0.75
      keep_negative: false
      block_size: 500
      cores: -1
    networking:
      network_mode: bipartite
      submodule_mode: louvain
      interactive: true
      interactive_layout: spring
    mofa:
      num_mofa_factors: 3
      num_mofa_iterations: 1000
      seed_for_training: 555
```

**Result:**

*   Only features with normalized abundance that was significantly different (FDR<0.01 and LFC>0.5) between samples based on "temperature" category (e.g., samples with low vs. medium vs. high) by a Kruskal-Wallis test by ranks are kept, up to the best 5,000 corrected p-values.
*   The correlation is performed with the Pearson method and pairs of features are only kept if they have a positive correlation rho score of ≥ 0.75. Calculations are made with a block size of 500 and all available cores.
*   The network includes only bipartite edges (i.e., edges between nodes of different data types) and submodules are extracted with the louvain (community) method. In the notebook, the network is rendered for interactivity using the default spring (Fruchterman Reingold) layout.
*   Multi-omics factor analysis is run with 3 factors, 1000 iterations, and a fixed random seed of 555 (for reproducibility).

**Final Output:** 

After these analysis steps, your results will include a subset of the integrated, QC-ed, and normalized features from the data processing step, a correlation network focused on strong cross-omics relationships, and a MOFA model summarizing features that are a major sources of variation across samples and datasets.