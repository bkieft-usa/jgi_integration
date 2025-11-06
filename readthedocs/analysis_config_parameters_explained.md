# Analysis Configuration Reference  

---  

## Overview

This document describes the practical effect of each option in the **analysis** configuration file (`/input_data/config/analysis.yml`). This file contains parameters for feature selection, correlation analysis, network analysis, functional enrichment, and MOFA modeling. Changes to any parameter in this file will generate a new analysis hash, ensuring results are never accidentally mixed between different analysis configurations.

---  

## Table of Contents
- [Configuration](#configuration)
- [Feature Selection](#feature-selection)  
   - [Shared options](#feature-selection-shared-options)  
   - [Methods](#feature-selection-methods)  
- [Feature Correlation](#feature-correlation)  
- [Network Analysis](#network-analysis)  
- [Functional Enrichment](#functional-enrichment)
- [MOFA Modeling](#mofa-modeling)  
- [Configuration Example](#configuration-example)  

---

## Configuration <a id="configuration"></a>

The workflow automatically generates an **analysis hash** from all parameters in this file. When you change any analysis parameter (correlation thresholds, feature selection methods, etc.), a new hash is generated, ensuring:

- Only analysis steps are re-run (data processing results are reused)
- Previous analysis results are preserved
- Clear tracking of which parameters produced which results

Example: Changing correlation cutoff from `0.65` to `0.7` will generate a new analysis hash like `f1g2h3i4`, creating a new directory `Dataset_Processing--a1b2c3d4/Analysis--f1g2h3i4/`. All results generated with the new parameter set are saved in their own analysis folder. To re-produce the exact results of an analysis again, use the configuration file path that was produced during the previous run.

---

## Feature Selection <a id="feature-selection"></a>

Configuration path: `analysis.analysis_parameters.feature_selection`

### Shared options <a id="feature-selection-shared-options"></a>

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `selected_method` | string | `"variance"` | Method used to pick informative features. Allowed values: `variance`, `glm`, `kruskalwallis`, `feature_list`, `lasso`, `random_forest`, `mutual_info`, `none`. See table below for details. |
| `max_features` | integer | `5000` | Upper bound on the number of features kept after selection (must be ≤ 10 000 for performance). |

### Methods <a id="feature-selection-methods"></a>

Below each method's *method‑specific* parameters are shown. All method blocks are optional **unless** that method is chosen via `selected_method`.

| Method | Config block | Parameter | Type | Default | Description |
|--------|----------------------------|-----------|------|---------|-------------|
| **variance** | `variance` | `top_n` | integer > 0 | `5000` | Number of highest‑variance features to retain. |
| **glm** | `glm` | `category` | string | – | Column in `user_settings.variable_list` used for group comparison. |
| | | `reference` | string | – | Reference group within the chosen metadata column. |
| | | `significance_level` | float ∈ (0, 1] | `0.05` | FDR‑adjusted p‑value cutoff. |
| | | `log2fc_cutoff` | float > 0 | `0.5` | Minimum absolute log₂ fold‑change. |
| **kruskalwallis** | `kruskalwallis` | `category` | string | – | Same meaning as for *glm*. |
| | | `significance_level` | float ∈ (0, 1] | `0.05` | FDR‑adjusted p‑value cutoff. |
| | | `lfc_cutoff` | float > 0 | `0.5` | Minimum absolute log₂ fold‑change. |
| **feature_list** | `feature_list` | `feature_list_file` | string (path) | – | File containing one feature ID per line. Must reside in the analysis output directory (`…/Analysis--<TAG>/`). |
| **lasso** | `lasso` | `category` | string | – | Continuous target variable for Lasso regression. |
| **random_forest** | `random_forest` | `category` | string | – | Target variable (continuous → regression, categorical → classification). |
| **mutual_info** | `mutual_info` | `category` | string | – | Target variable (continuous or categorical). |
| **none** | — | — | — | — | No selection; all features passed through (subject to `max_features`). |

---  

## Feature Correlation <a id="feature-correlation"></a>  

Configuration path: `analysis.analysis_parameters.correlation`

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `corr_method` | string | `"pearson"` | Correlation metric. Options: `pearson`, `spearman`, `cosine`, `centered_cosine`, `bicor`. |
| `corr_cutoff` | float ∈ [0, 1] | `0.5` | Minimum absolute correlation value for an edge to be kept. |
| `keep_negative` | boolean | `false` | If `true`, include negative correlations whose absolute value ≥ `corr_cutoff`. |
| `block_size` | integer > 0 | `500` | Number of features processed per block (affects memory & speed). |
| `cores` | integer | `1` | Parallel workers for block‑wise calculation (`-1` → use all cores). |
| `corr_mode` | string | `"bipartite"` | Edge topology. Options: `bipartite` (cross‑type only), `full` (all pairwise), or `<prefix>` (use data type prefix to select only those features). Note: `full` is not currently recommended due to computational constraints. |

---  

## Network Analysis <a id="network-analysis"></a>  

Configuration path: `analysis.analysis_parameters.networking`

**Background**: A datatype-specific or bipartite network is constructed from the integrated data. Before conducting network analysis, it is advisable to drop some un-important or non-relevant features (e.g., those without significant variation across groups in your experimental design) from the analysis because they are not meaningful to correlation analysis; large graphs are very difficult to interpret without sub-setting the data in a biologically or technically meaningful way (see feature selection above). Submodules (discrete sets of connected nodes) can also be extracted from the network, which represent tightly correlated clusters of features that had similar observed abundance across experimental samples. The networks are provided in .graphml format for uploading to CytoScape or a similar tool or are viewed by deafult in the Jupyterlab interface, and the node and edge files of each graph are provided for custom analysis.

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `submodule_mode` | string | `"louvain"` | Sub‑module extraction method. Options: `none`, `subgraphs`, `louvain`, `leiden`, `wgcna`. |
| `wgcna_params` *(required only if `submodule_mode: wgcna`)* | mapping | {5, 10, 0.25} | `beta`, `min_module_size`, `distance_cutoff`. |
| `interactive_plot` | boolean | `true` | Render interactive network in the notebook (`true`) or generate static files (`false`). |
| `interactive_layout` | string | `spring` | Layout algorithm for interactive view (`spring`, `fr`, `force`, `bipartite`, `pydot`, `random`, `circular`, `kamada_kawai`). Ignored if `interactive_plot` is `false`. |

---  

## Functional Enrichment <a id="functional-enrichment"></a>  

Configuration path: `analysis.analysis_parameters.functional_enrichment`

**Background**: Functional enrichment tests in correlation or coexpression networks are used to identify biological processes or pathways that are overrepresented among genes with highly correlated expression profiles. In the case of this workflow, the enrichment tests are performed on network submodules (if generated, see above), and the biological process can be any of the annotation categories from the feature annotation table. By testing for enriched functions, such as Gene Ontology terms, KEGG pathways, or metabolite classes, the "role" of the submodule and its relationship to your experimental design can be inferred. This approach can reveal insights into the biological functions of genes, metabolites, or their interactions, and the results of functional enrichment tests can be used to prioritize features for further investigation and to inform the development of new hypotheses.

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `annotation_column` | string | `"tx_goterm_acc"` | Column containing functional annotations. Options: `tx_goterm_acc` (Gene Ontology terms), `mx_subclass` (metabolite subclasses). |
| `pvalue_cutoff` | float ∈ (0, 1] | `0.05` | P‑value threshold for significant enrichment. |
| `min_genes_per_term` | integer > 0 | `1` | Minimum number of features required per functional term to be tested. |
| `correction_method` | string | `"fdr_bh"` | Multiple testing correction method. Options: `fdr_bh` (Benjamini-Hochberg), `bonferroni`, `holm`, `sidak`. |

---  

## MOFA Modeling <a id="mofa-modeling"></a>  

Configuration path: `analysis.analysis_parameters.mofa`

**Background**: MOFA (Multi-omics Factor Analysis). MOFA2 is a modelling tool that helps integrate and interpret ‘omics datasets in an unsupervised fashion. The software produces factors that are analogous to principal component analysis (PCA) axes but tailored for use on multi-omics datasets. It takes two or more data matrices with differing structure (e.g., RNA counts and metabolite peak heights) that have the same or at least overlapping samples and infers “interpretable low-dimensional representation in terms of a few latent factors”. More details on interpreting MOFA2 results can be found below. The source code for MOFA2 can be found here [https://github.com/bioFAM/MOFA2] and the publication describing the MOFA2 algorithm can be found here [doi:10.1101/2020.11.03.366674].

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `num_mofa_factors` | integer > 0 | `5` | Number of latent factors to learn. |
| `num_mofa_iterations` | integer > 0 | `1000` | Training iterations. |
| `seed_for_training` | integer > 0 | `555` | Random seed to guarantee reproducibility. |

---  

## Configuration Example <a id="configuration-example"></a>

Below is a complete `analysis.yml` file that exercises each major option:

```yaml
analysis:
  analysis_parameters:
    feature_selection:
      selected_method: kruskalwallis
      max_features: 5000
      variance:
        top_n: 5000
      glm:
        category:
        reference:
        significance_level: 0.05
        log2fc_cutoff: 0.25
      kruskalwallis:
        category: temperature
        significance_level: 0.01
        lfc_cutoff: 0.5
      feature_list:
        feature_list_file:
      lasso:
        category:
      random_forest:
        category:
      mutual_info:
        category:
    correlation:
      corr_method: pearson
      corr_cutoff: 0.75
      keep_negative: false
      block_size: 500
      cores: -1
      corr_mode: bipartite
    networking:
      submodule_mode: louvain
      wgcna_params:
        beta:
        min_module_size:
        distance_cutoff:
      interactive_plot: true
      interactive_layout: spring
    functional_enrichment:
      annotation_column: tx_goterm_acc
      pvalue_cutoff: 0.01
      min_genes_per_term: 2
      correction_method: fdr_bh
    mofa:
      num_mofa_factors: 3
      num_mofa_iterations: 1000
      seed_for_training: 555
```

**Result:**

*   Only features with normalized abundance that was significantly different (FDR<0.01 and LFC>0.5) between samples based on "temperature" category (e.g., samples with low vs. medium vs. high) by a Kruskal-Wallis test by ranks are kept, up to the best 5,000 corrected p-values.
*   The correlation is performed with the Pearson method and pairs of features are only kept if they have a positive correlation rho score of ≥ 0.75. Calculations are made with a block size of 500 and all available cores.
*   The network includes only bipartite edges (i.e., edges between nodes of different data types) and submodules are extracted with the louvain (community) method. In the notebook, the network is rendered for interactivity using the default spring (Fruchterman Reingold) layout.
*   Functional enrichment analysis is performed using Gene Ontology terms with FDR correction (Benjamini-Hochberg), requiring at least 2 features per term and a corrected p-value threshold of 0.01.
*   Multi-omics factor analysis is run with 3 factors, 1000 iterations, and a fixed random seed of 555 (for reproducibility).

**Final Output:** 

After these analysis steps, your results will include a subset of the integrated, QC-ed, and normalized features from the data processing step, a correlation network focused on strong cross-omics relationships, functional enrichment results highlighting biological pathways and processes, and a MOFA model summarizing features that are a major sources of variation across samples and datasets.