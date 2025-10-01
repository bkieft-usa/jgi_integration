# Analysis Parameters Explained

This document describes the practical effect of each option in the **analysis** section of the integration workflow configuration file (/input_data/config/project_config.yml), including the analysis tag, feature selection, network analysis, and MOFA modeling. Each section lists available methods, their options, and default values, and at the end there is an example of a full analysis workflow.

# Setup Parameters

## 1. Tagging

  **data_analysis_tag**

  Allows user to create a new analysis folder (Analysis--**TAG**) underneath the data processing folder to store analysis results. This is useful if analysis settings are changed (see below) and a new set of outputs should be produced that is separate from previous runs. Default: “0”.

  _Note_: if an analysis folder already exists with the supplied tag and overwriting is disabled, the workflow will return an error message indicating that you should change the tag

## 2. Feature Selection

  **selected_method**

  The type of feature selection method implemented to determine significant features used in downstream analysis (see methods below).

  **max_features**

  Maximum number of features to retain after feature selection (default: 5,000). This helps constrain the number of features passed to correlation calculations and network production and should be set to a value lower than 10,000 when possible.

### Options:

  **variance** \[default\]

  Selects the features with the highest variance across samples. Useful for keeping only the most variable and potentially informative features.

  #### Parameters:

  * _top_n_ (integer > 0)

    Number of highest variation features to return (default: 5,000)

  **glm**

  Generates a Generalized Linear Model (GLM) to identify features significantly associated with a metadata category. Filters by FDR-corrected p-value and minimum log2 fold change to keep only significant features.

  #### Parameters:

  * _metadata_category_

    Metadata column to use for group comparison (must match a variable in the user_settings->variable_list in the configuration file).

  * _metadata_category_reference_

    Reference group for the GLM - a specific group within the selected metadata category.

  *  _significance_level_ (real number between 0 and 1)

    FDR-corrected p-value cutoff (default: 0.05).

  * _log_fold_level_ (real number > 0)

    Minimum absolute log2 fold change to consider significant (default: 0.5).

  **kruskalwallis**

  Use the Kruskal-Wallis test to identify features significantly associated with a metadata category. Filters by FDR-corrected p-value and minimum log2 fold change.

  #### Parameters:

  * _metadata_category_

    Metadata column to use for group comparison (must match a variable in the user_settings->variable_list in the configuration file).

  * _significance_level_ (real number between 0 and 1)

    FDR-corrected p-value cutoff (default: 0.05).

  * _log_fold_level_ (real number > 0)

    Minimum absolute log2 fold change to consider significant (default: 0.5).

  **feature_list**

  Selects features from a user-provided list (one feature ID per line in a file, must match the feature IDs in the analysis.integrated_data table).

  #### Parameters:

  * _feature_list_file_

    Filename containing the list of features to keep. This file must be saved into the correct analysis output directory (e.g., /output_data/project_name/Data_Processing--TAG/Analysis--TAG/). You can drop this file directly into the folder via the JupyterLab interface.

  **lasso**

  Uses Lasso regression to select features most predictive of a continuous metadata variable. Features are ranked by absolute coefficient magnitude.

  #### Parameters:

  * _metadata_category_

    Metadata column to use as the target variable (must be continuous).

  **random_forest**

  Uses a random forest model to select features most predictive of a metadata variable (regression for continuous, classification for categorical). Features are ranked by importance.

  #### Parameters:

  * _metadata_category_

    Metadata column to use as the target variable.

  **mutual_info**

  Selects features with highest mutual information with a metadata variable (continuous or categorical). Captures non-linear associations.

  #### Parameters:

  * _metadata_category_

    Metadata column to use as the target variable.

  **none**

  No feature selection is performed. All features are retained, up to the max_features limit.

## 3. Feature Correlation

### Options:

  **corr_method**

  Determines the way that pairwise feature relationships are calculated.

  #### Parameters:

  * _pearson_ \[default\]

    Calculates Pearson correlation between features.

  * _spearman_

    Calculates Spearman rank correlation.

  * _cosine_

    Calculates cosine similarity between features.

  * _centered_cosine_

    Calculates centered cosine similarity (cosine similarity after mean-centering).

  * _bicor_

    Calculates biweight midcorrelation (robust correlation using median and MAD).

  **corr_cutoff** (real number between 0 and 1)

  Correlation threshold for including edges in the network (default: 0.5). Only feature pairs with correlation above this value are included.

  **keep_negative**

  If true, includes both positive and negative correlations above the absolute threshold. If false (default), only positive correlations are included.

  **block_size** (integer > 0)

  Number of features processed per block during correlation calculation (default: 500). Larger blocks may use more memory but can be faster.

  **n_jobs** (integer)

  Number of parallel jobs for block-wise correlation calculation (default: 1). Use -1 to utilize all available cores.

  **only_bipartite**

  If true (default), only calculates correlations between features of different types (e.g., transcript vs. metabolite). If false, calculates all pairwise correlations (including within type).

## 4. Network Analysis

### Options:

  **network_mode**

  Determines the architecture of the network by categorizing edges and selecting only subsets based on dataset type.

  #### Parameters:

  * _bipartite_ \[default\]

    Constructs a network only between features from different datasets (e.g., transcript and metabolite node edges).

  * _full_

    Constructs a network including all feature-feature correlations, regardless of dataset.

  **submodule_mode**

  Determines if submodules will be extracted from the network and if so by which method.

  **interactive**

  Determines if the network is printed to the notebook interface in interactive mode.

  #### Parameters:

  * _none_

    No submodules are extracted from the main graph.

  * _subgraphs_

    Extracts submodules as connected components.

  * _louvain_ \[default\]

    Extracts submodules using the Louvain community detection algorithm.

  * _leiden_

    Extracts submodules using the Leiden community detection algorithm (requires igraph and leidenalg).

  * _wgcna_

    Extracts submodules using WGCNA-style hierarchical clustering on the topological overlap matrix (TOM). Parameters include beta (soft-threshold power), min_module_size (minimum module size), and distance_cutoff (clustering cutoff).

## 5. MOFA Analysis

### Options:

  **num_mofa_factors** (integer > 0)

    Number of latent factors to compute in the MOFA model (default: 5). Controls model complexity.

  **num_mofa_iterations** (integer > 0)

    Number of training iterations for MOFA (default: 1,000). Higher values may improve convergence.

  **seed_for_training** (integer > 0)

    Random seed for reproducibility (default: 555). Ensures consistent results across runs – set a different random seed to produce a different (non-deterministic) result.

## Example

Suppose you run an analysis with the following configuration (removed some unused feature selection settings for this example):

```yaml
analysis:
    data_analysis_tag: VAR
    analysis_parameters:
        feature_selection:
            selected_method: kruskalwallis
            ...
            kruskalwallis:
                metadata_category: salinity
                significance_level: 0.01
                log_fold_level: 0.5
                max_features: 5000
            ...
        correlation:
            corr_method: pearson
            corr_cutoff: 0.75
            keep_negative: false
        networking:
            network_mode: bipartite
            submodule_mode: community
            interactive: false
        mofa:
            num_mofa_factors: 3
            num_mofa_iterations: 1000
            seed_for_training: 555
```

**Result:**

*   Only features with normalized abundance that was significantly different (FDR<0.01 and LFC>0.5) between samples of different "temperature" categories (e.g., samples with low vs. medium vs. high) by a Kruskal-Wallis test by ranks are kept, up to the best 5,000 corrected p-values.
*   The correlation is performed with the Pearson rho value and pairs of features are only kept if they have a positive correlation ≥ 0.75.
*   The network includes only bipartite edges (i.e., edges between nodes of different data types)
*   Multi-omics factor analysis is run with 3 factors, 1000 iterations, and a fixed random seed of 555 for reproducibility.

**Final Output:** 

After these analysis steps, your results will include a subset of the integrated, QC-ed, and normalized features from the data processing step, a correlation network focused on strong cross-omics relationships, and a MOFA model summarizing features that are a major sources of variation across samples and datasets.