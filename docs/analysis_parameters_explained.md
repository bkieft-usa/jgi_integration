# Analysis Parameters Explained

This document describes the practical effect of each option in the configuration file for the main analysis steps in omics data integration, including the analysis tag, feature selection, network analysis, and MOFA2 modeling. Each section lists available methods, their options, and default values, and at the end there is an example of a full analysis workflow.

---

# Setup Parameters

## 1. Tagging
- **data_analysis_tag**
  *Allows user to create a new analysis folder (Analysis--**TAG**) underneath the data processing folder to store analysis results. This is useful if analysis settings are changed (see below) and a new set of outputs should be produced that is separate from previous runs*
  *Note: if an analysis folder already exists with the supplied tag and overwriting is disabled, the workflow will return an error message indicating that you should change the tag* 

---

## 1. Feature Selection (`perform_feature_selection`)

### Options:

- **variance** [default]  
  *Selects the features with the highest variance across samples. Useful for keeping only the most variable and potentially informative features.*
  - **max_features**  
    *Maximum number of features to retain (integer > 0).*

- **glm**  
  *Generates a Generalized Linear Model (GLM) to identify features significantly associated with a metadata category. Filters by FDR-corrected p-value and minimum log2 fold change to keep only significant features.*
  - **metadata_category**  
    *Metadata column to use for group comparison (must match a variable in your analysis.integrated_metadata table - i.e., in the user_settings->variable_list in the config).*
  - **metadata_category_reference**  
    *Reference group for the GLM - a specific group within the selected metadata category.*
  - **significance_level**  
    *FDR-corrected p-value cutoff (real number between 0 and 1).*
  - **log_fold_level**  
    *Minimum absolute log2 fold change to consider significant (real number > 0).*
  - **max_features**  
    *Maximum number of features to retain (integer > 0).*

- **kruskalwallis**  
  *Uses the Kruskal-Wallis test to identify features significantly associated with a metadata category. Filters by FDR-corrected p-value and minimum log2 fold change.*
  - **metadata_category**  
    *Metadata column to use for group comparison (must match a variable in your analysis.integrated_metadata table - i.e., in the user_settings->variable_list in the config).*
  - **significance_level**  
    *FDR-corrected p-value cutoff (real number between 0 and 1).*
  - **log_fold_level**  
    *Minimum absolute log2 fold change to consider significant (real number > 0).*
  - **max_features**  
    *Maximum number of features to retain (integer > 0).*

- **feature_list**  
  *Selects features from a user-provided list (one feature ID per line in a file, must match the feature IDs in the analysis.integrated_data table).*
  - **feature_list_file**  
    *Filename containing the list of features to keep.*
  - **max_features**  
    *Maximum number of features to retain (integer > 0).*

- **none**  
  *No feature selection is performed; all features are retained.*

- **Note**
  *The max_features option during feature selection, which shows up in almost all modes, restricts the number of features that go into downstream correlation analysis and networking - this is designed to reduce the size and scale of calculations and should be set to a value lower than 10,000 when possible.*

---

## 2. Feature Correlation (`calculate_correlated_features`)

- **corr_method**
  - **pearson** [default]  
    *Calculates Pearson correlation between features.*
  - **spearman**  
    *Calculates Spearman rank correlation.*
  - **kendall**  
    *Calculates Kendall rank correlation.*

- **corr_cutoff**  
  *Correlation threshold for including edges in the network (real number between 0 and 1). Only feature pairs with correlation above this value are included.*

- **keep_negative**  
  *If true, includes both positive and negative correlations above the absolute threshold. If false, only positive correlations are included.*

## 3. Network Analysis (`plot_correlation_network`)

### Options:

- **network_mode**
  - **bipartite** [default]  
    *Constructs a network only between features from different datasets (e.g., transcript vs metabolite).*
  - **full**  
    *Constructs a network including all feature-feature correlations, regardless of dataset.*
  - **NOTE:** 
    Functionally, this option is also passed to feature correlation step to keep the cached correlation matrix as small as possible.

- **submodule_mode**
  - **community** [default]  
    *Extracts submodules using community detection algorithms (Louvain method).*
  - **subgraphs**  
    *Extracts submodules as connected components.*

---

## 4. MOFA2 Analysis (`run_full_mofa2_analysis`)

### Options:

- **num_mofa_factors**  
  *Number of latent factors to compute in the MOFA2 model (integer > 0). Controls model complexity.*

- **num_mofa_iterations**  
  *Number of training iterations for MOFA2 (integer > 0). Higher values may improve convergence.*

- **seed_for_training**  
  *Random seed for reproducibility (integer > 0). Ensures consistent results across runs.*

---

## Example

Suppose you run an analysis with the following configuration:

```yaml
analysis:
  data_analysis_tag: 0
  analysis_parameters:
    feature_selection:
      selected_method: kruskalwallis
      kruskalwallis:
        metadata_category: temperature
        significance_level: 0.05
        log_fold_level: 0.5
        max_features: 8000
    correlation:
      corr_method: pearson
      corr_cutoff: 0.75
      keep_negative: false
    networking:
      network_mode: bipartite
      submodule_mode: community
    mofa:
      num_mofa_factors: 5
      num_mofa_iterations: 1000
      seed_for_training: 555
```

**Result:**
- Only features with normalized abundance significantly associated with the "temperature" category (e.g., samples with low vs. medium vs. high) by a Kruskal-Wallis test are kept.
- The network includes only bipartite edges (between different data types) with Pearson correlation ≥ 0.75 and only positive correlations.
- MOFA2 is run with 5 factors, 1000 iterations, and a fixed random seed of 555 for reproducibility.

---

**Final Output:**  
After these analysis steps, your results will include a subset of the integrated, QC-ed, and normalized features from the data processing step, a correlation network focused on strong cross-omics relationships, and a MOFA2 model summarizing features that are a major sources of variation across samples and datasets.