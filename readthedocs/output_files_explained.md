# Guide to the Workflow Output Files

This guide on the outputs from the JGI multi-omics integration workflow explains:

1. how the results were generated (the processing and analysis steps, and the configuration settings that control each one),
2. what every file and folder contains, and
3. how to read the tables and figures, including what they can and cannot tell you.

> **Start here:** open the three files in `configs/` beside this guide. They record the exact settings used for the results you received. Each step below names the configuration key that controls it, so you can look up the value that was used.

---

## 1. Package layout

```
<project_name>/
├── configs/
│   ├── Dataset_Processing--<D>_Analysis--<A>_project_config.yml
│   ├── Dataset_Processing--<D>_Analysis--<A>_data_processing_config.yml
│   └── Dataset_Processing--<D>_Analysis--<A>_analysis_config.yml
├── notebooks/
│   └── Dataset_Processing--<D>_Analysis--<A>_notebook.ipynb
└── Dataset_Processing--<D>/
    ├── transcriptomics/          # tx: one folder per dataset type that was run
    ├── metabolomics/             # mx
    ├── proteomics/               # px (only if proteomics was included)
    └── Analysis--<A>/
        └── (integration, annotation, grouping and pathway tables and figures)
```

`<project_name>` comes from `user_settings.project_name` in the project config.

### What the hash tags mean

The `<D>` and `<A>` values are random characters that are derived from a hash of the configuration file. Different iterations of the workflow can be run with different parameter sets (producing different results), so you may see one or more hashed output directories (an Analysis folder is always inside a Data_Processing folder because of the order of the workflow steps).

- **`Dataset_Processing--<D>`**: `<D>` is the first 8 characters of a SHA-256 hash of the whole data processing config (`data_processing.yml`). Any change to a filtering, normalization or transformation setting produces a new folder.
- **`Analysis--<A>`**: `<A>` is the same kind of hash, made from the analysis config (`analysis.yml`). Several analyses can share one processed dataset, so several `Analysis--*` folders can sit inside one `Dataset_Processing--*` folder.
- The project config (`project.yml`) is **not** hashed. It holds paths, the project name and the list of metadata variables used for plots.

The hashes identify the **parameters**, not the code version. Two runs with the same hashes but different workflow code versions can differ. See [Section 8](#8-reproducing-the-results).

---

## 2. Where to look first

| Example Question | File(s) |
|---|---|
| What settings were used? | `configs/*.yml` |
| What exactly was run, and what did the run log say? | `notebooks/*_notebook.ipynb` (code cells with their saved outputs, including feature counts kept at each filter) |
| Which features grouped together? | `Analysis--<A>/feature_network_node_table.csv` (`group` column) |
| Which groups are enriched for which pathways? | `Analysis--<A>/heatmap_enrichment.png`, `full_padj_matrix.csv`, `full_fold_enrichment_matrix.csv` |
| What biological annotations are known about a feature? | `Analysis--<A>/feature_annotation_table.csv` |
| Do samples separate by treatment? | `Analysis--<A>/Integrated_Data_PCA___<variable>.pdf` |
| How many features survived each processing step? | Row counts of the CSVs in each dataset folder (Section 4), or the notebook log |

---

## 3. How the results were generated

The workflow runs the steps below in order. Steps 1–6 run **separately for each dataset type** (`tx` transcriptomics, `mx` metabolomics, `px` proteomics) and write to `Dataset_Processing--<D>/<dataset>/`. Steps 7–11 combine the datasets and write to `Analysis--<A>/`.

| # | Step | What it does | Controlled by | Output |
|---|---|---|---|---|
| 1 | Load raw data | Reads the quantitative table for each dataset (`counts.csv` for tx, `peak-height.csv` for px, `*_peak-height-filtered-3x-exctrl.csv` files for mx from different polarities). Adds a dataset prefix to feature IDs (Section 7). For mx, combines all configured chromatography/polarity files with a suffix indicating which one the feature came from. | `datasets.<ds>.*` (data processing) | `raw_data.csv` |
| 2 | Build per-dataset annotations | tx/px: merges every `*_annotation_table.tsv` file and maps EC numbers to ModelSEED reactions. mx: summarizes library matches (names, InChIKey, chemical classes) and maps them to ModelSEED compounds. | `datasets.<ds>.annotation_columns`, `gff_*` keys | `annotation_table.csv` |
| 3 | Link samples | Renames each dataset's sample columns (data-type and method-specific names that aren't guarenteed to match between data types) to a shared sample name (`unique_group`) using the link table, then keeps only samples present in **every** dataset. | `project.link_table` (project) | `linked_data.csv`, `Analysis--<A>/integrated_metadata.csv` |
| 4 | Check input scale and filter | Records value statistics and detects whether the input is linear, log-scaled or already scaled. For `input_state: raw`, removes features that are rarely detected or very low in abundance. Skipped for `prenormalized` input. | `input_state`, `normalization_parameters.filtering` | `input_report.json`, `filtered_data.csv` |
| 5 | Normalize and transform | Raw input only: corrects for sample-to-sample differences in depth or loading with size factors (defaults: median-of-ratios for tx; PQN for mx and px, computed separately for each mx chromatography/polarity). mx/px: zeros are treated as missing and filled with the feature's lowest observed value. Then transforms values to a log2-like scale (defaults: variance-stabilizing transform for tx, log2 for mx/px). | `normalization_method`, `transformation_method`, `pseudocount`, `recenter_samples` | `normalized_data.csv`, `detected_mask.csv`, `log_data.csv` |
| 6a | Remove low-variance features | Removes the bottom N% of features by variance. With `mean_adjusted: true`, a feature is ranked against features of similar mean abundance, so low-abundance features are not favored. | `devariancing` | `devarianced_data.csv` |
| 6b | Remove unreliable features | Within each sample group, measures the log-scale SD across replicates. Removes a feature when the SD exceeds `sd_threshold` in more than `majority_fraction` of the groups where it was detected in at least `min_replicates` replicates. | `replicate_handling` | `replicate_filtered_data.csv` |
| 6c | Scale for integration | **paired**: z-scores each feature across samples. **unpaired**: takes the median for each sample group, then z-scores each feature across groups. | `analysis.replicate_pairing` (analysis) | `scaled_data_paired.csv` or `scaled_data_unpaired.csv` |
| 7 | Integrate | Stacks the scaled matrices of all datasets into one table, keeping only the columns (samples or groups) shared by all of them. | `replicate_pairing` | `integrated_data.csv` |
| 8 | Select features | `variance`: picks the most variable features, ranked within each dataset on the log scale from step 6b. All datasets share one `max_features` budget, and smaller datasets are filled first. With `block_scale: true`, each dataset's block is then weighted so every data type carries equal total variance. `feature_list`: keeps only the features named in a supplied file. | `analysis.feature_selection` | `integrated_data_selected.csv` |
| 9 | Group features by annotation (knowledge-driven) | Joins every dataset's `annotation_table.csv` onto the integrated features, then adds pathway assignments from ModelSEED and KEGG (Section 5.2). | (automatic) | `feature_annotation_table.csv` |
| 10 | Group features by abundance (data-driven) | Clusters the selected features by their co-variation pattern across samples or groups. Six methods are available (Section 5.3). | `analysis.feature_grouping` | `feature_network_node_table.csv`, `feature_network_edge_table.csv`, plus method-specific files and folders |
| 11 | Compare data-driven groups with knowledge-driven pathways | Tests whether each data-driven group contains more members of a knowledge-driven biological pathway than expected by chance, and plots the results as heatmaps. | `analysis.pathway_enrichment` | `heatmap_*.png`, `full_*_matrix.csv`, `*_trend_values.csv`, `exploded_universe.csv`, `run_config.json` |
| 12 | Save provenance | Copies the three active configs and the notebook into `configs/` and `notebooks/` under hash-tagged names. | (automatic) | `configs/`, `notebooks/` |

### Two integration modes

`analysis.replicate_pairing` sets what a column means in every Analysis-level table:

- **`paired`** (sample resolution): each biological sample was measured by every omics platform, so samples can be matched one to one. Columns are **samples** (`unique_group` values). Correlations reflect co-variation across individual samples, so replicate 1 from sample group A should be paired with replicate 1 from sample group B.
- **`unpaired`** (condition resolution): replicates are not matched across omics platforms. Columns are **sample groups/conditions** (`group` values). Each value is the z-scored group median because replicate 1 from sample group A is equally related to replicate 1, 2, or 3 from sample group B (not paired), so correlations reflect co-variation across conditions only, with fewer points.

---

## 4. Per-dataset files: `Dataset_Processing--<D>/<dataset>/`

`<dataset>` is `transcriptomics`, `metabolomics` or `proteomics`. The tables are listed in pipeline order. Unless noted otherwise, rows are **features** and columns are **samples**, renamed to the shared `unique_group` names from step 3 onward.

| File | Scale of values | Contents |
|---|---|---|
| `raw_data.csv` | Raw (counts / peak heights) | Original quantitative table. The first column is the feature ID (`CompoundID` for mx), followed by the original sample names from the instrument or sequencing files. Feature IDs carry the dataset prefix. |
| `linked_data.csv` | Raw | Same values with sample columns renamed to `unique_group`. Only samples present in all datasets are kept. If two raw columns map to the same sample, they are summed. |
| `input_report.json` | – | Input diagnostics: number of features and samples; fractions of missing, zero and negative values; min, median and max; skew; whether values are integers; and the `detected_scale` (`linear`, `log` or `scaled`) that drives later steps. |
| `filtered_data.csv` | Raw | Features that passed the presence and magnitude filters. Identical to `linked_data.csv` for prenormalized input. |
| `normalized_data.csv` | Linear, size-factor normalized | For mx/px, zeros are left **blank** (missing) here, not imputed. |
| `detected_mask.csv` | True/False | `True` where the feature was actually observed (> 0, or non-missing for prenormalized input). Use it to tell real measurements apart from imputed values in `log_data.csv`. |
| `log_data.csv` | log2-like | Normalized, imputed (mx/px) and transformed values. This is the main "analysis-ready" abundance table for each dataset. |
| `devarianced_data.csv` | log2-like | `log_data` with the lowest-variance features removed. |
| `replicate_filtered_data.csv` | log2-like | Features that also passed the replicate-reliability filter. **This is the final per-dataset feature set** that goes into integration and feature selection. |
| `scaled_data_paired.csv` / `scaled_data_unpaired.csv` | z-score (unitless) | The integration representation (step 6c). For `unpaired`, columns are sample groups. Only the file matching the `replicate_pairing` setting is produced. |
| `annotation_table.csv` | – | Feature annotations for this dataset (see below). |
| `modelseed_reactions.tsv` (tx/px) / `modelseed_compounds.tsv` (mx) | – | Local copy of the ModelSEED biochemistry database used for ID mapping. This is reference data, not results. |
| `pca_plots/` | – | Produced only if PCA plots were requested separately (not part of the standard notebook): `pca_of_<dataset>_<stage>_by_<variable>.pdf` for the `linked` and `replicate_filtered` stages, plus `pca_grid.pdf`. |

**Tip:** to see how many features each step removed, compare the row counts of `linked_data` → `filtered_data` → `log_data` → `devarianced_data` → `replicate_filtered_data`. The notebook log reports the same counts.

**Note on units:** the tx variance-stabilizing transform (`arcsinh(sqrt(x)) / ln 2`) is on approximately the log2 scale for moderate-to-high counts but is compressed near zero. Values are comparable within a dataset, not across datasets. Cross-dataset comparison only becomes meaningful after the z-scoring in step 6c.

### `annotation_table.csv` columns

- **tx / px**: indexed by the gene/protein ID given by `gff_annot_id`. It contains **every column** of every `*_annotation_table.tsv` input file (for example, KEGG KO, EC, Pfam or GO, depending on what was supplied), plus:
  - `modelseed_rxn`: ModelSEED reaction IDs mapped from the EC column.
  - If `gff_quant_id` differs from `gff_annot_id`, a second ID column bridges the two through the GFF3 file.
- **mx**: indexed by `metabolome_id` (the mx feature ID). This table was generated by the JGI untargeted anlaysis pipeline with the FBMN tool at GNPS2. Columns: `molecular_formula`, `Compound_Name`, `Smiles`, `INCHI`, `InChiKey`, `superclass`, `class`, `subclass`, `npclassifier_superclass`, `npclassifier_class`, `npclassifier_pathway`, `library_usi`, `display_name`, and `modelseed_id`. `modelseed_id` is matched by exact InChIKey first, then by the first 25 InChIKey characters (structure plus stereo layer), and only if neither matches, by fuzzy matching on the compound name. Treat name-only matches as tentative. Features with no spectral library hit have empty annotation fields.

---

## 5. Analysis files: `Dataset_Processing--<D>/Analysis--<A>/`

### 5.1 Integration and feature selection

| File | Rows x columns | Contents |
|---|---|---|
| `integrated_metadata.csv` | samples x metadata | A copy of the link table: one row per `unique_group` sample, with the raw sample name in each dataset (`tx`, `mx_<chrom>_<pol>`, ...) and all metadata columns (for example `group`, `timepoint`). Use it to decode sample and group names, or you can review and make sure the samples from different omics data types were correctly associated with each other. |
| `integrated_data.csv` | features (all datasets) x samples or groups | All `scaled_data` matrices stacked together. Index column: `features`. Missing values are set to 0 (the feature mean, on the z-scale). |
| `integrated_data_selected.csv` | selected features x samples or groups | The subset chosen in step 8. **This matrix is the input to grouping and to the trend values in the heatmaps.** If `block_scale` was on, each dataset's values are multiplied by a constant weight (logged in the notebook as `Block weights applied per dataset`), so they are no longer exact z-scores. Relative patterns within a feature are unchanged. |

Note: in the current code, `feature_selection.params.max_features` sets how many features are kept.

### 5.2 Feature annotation: `feature_annotation_table.csv`

One row per feature in `integrated_data.csv` (all integrated features, not only the selected ones). Columns:

| Column(s) | Meaning |
|---|---|
| `feature_id` | Integrated feature ID (Section 7). |
| `tx_*`, `mx_*`, `px_*` | Every column of that dataset's `annotation_table.csv`, with the dataset name added as a prefix (for example `tx_KEGG_ko`, `tx_modelseed_rxn`, `mx_Compound_Name`, `mx_modelseed_id`). A feature only has values in its own dataset's columns. |
| `modelseed_pathway` | **The shared pathway key across data types.** Pathway IDs from the ModelSEED database (KEGG, MetaCyc and PlantCyc sources combined), reached through `*_modelseed_rxn` for genes/proteins and `mx_modelseed_id` for metabolites. A gene and a metabolite with the same pathway ID here are linked by known biochemistry. Values starting with `NOPATHWAY_rxn…` are placeholders: a reaction with known compounds but no named pathway. They keep gene–metabolite links that no named pathway covers, but they are not biological pathways. |
| `mx_kegg_id` | KEGG compound ID looked up from the InChIKey. |
| `kegg_pathway`, `kegg_pathway_name` | **KEGG level 1: pathway maps.** Reference pathway IDs (`path:map#####`) and names, including the broad global/overview maps (`map011xx`/`map012xx`, e.g. `map01100` Metabolic pathways). Organism-, KO- and EC-specific copies (`ko00860`, `ec00860`) are collapsed to `map00860`. All three KEGG levels are assigned the same way: metabolites through InChIKey → `mx_kegg_id` → KEGG, and transcripts/proteins through **both** their EC numbers and KO accessions. Every data type therefore shares one set of IDs at each level. Partial EC numbers (e.g. `EC:1.1.1.-`) are skipped. |
| `kegg_module`, `kegg_module_name` | **KEGG level 2: modules** (`M#####`, e.g. `M00001` glycolysis, `M00012` glyoxylate cycle). Smaller functional units than pathway maps, and the recommended level for group–pathway enrichment (`pathway_col: kegg_module`). Fewer features have a module than a pathway, because many KOs are not part of any module. |
| `kegg_reaction`, `kegg_reaction_name` | **KEGG level 3: reactions** (`R#####`, e.g. `R00200` pyruvate kinase). The most specific level. It directly links an enzyme to the metabolites it acts on, but each reaction has only a few features. Cofactor/currency compounds (water, ATP, NAD(P)H, CoA, CO₂, H⁺, …) are **not** linked at the module or reaction level, because they take part in thousands of reactions. They still receive pathways. |

`Unassigned` means no annotation was found.

### 5.3 Data-driven feature groups

The grouping method is set by `analysis.feature_grouping.method`. Only the parameters under `params` that belong to the chosen method are used. The config lists parameters for every method (for easy access), but the rest are ignored.

| Method | How features are grouped | Group labels |
|---|---|---|
| `hierarchical_clustering` | Agglomerative clustering on pairwise distances (`distance_metric`; `correlation` = 1 − Pearson r), using `linkage_method`. The tree is cut at `height_cutoff`. With correlation distance, a cutoff of 0.5 means that features in the same cluster are, roughly, correlated at r ≳ 0.5 (the exact meaning depends on the linkage method). | `group_<n>` |
| `network_modules` | Builds a network from pairwise correlations with \|r\| ≥ `corr_cutoff` (`corr_mode: bipartite` connects only features from **different** data types), drops components of fewer than 3 nodes, then finds communities with Louvain or Leiden. Communities larger than 400 nodes are split again. Features with no edge above the cutoff are **not** in any group. | `submodule_<n>` |
| `hdbscan` | Density-based clustering. Features that fit no dense cluster are labeled `noise`. | `group_<n>`, `noise` |
| `nmf` | Non-negative matrix factorization, with the number of components chosen at the elbow of the reconstruction-error curve between `k_min` and `k_max`. Each feature is assigned to its highest-loading component. | `group_<n>` |
| `leiden_knn` | k-nearest-neighbor graph (cosine similarity, `n_neighbors`), then Leiden community detection at the resolution that maximizes modularity. | `group_<n>` |
| `wgcna` | Weighted gene co-expression network analysis: soft-threshold power selection, topological overlap, dendrogram cut and merging of similar modules. Features that fit no module are labeled `grey`. | `module_<n>`, `grey` |

#### `feature_network_node_table.csv` (all methods)

One row per grouped feature (`node_id`). **Note that this file has "network_node" in the name, but it's simply the grouping from any of the above methods**. Columns:

- `group`: the group assignment used downstream. **Group numbers are arbitrary labels.** `group_3` in one analysis has nothing to do with `group_3` in another.
- `datatype_color`, `datatype_shape`: plotting attributes based on data type.
- All `feature_annotation_table.csv` columns, so you can filter groups by annotation directly. For `network_modules`, a feature with several annotations has them joined with `;;`.
- Method-specific extras:
  - `network_modules`: `node_size`, `submodule`, `submodule_color` (colors are assigned at random on each run).
  - `wgcna`: `wgcna_module`, `kIN` (connectivity within its module), and one `kME_*` column per module (correlation of the feature with that module's eigengene; values near ±1 mark hub features).

#### `feature_network_edge_table.csv`

**Note that this file is only produced when the network mode of grouping has been selected**.

- `network_modules`: one row per network edge with `source`, `target` and the correlation (`weight`). Use it with the node table in Cytoscape or a similar tool.
- All other methods: **empty** (header only). These methods do not produce edges.

#### Files from `network_modules` only

| File | Contents |
|---|---|
| `feature_correlation_table.csv` | All feature pairs that passed the correlation cutoff: `feature_1`, `feature_2`, `correlation`, `r_squared`, sorted by \|correlation\|. In `bipartite` mode, only pairs from different data types are included. |
| `feature_network_graph.graphml` | The whole network with node attributes. Opens in Cytoscape, Gephi, NetworkX or igraph. |
| `submodules/feature_network_graph_submodule<N>.graphml`, `..._nodes.csv`, `..._edges.csv` | One sub-network per community. |
| `submodules/feature_network_graph_submodule<N>_abundance.pdf` | Violin and strip plot of the mean scaled abundance of the community's members in each `group`. |

#### Files from `wgcna` only (`wgcna_results/`)

| File | Contents |
|---|---|
| `soft_threshold_plot.pdf` | Scale-free fit R² and mean connectivity for each candidate power. The chosen power is the smallest one with R² ≥ `r2_threshold`. |
| `dendrogram_modules.pdf` | Feature dendrogram with a color bar showing module membership. |
| `module_eigengenes.csv` | First principal component (eigengene) of each module x sample: the module's typical profile. |
| `module_membership_kme.csv` | Feature x module correlation with each eigengene. |
| `intramodular_connectivity_kin.csv` | Connectivity of each feature within its module. |
| `module_trait_correlation.csv` / `.pdf` | Correlation of each module eigengene with the numeric metadata columns. |
| `gs_vs_mm_plots/<module>_<trait>_gs_vs_mm.pdf` | For each module and trait: feature-trait correlation plotted against module membership. A strong positive trend means the module's core features are the ones linked to the trait. |

### 5.4 Group–pathway comparison (step 11)

This step asks: **does a data-driven group contain more members of a known pathway than chance would predict?**

#### Method

1. **Universe.** Starts from the node table. Each feature's pathway list (`pathway_enrichment.pathway_col`: `kegg_module` by default, or `kegg_pathway`, `kegg_reaction`, `modelseed_pathway`) is split into one row per feature–pathway pair. Filters, in order:
   - `exclude_nopathway`: drop `NOPATHWAY_*` placeholders.
   - `exclude_global_maps`: drop KEGG global/overview maps (`map011xx`/`map012xx`, e.g. `map01100` Metabolic pathways).
   - `bipartite_only`: keep only pathways with members from **every** data type present.
   - `min_features_per_pathway`: drop small pathways.
   - `min_features_per_group`: drop small groups.
   
   Unannotated features stay in the background count but are never tested.
2. **Test.** For every pathway x group pair with at least one shared feature, a one-sided hypergeometric test gives the probability of seeing at least that many pathway members in the group by chance. Here *M* = number of features in the universe, *n* = features in the pathway, *N* = features in the group and *k* = the overlap.
3. **Correction.** P-values are adjusted with Benjamini–Hochberg FDR across all tested pairs. Pairs with no overlap get p = padj = 1.
4. **Fold enrichment** = *k* / (*n* · *N* / *M*). A value of 1 means what chance predicts, and values above 1 mean over-representation.
5. **Display.** Only the top `top_n` pathways, ranked by `rank_by`, are drawn. **Ranking affects only the figures. Every pathway is tested and saved in the CSVs.**

#### Tables

| File | Contents |
|---|---|
| `exploded_universe.csv` | The tested universe: one row per feature–pathway pair (index = feature ID; columns `group`, `pathway`, and `dtype` when `bipartite_only` is on). |
| `full_overlap_matrix.csv` | Pathway x group: number of shared features (*k*). |
| `full_fold_enrichment_matrix.csv` | Pathway x group: fold enrichment. |
| `full_pvalue_matrix.csv` | Pathway x group: raw hypergeometric p-values. |
| `full_padj_matrix.csv` | Pathway x group: BH-adjusted p-values. **Use this table for significance calls** (compare it with `alpha` in the config). |
| `group_trend_values.csv` | Group x metadata category: the `trend_agg` (mean or median) of the selected-data values of the group's members, after pooling samples by `trend_collapse_by` (for example, timepoint). |
| `pathway_trend_values.csv` | The same summary for each pathway (all universe features in the pathway). |
| `run_config.json` | The `pathway_enrichment` settings used, plus the pathways and groups that were actually drawn. |

**Tip:** to list significant results, combine `full_padj_matrix.csv` (padj < `alpha`) with `full_fold_enrichment_matrix.csv` and `full_overlap_matrix.csv`. A significant result based on an overlap of 2 features is weak evidence even when its fold enrichment is high.

#### Figures: `heatmap_counts.png`, `heatmap_normalized.png`, `heatmap_enrichment.png`

All three share one layout, and each figure has a short legend printed above it:

- **Rows** are feature groups and **columns** are the displayed pathways. Both are ordered by hierarchical clustering (the dendrograms).
- **Grey squares with numbers** beside the dendrograms show how many features of each data type (labeled `tx`, `mx`, `px`) are in that row or column.
- **Trend tracks** (right of the heatmap for groups, below it for pathways) show the summarized abundance of the members in each metadata category. Red means above the feature average and blue means below. Each track has its own color scale, and the printed numbers are the actual values. Use them to see *when* or *where* a group or pathway is active.

| Figure | Cell color shows | How to read it |
|---|---|---|
| `heatmap_counts.png` | Number of the group's features in the pathway (viridis) | Where the shared features are. |
| `heatmap_normalized.png` | Fraction of the displayed pathway features that fall in each group (each column sums to 1; magma) | Bright cells: one group holds most of the pathway. |
| `heatmap_enrichment.png` | Fold enrichment (RdBu_r). Stars mark BH-adjusted significance: `*` < 0.05, `**` < 0.01, `***` < 0.001 | Dark-red starred cells are the main result: a co-varying group that is significantly over-represented in a known pathway. |

### 5.5 Other figures in the Analysis folder

| File | Contents |
|---|---|
| `TX_data_(raw).pdf`, `MX_data_(raw).pdf` (and `PX_...`) | Histogram of every value in `linked_data` (raw scale, log-scaled y-axis). Use it to check the input for a sensible distribution and for an excess of zeros. |
| `distribution_of_integrated_datasets.pdf` | Overlaid histograms of the scaled values of each dataset. After z-scoring, the datasets should overlap, centered at 0. A dataset that is visibly offset or much wider suggests a processing problem. |
| `Integrated_Data_PCA___<variable>.pdf` | PCA of `integrated_data_selected.csv` (PC1 against PC2), one file per variable in `user_settings.variable_list`. Points are samples or groups, colored by that variable, with density contours. Clear separation by a treatment variable means the selected features carry a treatment signal. Separation by a technical variable (batch, run order) is a warning sign. |

The notebook also shows PCAs of the full integrated data and of each dataset's `linked_data`, but these appear only in the notebook, not as files.

---

## 6. Configuration files: `configs/`

| File | Contents |
|---|---|
| `..._project_config.yml` | Project name, PI and proposal ID, list of dataset types, input and output paths, the link table location, and `variable_list` (metadata variables used for plots). |
| `..._data_processing_config.yml` | For each dataset: input folder, `input_state`, ID and annotation column mapping, filtering, normalization, transformation, variance filtering and replicate-filter settings. A dataset block such as `px:` can be present even if that dataset was not run. `project.dataset_list` says which ones were. |
| `..._analysis_config.yml` | `replicate_pairing`, `feature_selection`, `feature_grouping` and `pathway_enrichment`. |

See `project_config_parameters_explained.md`, `data_processing_config_parameters_explained.md` and `analysis_config_parameters_explained.md` for details on each parameter.

### Notebook: `notebooks/`

`..._notebook.ipynb` is a copy of the notebook as it was saved at the end of the run, including the printed log of every step: features kept and removed at each filter, size-factor ranges, number of groups found, annotation coverage and any warnings. It opens in Jupyter, VS Code or GitHub without the data. If something in the outputs looks surprising, read this log first.

---

## 7. Conventions used across files

**Feature IDs** always start with the dataset prefix:

| Prefix | Example | Origin |
|---|---|---|
| `tx_` | `tx_Os01g0100100` | Gene ID from the first column of `counts.csv`. |
| `px_` | `px_Os01g0100100` | Protein ID from the first column of `peak-height.csv`. |
| `mx_` | `mx_1234` or `mx_1234_hilic_pos` | FBMN scan/compound number. When more than one chromatography or polarity was run, a `_<chromatography>_<polarity>` suffix is added. The same number in two suffixes refers to **different** features. |

**Sample IDs**: columns from `linked_data.csv` onward use the `unique_group` names from the link table (see `integrated_metadata.csv`). In `unpaired` mode, Analysis-level columns are `group` values instead.

---

## 8. Reproducing the results

*Note: this is only used when you have the source code in a container that is running locally*

1. Note the two hashes from the folder names (for example `Dataset_Processing--5d861c04`, `Analysis--e340140e`).
2. Put the original raw inputs and link table back at the paths listed in the project config.
3. Place the `configs/` folder at `output_data/<project_name>/configs/` and start the notebook with:
   ```python
   project = objs.Project(data_processing_hash='5d861c04', analysis_hash='e340140e')
   ```
   This copies the hash-tagged configs back to the active config names. The workflow warns if the recomputed hashes do not match.
4. Run the notebook from top to bottom.

What may still differ:

- **Code version.** The hashes do not include the workflow code. Record the git commit used for the run (`git log -1` in the workflow repository) alongside the outputs.
- **External databases.** ModelSEED (downloaded from the `dev` branch of the ModelSEED database) and KEGG (REST API) are fetched live the first time and then cached. A fresh run without the cache can produce different pathway annotations if those databases have changed.
- **Cached outputs.** With `overwrite=False`, existing files in the hashed folders are reused instead of recomputed. To regenerate a step, delete its output file or run with `overwrite=True`.
- **Network colors** and the numbering of communities or clusters can change between runs. Group membership is what matters.

---

## 9. Interpretation guidance and caveats

- **Groups show co-variation, not causation.** Features in one group change together across samples or conditions. A gene and a metabolite in the same group are candidates for a functional link, not evidence of one.
- **`unpaired` mode has few data points.** With, for example, 6 conditions, each feature profile has 6 values, so strong correlations can arise by chance. Interpret groups more cautiously than in `paired` mode.
- **Feature selection shapes everything downstream.** Only `integrated_data_selected.csv` features are grouped and tested. A feature absent from the node table may simply not have been selected.
- **The background for enrichment is the grouped, filtered universe,** not the genome or metabolome. Enrichment p-values are relative to the features that were measured, passed filtering, were selected and were grouped.
- **Annotation coverage differs by data type.** Typically far more genes than metabolites have pathway annotations. A group may look like a gene-only cluster simply because its metabolites are unannotated. Check `feature_annotation_table.csv`.
- **Name-matched metabolites** (fuzzy matches to ModelSEED, Section 4) and `NOPATHWAY_*` placeholder pathways are lower-confidence links.
- **Imputed values.** For mx and px, values missing in a sample were filled with the feature's lowest observed value before log-transforming. Use `detected_mask.csv` to check whether a pattern rests on real detections.

---

## 10. Troubleshooting and FAQs

| Symptom | Likely explanation |
|---|---|
| `feature_network_edge_table.csv` is empty | The grouping method was not `network_modules`. These methods produce no edges. |
| `submodules/` is empty | Same reason. It is filled only by `network_modules`. |
| No `wgcna_results/` folder | The grouping method was not `wgcna`. |
| No `heatmap_*.png` or `full_*_matrix.csv` files | The group–pathway step did not finish. Check the end of the notebook for errors. |
| A pathway appears in the CSVs but not in the heatmap | It was outside the `top_n` displayed pathways. |
| A feature is missing from the node table | It was removed by filtering, not selected in step 8, labeled as noise, or (for `network_modules`) had no edge above the correlation cutoff. Trace it through the per-dataset tables in Section 4. |
| The number of samples is lower than expected | Only samples present in **all** datasets and listed in the link table are kept (step 3). |
| Two `Analysis--*` folders under one processing folder | Two analyses with different settings were run on the same processed data. Compare their `analysis_config.yml` files. |

---
