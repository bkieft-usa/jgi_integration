# Integration Explorer

The explorer is an interactive web app for one finished run. It shows:
- which features, data-driven groups and knowledge-driven terms change with your experimental variables;
- how well the groups agree with known biology;
- where unannotated features fall among annotated ones.

Everything is precomputed when the run finishes (`explorer/` folder), so the app responds immediately and never recomputes anything.

Start it from a terminal:

```bash
jgi-integrate explore --project-dir <results>/<project> -d <data hash> -a <analysis hash>
jgi-integrate explore --bundle <path to explorer/>     # a copied bundle works on its own
```

Or start it from the last cell of `workflow.ipynb` with `start_explorer(project.analysis_dir)`; that cell runs the explorer in the background, and `stop_explorer()` stops it. In the JupyterLab container, either way works without extra options (see [setup.md](setup.md), step 4).

Open `http://localhost:8050`. The sidebar sets the false discovery rate, the minimum group size and whether HDBSCAN noise / WGCNA grey features are shown. The search box in the top bar finds features, groups and terms of any annotation type. Every page has a link you can share: the URL holds the selection (e.g. `/groups?group=group_12`).

## Pages

| Page | Question it answers |
|---|---|
| **Overview** | How do samples relate on the integrated data (PCA/UMAP, correlation heatmap)? Overall, do data-driven groups recover each annotation type better than random groups would (plot, with the same numbers in the table below it)? |
| **Feature-Metadata Associations** | Pick a metadata variable, then "all levels" (F test) or two levels to compare (A vs B). Which features, groups or terms differ significantly? Volcano plot (or ranked F statistics) and a table; click to open any result. |
| **Data-Driven Groupings** | The full feature network: the run's own correlation network for network methods, or a nearest-neighbour correlation network (labelled as derived) for hierarchical clustering, HDBSCAN and others, which also get their clustering tree as a second view. Color by group, data type or annotated/unannotated. Highlight several terms or features at once. Click a feature to open its group: abundance profile, annotation composition with enrichment, members (annotated or not), within-group network, and the distribution of group sizes. |
| **Knowledge-Driven Groupings** | Type a pathway, module, reaction, KO, EC, COG or compound class. See (a) how its features divide among groups, (b) its eigengene across metadata with tests, (c) every feature carrying it, and (d) summary statistics. You can also browse all terms of one annotation type. |
| **Enrichment Matrix** | A heatmap of significant term × group pairs for one annotation type. Click a cell to list the overlapping features. |
| **Unannotated Feature Discovery** | Three ways in: (1) pick an annotation and list unannotated features directly connected to (or in the same group as) features that carry it; (2) pick a feature and see the annotations of its network neighbours and of its most correlated annotated features; (3) a ranked list of every neighbour-voting hypothesis. |
| **Methods** | Every statistic in plain language, why it was chosen, references, and a methods paragraph for this run with its numbers filled in, ready to adapt for a publication. |
| **Feature** | One feature: annotations, group, abundance by metadata, test results, network neighbours. |

The sidebar's "This run" section lists the samples, groups and annotation types; click a line to expand it.

## Statistics

The explorer uses three tests; the Methods tab describes them in full, with references.

1. **Moderated linear model (limma method)** for features, data-driven groups and knowledge-driven terms against each metadata variable. Groups and terms are summarized by their eigengene (first principal component of their members' z-scores) and tested exactly like a feature. Categorical variables get an F test across all levels and a t test for every pair of levels; continuous variables get a slope test. Nested variables (e.g. lineage over organism) are tested on means of the finer level, so replicates are not counted as independent evidence. Variables that split the samples identically (e.g. `group` = `organism`) are tested once.
2. **One-sided hypergeometric test** for a term in a group (background: grouped features annotated with that type). Discovery reuses it on a feature's network neighbours (neighbour voting).
3. **Label-permutation test** for overall agreement between groups and an annotation type.

P-values are Benjamini–Hochberg corrected: within each comparison for metadata tests (and within each annotation type for terms), and within each annotation type for enrichment and discovery. Adding pairwise comparisons does not change the correction for any single comparison.

**Unpaired runs.** The integrated matrix has one column per condition, so group and term tests have little or no replication. The explorer labels them descriptive. Feature tests still use the per-sample data.

## Settings (`explorer.yml`, optional)

These settings are not part of the run hash. Pass the file with `--explorer-config` to `run`, `build-explorer` or `explore`. See `config_examples/explorer.yml`.

| Key | Default | Meaning |
|---|---|---|
| `n_permutations` | 1000 | Label permutations for the agreement test. |
| `min_term_size` / `min_group_size` | 2 / 2 | Smallest term / group tested for enrichment. |
| `min_set_size_tests` | 3 | Smallest term that gets an eigengene (metadata tests). |
| `fdr` | 0.05 | Default FDR in the app (can be changed in the sidebar). |
| `variable_types` | auto | Force a metadata variable to be `categorical` or `continuous` (numeric with ≥ 6 distinct values is continuous by default). |
| `network_k` | 5 | Neighbours per feature in the derived network (methods without edges). |
| `max_votes_per_feature` | 20 | Best discovery terms kept per feature and annotation type. |

Metadata variables default to `user_settings.variable_list` if set, otherwise every link-table column except sample IDs.
