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
| **Overview** | How do samples relate on the integrated data (PCA/UMAP, correlation heatmap)? Overall, how well do data-driven groups agree with each annotation type? |
| **Associations** | Pick a metadata variable and comparison. Which features, groups or terms are significantly associated with it? Shows a volcano plot (or a ranked F-statistic for "all levels") and a table; click to open any result. |
| **Data-driven groups** | A map of the groups. Three views: a network of correlated groups, a UMAP of all features, or the clustering's own tree. Highlight a KO, pathway or feature to see where it falls. Click a group to see its abundance profile across metadata, its annotation composition with enrichment, its members (annotated or not) and a network of its members. |
| **Knowledge terms** | Type a pathway, module, reaction, KO, EC, COG or compound class. See (a) how its features divide among groups, (b) its abundance across metadata with tests, (c) every feature carrying it, and (d) summary statistics. You can also browse all terms of one annotation type. |
| **Enrichment matrix** | A heatmap of significant term × group pairs for one annotation type. Click a cell to list the overlapping features. |
| **Discovery** | Unannotated features in groups with significant enrichment, ranked as functional hypotheses. |
| **Feature** | One feature: annotations, group, abundance by metadata, test results, most correlated group members. |

## Statistics and why they were chosen

**Features vs metadata.** A linear model is fitted per feature on log-scale abundance, with empirical-Bayes variance moderation (the limma method, Smyth 2004). Moderation borrows information across features, so it stays reliable with 3–5 replicates, where per-feature tests are underpowered or unstable.
- Categorical variables get a moderated F test across all levels ("omnibus") and a moderated t test for each level vs the mean of the other levels. The effect is the log2 fold change.
- Continuous variables get a test of the slope.
- P-values are Benjamini–Hochberg corrected within each variable and comparison, across all data types.

**Nested variables (pseudo-replication).** If every level of one variable falls inside a single level of another (e.g. organisms within lineages), replicates of one organism are not independent observations of its lineage. Such variables are tested on means at the finer level, so n is the number of organisms, not the number of samples. Nesting is detected automatically, listed on the Overview page, and noted wherever it applies. Without this correction, false positives for lineage-type variables are typically several times the nominal rate.

**Groups vs metadata.** Each group is summarized by its eigengene: the first principal component of its members' z-scores, sign-aligned with the mean profile. The eigengene is tested with the same moderated model. Groups are **not** compared against random features: their members are correlated by construction, so that comparison would always be significant.

**Terms vs metadata.** Two complementary tests:
- **Self-contained:** the term's mean z-score profile is tested like a group. Is the term, as a set, associated with the variable?
- **Competitive (CAMERA, Wu & Smyth 2012):** are the term's features more associated than other features? CAMERA corrects for correlation among the term's features. Without that correction, co-regulated pathway members make random-set comparisons wildly overconfident. Use CAMERA to find pathways enriched along an experimental axis.

**Groups vs terms (enrichment).** A one-sided hypergeometric test of every term in every group, Benjamini–Hochberg corrected within each annotation type. The background is the grouped features annotated with that type, so differences in annotation coverage between groups do not create or hide enrichment. Noise/grey features stay in the background but are not tested as a group.

**Overall agreement (Overview).** For each annotation type:
- **Co-membership.** For every term, the share of its feature pairs that fall in the same group, averaged over terms. This is compared with the same statistic after shuffling group labels 1,000 times (group sizes and annotations kept), giving a z-score and a permutation p-value.
- **AMI and ARI.** These compare the group partition with each feature's most specific term (0 = random, 1 = identical). They are usually small, because a few hundred groups cannot reproduce thousands of terms; read them relative to each other.

**Per term:** *concentration* is the largest share of the term in one group, with a permutation p-value. **Per group:** *purity* and *entropy* of its term composition, and *coherence*, the mean correlation among its members.

**Unpaired runs.** The integrated matrix has one column per condition, so group and term tests have little or no replication. The explorer labels them descriptive. Feature tests still use the per-sample data.

## Structure views of the groups

The view that best represents how groups were formed depends on the method:

| Method | Faithful view |
|---|---|
| hierarchical clustering | The clustering's own linkage, cut at the group level (dendrogram). |
| HDBSCAN | The condensed cluster tree. Clusters persist over a range of density (λ); selected clusters are the groups. |
| network modules | The real correlation edges (within-group network). |
| WGCNA | Eigengene network (the standard WGCNA module–module view). |

A graph derived after the fact (e.g. from a hierarchical clustering) is a valid **description** of how features or groups relate, but it is not what formed the groups. The explorer labels such views accordingly:
- The **group network** links groups whose eigengenes correlate. It is valid for every method and describes relationships between groups, not group formation.
- Inside a group without real edges, the member network is a "correlation view": each member is linked to its most correlated members.

Runs made before the structure was saved get a dendrogram of group centroids, marked "approximate", or an HDBSCAN tree recomputed and verified identical to the run.

## Settings (`explorer.yml`, optional)

These settings are not part of the run hash. Pass the file with `--explorer-config` to `run`, `build-explorer` or `explore`. See `config_examples/explorer.yml`.

| Key | Default | Meaning |
|---|---|---|
| `n_permutations` | 1000 | Label permutations for agreement null distributions. |
| `min_term_size` / `min_group_size` | 2 / 2 | Smallest term / group tested for enrichment. |
| `min_set_size_tests` | 3 | Smallest term for term-score and CAMERA tests. |
| `fdr` | 0.05 | Default FDR in the app (can be changed in the sidebar). |
| `variable_types` | auto | Force a metadata variable to be `categorical` or `continuous` (numeric with ≥ 6 distinct values is continuous by default). |
| `eigengene_network.min_abs_r`, `.top_k` | 0.7, 5 | Edge threshold and partners kept per group in the group network. |

Metadata variables default to `user_settings.variable_list` if set, otherwise every link-table column except sample IDs.
