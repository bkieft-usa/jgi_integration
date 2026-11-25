"""Methods: every statistic used in the explorer, in plain language, plus a paragraph for this run."""
from __future__ import annotations

from dash import dcc, html

from explorer import components as ui

REFERENCES = """
1. Smyth GK (2004). Linear models and empirical Bayes methods for assessing differential expression in microarray
   experiments. *Statistical Applications in Genetics and Molecular Biology* 3:Article 3.
2. Ritchie ME, Phipson B, Wu D, *et al.* (2015). limma powers differential expression analyses for RNA-sequencing
   and microarray studies. *Nucleic Acids Research* 43:e47.
3. Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical and powerful approach to
   multiple testing. *Journal of the Royal Statistical Society B* 57:289–300.
4. Lazic SE (2010). The problem of pseudoreplication in neuroscientific studies: is it affecting your analysis?
   *BMC Neuroscience* 11:5.
5. Langfelder P, Horvath S (2008). WGCNA: an R package for weighted correlation network analysis.
   *BMC Bioinformatics* 9:559.
6. Rivals I, Personnaz L, Taing L, Potier M-C (2007). Enrichment or depletion of a GO category within a class of
   genes: which test? *Bioinformatics* 23:401–407.
7. Phipson B, Smyth GK (2010). Permutation p-values should never be zero: calculating exact p-values when
   permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology* 9:Article 39.
8. Gillis J, Pavlidis P (2012). "Guilt by association" is the exception rather than the rule in gene networks.
   *PLoS Computational Biology* 8:e1002444.
9. Fruchterman TMJ, Reingold EM (1991). Graph drawing by force-directed placement.
   *Software: Practice and Experience* 21:1129–1164.
"""


def _levels_text(b, var):
    if b.var_types.get(var) == "continuous":
        return f"{var} (continuous)"
    n = len(b.levels(var))
    pairs = n * (n - 1) // 2
    return f"{var} ({n} levels; {pairs} pairwise comparison{'s' if pairs != 1 else ''})"


def _run_paragraph(b) -> str:
    m = b.manifest
    cfg = m.get("config", {})
    tested = [v for v in b.variables if v not in b.aliases]
    var_text = "; ".join(_levels_text(b, v) for v in tested) or "none"
    alias_text = "".join(f" '{a}' divided the samples exactly like '{k}' and was therefore not tested separately."
                         for a, k in b.aliases.items())
    nest_text = "".join(f" Because each {u} belongs to a single {v}, {v} was tested on {u}-level means "
                        f"(n = number of {u}s) to avoid pseudoreplication." for v, u in b.nesting.items())
    netw = m.get("network", {})
    if netw.get("kind") == "real":
        net_text = (f"The feature network shown is the correlation network used for grouping "
                    f"({netw.get('n_nodes', 0):,} features, {netw.get('n_edges', 0):,} edges).")
    else:
        net_text = (f"Because {m.get('grouping_method')} does not produce a network, a display network was derived by "
                    f"linking each grouped feature to its {netw.get('k')} most correlated grouped features (Pearson "
                    f"correlation of integrated z-scores; {netw.get('n_edges', 0):,} edges).")
    fdr = cfg.get("fdr", 0.05)
    return f"""
Feature abundance was tested against sample metadata with linear models and empirical-Bayes variance moderation
(limma method [1,2]), fitted separately for each data type on log-scale abundance. Variables tested: {var_text}.
For categorical variables, a moderated F test assessed any difference among levels, and moderated t tests compared
every pair of levels (effect: log2 fold change); continuous variables were tested for a non-zero slope.{alias_text}{nest_text}
P-values were adjusted with the Benjamini–Hochberg procedure [3] within each comparison across all features of all
data types. Each of the {m.get('n_groups', 0):,} data-driven groups ({m.get('grouping_method')}) and each annotation
term with at least {cfg.get('min_set_size_tests', 3)} features was summarized by its eigengene, the first principal
component of its members' standardized abundances, sign-aligned with their mean [5], and tested with the same model
(adjusted within each comparison, and for terms within each annotation type). Over-representation of each annotation
term in each group was assessed with a one-sided hypergeometric test [6] (background: grouped features annotated with
that annotation type; terms and groups with fewer than {cfg.get('min_term_size', 2)} and
{cfg.get('min_group_size', 2)} features excluded{'; KEGG global overview maps excluded' if cfg.get('exclude_global_maps') else ''}),
adjusted within each annotation type. Overall agreement between groups and each annotation type was measured as the
share of co-annotated feature pairs falling in the same group, compared with {cfg.get('n_permutations', 1000):,}
random permutations of group labels that preserved group sizes (permutation p-values as in [7]). {net_text}
Candidate functions for unannotated features were obtained by neighbour voting [8]: for each feature lacking an
annotation type, the number of its network neighbours carrying each term of that type was tested with the
hypergeometric test (background: network features annotated with that type), adjusted within each annotation type.
Significance was assessed at FDR < {fdr}.
""".strip()


def layout(b, params):
    m = b.manifest
    cfg = m.get("config", {})
    n_tests = m.get("n_tests", {})
    body = f"""
#### What is tested, and how

The explorer uses **three statistical tests**. Everything else on its pages is descriptive.

| Question | Method | Where |
|---|---|---|
| Does a feature, a data-driven group or a knowledge-driven term change with a metadata variable? | Moderated linear model (limma): F test across all levels, t test for every pair of levels, or slope test | Feature-Metadata Associations, group and term pages |
| Is a knowledge-driven term over-represented in a data-driven group? | One-sided hypergeometric test (Fisher's exact test) | Enrichment Matrix, group and term pages |
| Overall, do the groups recover an annotation type better than chance? | Label-permutation test | Overview |

Discovery reuses the hypergeometric test on a feature's network neighbours instead of a group.
All p-values are corrected with the Benjamini–Hochberg false discovery rate (FDR); the FDR threshold in the sidebar
(default {cfg.get('fdr', 0.05)}) only changes what is highlighted.

#### 1. Features vs metadata: moderated linear model

* **Data.** Each data type's normalized, log-scale abundance per sample, as produced by the workflow's
  data-processing stages (the replicate-filtered data).
* **Model.** For each feature, abundance = mean of its sample's level (categorical variable) or a straight line
  (continuous variable). The residual variance of each feature is **moderated**: shrunk toward the typical variance
  of all features (empirical Bayes). With 3–5 replicates per level, per-feature variances are very noisy; borrowing
  strength across thousands of features makes the tests reliable without inflating false positives. This is the
  standard limma method for omics data.
* **Comparisons.** For a categorical variable with *k* levels: one **moderated F test** ("is there any difference among
  the levels?") and a **moderated t test for each of the k(k−1)/2 pairs** of levels (effect = log2 fold change, first
  level minus second). Choose the order in the app; reversing it only flips the sign. For a continuous variable:
  a t test of the slope.
* **Multiple testing.** Benjamini–Hochberg within each comparison, across all features of all data types. Adding more
  comparisons does not change the correction for any single comparison.
* **Tests in this bundle:** {n_tests.get('moderated_t', 0):,} t tests and {n_tests.get('moderated_F', 0):,} F tests.

#### 2. Nested variables (pseudo-replication)

If every level of one variable lies inside a single level of another (e.g. organisms within lineages), replicate
samples of one organism are not independent evidence about its lineage. Such variables are tested on the means of the
finer level, so *n* is the number of organisms, not samples. This is detected automatically from the metadata.
{"Detected here: " + "; ".join(f"**{v}** tested on **{u}**-level means" for v, u in b.nesting.items()) + "." if b.nesting else "None detected in this run."}
Variables that divide the samples identically are tested once
{"(here: " + ", ".join(f"**{a}** = **{k}**" for a, k in b.aliases.items()) + ")." if b.aliases else "(none in this run)."}

#### 3. Groups and terms vs metadata: eigengene + the same model

A group or term contains many features. Each is summarized by its **eigengene**: the first principal component of
its members' integrated z-scores, scaled to unit variance and signed to follow the members' average. The eigengene
is the single profile that best represents the set (the standard WGCNA "module eigengene"). It is then tested
exactly like a feature (section 1). Terms need at least {cfg.get('min_set_size_tests', 3)} features. Effects are in
eigengene standard deviations.

#### 4. Group vs term enrichment: hypergeometric test

For every term in every group: are more of the group's features annotated with the term than expected by chance?
One-sided hypergeometric test (equivalent to a one-sided Fisher's exact test) with

* *k* = group members with the term, *n* = group members annotated with this annotation type,
* *K* = all grouped features with the term, *N* = all grouped features annotated with this annotation type.

Restricting the background to annotated features prevents groups with many unannotated features from looking
depleted. HDBSCAN noise / WGCNA grey features stay in the background but are never tested as a group. Terms with
fewer than {cfg.get('min_term_size', 2)} grouped features and groups with fewer than {cfg.get('min_group_size', 2)}
features are not tested. BH within each annotation type.

#### 5. Overall agreement: permutation test

For one annotation type, take every pair of features that share a term and ask what share of those pairs fall in the
same group (averaged over terms). Then shuffle the group labels {cfg.get('n_permutations', 1000):,} times (group sizes
and annotations unchanged) to get the share expected by chance. p = (1 + shuffles at least as high as observed) /
(1 + shuffles).

#### 6. The feature network

{"The run grouped features with a correlation network, so the network shown is that network: its edges are what defined the groups." if b.network_kind == "real" else f"The run's grouping method does not produce a network, so for display and discovery each grouped feature is linked to its {m.get('network', {}).get('k', 5)} most correlated grouped features (Pearson correlation of integrated z-scores). This describes relationships; it is not how the groups were formed. The 'Clustering tree' view shows that."}
Positions come from a force-directed layout (Fruchterman–Reingold) of each connected component. Distances on the
picture are approximate.

#### 7. Unannotated feature discovery: neighbour voting

"Guilt by association": a feature whose neighbours share a function may share it too. For a feature that lacks an
annotation type, count its directly connected neighbours that are annotated with that type (*n*) and how many of
them carry each term (*k*), then apply the hypergeometric test of section 4 with this neighbourhood in place of a
group. BH within each annotation type over all (feature, term) pairs tested. The "same data-driven group" option uses
group enrichment (section 4) instead. Results are hypotheses to test experimentally, not annotations.

#### Descriptive numbers (not tests)

* **kME**: correlation of a feature with its group's eigengene (how typical it is of the group).
* **Coherence**: mean pairwise correlation among a group's members.
* **Largest share in one group**: the fraction of a term's grouped features that sit in its most common group.
* **Fold enrichment**: observed / expected overlap, k / (K·n/N).

#### Methods text for this run

Ready to adapt for a publication. Numbers and settings are filled in from this bundle.
"""
    return html.Div([
        ui.page_header("Methods", "Every statistic in the explorer, why it was chosen, and a methods paragraph for "
                                  "this run."),
        html.Div([
            dcc.Markdown(body, className="methods-md"),
            html.Div(dcc.Markdown(_run_paragraph(b)), className="methods-paragraph"),
            dcc.Markdown("#### References\n" + REFERENCES, className="methods-md mt-3"),
            html.Div(f"Explorer bundle schema {m.get('schema_version')}, built {m.get('built_at')}.",
                     className="small text-secondary mt-3"),
        ], className="methods"),
    ])


def register(app, b):
    return None
