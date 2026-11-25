"""Overview: run summary, sample structure, and knowledge vs data-driven agreement."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from explorer import components as ui
from explorer import figures as fg
from explorer import theme as th


def layout(b, params):
    m = b.manifest
    f = b.features
    by_dtype = f.groupby("dtype").size()
    annotated = f.groupby("dtype")["is_annotated"].mean()
    coords = [c for c in ("PCA", "UMAP") if (c == "PCA" and "PC1" in b.samples) or (c == "UMAP" and "UMAP1" in b.samples)]
    stats = ui.stat_row([
        ui.stat("Samples" if not m.get("descriptive_only") else "Conditions", m.get("n_samples")),
        *[ui.stat(th.DTYPE_LABEL.get(d, d), f"{n:,}", f"{annotated.get(d, 0):.0%} annotated")
          for d, n in by_dtype.items()],
        ui.stat("Grouped features", f"{m.get('n_grouped'):,}", f"of {m.get('n_selected'):,} selected"),
        ui.stat("Data-driven groups", f"{m.get('n_groups'):,}",
                f"{m.get('n_noise'):,} noise features" if m.get("n_noise") else None),
        ui.stat("Annotation types", len(b.ann_types)),
    ])
    notes = []
    if m.get("descriptive_only"):
        notes.append(ui.note("Unpaired run: columns are conditions, so group and term association tests have little "
                             "or no replication. Treat their p-values as descriptive.", "warn"))
    if b.nesting:
        notes.append(ui.note("Nested variables: " + "; ".join(
            f"'{v}' is tested on '{u}'-level means (replicates of one {u} are not independent observations of {v})"
            for v, u in b.nesting.items()) + "."))
    return html.Div([
        ui.page_header("Overview", "How the integrated samples relate to each other, and how well the data-driven "
                                   "groups agree with knowledge-driven annotations overall."),
        stats, *notes,
        dbc.Row([
            dbc.Col(ui.card("Samples on integrated data", [
                html.Div([
                    ui.labeled("Embedding", ui.dropdown("ov-embed", coords, coords[0] if coords else None)),
                    ui.labeled("Color by", ui.dropdown("ov-color", b.variables, b.variables[0] if b.variables else None)),
                ], className="controls"),
                ui.graph("ov-samples", 430),
            ], "Each point is a sample, placed by its profile across all selected integrated features."), lg=6),
            dbc.Col(ui.card("Sample-to-sample correlation", ui.graph("ov-sim", 488),
                            "Pearson correlation on integrated features, clustered (average linkage)."), lg=6),
        ]),
        dbc.Row([
            dbc.Col(ui.card("Do data-driven groups recover knowledge-driven terms?", [
                ui.graph("ov-agree", 520),
                html.Div("Each row is an annotation type. For every term, the share of its co-annotated feature "
                         "pairs that fall in the same group is averaged over terms (dot). Gray: the same statistic "
                         "when group labels are shuffled (sizes kept). Blue = better than random (permutation "
                         "p < 0.05). AMI compares the partitions directly (0 = random, 1 = identical).",
                         className="small text-secondary mt-2"),
            ]), lg=7),
            dbc.Col([
                ui.card("Agreement by annotation type", html.Div(id="ov-agree-table")),
                ui.card("Group sizes", ui.graph("ov-sizes", 260)),
            ], lg=5),
        ]),
    ])


def register(app, b):
    @app.callback(Output("ov-samples", "figure"), Input("ov-embed", "value"), Input("ov-color", "value"),
                  Input("theme", "data"))
    def samples_fig(embed, color, theme):
        s = b.samples
        if embed == "UMAP":
            x, y, xl, yl = "UMAP1", "UMAP2", "UMAP 1", "UMAP 2"
        else:
            ev = b.manifest.get("sample_pca_explained", [0, 0])
            x, y = "PC1", "PC2"
            xl, yl = f"PC1 ({ev[0]:.0%})", f"PC2 ({ev[1]:.0%})" if len(ev) > 1 else "PC2"
        levels = b.levels(color) if color and b.var_types.get(color) != "continuous" else []
        return fg.sample_scatter(s, x, y, color if color else None, theme, levels, xl, yl, 430)

    @app.callback(Output("ov-sim", "figure"), Input("theme", "data"))
    def sim_fig(theme):
        return fg.similarity_heatmap(b.similarity, theme, 488)

    @app.callback(Output("ov-agree", "figure"), Output("ov-agree-table", "children"), Output("ov-sizes", "figure"),
                  Input("theme", "data"))
    def agree(theme):
        t = b.agree_types
        if t.empty:
            empty = th.empty_figure(theme, "No annotation types with enough grouped features")
            return empty, html.Div(), fg.size_histogram(b.groups, theme)
        tab = t.assign(annotation=t["ann_type"].map(b.ann_label)).sort_values("z", ascending=False)
        grid = ui.grid("ov-agree-grid", tab, [("annotation", "Annotation type", None, 190), ("ami", "AMI", "num"),
                                             ("z", "z vs null", "num"), ("p", "perm p", "p"),
                                             ("n_features", "features", "int"), ("n_terms", "terms", "int")],
                       height=330, export_name="agreement")
        return fg.agreement_dots(t, b.ann_label, theme, 520), grid, fg.size_histogram(b.groups, theme)
