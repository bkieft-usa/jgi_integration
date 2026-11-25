"""Associations: which features, groups, or terms differ across a metadata variable."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Input, Output, State, html, no_update

from explorer import components as ui
from explorer import figures as fg
from explorer import theme as th

ENTITIES = [{"label": "Features", "value": "feature"}, {"label": "Data-driven groups", "value": "group"},
            {"label": "Knowledge terms", "value": "term"}]
TESTS = [{"label": "Self-contained (term score vs metadata)", "value": "moderated"},
         {"label": "Competitive (CAMERA: vs other features)", "value": "camera"}]
HELP = {
    "feature": "Moderated linear model per feature on log-scale abundance (limma-style empirical Bayes). "
               "Effect = log2 fold change of the level vs the mean of the other levels (or slope for continuous "
               "variables).",
    "group": "The same model on each group's eigengene (first principal component of its members' z-scores). "
             "Effect is in eigengene units (standard deviations). Groups are not tested against random features: "
             "they are co-varying by construction, so that test would always pass.",
    "term": "Self-contained: the term's mean z-score profile is tested like a group. Competitive (CAMERA): is the "
            "term more associated than the other features, after correcting for correlation among its members? "
            "Use CAMERA to find pathways enriched along an experimental axis.",
}


def layout(b, params):
    entity = params.get("entity", "feature")
    variable = params.get("variable", b.variables[0] if b.variables else None)
    at_opts = [{"label": a["label"], "value": a["key"]} for a in b.ann_types]
    return html.Div([
        ui.page_header("Associations with metadata",
                       "Pick a metadata variable and comparison to list the features, groups, or knowledge terms "
                       "that are significantly associated with it. Click a point or row to open it."),
        html.Div([
            ui.labeled("Test", ui.dropdown("as-entity", ENTITIES, entity, style={"minWidth": "190px"})),
            ui.labeled("Metadata variable", ui.dropdown("as-variable", b.variables, variable)),
            ui.labeled("Comparison", ui.dropdown("as-contrast", [], None, style={"minWidth": "260px"})),
            html.Div(ui.labeled("Annotation type", ui.dropdown("as-anntype", at_opts,
                                                               params.get("ann_type", at_opts[0]["value"] if at_opts else None),
                                                               style={"minWidth": "220px"})), id="as-anntype-wrap"),
            html.Div(ui.labeled("Term test", ui.dropdown("as-test", TESTS, params.get("test", "camera"),
                                                         style={"minWidth": "300px"})), id="as-test-wrap"),
            html.Div(ui.labeled("Data type", ui.dropdown("as-dtype", [{"label": "All", "value": "all"}] +
                                                         [{"label": th.DTYPE_LABEL.get(d, d), "value": d} for d in b.dtypes],
                                                         "all")), id="as-dtype-wrap"),
        ], className="controls"),
        html.Div(id="as-help", className="small text-secondary mb-2"),
        html.Div(id="as-note"),
        ui.card("Results", ui.graph("as-plot", 470), subtitle=None),
        ui.card("Significant results", html.Div(id="as-table")),
    ])


def register(app, b):
    @app.callback(Output("as-contrast", "options"), Output("as-contrast", "value"),
                  Output("as-anntype-wrap", "style"), Output("as-test-wrap", "style"), Output("as-dtype-wrap", "style"),
                  Output("as-help", "children"),
                  Input("as-entity", "value"), Input("as-variable", "value"), State("url", "search"),
                  State("as-contrast", "value"))
    def contrasts(entity, variable, search, current):
        if not variable:
            return [], None, {}, {}, {}, ""
        from urllib.parse import parse_qs
        wanted = parse_qs((search or "").lstrip("?")).get("contrast", [None])[0]
        opts = b.contrasts(entity, variable)
        value = next((v for v in (current, wanted) if v in opts), None) or \
            next((o for o in opts if o != "omnibus"), opts[0] if opts else None)
        hide, show = {"display": "none"}, {}
        return ([{"label": "All levels (omnibus F test)" if o == "omnibus" else o, "value": o} for o in opts], value,
                show if entity == "term" else hide, show if entity == "term" else hide,
                show if entity == "feature" else hide, HELP.get(entity, ""))

    @app.callback(Output("as-plot", "figure"), Output("as-table", "children"), Output("as-note", "children"),
                  Input("as-entity", "value"), Input("as-variable", "value"), Input("as-contrast", "value"),
                  Input("as-anntype", "value"), Input("as-test", "value"), Input("as-dtype", "value"),
                  Input("filters", "data"), Input("theme", "data"))
    def results(entity, variable, contrast, ann_type, test, dtype, filters, theme):
        fdr = (filters or {}).get("fdr", b.fdr)
        if not (variable and contrast):
            return th.empty_figure(theme, "Choose a variable and comparison"), html.Div(), None
        kw = {}
        if entity == "term":
            kw["ann_type"] = ann_type
            kw["test"] = "camera" if test == "camera" and contrast != "omnibus" else (
                "moderated_F" if contrast == "omnibus" else "moderated_t")
        if entity == "feature" and dtype and dtype != "all":
            kw["dtype"] = dtype
        df = b.association_table(entity, variable, contrast, **kw)
        notes = []
        if b.nesting.get(variable):
            notes.append(ui.note(f"'{variable}' is nested over '{b.nesting[variable]}': tested on "
                                 f"{b.nesting[variable]}-level means (n = number of {b.nesting[variable]} levels)."))
        if b.manifest.get("descriptive_only") and entity != "feature":
            notes.append(ui.note("Unpaired run: these tests have little replication; treat p-values as descriptive.",
                                 "warn"))
        if df.empty:
            return th.empty_figure(theme, "No tests for this selection (too little replication?)"), html.Div(), notes
        if entity == "feature":
            df["name"] = df["entity_id"].map(lambda f: b.feature_label(f))
            df["group"] = df["entity_id"].map(b.features["group"])
            id_col, extra = "feature_id", [("display_name", "name"), ("group", "group")]
            df["display_name"] = df["entity_id"].map(b.features["display_name"])
        elif entity == "group":
            df["name"] = df["entity_id"] + "  " + df["entity_id"].map(b.groups["top_term"]).fillna("")
            df["size"] = df["entity_id"].map(b.groups["size"])
            id_col, extra = "group", [("size", "features", "int"), ("top_term", "top enriched term")]
            df["top_term"] = df["entity_id"].map(b.groups["top_term"])
        else:
            df["name"] = df["entity_id"].map(lambda e: b.term_label(e))
            df["term_name"] = df["entity_id"].map(b.terms["term_name"])
            df["n_features"] = df["n"]
            id_col, extra = "entity_id", [("term_name", "name"), ("n_features", "features in test", "int")]
        df[id_col] = df["entity_id"]
        if contrast == "omnibus":
            top = df.sort_values("p").head(40)
            fig = fg.ranked_bars(top, theme, fdr, "statistic", "name", "moderated F")
            effect_cols = [("statistic", "F", "num")]
        else:
            label = {"feature": "log2 fold change" if b.var_types.get(variable) != "continuous" else "slope (log2 / unit)",
                     "group": "Δ eigengene (SD)", "term": "mean z of term members (CAMERA)" if test == "camera" else "Δ term score (SD)"}
            fig = fg.volcano(df, theme, fdr, label[entity])
            is_camera = entity == "term" and test == "camera"
            effect_cols = [("effect", "effect", "num"), ("statistic", "CAMERA stat" if is_camera else "moderated t", "num")]
        sig = df[df["padj"] < fdr].sort_values("p")
        shown = sig if len(sig) else df.sort_values("p").head(200)
        caption = f"{len(sig):,} of {len(df):,} significant at FDR < {fdr}" if len(sig) else \
            f"None significant at FDR < {fdr}; showing the 200 smallest p-values"
        cols = [(id_col, None, None, 230)] + extra + effect_cols + [("p", "p", "p"), ("padj", "padj", "p")]
        if entity == "term" and test == "camera" and contrast != "omnibus" and "aux" in shown:
            shown = shown.assign(correlation=shown["aux"])
            cols.append(("correlation", "within-set corr.", "num"))
        table = html.Div([html.Div(caption, className="small text-secondary mb-1"),
                          ui.grid("as-grid", shown, cols, height=460, export_name=f"associations_{variable}")])
        return fig, table, notes

    @app.callback(Output("url", "href", allow_duplicate=True), Input("as-plot", "clickData"), State("as-entity", "value"),
                  prevent_initial_call=True)
    def click(click, entity):
        if not click:
            return no_update
        cd = click["points"][0].get("customdata")
        if cd is None:
            return no_update
        key = cd[0]
        return {"feature": ui.href("/feature", id=key), "group": ui.href("/groups", group=key),
                "term": ui.href("/terms", term=key)}[entity]
