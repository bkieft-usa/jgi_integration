"""Feature-metadata associations: which features, groups, or terms differ across a metadata variable."""
from __future__ import annotations

from urllib.parse import parse_qs

from dash import Input, Output, State, html, no_update

from explorer import components as ui
from explorer import figures as fg
from explorer import theme as th

ENTITIES = [{"label": "Features", "value": "feature"}, {"label": "Data-driven groups", "value": "group"},
            {"label": "Knowledge-driven terms", "value": "term"}]
ALL_LEVELS = "__all__"
HELP = {
    "feature": "Each feature is tested on its log-scale abundance with a moderated linear model (limma-style). "
               "Pairwise effect = log2 fold change (first level minus second).",
    "group": "Each data-driven group is summarized by its eigengene (the first principal component of its members' "
             "z-scores) and tested with the same model. Effect is in eigengene units (standard deviations).",
    "term": "Each knowledge-driven term with at least 3 features is summarized by its eigengene, exactly like a "
            "group, and tested with the same model. Use this to find pathways that change along an experimental axis.",
}


def layout(b, params):
    entity = params.get("entity", "feature")
    variable = params.get("variable", b.variables[0] if b.variables else None)
    at_opts = [{"label": a["label"], "value": a["key"]} for a in b.ann_types]
    return html.Div([
        ui.page_header("Feature-Metadata Associations",
                       "Pick a metadata variable and a comparison to list the features, data-driven groups, or "
                       "knowledge-driven terms that differ significantly. Click a point or a row to open it."),
        html.Div([
            ui.labeled("Test", ui.dropdown("as-entity", ENTITIES, entity, style={"minWidth": "210px"})),
            ui.labeled("Metadata variable", ui.dropdown("as-variable", b.variable_options(), variable,
                                                        style={"minWidth": "220px"})),
            html.Div(ui.labeled("Compare", ui.dropdown("as-a", [], None, style={"minWidth": "220px"})),
                     id="as-a-wrap"),
            html.Div(ui.labeled("against", ui.dropdown("as-b", [], None, style={"minWidth": "220px"})),
                     id="as-b-wrap"),
            html.Div(ui.labeled("Annotation type", ui.dropdown("as-anntype", at_opts,
                                                               params.get("ann_type", at_opts[0]["value"] if at_opts else None),
                                                               style={"minWidth": "220px"})), id="as-anntype-wrap"),
            html.Div(ui.labeled("Data type", ui.dropdown("as-dtype", [{"label": "All", "value": "all"}] +
                                                         [{"label": th.DTYPE_LABEL.get(d, d), "value": d} for d in b.dtypes],
                                                         "all")), id="as-dtype-wrap"),
        ], className="controls"),
        html.Div(id="as-help", className="small text-secondary mb-2"),
        html.Div(id="as-note"),
        ui.card("Results", ui.graph("as-plot", 470)),
        ui.card("Significant results", html.Div(id="as-table")),
    ])


def _short(text: str, n: int = 48) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _comparison(b, entity, variable, a, bb):
    """``(stored contrast, sign, label)`` for the current controls."""
    if b.var_types.get(b.canonical(variable)) == "continuous":
        return "slope", 1, f"slope of {variable}"
    if a == ALL_LEVELS or a is None:
        return "omnibus", 1, f"any difference among {variable} levels (F test)"
    contrast, sign = b.resolve_pair(entity, variable, a, bb)
    return contrast, sign, f"{a} vs {bb}"


def register(app, b):
    @app.callback(Output("as-a", "options"), Output("as-a", "value"), Output("as-a-wrap", "style"),
                  Output("as-anntype-wrap", "style"), Output("as-dtype-wrap", "style"), Output("as-help", "children"),
                  Input("as-entity", "value"), Input("as-variable", "value"), State("as-a", "value"),
                  State("url", "search"))
    def first_level(entity, variable, current, search):
        hide, show = {"display": "none"}, {}
        vis = (show if entity == "term" else hide, show if entity == "feature" else hide, HELP.get(entity, ""))
        if not variable or b.var_types.get(b.canonical(variable)) == "continuous":
            return [], None, hide, *vis
        levels = b.comparison_levels(entity, variable)
        opts = ([{"label": "All levels (F test)", "value": ALL_LEVELS}] if len(levels) > 2 else []) + \
            [{"label": str(lv), "value": lv} for lv in levels]
        wanted = parse_qs((search or "").lstrip("?")).get("a", [None])[0]
        values = [o["value"] for o in opts]
        value = next((v for v in (current, wanted) if v in values), values[0] if values else None)
        return opts, value, show, *vis

    @app.callback(Output("as-b", "options"), Output("as-b", "value"), Output("as-b-wrap", "style"),
                  Input("as-a", "value"), Input("as-entity", "value"), Input("as-variable", "value"),
                  State("as-b", "value"))
    def second_level(a, entity, variable, current):
        if not a or a == ALL_LEVELS or not variable:
            return [], None, {"display": "none"}
        others = [lv for lv in b.comparison_levels(entity, variable) if lv != a]
        return ([{"label": str(lv), "value": lv} for lv in others],
                current if current in others else (others[0] if others else None), {})

    @app.callback(Output("as-plot", "figure"), Output("as-table", "children"), Output("as-note", "children"),
                  Input("as-entity", "value"), Input("as-variable", "value"), Input("as-a", "value"),
                  Input("as-b", "value"), Input("as-anntype", "value"), Input("as-dtype", "value"),
                  Input("filters", "data"), Input("theme", "data"))
    def results(entity, variable, a, bb, ann_type, dtype, filters, theme):
        fdr = (filters or {}).get("fdr", b.fdr)
        if not variable:
            return th.empty_figure(theme, "Choose a variable"), html.Div(), None
        contrast, sign, label = _comparison(b, entity, variable, a, bb)
        if contrast is None:
            return th.empty_figure(theme, "Choose two levels to compare"), html.Div(), None
        kw = {"sign": sign}
        if entity == "term":
            kw["ann_type"] = ann_type
        if entity == "feature" and dtype and dtype != "all":
            kw["dtype"] = dtype
        df = b.association_table(entity, variable, contrast, **kw)
        notes = []
        canon = b.canonical(variable)
        if canon != variable:
            notes.append(ui.note(f"'{variable}' splits the samples exactly like '{canon}', so it is tested once; "
                                 f"these are the '{canon}' results."))
        if b.nesting.get(canon):
            u = b.nesting[canon]
            notes.append(ui.note(f"'{canon}' is nested over '{u}': tested on {u}-level means "
                                 f"(n = number of {u} levels), because replicates of one {u} are not independent "
                                 f"observations of {canon}."))
        if b.manifest.get("descriptive_only") and entity != "feature":
            notes.append(ui.note("Unpaired run: these tests have little replication; treat p-values as descriptive.",
                                 "warn"))
        if df.empty:
            return th.empty_figure(theme, "No tests for this selection (too little replication?)"), html.Div(), notes
        if entity == "feature":
            df["name"] = df["entity_id"].map(lambda f: _short(b.feature_label(f)))
            df["group"] = df["entity_id"].map(b.features["group"])
            df["display_name"] = df["entity_id"].map(b.features["display_name"])
            id_col, extra = "feature_id", [("display_name", "name"), ("group", "group", None, 120)]
        elif entity == "group":
            df["name"] = (df["entity_id"] + "  " + df["entity_id"].map(b.groups["top_term"]).fillna("")).map(_short)
            df["size"] = df["entity_id"].map(b.groups["size"])
            df["top_term"] = df["entity_id"].map(b.groups["top_term"])
            id_col, extra = "group", [("size", "features", "int", 90), ("top_term", "top enriched term")]
        else:
            df["name"] = df["entity_id"].map(lambda e: b.term_label(e, 48))
            df["term_name"] = df["entity_id"].map(b.terms["term_name"])
            df["n_features"] = df["entity_id"].map(b.terms["n_features"])
            id_col, extra = "entity_id", [("term_name", "name"), ("n_features", "features", "int", 90)]
        df[id_col] = df["entity_id"]
        if contrast == "omnibus":
            top = df.sort_values("p").head(40)
            fig = fg.ranked_bars(top, theme, fdr, "statistic", "name", "moderated F")
            effect_cols = [("statistic", "F", "num", 90)]
        else:
            unit = {"feature": "log2 fold change", "group": "Δ eigengene (SD)", "term": "Δ term eigengene (SD)"}[entity]
            if contrast == "slope":
                unit = {"feature": "slope (log2 per unit)", "group": "slope (SD per unit)",
                        "term": "slope (SD per unit)"}[entity]
            fig = fg.volcano(df, theme, fdr, f"{unit}: {label}" if contrast != "slope" else unit)
            effect_cols = [("effect", "effect", "num", 90), ("statistic", "moderated t", "num", 110)]
        sig = df[df["padj"] < fdr].sort_values("p")
        shown = sig if len(sig) else df.sort_values("p").head(200)
        caption = f"{label}: {len(sig):,} of {len(df):,} significant at FDR < {fdr}" if len(sig) else \
            f"{label}: none significant at FDR < {fdr}; showing the 200 smallest p-values"
        cols = [(id_col, None, None, 230)] + extra + effect_cols + [("p", "p", "p", 90), ("padj", "padj", "p", 90)]
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
