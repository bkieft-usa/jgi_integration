"""Knowledge terms: look up a pathway / KO / EC / COG / compound class and see its project statistics."""
from __future__ import annotations

from urllib.parse import parse_qs

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, html, no_update

from explorer import components as ui
from explorer import figures as fg
from explorer import theme as th


def _selected(search):
    return parse_qs((search or "").lstrip("?")).get("term", [None])[0]


def layout(b, params):
    term = params.get("term")
    opts = [{"label": b.term_label(term, 120), "value": term}] if term else []
    return html.Div([
        ui.page_header("Knowledge-driven terms",
                       "Type a pathway, module, reaction, KO, EC number, COG, or compound class to see how its "
                       "features split across data-driven groups, whether it changes across your metadata, and which "
                       "features carry it."),
        html.Div([
            ui.labeled("Term", ui.dropdown("tm-pick", opts, term, placeholder="Search terms…", clearable=True,
                                           searchable=True, style={"minWidth": "520px"})),
            ui.labeled("Browse by annotation type", ui.dropdown(
                "tm-browse", [{"label": a["label"], "value": a["key"]} for a in b.ann_types], None, clearable=True,
                placeholder="(optional) list terms of one type", style={"minWidth": "280px"})),
        ], className="controls"),
        html.Div(id="tm-body"),
    ])


def register(app, b):
    @app.callback(Output("tm-pick", "options"), Input("tm-pick", "search_value"), State("tm-pick", "value"))
    def options(query, current):
        opts = b.search(query or "", kinds=["term"], limit=80)
        if current and not any(o["value"] == f"term:{current}" for o in opts):
            opts = [{"label": b.term_label(current, 120), "value": current}] + \
                [{"label": o["label"], "value": o["value"].split(":", 1)[1]} for o in opts]
        else:
            opts = [{"label": o["label"], "value": o["value"].split(":", 1)[1]} for o in opts]
        return opts

    @app.callback(Output("url", "href", allow_duplicate=True), Input("tm-pick", "value"), State("url", "search"),
                  prevent_initial_call=True)
    def pick(value, search):
        if not value or value == _selected(search):
            return no_update
        return ui.href("/terms", term=value)

    @app.callback(Output("tm-body", "children"), Input("url", "search"), Input("tm-browse", "value"),
                  Input("filters", "data"))
    def body(search, browse, filters):
        term = _selected(search)
        fdr = (filters or {}).get("fdr", b.fdr)
        if browse and not term:
            t = b.terms[b.terms["ann_type"] == browse].reset_index().sort_values("n_features", ascending=False)
            return ui.card(f"{b.ann_label.get(browse, browse)} terms", ui.grid(
                "tm-browse-grid", t, [("entity_id", None, None, 160), ("term_name", "name"),
                                      ("n_features", "features", "int", 100), ("n_grouped", "grouped", "int", 100),
                                      ("concentration", "top-group share", "num", 130),
                                      ("perm_padj", "concentration padj", "p", 150)], height=560,
                export_name=f"terms_{browse}"))
        if not term or term not in b.terms.index:
            return ui.note("Search for a term above, or pick an annotation type to browse its terms.")
        r = b.terms.loc[term]
        members = b.term_members(term)
        dtype_bits = " · ".join(f"{int(r[f'n_{d}'])} {d}" for d in b.dtypes if f"n_{d}" in r and r[f"n_{d}"] > 0)
        stats = ui.stat_row([
            ui.stat("Features", f"{int(r['n_features']):,}", dtype_bits),
            ui.stat("In data-driven groups", f"{int(r['n_grouped']):,}"),
            ui.stat("Groups spanned", int(r["n_groups"]) if pd.notna(r.get("n_groups")) else "–"),
            ui.stat("Largest share in one group", f"{r['concentration']:.0%}" if pd.notna(r.get("concentration")) else "–",
                    f"permutation padj {ui.fmt_p(r.get('perm_padj'))}" if pd.notna(r.get("perm_padj")) else None),
        ])
        name = r["term_name"] if isinstance(r["term_name"], str) and r["term_name"] != r["term_id"] else ""
        header = html.Div([
            html.H5([r["term_id"], html.Span(f"  {name}", className="text-secondary")], className="mb-0"),
            html.Div(b.ann_label.get(r["ann_type"], r["ann_type"]), className="small text-secondary mb-2"),
        ])
        return html.Div([
            header, stats,
            dbc.Row([
                dbc.Col(ui.card("(a) How the term divides among data-driven groups",
                                [ui.graph("tm-groups", 420), html.Div(id="tm-groups-note",
                                                                      className="small text-secondary")]), lg=6),
                dbc.Col(ui.card("(b) Term abundance across metadata", [
                    html.Div([ui.labeled("Metadata variable", ui.dropdown("tm-var", b.variables,
                                                                          b.variables[0] if b.variables else None))],
                             className="controls"),
                    ui.graph("tm-profile", 330), html.Div(id="tm-assoc")]), lg=6),
            ]),
            ui.card("(c) Features with this term", ui.grid(
                "tm-members", members.reset_index().sort_values(["group", "kME"], ascending=[True, False]),
                [("feature_id", None, None, 220), ("dtype", None, None, 70), ("display_name", "name"),
                 ("group", None, None, 130), ("kME", "kME", "num", 90), ("is_selected", "selected", None, 100)],
                height=420, export_name=f"{r['term_id']}_features")),
        ])

    @app.callback(Output("tm-groups", "figure"), Output("tm-groups-note", "children"),
                  Input("url", "search"), Input("filters", "data"), Input("theme", "data"))
    def groups_fig(search, filters, theme):
        term = _selected(search)
        fdr = (filters or {}).get("fdr", b.fdr)
        if not term or term not in b.terms.index:
            return th.empty_figure(theme, ""), None
        members = b.term_members(term)
        counts = members["group"].dropna()
        if not (filters or {}).get("include_noise"):
            counts = counts[~members.loc[counts.index, "is_noise"].astype(bool)]
        counts = counts.value_counts().rename("k").rename_axis("group").reset_index()
        if counts.empty:
            return th.empty_figure(theme, "None of this term's features are in data-driven groups"), None
        e = b.enrichment_for_term(term)[["group", "n", "fold", "padj"]]
        counts = counts.merge(e, on="group", how="left")
        top = counts.sort_values(["k", "padj"], ascending=[False, True]).head(25)
        ungrouped = int(len(members) - members["group"].notna().sum())
        note = f"{ungrouped:,} of the term's features were not selected for grouping. " if ungrouped else ""
        note += "Blue bars: the term is significantly enriched in that group. Click a bar to open the group."
        return fg.term_group_bars(top, theme, fdr), note

    @app.callback(Output("tm-profile", "figure"), Output("tm-assoc", "children"),
                  Input("url", "search"), Input("tm-var", "value"), Input("theme", "data"))
    def profile(search, var, theme):
        term = _selected(search)
        if not term or not var:
            return th.empty_figure(theme, ""), None
        if term not in b.term_scores.index:
            return th.empty_figure(theme, "Too few features for a term score"), None
        fig = fg.profile_strip(b.term_scores.loc[term], b.samples, var, b.levels(var), theme,
                               "Term score (mean z of its features)")
        a = b.associations_for("term", term)
        a = a[a["variable"] == var].copy()
        a["test"] = a["test"].map({"moderated_t": "self-contained", "moderated_F": "self-contained (F)",
                                   "camera": "competitive (CAMERA)"})
        return fig, ui.grid("tm-assoc-grid", a.sort_values("p"),
                            [("contrast", None, None, 190), ("test", None, None, 170), ("effect", "effect", "num", 80),
                             ("padj", "padj", "p", 80)], height=260, export_name="term_associations")

    @app.callback(Output("url", "href", allow_duplicate=True), Input("tm-groups", "clickData"),
                  prevent_initial_call=True)
    def click_group(click):
        if not click:
            return no_update
        return ui.href("/groups", group=click["points"][0]["y"])
