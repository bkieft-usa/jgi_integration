"""Data-driven groupings: the full feature network, then zoom into one group's biology."""
from __future__ import annotations

from urllib.parse import parse_qs

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import numpy as np
import pandas as pd
from dash import Input, Output, State, dcc, html, no_update

from explorer import components as ui
from explorer import figures as fg
from explorer import theme as th
from explorer.data import contrast_label

MAX_WITHIN = 300
N_COLORED_GROUPS = 8


def _selected(search) -> str | None:
    return parse_qs((search or "").lstrip("?")).get("group", [None])[0]


def _network_caption(b) -> str:
    n = b.manifest.get("network", {})
    if b.network_kind == "real":
        return (f"The run's own correlation network ({n.get('n_nodes', 0):,} features, {n.get('n_edges', 0):,} edges): "
                "these edges are what defined the groups. Each point is a feature; nearby features are connected.")
    return (f"Derived network for display ({n.get('n_nodes', 0):,} features, {n.get('n_edges', 0):,} edges): the "
            f"{b.manifest.get('grouping_method')} method does not produce edges, so each feature is linked to its "
            f"{n.get('k', 5)} most correlated grouped features. Groups were formed by the clustering itself (see "
            "'Clustering tree'), not by these edges.")


def layout(b, params):
    native = b.native or {}
    views = [{"label": "Feature network", "value": "network"}]
    if native.get("kind") == "dendrogram":
        views.append({"label": "Clustering tree (dendrogram)" + (" — approximate" if native.get("approximate") else ""),
                      "value": "native"})
    elif native.get("kind") == "hdbscan_tree":
        views.append({"label": "Clustering tree (HDBSCAN condensed tree)", "value": "native"})
    group = params.get("group")
    hl_values = [v for v in (params.get("highlight") or "").split("|") if v]
    return html.Div([
        ui.page_header("Data-Driven Groupings",
                       "Every grouped feature and how it relates to the others. Highlight knowledge-driven terms or "
                       "features to see where they fall, then click a feature to open its group: abundance profile, "
                       "annotations, and members (including unannotated ones)."),
        html.Div([
            ui.labeled("View", ui.dropdown("gr-view", views, "network", style={"minWidth": "280px"})),
            ui.labeled("Color", ui.dropdown("gr-color", [
                {"label": f"Group (largest {N_COLORED_GROUPS} in color)", "value": "group"},
                {"label": "Data type", "value": "dtype"},
                {"label": "Annotated vs not", "value": "annotation"}], "group", style={"minWidth": "220px"})),
            ui.labeled("Highlight terms or features (several allowed)",
                       ui.dropdown("gr-highlight", [{"label": v.split(":", 1)[1], "value": v} for v in hl_values],
                                   hl_values, multi=True, clearable=True, searchable=True,
                                   placeholder="type a pathway, KO, EC, compound class, feature ID…",
                                   style={"minWidth": "480px"})),
        ], className="controls"),
        ui.card("Feature network", [html.Div(ui.graph("gr-net", 640), id="gr-net-wrap"),
                                    html.Div(ui.graph("gr-native", 640), id="gr-native-wrap", style={"display": "none"}),
                                    html.Div(id="gr-map-note", className="small text-secondary mt-1")]),
        html.Div(id="gr-highlight-summary"),
        dbc.Row([
            dbc.Col(html.Div(id="gr-summary"), lg=7),
            dbc.Col(ui.card("Group sizes", ui.graph("gr-sizes", 260),
                            "Number of features in each data-driven group."), lg=5),
        ]),
        html.Div(id="gr-detail"),
        dcc.Store(id="gr-init", data=group),
    ])


def _highlight_sets(b, values):
    """``[(label, feature_ids)]`` for the highlight dropdown values."""
    out = []
    for v in values or []:
        kind, key = v.split(":", 1)
        if kind == "term":
            ids = set(b.term_members(key).index)
            out.append((b.term_label(key, 40), ids))
        elif kind == "feature":
            out.append((key, {key}))
    return out


def _colored_groups(b):
    g = b.groups[~b.groups["is_noise"]]
    return g.nlargest(N_COLORED_GROUPS, "size").index.tolist()


def register(app, b):
    @app.callback(Output("gr-highlight", "options"), Input("gr-highlight", "search_value"),
                  State("gr-highlight", "value"), State("gr-highlight", "options"))
    def hl_options(query, current, existing):
        keep = [o for o in (existing or []) if o["value"] in (current or [])]
        opts = b.search(query or "", kinds=["term", "feature"]) if query else []
        seen = {o["value"] for o in keep}
        return keep + [o for o in opts if o["value"] not in seen]

    @app.callback(Output("gr-net-wrap", "style"), Output("gr-native-wrap", "style"), Output("gr-map-note", "children"),
                  Input("gr-view", "value"))
    def map_container(view):
        hide = {"display": "none"}
        if view == "native":
            return hide, {}, (b.native or {}).get("note", "")
        return {}, hide, [
            _network_caption(b), " Scroll to zoom, drag to pan, click a feature to open its group. ",
            "Use the sidebar to include noise / unassigned features."]

    @app.callback(Output("gr-net", "figure"), Output("gr-highlight-summary", "children"),
                  Input("url", "search"), Input("gr-highlight", "value"), Input("gr-color", "value"),
                  Input("filters", "data"), Input("theme", "data"))
    def network(search, hl, color, filters, theme):
        f = b.features.loc[b.network_nodes]
        if not (filters or {}).get("include_noise"):
            f = f[~f["is_noise"].fillna(False).astype(bool)]
        min_size = (filters or {}).get("min_group_size", 1)
        f = f[f["group"].map(b.groups["size"]).fillna(0) >= min_size]
        sets = _highlight_sets(b, hl)
        fig = fg.network_figure(f, b.feature_edges, theme, color or "group", _colored_groups(b), _selected(search),
                                sets)
        fig.update_layout(uirevision="network")
        summary = None
        if sets:
            rows = []
            for label, ids in sets:
                in_net = f.index.intersection(list(ids))
                groups = b.features.loc[in_net, "group"].value_counts()
                rows.append(html.Li([html.B(label), f": {len(ids):,} features, {len(in_net):,} in the network",
                                     (", mostly in " + ", ".join(f"{g} ({n})" for g, n in groups.head(3).items()))
                                     if len(groups) else ""]))
            summary = ui.note(html.Ul(rows, className="mb-0 ps-3"))
        return fig, summary

    @app.callback(Output("gr-sizes", "figure"), Input("theme", "data"))
    def sizes(theme):
        return fg.size_histogram(b.groups, theme)

    @app.callback(Output("gr-native", "figure"), Input("url", "search"), Input("theme", "data"))
    def native_fig(search, theme):
        n = b.native or {}
        if n.get("kind") == "dendrogram":
            return fg.dendrogram(n, theme, _selected(search), height=640)
        if n.get("kind") == "hdbscan_tree":
            return fg.hdbscan_icicle(n, theme, _selected(search), height=640)
        return th.empty_figure(theme, "No clustering tree for this grouping method")

    @app.callback(Output("url", "href", allow_duplicate=True), Input("gr-net", "clickData"), State("url", "search"),
                  State("gr-highlight", "value"), prevent_initial_call=True)
    def click_feature(click, search, hl):
        if not click or "customdata" not in click["points"][0]:
            return no_update
        g = click["points"][0]["customdata"][1]
        if not g or g == "–" or g == _selected(search):
            return no_update
        return ui.href("/groups", group=g, highlight="|".join(hl or []))

    @app.callback(Output("url", "href", allow_duplicate=True), Input("gr-native", "clickData"),
                  prevent_initial_call=True)
    def click_native(click):
        if not click:
            return no_update
        cd = click["points"][0].get("customdata")
        # dendrogram leaves carry the group name; condensed-tree segments carry [label, size, stability, group]
        g = cd if isinstance(cd, str) else (cd[3] if isinstance(cd, list) and len(cd) > 3 else None)
        return ui.href("/groups", group=g) if g else no_update

    # ---- selected group -------------------------------------------------------------------
    @app.callback(Output("gr-summary", "children"), Output("gr-detail", "children"), Input("url", "search"))
    def detail(search):
        g = _selected(search)
        if not g or g not in b.groups.index:
            hint = ui.card("No group selected", html.Div([
                html.P("Click a feature in the network, a leaf of the clustering tree, or a row below. You can also "
                       "search for a group in the top bar.", className="small"),
                ui.grid("gr-top", b.groups.reset_index().head(500),
                        [("group", None, None, 130), ("size", "features", "int", 90),
                         ("frac_annotated", "annotated", "num", 100), ("top_term", "top enriched term")],
                        height=300, export_name="groups"),
            ]))
            return hint, None
        r = b.groups.loc[g]
        dtype_bits = " · ".join(f"{int(r[f'n_{d}'])} {d}" for d in b.dtypes if f"n_{d}" in r and r[f"n_{d}"] > 0)
        summary = ui.card(f"{g}", [
            ui.stat_row([
                ui.stat("Features", f"{int(r['size']):,}", dtype_bits),
                ui.stat("Annotated", f"{r['frac_annotated']:.0%}" if pd.notna(r["frac_annotated"]) else "–"),
                ui.stat("Coherence", f"{r['coherence']:.2f}" if pd.notna(r.get("coherence")) else "–",
                        "mean member correlation"),
                ui.stat("Enriched terms", int(r.get("n_enriched_terms", 0))),
            ]),
            html.Div([html.Span("Top enriched term: ", className="text-secondary"),
                      html.Span(r["top_term"] if isinstance(r.get("top_term"), str) else "none at this FDR")],
                     className="small mb-1"),
            html.Div([html.Span("Strongest metadata association: ", className="text-secondary"),
                      html.Span(f"{r['best_variable']} · {contrast_label(r['best_contrast'])} (padj {ui.fmt_p(r['best_padj'])})"
                                if isinstance(r.get("best_variable"), str) else "–")], className="small"),
        ])
        at_counts = b.enrichment_for_group(g)
        at_default = (at_counts[at_counts["padj"] < b.fdr]["ann_type"].value_counts().index.tolist() or
                      at_counts["ann_type"].value_counts().index.tolist() or [b.ann_types[0]["key"] if b.ann_types else None])[0]
        tabs = dbc.Tabs([
            dbc.Tab(label="Abundance profile", tab_id="profile", children=html.Div([
                html.Div([ui.labeled("Metadata variable", ui.dropdown("gr-var", b.variable_options(),
                                                                      b.variables[0] if b.variables else None))],
                         className="controls mt-2"),
                dbc.Row([dbc.Col(ui.graph("gr-profile", 360), lg=7), dbc.Col(html.Div(id="gr-assoc"), lg=5)]),
            ])),
            dbc.Tab(label="Knowledge-driven annotations", tab_id="annotations", children=html.Div([
                html.Div([ui.labeled("Annotation type", ui.dropdown(
                    "gr-anntype", [{"label": a["label"], "value": a["key"]} for a in b.ann_types], at_default,
                    style={"minWidth": "260px"}))], className="controls mt-2"),
                dbc.Row([dbc.Col(ui.graph("gr-comp", 440), lg=7), dbc.Col(html.Div(id="gr-enriched"), lg=5)]),
            ])),
            dbc.Tab(label="Members", tab_id="members", children=html.Div(id="gr-members", className="mt-2")),
            dbc.Tab(label="Within-group network", tab_id="within", children=html.Div(id="gr-within", className="mt-2")),
        ], active_tab="profile", className="mt-2")
        return summary, ui.card(f"Inside {g}", tabs)

    @app.callback(Output("gr-profile", "figure"), Output("gr-assoc", "children"),
                  Input("url", "search"), Input("gr-var", "value"), Input("filters", "data"), Input("theme", "data"))
    def profile(search, var, filters, theme):
        g = _selected(search)
        if not g or g not in b.eig.index or not var:
            return th.empty_figure(theme, "No eigengene for this group"), None
        fig = fg.profile_strip(b.eig.loc[g], b.samples, var, b.levels(var), theme, "Group eigengene (SD)")
        a = b.associations_for("group", g)
        a = a[a["variable"] == b.canonical(var)].sort_values("p")
        tbl = ui.grid("gr-assoc-grid", a, [("contrast", "comparison", None, 220), ("effect", "Δ eigengene", "num"),
                                           ("padj", "padj", "p")], height=320, export_name=f"{g}_associations")
        return fig, tbl

    @app.callback(Output("gr-comp", "figure"), Output("gr-enriched", "children"),
                  Input("url", "search"), Input("gr-anntype", "value"), Input("filters", "data"), Input("theme", "data"))
    def composition(search, at, filters, theme):
        g = _selected(search)
        fdr = (filters or {}).get("fdr", b.fdr)
        if not g or not at:
            return th.empty_figure(theme, "Select an annotation type"), None
        members = b.group_members(g).index
        sub = b.ann[(b.ann["ann_type"] == at) & b.ann["feature_id"].isin(members)]
        if sub.empty:
            return th.empty_figure(theme, f"No {b.ann_label.get(at, at)} annotations in this group"), \
                ui.note(f"{len(members)} members, none annotated with {b.ann_label.get(at, at)}.")
        k = sub.groupby("term_id").agg(k=("feature_id", "nunique"), term_name=("term_name", "first")).reset_index()
        e = b.enrichment_for_group(g, at)[["term_id", "K", "fold", "padj"]]
        k = k.merge(e, on="term_id", how="left")
        k["entity_id"] = at + "::" + k["term_id"]
        k["label"] = k["entity_id"].map(lambda x: b.term_label(x, 48))
        k = k.sort_values(["k", "padj"], ascending=[False, True]).head(20)
        fig = fg.composition_bars(k, theme, fdr)
        allsig = b.enrichment_for_group(g)
        allsig = allsig[allsig["padj"] < fdr].sort_values("padj").assign(
            entity_id=lambda d: d["ann_type"] + "::" + d["term_id"],
            annotation=lambda d: d["ann_type"].map(b.ann_label))
        tbl = html.Div([html.Div(f"Significantly enriched terms, all annotation types (FDR < {fdr})",
                                 className="small text-secondary mb-1"),
                        ui.grid("gr-enr-grid", allsig, [("entity_id", None, None, 120), ("term_name", "name"),
                                                        ("annotation", "type", None, 150), ("k", "k", "int", 60),
                                                        ("fold", "fold", "num", 70), ("padj", "padj", "p", 80)],
                                height=400, export_name=f"{g}_enrichment")])
        return fig, tbl

    @app.callback(Output("gr-members", "children"), Input("url", "search"))
    def members(search):
        g = _selected(search)
        if not g:
            return None
        m = b.group_members(g).reset_index().sort_values("kME", ascending=False)
        n_un = int((~m["is_annotated"]).sum())
        return html.Div([
            html.Div(f"{len(m):,} members; {n_un:,} have no annotation of any type. kME = correlation with the "
                     "group eigengene (how typical the feature is of the group). Degree = network neighbours.",
                     className="small text-secondary mb-1"),
            ui.grid("gr-mem-grid", m, [("feature_id", None, None, 220), ("dtype", None, None, 70), ("display_name", "name"),
                                       ("kME", "kME", "num", 90), ("degree", "degree", "int", 90),
                                       ("is_annotated", "annotated", None, 100), ("n_terms", "terms", "int", 80)],
                    height=480, export_name=f"{g}_members"),
        ])

    @app.callback(Output("gr-within", "children"), Input("url", "search"), Input("theme", "data"))
    def within(search, theme):
        g = _selected(search)
        if not g:
            return None
        m = b.group_members(g)
        e = b.feature_edges
        inside = e[e["source"].isin(m.index) & e["target"].isin(m.index)]
        degree = pd.concat([inside["source"], inside["target"]]).value_counts()
        m = m.assign(deg_in=degree.reindex(m.index).fillna(0)).sort_values(["deg_in", "kME"], ascending=False)
        ids = list(m.index[:MAX_WITHIN])
        idset = set(ids)
        sub = inside[inside["source"].isin(idset) & inside["target"].isin(idset)]
        caption = ("Edges of the run's correlation network between members of this group." if b.network_kind == "real"
                   else "Edges of the derived nearest-neighbour network between members of this group "
                        "(a display aid; the clustering did not use edges).")
        if len(m) > MAX_WITHIN:
            caption += f" Showing the {MAX_WITHIN} best-connected members of {len(m):,}."
        connected = set(sub["source"]) | set(sub["target"])
        if len(ids) - len(connected) > 0:
            caption += f" {len(ids) - len(connected):,} shown members have no edge to another member."
        nodes = [{"data": {"id": f, "label": (m.at[f, "display_name"] or "")[:24] if isinstance(m.at[f, "display_name"], str) else "",
                           "size": 10 + 14 * max(float(m.at[f, "kME"]) if pd.notna(m.at[f, "kME"]) else 0, 0),
                           "dtype": m.at[f, "dtype"], "ann": "yes" if m.at[f, "is_annotated"] else "no"}}
                 for f in ids]
        els = nodes + [{"data": {"source": s, "target": t, "width": 1.0, "color": th.chrome(theme)["muted"]}}
                       for s, t in sub[["source", "target"]].itertuples(index=False)]
        ss = th.cytoscape_stylesheet(theme) + [
            {"selector": f'[dtype = "{d}"]', "style": {"background-color": th.dtype_color(theme, d)}} for d in b.dtypes
        ] + [{"selector": '[ann = "no"]', "style": {"background-opacity": 0.25, "border-width": 2,
                                                    "border-color": th.chrome(theme)["ink2"]}},
             {"selector": "node", "style": {"min-zoomed-font-size": 14, "font-size": 10}}]
        legend = html.Div([
            *[html.Span([html.Span(className="legend-dot", style={"background": th.dtype_color(theme, d)}), d],
                        className="me-3") for d in b.dtypes],
            html.Span("hollow = unannotated · size = kME · zoom in to read names · click a node to open the feature",
                      className="text-secondary"),
        ], className="small mb-1")
        return html.Div([html.Div(caption, className="small text-secondary mb-1"), legend,
                         cyto.Cytoscape(id="gr-within-cyto", elements=els, stylesheet=ss,
                                        layout={"name": "cose", "animate": False, "numIter": 600},
                                        style={"width": "100%", "height": "520px"}, wheelSensitivity=0.2)])

    @app.callback(Output("url", "href", allow_duplicate=True), Input("gr-within-cyto", "tapNodeData"),
                  prevent_initial_call=True)
    def tap_feature(data):
        return ui.href("/feature", id=data["id"]) if data else no_update
