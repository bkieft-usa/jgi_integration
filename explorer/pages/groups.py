"""Data-driven groups: map of groups, zoom into one group's biology."""
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

MAX_WITHIN = 300


def _selected(search) -> str | None:
    return parse_qs((search or "").lstrip("?")).get("group", [None])[0]


def layout(b, params):
    native = b.native or {}
    views = [{"label": "Group network (eigengene correlation)", "value": "network"},
             {"label": "Feature map (UMAP)", "value": "features"}]
    if native.get("kind") == "dendrogram":
        views.append({"label": "Clustering dendrogram" + (" (approximate)" if native.get("approximate") else ""),
                      "value": "native"})
    elif native.get("kind") == "hdbscan_tree":
        views.append({"label": "HDBSCAN cluster tree", "value": "native"})
    group = params.get("group")
    return html.Div([
        ui.page_header("Data-driven groups",
                       "Explore how groups of co-varying features relate, then click a group to zoom into its "
                       "annotations, abundance profile, and members (including unannotated ones)."),
        html.Div([
            ui.labeled("View", ui.dropdown("gr-view", views, "network", style={"minWidth": "300px"})),
            ui.labeled("Color", ui.dropdown("gr-color", [{"label": "Selected group", "value": "group"},
                                                        {"label": "Data type", "value": "dtype"}], "group")),
            ui.labeled("Groups in network", ui.dropdown("gr-maxn", [{"label": f"largest {n}", "value": n}
                                                                   for n in (100, 300, 600)] + [{"label": "all", "value": 0}],
                                                        300, style={"minWidth": "150px"})),
            dbc.Switch(id="gr-neg", label="Show anti-correlated edges", value=False, className="mb-1 small"),
            ui.labeled("Highlight a feature or term", ui.dropdown("gr-highlight", [], None, clearable=True,
                                                                  placeholder="type a KO, pathway, compound, feature ID…",
                                                                  style={"minWidth": "360px"}, searchable=True)),
        ], className="controls"),
        dbc.Row([
            dbc.Col(ui.card("Map", [html.Div(id="gr-map-wrap"), html.Div(id="gr-map-note", className="small text-secondary mt-1")]),
                    lg=7),
            dbc.Col(html.Div(id="gr-summary"), lg=5),
        ]),
        html.Div(id="gr-detail"),
        dcc.Store(id="gr-init", data=group),
    ])


def _network_elements(b, filters, max_groups=300, negative=False):
    g = b.groups
    min_size = (filters or {}).get("min_group_size", 2)
    g = g[(g["size"] >= min_size) & ~g["is_noise"]].dropna(subset=["x"])
    if max_groups:
        g = g.nlargest(int(max_groups), "size")
    # normalize positions to a fixed canvas so node spacing does not depend on embedding scale
    span = max(float(g["x"].max() - g["x"].min()), float(g["y"].max() - g["y"].min()), 1e-9) if len(g) else 1.0
    scale = 1800.0 / span
    x0, y0 = (g["x"].min(), g["y"].min()) if len(g) else (0, 0)
    nodes = []
    dtype_cols = [c for c in g.columns if c.startswith("n_") and c[2:] in b.dtypes]
    for gid, r in g.iterrows():
        counts = {c[2:]: r[c] for c in dtype_cols}
        total = max(sum(counts.values()), 1)
        dom = max(counts, key=counts.get) if counts else None
        kind = dom if dom and counts[dom] / total >= 0.8 else "mixed"
        nodes.append({"data": {"id": gid, "label": "", "size": float(min(44, 6 + 2.2 * np.sqrt(r["size"]))),
                               "kind": kind, "n": int(r["size"])},
                      "position": {"x": float(r["x"] - x0) * scale, "y": float(y0 - r["y"]) * scale}})
    keep = set(g.index)
    e = b.group_net
    e = e[e["source"].isin(keep) & e["target"].isin(keep)] if len(e) else e
    if len(e) and not negative:
        e = e[e["r"] > 0]
    edges = [{"data": {"source": s, "target": t, "r": float(r), "width": 0.6 + 3 * max(abs(r) - 0.7, 0) / 0.3,
                       "sign": "pos" if r > 0 else "neg"}} for s, t, r in e[["source", "target", "r"]].itertuples(index=False)]
    return nodes + edges


def _network_stylesheet(theme, selected, highlight_groups):
    c = th.chrome(theme)
    ss = th.cytoscape_stylesheet(theme)
    ss += [
        {"selector": '[kind = "tx"]', "style": {"background-color": th.dtype_color(theme, "tx")}},
        {"selector": '[kind = "mx"]', "style": {"background-color": th.dtype_color(theme, "mx")}},
        {"selector": '[kind = "px"]', "style": {"background-color": th.dtype_color(theme, "px")}},
        {"selector": '[kind = "mixed"]', "style": {"background-color": th.cat(theme, 2)}},
        {"selector": '[sign = "pos"]', "style": {"line-color": c["muted"]}},
        {"selector": '[sign = "neg"]', "style": {"line-color": c["red"]}},
    ]
    for gid in list(highlight_groups)[:400]:
        ss.append({"selector": f'node[id = "{gid}"]', "style": {"border-width": 3, "border-color": th.cat(theme, 3)}})
    if selected:
        ss.append({"selector": f'node[id = "{selected}"]',
                   "style": {"border-width": 4, "border-color": c["ink"], "label": "data(id)", "font-size": 24,
                             "min-zoomed-font-size": 0, "color": c["ink"], "z-index": 99,
                             "text-background-color": c["surface"], "text-background-opacity": 0.8}})
    return ss


def _highlight_ids(b, value):
    if not value:
        return set()
    kind, key = value.split(":", 1)
    if kind == "feature":
        return {key}
    if kind == "term":
        return set(b.term_members(key).index)
    return set()


def register(app, b):
    @app.callback(Output("gr-highlight", "options"), Input("gr-highlight", "search_value"),
                  State("gr-highlight", "value"))
    def hl_options(query, current):
        opts = b.search(query or "", kinds=["term", "feature"])
        if current and not any(o["value"] == current for o in opts):
            opts = [{"label": current.split(":", 1)[1], "value": current}] + opts
        return opts

    @app.callback(Output("gr-map-wrap", "children"), Output("gr-map-note", "children"),
                  Input("gr-view", "value"), Input("filters", "data"), Input("gr-maxn", "value"), Input("gr-neg", "value"))
    def map_container(view, filters, max_groups, negative):
        if view == "network":
            els = _network_elements(b, filters, max_groups, negative)
            n_nodes = sum(1 for e in els if "position" in e)
            legend = html.Div([
                html.Span("Node color: ", className="me-1"),
                *[html.Span([html.Span(className="legend-dot", style={"background": th.dtype_color("light", d)}),
                             f"mostly {d}"], className="me-3") for d in b.dtypes],
                html.Span([html.Span(className="legend-dot", style={"background": th.cat("light", 2)}), "mixed"],
                          className="me-3"),
                html.Span("· size = features · edges = eigengene correlation |r| ≥ "
                          f"{b.manifest.get('config', {}).get('eigengene_network', {}).get('min_abs_r', 0.7)} "
                          "(gray positive, red negative) · nearby groups have similar abundance profiles (UMAP of eigengenes)"),
            ])
            return (cyto.Cytoscape(id="gr-cyto", elements=els, layout={"name": "preset", "fit": True},
                                   style={"width": "100%", "height": "560px"}, minZoom=0.05, maxZoom=4,
                                   boxSelectionEnabled=False, wheelSensitivity=0.2),
                    [legend, html.Div(f"{n_nodes:,} groups shown (largest first; minimum size in the sidebar). "
                                      "Scroll to zoom, drag to pan, click a group to open it.")])
        if view == "features":
            return ui.graph("gr-fmap", 560), "Each point is a feature (UMAP of its profile, correlation metric). " \
                                             "Scroll to zoom, drag to pan, click a point to open its group."
        note = (b.native or {}).get("note", "")
        return ui.graph("gr-native", 560), note

    @app.callback(Output("gr-cyto", "stylesheet"), Input("url", "search"), Input("gr-highlight", "value"),
                  Input("theme", "data"))
    def cyto_style(search, hl, theme):
        ids = _highlight_ids(b, hl)
        hgroups = set(b.features.loc[b.features.index.isin(ids), "group"].dropna())
        return _network_stylesheet(theme, _selected(search), hgroups)

    @app.callback(Output("gr-fmap", "figure"), Input("url", "search"), Input("gr-highlight", "value"),
                  Input("gr-color", "value"), Input("filters", "data"), Input("theme", "data"))
    def fmap(search, hl, color, filters, theme):
        f = b.features
        if not (filters or {}).get("include_noise"):
            f = f[~f["is_noise"].fillna(False).astype(bool)]
        fig = fg.feature_map(f, theme, _selected(search), _highlight_ids(b, hl), color or "group")
        fig.update_layout(uirevision="fmap")
        return fig

    @app.callback(Output("gr-native", "figure"), Input("url", "search"), Input("theme", "data"))
    def native_fig(search, theme):
        n = b.native or {}
        if n.get("kind") == "dendrogram":
            return fg.dendrogram(n, theme, _selected(search))
        if n.get("kind") == "hdbscan_tree":
            return fg.hdbscan_icicle(n, theme, _selected(search))
        return th.empty_figure(theme, "No native structure view for this grouping method")

    @app.callback(Output("url", "href", allow_duplicate=True), Input("gr-cyto", "tapNodeData"),
                  prevent_initial_call=True)
    def tap_node(data):
        return ui.href("/groups", group=data["id"]) if data else no_update

    @app.callback(Output("url", "href", allow_duplicate=True), Input("gr-fmap", "clickData"),
                  prevent_initial_call=True)
    def click_feature(click):
        if not click:
            return no_update
        g = click["points"][0]["customdata"][1]
        return ui.href("/groups", group=g) if g else no_update

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
                html.P("Click a node in the group network, a point on the feature map, or a leaf of the tree. "
                       "You can also search for a group in the top bar."),
                html.P("Largest groups:", className="mb-1 small text-secondary"),
                ui.grid("gr-top", b.groups.reset_index().head(200),
                        [("group", None, None, 120), ("size", "features", "int", 90), ("top_term", "top enriched term")],
                        height=420, export_name="groups"),
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
                      html.Span(f"{r['best_variable']} · {r['best_contrast']} (padj {ui.fmt_p(r['best_padj'])})"
                                if isinstance(r.get("best_variable"), str) else "–")], className="small"),
        ])
        at_counts = b.enrichment_for_group(g)
        at_default = (at_counts[at_counts["padj"] < b.fdr]["ann_type"].value_counts().index.tolist() or
                      at_counts["ann_type"].value_counts().index.tolist() or [b.ann_types[0]["key"] if b.ann_types else None])[0]
        tabs = dbc.Tabs([
            dbc.Tab(label="Abundance profile", tab_id="profile", children=html.Div([
                html.Div([ui.labeled("Metadata variable", ui.dropdown("gr-var", b.variables,
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
        a = a[a["variable"] == var].sort_values("p")
        tbl = ui.grid("gr-assoc-grid", a, [("contrast", None, None, 200), ("effect", "Δ eigengene", "num"),
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
                     "group eigengene (how typical the feature is of the group).", className="small text-secondary mb-1"),
            ui.grid("gr-mem-grid", m, [("feature_id", None, None, 220), ("dtype", None, None, 70), ("display_name", "name"),
                                       ("kME", "kME", "num", 90), ("is_annotated", "annotated", None, 100),
                                       ("n_terms", "terms", "int", 80)], height=480, export_name=f"{g}_members"),
        ])

    @app.callback(Output("gr-within", "children"), Input("url", "search"), Input("theme", "data"))
    def within(search, theme):
        g = _selected(search)
        if not g:
            return None
        m = b.group_members(g).sort_values("kME", ascending=False)
        e = b.feature_edges
        real = len(e) > 0
        if real:  # show the best-connected members of the real network
            inside = e[e["source"].isin(m.index) & e["target"].isin(m.index)]
            degree = pd.concat([inside["source"], inside["target"]]).value_counts()
            m = m.assign(degree=degree.reindex(m.index).fillna(0)).sort_values(["degree", "kME"], ascending=False)
        ids = list(m.index[:MAX_WITHIN])
        idset = set(ids)
        if real:
            sub = e[e["source"].isin(idset) & e["target"].isin(idset)]
            edges = [(s, t, w) for s, t, w in sub[["source", "target", "weight"]].itertuples(index=False)]
            caption = "Real correlation-network edges used to define the modules."
        else:
            Z = b.z.reindex(ids).to_numpy(dtype=float)
            Zc = np.nan_to_num(Z - Z.mean(axis=1, keepdims=True))
            Zc /= np.maximum(np.linalg.norm(Zc, axis=1, keepdims=True), 1e-12)
            R = Zc @ Zc.T
            np.fill_diagonal(R, -np.inf)
            edges = set()
            for i in range(len(ids)):
                for j in np.argsort(-R[i])[:3]:
                    if R[i, j] >= 0.7:
                        edges.add((min(i, j), max(i, j)))
            edges = [(ids[i], ids[j], float(R[i, j])) for i, j in edges]
            caption = ("Correlation view (not the clustering itself): each member linked to its 3 most correlated "
                       "members (r ≥ 0.7).")
        if len(m) > MAX_WITHIN:
            caption += (f" Showing the {MAX_WITHIN} best-connected members of {len(m):,}." if real else
                        f" Showing the {MAX_WITHIN} most typical members (highest kME) of {len(m):,}.")
        connected = {s for s, t, _ in edges} | {t for s, t, _ in edges}
        if len(ids) - len(connected) > 0:
            caption += f" {len(ids) - len(connected):,} shown members have no edge to another shown member."
        nodes = [{"data": {"id": f, "label": (m.at[f, "display_name"] or "")[:24] if isinstance(m.at[f, "display_name"], str) else "",
                           "size": 10 + 14 * max(float(m.at[f, "kME"]) if pd.notna(m.at[f, "kME"]) else 0, 0),
                           "dtype": m.at[f, "dtype"], "ann": "yes" if m.at[f, "is_annotated"] else "no"}}
                 for f in ids]
        els = nodes + [{"data": {"source": s, "target": t, "width": 1.0, "color": th.chrome(theme)["muted"]}}
                       for s, t, _ in edges]
        ss = th.cytoscape_stylesheet(theme) + [
            {"selector": f'[dtype = "{d}"]', "style": {"background-color": th.dtype_color(theme, d)}} for d in b.dtypes
        ] + [{"selector": '[ann = "no"]', "style": {"background-opacity": 0.25, "border-width": 2,
                                                    "border-color": th.chrome(theme)["ink2"]}},
             # names appear only once zoomed in far enough to read them
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
