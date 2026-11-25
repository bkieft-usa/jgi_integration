"""Dash application factory for the integration explorer."""
from __future__ import annotations

import os
from urllib.parse import parse_qs

import dash_bootstrap_components as dbc
from dash import ALL, MATCH, Dash, Input, Output, State, ctx, dcc, html, no_update

from explorer import components as ui
from explorer.data import Bundle
from explorer.pages import associations, discovery, enrichment, feature, groups, methods, overview, terms

PAGES = {
    "/": ("Overview", overview),
    "/associations": ("Feature-Metadata Associations", associations),
    "/groups": ("Data-Driven Groupings", groups),
    "/terms": ("Knowledge-Driven Groupings", terms),
    "/enrichment": ("Enrichment Matrix", enrichment),
    "/discovery": ("Unannotated Feature Discovery", discovery),
    "/methods": ("Methods", methods),
    "/feature": ("Feature", feature),
}
NAV_ORDER = ["/", "/associations", "/groups", "/terms", "/enrichment", "/discovery", "/methods"]


def _navbar(b: Bundle) -> html.Div:
    links = [dbc.NavLink(PAGES[p][0], href=p, id=f"nav-{p.strip('/') or 'home'}", active="exact") for p in NAV_ORDER]
    return html.Div([
        html.Div([
            html.Div([html.Span("Integration Explorer", className="brand"),
                      html.Span(b.title, className="run-id text-secondary")], className="d-flex flex-column"),
            dcc.Dropdown(id="global-search", placeholder="Search features, groups, pathways, KO, EC, COG …",
                         options=[], searchable=True, clearable=True, className="global-search",
                         optionHeight=44, maxHeight=420),
            dbc.Switch(id="theme-switch", label="Dark", value=False, className="mb-0 ms-2 theme-switch"),
        ], className="topbar-row"),
        dbc.Nav(links, pills=True, className="mainnav"),
    ], className="topbar")


def _details(summary: str, items: list) -> html.Details:
    """A collapsed list: the summary line is always shown, the items on click."""
    return html.Details([html.Summary(summary), html.Div(items, className="details-list")], className="sidebar-details")


def _sidebar(b: Bundle) -> html.Div:
    return html.Div([
        html.Div("Global filters", className="sidebar-title"),
        ui.labeled("False discovery rate", dcc.Dropdown(
            id="flt-fdr", options=[{"label": f"{v:g}", "value": v} for v in (0.001, 0.01, 0.05, 0.1, 0.25)],
            value=b.fdr, clearable=False)),
        ui.labeled("Minimum group size", dcc.Input(id="flt-min-size", type="number", min=1, step=1, value=2,
                                                   className="form-control form-control-sm")),
        dbc.Switch(id="flt-noise", label="Include noise / unassigned features", value=False, className="mt-2 small"),
        html.Hr(),
        html.Div("This run", className="sidebar-title"),
        html.Div([
            html.Div(f"{b.manifest.get('grouping_method') or 'unknown'} grouping"),
            html.Div(f"{b.manifest.get('n_features'):,} integrated features"),
            _details(f"{b.manifest.get('n_samples')} {'samples' if not b.manifest.get('descriptive_only') else 'conditions'}",
                     [html.Div(sid) for sid in b.manifest.get("sample_ids", b.samples.index.tolist())]),
            _details(f"{b.manifest.get('n_groups'):,} groups",
                     [html.Div([html.A(g, href=ui.href("/groups", group=g)),
                                html.Span(f" {int(n):,}", className="text-secondary")])
                      for g, n in b.groups["size"].items()]),
            _details(f"{len(b.ann_types)} annotation types",
                     [html.Div([html.A(a["label"], href=ui.href("/terms", browse=a["key"])),
                                html.Span(f" {int((b.terms['ann_type'] == a['key']).sum()):,} terms",
                                          className="text-secondary")]) for a in b.ann_types]),
        ], className="small sidebar-facts"),
    ], className="sidebar")


def create_app(bundle_dir: str) -> Dash:
    b = Bundle(bundle_dir)
    app = Dash(__name__, assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
               suppress_callback_exceptions=True, title="Integration Explorer", update_title=None)
    app.layout = html.Div([
        dcc.Location(id="url", refresh="callback-nav"),
        dcc.Store(id="theme", storage_type="local", data="light"),
        dcc.Store(id="filters", storage_type="session"),
        html.Div(id="theme-dummy", style={"display": "none"}),
        _navbar(b),
        html.Div([_sidebar(b), html.Main(id="page", className="page")], className="shell"),
    ])

    @app.callback(Output("page", "children"), Input("url", "pathname"), State("url", "search"))
    def route(pathname, search):
        params = {k: v[0] for k, v in parse_qs((search or "").lstrip("?")).items()}
        _, module = PAGES.get(pathname or "/", PAGES["/"])
        return module.layout(b, params)

    @app.callback(Output("filters", "data"), Input("flt-fdr", "value"), Input("flt-min-size", "value"),
                  Input("flt-noise", "value"))
    def set_filters(fdr, min_size, noise):
        return {"fdr": fdr or b.fdr, "min_group_size": int(min_size or 1), "include_noise": bool(noise)}

    @app.callback(Output("theme-switch", "value"), Output("theme", "data"),
                  Input("theme-switch", "value"), Input("theme", "data"))
    def sync_theme(switch, stored):
        if ctx.triggered_id == "theme-switch":
            return no_update, "dark" if switch else "light"
        return stored == "dark", no_update

    app.clientside_callback(
        "function(t){document.documentElement.setAttribute('data-bs-theme', t || 'light'); return '';}",
        Output("theme-dummy", "children"), Input("theme", "data"))

    @app.callback(Output("global-search", "options"), Input("global-search", "search_value"),
                  State("global-search", "value"))
    def search_options(query, current):
        opts = b.search(query or "")
        if current and not any(o["value"] == current for o in opts):
            opts = [{"label": current.split(":", 1)[1], "value": current}] + opts
        return opts

    @app.callback(Output("url", "href", allow_duplicate=True), Output("global-search", "value"),
                  Input("global-search", "value"), prevent_initial_call=True)
    def search_nav(value):
        if not value:
            return no_update, no_update
        kind, key = value.split(":", 1)
        target = {"feature": ui.href("/feature", id=key), "group": ui.href("/groups", group=key),
                  "term": ui.href("/terms", term=key)}[kind]
        return target, None

    @app.callback(Output("url", "href", allow_duplicate=True),
                  Input({"type": "nav-grid", "id": ALL}, "cellClicked"), prevent_initial_call=True)
    def grid_nav(_):
        if not ctx.triggered or not ctx.triggered[0]["value"]:
            return no_update
        cell = ctx.triggered[0]["value"]
        col, value = cell.get("colId"), cell.get("value")
        if col not in ui.NAV_COLUMNS or value in (None, ""):
            return no_update
        route, key = ui.NAV_COLUMNS[col]
        return ui.href(route, **{key: value})

    @app.callback(Output({"type": "nav-grid", "id": MATCH}, "exportDataAsCsv"),
                  Input({"type": "grid-export", "id": MATCH}, "n_clicks"), prevent_initial_call=True)
    def export_grid(n):
        return bool(n)

    for _, module in PAGES.values():
        module.register(app, b)
    app.bundle = b
    return app
