"""Feature detail: annotations, group, abundance across metadata, neighbors."""
from __future__ import annotations

from urllib.parse import parse_qs

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, html

from explorer import components as ui
from explorer import figures as fg
from explorer import theme as th


def _selected(search):
    return parse_qs((search or "").lstrip("?")).get("id", [None])[0]


def layout(b, params):
    fid = params.get("id")
    if not fid or fid not in b.features.index:
        return html.Div([ui.page_header("Feature", "Search for a feature in the top bar."),
                         ui.note(f"Feature '{fid}' is not in this run." if fid else "No feature selected.")])
    r = b.features.loc[fid]
    ann = b.feature_annotations(fid).assign(entity_id=lambda d: d["ann_type"] + "::" + d["term_id"],
                                            annotation=lambda d: d["ann_type"].map(b.ann_label))
    group = r["group"] if isinstance(r["group"], str) else None
    head = html.Div([
        html.H5([fid, html.Span(f"  {r['display_name']}" if r["display_name"] else "", className="text-secondary")]),
        ui.stat_row([
            ui.stat("Data type", th.DTYPE_LABEL.get(r["dtype"], r["dtype"])),
            ui.stat("Group", html.A(group, href=ui.href("/groups", group=group)) if group else "not grouped",
                    "noise" if r.get("is_noise") else None),
            ui.stat("kME", f"{r['kME']:.2f}" if pd.notna(r.get("kME")) else "–", "correlation with group eigengene"),
            ui.stat("Annotations", int(r["n_terms"])),
        ]),
    ])
    return html.Div([
        ui.page_header("Feature", "One feature's annotations, abundance across metadata, and co-varying neighbors."),
        head,
        dbc.Row([
            dbc.Col(ui.card("Abundance across metadata", [
                html.Div([ui.labeled("Metadata variable", ui.dropdown("ft-var", b.variables,
                                                                      b.variables[0] if b.variables else None)),
                          ui.labeled("Scale", ui.dropdown("ft-scale", [{"label": "log-scale abundance", "value": "log"},
                                                                       {"label": "integrated z-score", "value": "z"}],
                                                          "log" if fid in b.logab.index else "z"))],
                         className="controls"),
                ui.graph("ft-profile", 340), html.Div(id="ft-assoc")]), lg=7),
            dbc.Col([
                ui.card("Annotations", ui.grid("ft-ann", ann, [("annotation", "type", None, 170), ("entity_id", None, None, 130),
                                                               ("term_name", "name")], height=300,
                                               export_name=f"{fid}_annotations") if len(ann) else
                        ui.note("No annotations of any type. See the group's enriched terms for hypotheses.")),
                ui.card("Most correlated members of its group", html.Div(id="ft-neighbors")),
            ], lg=5),
        ]),
    ])


def register(app, b):
    @app.callback(Output("ft-profile", "figure"), Output("ft-assoc", "children"), Input("url", "search"),
                  Input("ft-var", "value"), Input("ft-scale", "value"), Input("theme", "data"))
    def profile(search, var, scale, theme):
        fid = _selected(search)
        if not fid or not var:
            return th.empty_figure(theme, ""), None
        if scale == "log" and fid in b.logab.index:
            vals, ylab, meta = b.logab.loc[fid].dropna(), "log-scale abundance", b.sample_meta
        elif fid in b.z.index:
            vals, ylab, meta = b.z.loc[fid], "integrated z-score", b.samples
        else:
            return th.empty_figure(theme, "No abundance values"), None
        fig = fg.profile_strip(vals, meta, var, b.levels(var, meta), theme, ylab)
        a = b.associations_for("feature", fid)
        a = a[a["variable"] == var]
        return fig, ui.grid("ft-assoc-grid", a, [("contrast", None, None, 220), ("effect", "log2FC", "num", 90),
                                                 ("statistic", "t / F", "num", 80), ("padj", "padj", "p", 90)],
                            height=260, export_name=f"{fid}_associations")

    @app.callback(Output("ft-neighbors", "children"), Input("url", "search"))
    def neighbors(search):
        fid = _selected(search)
        if not fid or fid not in b.features.index:
            return None
        g = b.features.at[fid, "group"]
        if not isinstance(g, str):
            return ui.note("Not in a data-driven group.")
        cand = list(b.group_members(g).index)
        r = b.correlated_features(fid, cand, top=20)
        df = b.features.loc[r.index, ["dtype", "display_name", "is_annotated"]].assign(r=r.values) \
            .rename_axis("feature_id").reset_index()
        return ui.grid("ft-nb-grid", df, [("feature_id", None, None, 200), ("dtype", None, None, 60),
                                          ("display_name", "name"), ("r", "r", "num", 70)], height=300,
                       export_name=f"{fid}_neighbors")
