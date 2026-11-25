"""Enrichment matrix: knowledge-driven terms x data-driven groups."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Input, Output, State, html, no_update

from explorer import components as ui
from explorer import figures as fg
from explorer import theme as th


def layout(b, params):
    at_opts = [{"label": a["label"], "value": a["key"]} for a in b.ann_types]
    sig_counts = b.enrich[b.enrich["padj"] < b.fdr]["ann_type"].value_counts()
    default = params.get("ann_type") or (sig_counts.index[0] if len(sig_counts) else (at_opts[0]["value"] if at_opts else None))
    return html.Div([
        ui.page_header("Enrichment matrix",
                       "Which knowledge-driven terms are over-represented in which data-driven groups "
                       "(one-sided hypergeometric test, background = grouped features annotated with that type, "
                       "BH-corrected per annotation type). Click a cell to list the overlapping features."),
        html.Div([
            ui.labeled("Annotation type", ui.dropdown("en-anntype", at_opts, default, style={"minWidth": "260px"})),
            ui.labeled("Color", ui.dropdown("en-metric", [{"label": "−log10 padj", "value": "nlp"},
                                                         {"label": "fold enrichment", "value": "fold"}], "nlp")),
            ui.labeled("Minimum overlap (k)", ui.dropdown("en-mink", [1, 2, 3, 5, 10], 2)),
            ui.labeled("Show at most", ui.dropdown("en-max", [{"label": f"{n} terms × {2 * n} groups", "value": n}
                                                             for n in (20, 40, 60)], 40)),
        ], className="controls"),
        ui.card("Significant term × group pairs", [ui.graph("en-heat", 720),
                                                   html.Div(id="en-caption", className="small text-secondary")]),
        ui.card("Overlapping features", html.Div(id="en-cell", children=ui.note("Click a heatmap cell."))),
    ])


def register(app, b):
    @app.callback(Output("en-heat", "figure"), Output("en-caption", "children"),
                  Input("en-anntype", "value"), Input("en-metric", "value"), Input("en-mink", "value"),
                  Input("en-max", "value"), Input("filters", "data"), Input("theme", "data"))
    def heat(at, metric, mink, max_terms, filters, theme):
        fdr = (filters or {}).get("fdr", b.fdr)
        e = b.enrich[(b.enrich["ann_type"] == at) & (b.enrich["k"] >= (mink or 1))]
        sig = e[e["padj"] < fdr]
        if sig.empty:
            return th.empty_figure(theme, f"No significant pairs at FDR < {fdr} for this annotation type"), ""
        terms = sig.groupby("term_id")["padj"].min().nsmallest(max_terms).index
        groups = sig[sig["term_id"].isin(terms)].groupby("group")["padj"].min().nsmallest(2 * max_terms).index
        sub = e[e["term_id"].isin(terms) & e["group"].isin(groups)].copy()
        sub["nlp"] = -np.log10(sub["padj"].clip(lower=1e-300))
        sub.loc[sub["padj"] >= fdr, ["nlp", "fold"]] = np.nan  # only significant cells are colored
        mat = sub.pivot_table(index="term_id", columns="group", values=metric, aggfunc="first")
        kmat = sub.pivot_table(index="term_id", columns="group", values="k", aggfunc="first")
        # order rows/columns by similarity of their significance patterns
        filled = mat.fillna(0)
        order_r = filled.sum(axis=1).sort_values(ascending=False).index
        order_c = filled.loc[order_r].idxmax(axis=0).map({t: i for i, t in enumerate(order_r)}).sort_values().index
        mat, kmat = mat.loc[order_r, order_c], kmat.reindex(index=order_r, columns=order_c)
        labels = [b.term_label(f"{at}::{t}", 55) for t in mat.index]
        hover = np.where(kmat.notna(), "k = " + kmat.fillna(0).astype(int).astype(str), "")
        fig = fg.enrichment_heatmap(mat.set_axis(labels, axis=0), theme,
                                    "−log10 padj" if metric == "nlp" else "fold", hover=hover)
        fig.update_layout(meta={"term_ids": list(mat.index)})
        return fig, (f"{len(sig):,} significant pairs at FDR < {fdr}; showing the {len(mat)} most significant terms "
                     f"and {mat.shape[1]} groups. Blank cells are not significant.")

    @app.callback(Output("en-cell", "children"), Input("en-heat", "clickData"), State("en-heat", "figure"),
                  State("en-anntype", "value"), prevent_initial_call=True)
    def cell(click, fig, at):
        if not click:
            return no_update
        p = click["points"][0]
        term_ids = (fig.get("layout", {}).get("meta") or {}).get("term_ids", [])
        labels = fig["data"][0]["y"]
        term = term_ids[labels.index(p["y"])] if p["y"] in labels else None
        group = p["x"]
        if term is None:
            return no_update
        ent = f"{at}::{term}"
        members = b.term_members(ent)
        both = members[members["group"] == group].reset_index()
        return html.Div([
            html.Div([html.Span(f"{b.term_label(ent, 90)} in "), html.A(group, href=ui.href("/groups", group=group)),
                      html.Span("  ·  "), html.A("open term", href=ui.href("/terms", term=ent))], className="mb-2"),
            ui.grid("en-cell-grid", both, [("feature_id", None, None, 220), ("dtype", None, None, 70),
                                           ("display_name", "name"), ("kME", "kME", "num", 90)], height=320,
                    export_name=f"{term}_{group}"),
        ])
