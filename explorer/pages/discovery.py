"""Discovery: unannotated features that co-vary with annotated, enriched groups (guilt by association)."""
from __future__ import annotations

from dash import Input, Output, html

from explorer import components as ui
from explorer import theme as th


def layout(b, params):
    return html.Div([
        ui.page_header("Discovery: unannotated features in annotated groups",
                       "Features with no annotation of any type that belong to a data-driven group with significantly "
                       "enriched knowledge-driven terms. Ranked by how typical the feature is of its group (kME) times "
                       "the strength of the group's best enrichment (−log10 padj). These are candidates for functional "
                       "hypotheses, not annotations."),
        html.Div([
            ui.labeled("Data type", ui.dropdown("dc-dtype", [{"label": "All", "value": "all"}] +
                                                [{"label": th.DTYPE_LABEL.get(d, d), "value": d} for d in b.dtypes], "all")),
            ui.labeled("Minimum kME", ui.dropdown("dc-kme", [0, 0.5, 0.7, 0.8, 0.9], 0.7)),
        ], className="controls"),
        html.Div(id="dc-table"),
    ])


def register(app, b):
    @app.callback(Output("dc-table", "children"), Input("dc-dtype", "value"), Input("dc-kme", "value"),
                  Input("filters", "data"))
    def table(dtype, kme, filters):
        c = b.candidates
        if c.empty:
            return ui.note("No candidates: no group has a significantly enriched term at the bundle's FDR.")
        if dtype and dtype != "all":
            c = c[c["dtype"] == dtype]
        c = c[c["kME"] >= (kme or 0)]
        min_size = (filters or {}).get("min_group_size", 1)
        c = c[c["group"].map(b.groups["size"]) >= min_size]
        return ui.card(f"{len(c):,} candidate features", ui.grid(
            "dc-grid", c, [("feature_id", None, None, 220), ("dtype", None, None, 70), ("group", None, None, 120),
                           ("kME", "kME", "num", 80), ("group_evidence", "group −log10 padj", "num", 140),
                           ("score", "score", "num", 80), ("enriched_terms", "group's top enriched terms")],
            height=620, export_name="discovery_candidates"))
