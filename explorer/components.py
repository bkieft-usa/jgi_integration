"""Shared UI building blocks."""
from __future__ import annotations

from typing import Dict, List, Optional
from urllib.parse import quote

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html

# Columns whose cells navigate when clicked: column -> (route, query key)
NAV_COLUMNS = {
    "feature_id": ("/feature", "id"),
    "group": ("/groups", "group"),
    "top_group": ("/groups", "group"),
    "entity_id": ("/terms", "term"),
}


def href(route: str, **params) -> str:
    q = "&".join(f"{k}={quote(str(v), safe='')}" for k, v in params.items() if v not in (None, ""))
    return f"{route}?{q}" if q else route


def page_header(title: str, help_text: str, extra=None) -> html.Div:
    return html.Div([
        html.Div([html.H4(title, className="mb-1"), extra], className="d-flex align-items-baseline gap-3"),
        html.P(help_text, className="text-secondary small mb-3", style={"maxWidth": "980px"}),
    ])


def card(title: str, body, subtitle: Optional[str] = None, className: str = "") -> dbc.Card:
    head = [html.Div(title, className="card-title-text")]
    if subtitle:
        head.append(html.Div(subtitle, className="text-secondary small"))
    return dbc.Card([dbc.CardHeader(head), dbc.CardBody(body)], className=f"mb-3 viz-card {className}")


def stat(label: str, value, note: Optional[str] = None) -> html.Div:
    return html.Div([
        html.Div(label, className="stat-label"),
        html.Div(value, className="stat-value"),
        html.Div(note, className="stat-note") if note else None,
    ], className="stat-tile")


def stat_row(items: List) -> html.Div:
    return html.Div(items, className="stat-row")


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "–"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def _col_def(field: str, header: Optional[str] = None, kind: Optional[str] = None, width: Optional[int] = None) -> Dict:
    d: Dict = {"field": field, "headerName": header or field.replace("_", " "), "filter": True}
    if kind == "p":
        d.update(valueFormatter={"function": "params.value == null ? '' : (params.value < 0.001 ? "
                                             "d3.format('.1e')(params.value) : d3.format('.3f')(params.value))"},
                 filter="agNumberColumnFilter", type="numericColumn", minWidth=90)
    elif kind == "num":
        d.update(valueFormatter={"function": "params.value == null ? '' : d3.format('.3~f')(params.value)"},
                 filter="agNumberColumnFilter", type="numericColumn")
    elif kind == "int":
        d.update(filter="agNumberColumnFilter", type="numericColumn")
    if field in NAV_COLUMNS:
        d["cellClass"] = "nav-cell"
        if field == "entity_id":
            d["valueFormatter"] = {"function": "params.value ? params.value.split('::').slice(1).join('::') : ''"}
            d["headerName"] = header or "term"
    if isinstance(width, (int, float)) and width:
        d["width"] = width
    else:
        d["flex"] = 1
        d["minWidth"] = 110
    return d


def grid(grid_id: str, df: pd.DataFrame, columns: List, height: int = 420, export_name: str = "table") -> html.Div:
    """Sortable/filterable table. ``columns``: ``field`` or ``(field, header, kind[, width])`` tuples.

    Clicking a cell in a navigable column (feature_id, group, entity_id) opens its page.
    """
    defs = []
    for c in columns:
        if isinstance(c, str):
            defs.append(_col_def(c))
        else:
            defs.append(_col_def(*c))
    fields = [d["field"] for d in defs]
    data = df[[f for f in fields if f in df.columns]].replace({np.nan: None}).to_dict("records") if len(df) else []
    return html.Div([
        html.Div([
            html.Span(f"{len(df):,} rows", className="text-secondary small"),
            dbc.Button("Download CSV", id={"type": "grid-export", "id": grid_id}, size="sm", color="link",
                       className="p-0 ms-auto"),
        ], className="d-flex align-items-center mb-1"),
        dag.AgGrid(
            id={"type": "nav-grid", "id": grid_id},
            rowData=data, columnDefs=defs,
            defaultColDef={"sortable": True, "resizable": True, "floatingFilter": False},
            dashGridOptions={"pagination": True, "paginationPageSize": 50, "animateRows": False,
                             "tooltipShowDelay": 300, "enableCellTextSelection": True},
            csvExportParams={"fileName": f"{export_name}.csv"},
            style={"height": f"{height}px"},
            className="ag-theme-quartz",
        ),
    ])


def dropdown(id_, options, value=None, placeholder=None, clearable=False, multi=False, style=None, **kw):
    opts = [o if isinstance(o, dict) else {"label": str(o), "value": o} for o in options]
    return dcc.Dropdown(id=id_, options=opts, value=value, placeholder=placeholder, clearable=clearable, multi=multi,
                        style=style or {"minWidth": "200px"}, **kw)


def labeled(label: str, control) -> html.Div:
    return html.Div([html.Label(label, className="form-label small text-secondary mb-1"), control],
                    className="control")


def note(text: str, kind: str = "info") -> html.Div:
    return dbc.Alert(text, color="light" if kind == "info" else "warning", className="small py-2 mb-2")


def graph(id_, height: int = 380, **kw) -> dcc.Graph:
    return dcc.Loading(dcc.Graph(id=id_, style={"height": f"{height}px"},
                                 config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}}, **kw),
                       type="dot", delay_show=300)
