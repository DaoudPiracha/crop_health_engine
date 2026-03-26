"""
vi_dash_app.py — Interactive Dash app for VI time-series exploration.

Click any field polygon on the map to display its NDVI time series in
the sidebar. Use Compare mode to overlay two fields on the same chart.

Usage:
    python -m engine.vi_analysis.app.vi_dash_app
"""
from __future__ import annotations

import dash
import dash_leaflet as dl
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html

from engine.vi_analysis.app.data import app_data
from engine.vi_analysis.app.theme import (
    COLOR_BG, COLOR_DIVIDER, COLOR_HEADING, COLOR_MUTED, COLOR_TEXT,
    HIGHLIGHT_COLOR_A, HIGHLIGHT_COLOR_B,
    TOGGLE_STYLE, TOGGLE_STYLE_ON,
    TRACE_COLOR_A, TRACE_COLOR_B,
    empty_figure,
)

# Unpack singletons for convenience
_vi_log            = app_data.vi_log
_layers            = app_data.layers
_z_score_layers    = app_data.z_score_layers
_wwf_geojson       = app_data.wwf_geojson
_map_center        = app_data.map_center
_field_props_map   = app_data.field_props_map
_field_geojson_map = app_data.field_geojson_map

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

# compare_on lives in the store so both callbacks share a single source of truth
_EMPTY_STORE = {"field_a": None, "field_b": None, "next_slot": "a", "compare_on": False, "vi": "ndvi"}

_VI_OPTIONS = [
    {"label": "NDVI", "value": "ndvi"},
    {"label": "EVI",  "value": "evi"},
    {"label": "NDRE", "value": "ndre"},
    {"label": "CIRE", "value": "cire"},
]

# Y-axis range per VI
_VI_YRANGE = {"ndvi": [0, 1], "evi": [0, 2], "ndre": [0, 1], "cire": [0, 5]}

app = dash.Dash(__name__)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _map_panel() -> html.Div:
    field_layers = [
        dl.GeoJSON(
            id={"type": "field-layer", "index": i},
            data=layer.geojson,
            style={"fillColor": layer.color, "color": "black",
                   "weight": 0.8, "fillOpacity": layer.fill_opacity},
            hoverStyle={"weight": 2, "color": "white", "fillOpacity": 0},
        )
        for i, layer in enumerate(_layers)
    ]
    zscore_layers = [
        dl.GeoJSON(
            id={"type": "zscore-layer", "index": i},
            data=layer.geojson,
            style={"fillColor": layer.color, "color": "black",
                   "weight": 0.8, "fillOpacity": 0},  # hidden by default
            hoverStyle={"weight": 2, "color": "white", "fillOpacity": 0},
        )
        for i, layer in enumerate(_z_score_layers)
    ]
    return html.Div(
        style={"flex": "1", "position": "relative"},
        children=[
            dl.Map(
                center=_map_center,
                zoom=14,
                style={"height": "100%", "width": "100%"},
                children=[
                    dl.TileLayer(
                        url="https://server.arcgisonline.com/ArcGIS/rest/services/"
                            "World_Imagery/MapServer/tile/{z}/{y}/{x}",
                        attribution="Esri World Imagery",
                    ),
                    *field_layers,
                    *zscore_layers,
                    dl.GeoJSON(
                        id="wwf-layer",
                        data=_wwf_geojson,
                        style={"fillColor": "none", "color": "white",
                               "weight": 2.5, "fillOpacity": 0},
                    ) if _wwf_geojson else None,
                    dl.GeoJSON(id="highlight-a", data=None,
                               style={"fillColor": "none", "color": HIGHLIGHT_COLOR_A,
                                      "weight": 2.5, "fillOpacity": 0}),
                    dl.GeoJSON(id="highlight-b", data=None,
                               style={"fillColor": "none", "color": HIGHLIGHT_COLOR_B,
                                      "weight": 2.5, "fillOpacity": 0}),
                ],
            ),
        ],
    )


def _sidebar() -> html.Div:
    return html.Div(
        style={
            "width": "380px", "padding": "16px", "boxSizing": "border-box",
            "overflowY": "auto", "background": COLOR_BG, "color": COLOR_TEXT,
            "display": "flex", "flexDirection": "column", "gap": "12px",
        },
        children=[
            html.H3("VI Analysis", style={"margin": "0", "color": COLOR_HEADING}),
            html.Div(
                style={"display": "flex", "gap": "8px",
                       "borderBottom": f"1px solid {COLOR_DIVIDER}",
                       "paddingBottom": "10px"},
                children=[
                    html.Button("Fields",         id="btn-fields",   n_clicks=0, style=TOGGLE_STYLE_ON),
                    html.Button("Outlines only",  id="btn-outlines", n_clicks=0, style=TOGGLE_STYLE),
                    html.Button("Z-Score",        id="btn-zscore",   n_clicks=0,
                                style=TOGGLE_STYLE if _z_score_layers else {**TOGGLE_STYLE, "opacity": "0.4", "cursor": "not-allowed"}),
                    html.Button("WWF boundaries", id="btn-wwf",      n_clicks=0, style=TOGGLE_STYLE),
                    html.Button("Compare",        id="btn-compare",  n_clicks=0, style=TOGGLE_STYLE),
                ],
            ),
            html.P(id="sidebar-hint",
                   children="Click a field to view its VI time series.",
                   style={"color": COLOR_MUTED, "fontSize": "13px", "margin": "0"}),
            dcc.RadioItems(
                id="vi-selector",
                options=_VI_OPTIONS,
                value="ndvi",
                inline=True,
                style={"fontSize": "12px", "color": COLOR_TEXT, "gap": "10px"},
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "12px"},
            ),
            html.Div(id="field-info", style={"fontSize": "13px"}),
            dcc.Graph(id="vi-chart", config={"displayModeBar": False},
                      style={"height": "340px"}),
        ],
    )


app.layout = html.Div(
    style={"display": "flex", "height": "100vh", "fontFamily": "sans-serif"},
    children=[
        dcc.Store(id="compare-store", data=_EMPTY_STORE),
        _map_panel(),
        _sidebar(),
    ],
)

# ---------------------------------------------------------------------------
# Sidebar rendering helpers
# ---------------------------------------------------------------------------

def _field_card(field_id: str, label: str, label_color: str) -> html.Div:
    p = _field_props_map.get(field_id, {"block_id": "N/A", "cluster": "N/A", "wwf_name": "—"})
    return html.Div(style={"flex": "1", "minWidth": "0"}, children=[
        html.Span(label, style={"color": label_color, "fontSize": "11px", "fontWeight": "bold"}),
        html.Br(),
        html.Strong(field_id, style={"color": COLOR_TEXT}),
        html.Br(),
        html.Span(f"Block ID {p['block_id']}  ·  Crop ID {p['cluster']}",
                  style={"color": COLOR_MUTED, "fontSize": "12px"}),
        html.Br(),
        html.Span("WWF ID: ", style={"color": COLOR_MUTED, "fontSize": "12px"}),
        html.Span(p["wwf_name"], style={"color": COLOR_TEXT, "fontSize": "12px"}),
    ])


def _vi_trace(field_id: str, label: str, color: str, vi: str) -> go.Scatter | None:
    col = f"{vi}_mean"
    data = _vi_log[_vi_log["name"] == field_id].sort_values("date")
    if data.empty or col not in data.columns:
        return None
    return go.Scatter(
        x=pd.to_datetime(data["date"]),
        y=data[col],
        mode="lines+markers",
        name=f"{label}: {field_id}",
        line={"color": color, "width": 2},
        marker={"size": 4},
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output({"type": "field-layer",  "index": ALL}, "data"),
    Output({"type": "field-layer",  "index": ALL}, "style"),
    Output({"type": "field-layer",  "index": ALL}, "hoverStyle"),
    Output({"type": "zscore-layer", "index": ALL}, "data"),
    Output({"type": "zscore-layer", "index": ALL}, "style"),
    Output({"type": "zscore-layer", "index": ALL}, "hoverStyle"),
    Output("btn-fields",  "style"),
    Output("btn-outlines","style"),
    Output("btn-zscore",  "style"),
    Input("btn-fields",  "n_clicks"),
    Input("btn-outlines","n_clicks"),
    Input("btn-zscore",  "n_clicks"),
)
def toggle_fields(fields_clicks, outlines_clicks, zscore_clicks):
    fields_on   = (fields_clicks   % 2) == 0
    outlines_on = (outlines_clicks % 2) == 1
    zscore_on   = (zscore_clicks   % 2) == 1 and bool(_z_score_layers)

    hover_color = "red" if outlines_on else "white"
    hover        = {"weight": 2, "color": hover_color, "fillOpacity": 0}
    hover_hidden = {"weight": 0, "color": "transparent", "fillOpacity": 0}

    def _layer_style(layer: "Layer") -> dict:
        if outlines_on:
            return {"fillColor": layer.color, "color": "white", "weight": 1, "fillOpacity": 0}
        return {"fillColor": layer.color, "color": "black",
                "weight": 0.8, "fillOpacity": layer.fill_opacity}

    # Use data=None to fully remove a layer from the map (prevents event interception)
    if not fields_on:
        block_data  = [None] * len(_layers)
        zscore_data = [None] * len(_z_score_layers)
    elif zscore_on:
        block_data  = [None] * len(_layers)
        zscore_data = [l.geojson for l in _z_score_layers]
    else:
        block_data  = [l.geojson for l in _layers]
        zscore_data = [None] * len(_z_score_layers)

    block_hover  = [hover if fields_on and not zscore_on else hover_hidden for _ in _layers]
    zscore_hover = [hover if fields_on and zscore_on     else hover_hidden for _ in _z_score_layers]

    return (
        block_data,
        [_layer_style(l) for l in _layers],
        block_hover,
        zscore_data,
        [_layer_style(l) for l in _z_score_layers],
        zscore_hover,
        TOGGLE_STYLE_ON if fields_on   else TOGGLE_STYLE,
        TOGGLE_STYLE_ON if outlines_on else TOGGLE_STYLE,
        TOGGLE_STYLE_ON if zscore_on   else TOGGLE_STYLE,
    )


if _wwf_geojson:
    @app.callback(
        Output("wwf-layer", "style"),
        Output("btn-wwf", "style"),
        Input("btn-wwf", "n_clicks"),
    )
    def toggle_wwf(n_clicks):
        visible = (n_clicks % 2) == 1
        return (
            {"fillColor": "none", "color": "white",
             "weight": 2.5 if visible else 0, "fillOpacity": 0},
            TOGGLE_STYLE_ON if visible else TOGGLE_STYLE,
        )
else:
    @app.callback(
        Output("btn-wwf", "style"),
        Input("btn-wwf", "n_clicks"),
    )
    def toggle_wwf(_):
        return {**TOGGLE_STYLE, "opacity": "0.4", "cursor": "not-allowed"}


@app.callback(
    Output("compare-store", "data"),
    Input({"type": "field-layer",  "index": ALL}, "clickData"),
    Input({"type": "zscore-layer", "index": ALL}, "clickData"),
    Input("btn-compare", "n_clicks"),
    Input("vi-selector", "value"),
    State("compare-store", "data"),
)
def update_store(field_click_data, zscore_click_data, compare_clicks, vi_value, store):
    if ctx.triggered_id == "vi-selector":
        return {**store, "vi": vi_value}

    compare_on = (compare_clicks % 2) == 1

    if ctx.triggered_id == "btn-compare":
        if not compare_on:
            return {**store, "compare_on": False, "field_b": None, "next_slot": "b"}
        return {**store, "compare_on": True}

    if not ctx.triggered:
        return store
    triggered_prop = ctx.triggered[0].get("prop_id", "")
    if "clickData" not in triggered_prop:
        return store
    click_data = ctx.triggered[0]["value"]
    if not click_data or not isinstance(click_data, dict):
        return store

    field_id = click_data.get("properties", {}).get("field_id", "")
    if not field_id:
        return store

    if not compare_on:
        return {**store, "compare_on": False, "field_a": field_id, "field_b": None, "next_slot": "b"}

    if store.get("next_slot", "a") == "a":
        return {**store, "compare_on": True, "field_a": field_id, "next_slot": "b"}
    return {**store, "compare_on": True, "field_b": field_id, "next_slot": "a"}


@app.callback(
    Output("field-info", "children"),
    Output("vi-chart", "figure"),
    Output("btn-compare", "style"),
    Output("sidebar-hint", "children"),
    Output("highlight-a", "data"),
    Output("highlight-b", "data"),
    Input("compare-store", "data"),
)
def render_selection(store):
    compare_on = store.get("compare_on", False)
    field_a    = store.get("field_a")
    field_b    = store.get("field_b")
    vi         = store.get("vi", "ndvi")

    hint = (
        f"Click to set Field {'A' if store.get('next_slot', 'a') == 'a' else 'B'}."
        if compare_on else
        "Click a field to view its VI time series."
    )

    if not field_a:
        return "", empty_figure(), TOGGLE_STYLE_ON if compare_on else TOGGLE_STYLE, hint, None, None

    fig = empty_figure()
    trace_a = _vi_trace(field_a, "A", TRACE_COLOR_A, vi)
    trace_b = _vi_trace(field_b, "B", TRACE_COLOR_B, vi) if field_b else None

    if trace_a:
        fig.add_trace(trace_a)
    else:
        fig.add_annotation(text=f"No data for {field_a}", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font={"color": COLOR_MUTED})
    if trace_b:
        fig.add_trace(trace_b)
    elif field_b:
        fig.add_annotation(text=f"No data for {field_b}", xref="paper", yref="paper",
                           x=0.5, y=0.3, showarrow=False, font={"color": COLOR_MUTED})
    fig.update_layout(
        legend={"font": {"size": 11}, "bgcolor": "rgba(0,0,0,0)"},
        xaxis_title="Date",
        yaxis_title=vi.upper(),
        yaxis={"range": _VI_YRANGE.get(vi, [0, 1]), "gridcolor": COLOR_DIVIDER},
    )

    if field_b:
        info = html.Div(
            style={"display": "flex", "gap": "10px"},
            children=[
                _field_card(field_a, "A", HIGHLIGHT_COLOR_A),
                html.Div(style={"width": "1px", "background": COLOR_DIVIDER}),
                _field_card(field_b, "B", HIGHLIGHT_COLOR_B),
            ],
        )
    else:
        info = _field_card(field_a, "A", HIGHLIGHT_COLOR_A if compare_on else TRACE_COLOR_A)

    return (
        info, fig,
        TOGGLE_STYLE_ON if compare_on else TOGGLE_STYLE,
        hint,
        _field_geojson_map.get(field_a) if compare_on else None,
        _field_geojson_map.get(field_b) if compare_on else None,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
