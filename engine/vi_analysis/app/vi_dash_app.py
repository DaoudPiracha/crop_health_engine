"""
vi_dash_app.py — Interactive Dash app for VI time-series exploration.

Click any field polygon on the map to display its NDVI time series in
the sidebar. Use Compare mode to overlay two fields on the same chart.

Usage:
    python -m engine.vi_analysis.app.vi_dash_app
"""
from __future__ import annotations

import re
import requests
import dash
import dash_leaflet as dl
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html
from datetime import datetime

from engine.vi_analysis.app.config import API_URL, LOGS_URL
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
_markers_geojson   = app_data.markers_geojson
_map_center        = app_data.map_center
_field_props_map   = app_data.field_props_map
_field_geojson_map = app_data.field_geojson_map

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_whatsapp_markers() -> list:
    """Build dl.Marker components from the WhatsApp markers GeoJSON (called once at startup)."""
    if not _markers_geojson:
        return []
    markers = []
    for i, feat in enumerate(_markers_geojson.get("features", [])):
        coords = feat["geometry"]["coordinates"]  # [lon, lat, ...]
        props  = feat.get("properties", {})
        name   = props.get("Name") or ""
        desc   = props.get("description")

        popup_children = [html.Strong(name, style={"fontSize": "13px"})]
        if desc:
            plain_desc = re.sub(r"<[^>]+>", " ", desc).strip()
            plain_desc = re.sub(r"\s{2,}", " ", plain_desc)
            popup_children += [
                html.Hr(style={"margin": "4px 0"}),
                html.Span(plain_desc, style={"fontSize": "12px"}),
            ]

        markers.append(
            dl.Marker(
                id={"type": "whatsapp-marker", "index": i},
                position=[coords[1], coords[0]],
                children=[
                    dl.Tooltip(name, sticky=True),
                    dl.Popup(children=popup_children, maxWidth=300),
                ],
            )
        )
    return markers


_WHATSAPP_MARKERS = _build_whatsapp_markers()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_EMPTY_STORE = {"field_a": None, "field_b": None, "next_slot": "a", "compare_on": False, "vi": "ndvi"}

_VI_OPTIONS = [
    {"label": "NDVI", "value": "ndvi"},
    {"label": "EVI",  "value": "evi"},
    {"label": "NDRE", "value": "ndre"},
    {"label": "CIRE", "value": "cire"},
]

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
            options={"bubblingMouseEvents": False},
        )
        for i, layer in enumerate(_layers)
    ]
    zscore_layers = [
        dl.GeoJSON(
            id={"type": "zscore-layer", "index": i},
            data=layer.geojson,
            style={"fillColor": layer.color, "color": "black",
                   "weight": 0.8, "fillOpacity": 0},
            hoverStyle={"weight": 2, "color": "white", "fillOpacity": 0},
            options={"bubblingMouseEvents": False},
        )
        for i, layer in enumerate(_z_score_layers)
    ]
    wwf_children = [
        dl.GeoJSON(
            id="wwf-layer",
            data=_wwf_geojson,
            style={"fillColor": "none", "color": "white",
                   "weight": 2.5, "fillOpacity": 0},
        )
    ] if _wwf_geojson else []
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
                    *wwf_children,
                    dl.GeoJSON(id="highlight-a", data=None,
                               style={"fillColor": "none", "color": HIGHLIGHT_COLOR_A,
                                      "weight": 2.5, "fillOpacity": 0}),
                    dl.GeoJSON(id="highlight-b", data=None,
                               style={"fillColor": "none", "color": HIGHLIGHT_COLOR_B,
                                      "weight": 2.5, "fillOpacity": 0}),
                    dl.GeoJSON(id="submitted-highlights", data=None,
                               style={"fillColor": "none", "color": "#2ecc71",
                                      "weight": 4, "fillOpacity": 0}),
                    dl.GeoJSON(id="uncertain-highlights", data=None,
                               style={"fillColor": "none", "color": "#e74c3c",
                                      "weight": 4, "fillOpacity": 0}),
                    dl.LayerGroup(id="markers-layer", children=_WHATSAPP_MARKERS),
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
                style={"display": "flex", "flexWrap": "wrap", "gap": "8px",
                       "borderBottom": f"1px solid {COLOR_DIVIDER}",
                       "paddingBottom": "10px"},
                children=[
                    html.Button("Fields",         id="btn-fields",   n_clicks=0, style=TOGGLE_STYLE_ON),
                    html.Button("Outlines only",  id="btn-outlines", n_clicks=0, style=TOGGLE_STYLE),
                    html.Button("Z-Score",        id="btn-zscore",   n_clicks=0,
                                style=TOGGLE_STYLE if _z_score_layers else {**TOGGLE_STYLE, "opacity": "0.4", "cursor": "not-allowed"}),
                    html.Button("WWF boundaries", id="btn-wwf",      n_clicks=0, style=TOGGLE_STYLE),
                    html.Button("Markers",        id="btn-markers",  n_clicks=0, style=TOGGLE_STYLE),
                    html.Button("Compare",        id="btn-compare",      n_clicks=0, style=TOGGLE_STYLE),
                    html.Button("Annotations",    id="btn-annotations",  n_clicks=0, style=TOGGLE_STYLE),
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
            html.Div(
                id="assessment-panel",
                style={"display": "none"},
                children=[
                    html.Div(
                        style={
                            "background": "#ffffff", "borderRadius": "8px",
                            "padding": "14px 16px", "boxShadow": "0 1px 4px rgba(0,0,0,0.3)",
                        },
                        children=[
                            html.P("Field Assessment",
                                   style={"color": "#1a1a2e", "fontSize": "13px",
                                          "fontWeight": "bold", "margin": "0 0 10px 0"}),
                            dcc.RadioItems(
                                id="field-status",
                                options=[
                                    {"label": "  Correct",                "value": "correct"},
                                    {"label": "  Incorrect — needs revisit", "value": "revisit"},
                                ],
                                value="correct",
                                style={"fontSize": "13px", "color": "#1a1a2e"},
                                inputStyle={"marginRight": "6px"},
                                labelStyle={"display": "block", "marginBottom": "8px"},
                            ),
                            html.P("Notes", style={"color": "#555", "fontSize": "12px",
                                                   "margin": "8px 0 4px 0"}),
                            dcc.Textarea(
                                id="field-notes",
                                placeholder="Add notes...",
                                style={
                                    "width": "100%", "height": "80px", "boxSizing": "border-box",
                                    "background": "#f5f5f5", "color": "#1a1a2e",
                                    "border": "1px solid #ddd", "borderRadius": "4px",
                                    "padding": "6px", "fontSize": "12px", "resize": "vertical",
                                },
                            ),
                            html.Button(
                                "Submit Assessment", id="btn-submit", n_clicks=0,
                                style={
                                    "marginTop": "10px", "width": "100%",
                                    "background": "#4a90d9", "color": "#ffffff",
                                    "border": "none", "borderRadius": "4px",
                                    "padding": "7px 0", "fontSize": "13px",
                                    "fontWeight": "bold", "cursor": "pointer",
                                },
                            ),
                            html.Div(id="submit-status",
                                     style={"fontSize": "12px", "marginTop": "8px",
                                            "textAlign": "center"}),
                            # Log entry shown below the form when a saved assessment exists
                            html.Div(id="log-entry"),
                        ],
                    ),
                ],
            ),
        ],
    )


app.layout = html.Div(
    style={"display": "flex", "height": "100vh", "fontFamily": "sans-serif"},
    children=[
        dcc.Store(id="compare-store", data=_EMPTY_STORE),
        dcc.Store(id="submitted-fields", data=[]),
        # Fires once on load to populate green highlights from the DB
        dcc.Interval(id="highlights-interval", interval=30_000, n_intervals=0),
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


def _log_entry_card(status: str, notes: str, ts_raw: str) -> html.Div:
    badge_color = "#2d8a4e" if status == "correct" else "#e74c3c"
    badge_label = "Correct" if status == "correct" else "Needs revisit"
    try:
        dt     = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        ts_fmt = dt.strftime("%b %d, %Y  %H:%M")
    except Exception:
        ts_fmt = ts_raw

    return html.Div(
        style={"borderTop": f"1px solid {COLOR_DIVIDER}", "paddingTop": "10px", "marginTop": "6px"},
        children=[
            html.P("Last saved",
                   style={"color": COLOR_MUTED, "fontSize": "11px", "margin": "0 0 6px 0",
                          "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                children=[
                    html.Span(ts_fmt, style={"color": "#555", "fontSize": "12px"}),
                    html.Span(badge_label, style={
                        "background": badge_color, "color": "#fff",
                        "borderRadius": "4px", "padding": "2px 7px",
                        "fontSize": "11px", "fontWeight": "bold",
                    }),
                ],
            ),
            *(
                [html.Div(notes, style={"color": "#555", "fontSize": "12px", "marginTop": "4px"})]
                if notes else []
            ),
        ],
    )


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
    Output("markers-layer", "children"),
    Output("btn-markers", "style"),
    Input("btn-markers", "n_clicks"),
)
def toggle_markers(n_clicks):
    visible = (n_clicks % 2) == 1
    return (
        _WHATSAPP_MARKERS if visible else [],
        TOGGLE_STYLE_ON if visible else TOGGLE_STYLE,
    )


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
            return {**store, "compare_on": False, "field_b": None, "next_slot": "a"}
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
    Output("assessment-panel", "style"),
    Output("field-status", "value"),
    Output("field-notes", "value"),
    Output("submit-status", "children"),
    Output("log-entry", "children"),
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

    panel_hidden = {"display": "none"}
    panel_shown  = {"display": "block"}

    if not field_a:
        return ("", empty_figure(), TOGGLE_STYLE_ON if compare_on else TOGGLE_STYLE,
                hint, None, None, panel_hidden, "correct", "", "", "")

    # Fetch existing assessment for field_a
    has_saved    = False
    saved_status = "correct"
    saved_notes  = ""
    saved_ts     = ""
    try:
        r = requests.get(f"{API_URL}/{field_a}", timeout=3)
        if r.ok:
            assessment = r.json().get("data")
            if assessment:
                has_saved    = True
                saved_status = assessment.get("status", "correct")
                saved_notes  = assessment.get("notes", "")
                saved_ts     = assessment.get("updated_at", "")
    except requests.exceptions.RequestException:
        pass

    log_entry = _log_entry_card(saved_status, saved_notes, saved_ts) if has_saved else ""

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
        panel_shown, saved_status, saved_notes, "",
        log_entry,
    )


@app.callback(
    Output("submit-status", "children", allow_duplicate=True),
    Output("submitted-fields", "data"),
    Input("btn-submit", "n_clicks"),
    State("compare-store", "data"),
    State("field-status", "value"),
    State("field-notes", "value"),
    State("submitted-fields", "data"),
    prevent_initial_call=True,
)
def submit_assessment(n_clicks, store, status, notes, submitted):
    field_id = store.get("field_a")
    if not field_id:
        return html.Span("No field selected.", style={"color": "#888"}), submitted

    payload = {
        "field_id": field_id,
        "status":   status or "correct",
        "notes":    notes or "",
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=5)
        resp.raise_for_status()
        updated = submitted if field_id in submitted else submitted + [field_id]
        return html.Span("Saved successfully.", style={"color": "#2d8a4e", "fontWeight": "bold"}), updated
    except requests.exceptions.ConnectionError:
        return html.Span("Error: could not reach API.", style={"color": "#c0392b"}), submitted
    except requests.exceptions.HTTPError as e:
        return html.Span(f"Error: {e}", style={"color": "#c0392b"}), submitted
    except requests.exceptions.RequestException as e:
        return html.Span(f"Error: {e}", style={"color": "#c0392b"}), submitted


@app.callback(
    Output("submitted-highlights", "data"),
    Output("uncertain-highlights", "data"),
    Output("btn-annotations", "style"),
    Input("submitted-fields", "data"),
    Input("highlights-interval", "n_intervals"),
    Input("btn-annotations", "n_clicks"),
)
def update_submitted_highlights(submitted, _, annotations_clicks):
    annotations_on = (annotations_clicks % 2) == 1
    if not annotations_on:
        return None, None, TOGGLE_STYLE
    # Fetch all assessments from the DB so highlights persist across sessions
    entries: list[dict] = []
    try:
        r = requests.get(LOGS_URL, timeout=5)
        if r.ok:
            entries = r.json().get("data", [])
    except requests.exceptions.RequestException:
        # Fallback: treat all session field IDs as "correct"
        entries = [{"field_id": fid, "status": "correct"} for fid in submitted]

    def _geojson(field_ids):
        features = []
        for fid in field_ids:
            fc = _field_geojson_map.get(str(fid))
            if fc:
                features.extend(fc.get("features", []))
        return {"type": "FeatureCollection", "features": features} if features else None

    assessed  = [e["field_id"] for e in entries if e.get("status") == "correct"]
    uncertain = [e["field_id"] for e in entries if e.get("status") != "correct"]

    return _geojson(assessed), _geojson(uncertain), TOGGLE_STYLE_ON


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
