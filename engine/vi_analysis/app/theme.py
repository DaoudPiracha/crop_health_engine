"""
theme.py — Visual constants for the VI Dash app.
"""

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Map layer colors
# ---------------------------------------------------------------------------

UNASSIGNED_COLOR = "#bbbbbb"

# Field highlight outlines in compare mode
HIGHLIGHT_COLOR_A = "red"
HIGHLIGHT_COLOR_B = "yellow"

# NDVI trace colors
TRACE_COLOR_A = "#a6e3a1"
TRACE_COLOR_B = "#89dceb"

# ---------------------------------------------------------------------------
# Sidebar toggle button style
# ---------------------------------------------------------------------------

TOGGLE_STYLE: dict = {
    "fontSize": "12px",
    "background": "#313244",
    "color": "#cdd6f4",
    "border": "none",
    "borderRadius": "4px",
    "padding": "4px 10px",
    "cursor": "pointer",
    "textAlign": "left",
}

TOGGLE_STYLE_ON: dict = {**TOGGLE_STYLE, "background": "#45475a"}

# ---------------------------------------------------------------------------
# Sidebar text colors
# ---------------------------------------------------------------------------

COLOR_MUTED = "#a6adc8"
COLOR_TEXT = "#cdd6f4"
COLOR_HEADING = "#cba6f7"
COLOR_BG = "#1e1e2e"
COLOR_DIVIDER = "#313244"

# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def empty_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font_color=COLOR_TEXT,
        margin={"l": 40, "r": 10, "t": 10, "b": 40},
        xaxis={"showgrid": False},
        yaxis={"showgrid": True, "gridcolor": COLOR_DIVIDER},
    )
    return fig
