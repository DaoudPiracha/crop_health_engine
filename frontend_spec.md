# Frontend Spec — VI Analysis Dashboard

This document describes the full functionality and data flow of the Dash app (`engine/vi_analysis/app/vi_dash_app.py`) as a reference for rebuilding in React or any other frontend framework.

---

## Overview

A two-panel agronomic review tool. Left panel is a full-height satellite map with interactive field polygons. Right panel is a fixed-width sidebar showing field metadata, a vegetation index time-series chart, and an assessment form. All assessment data is persisted to Supabase via an Express backend.

---

## Layout

```
┌─────────────────────────────────┬──────────────────────┐
│                                 │  Sidebar (380px)     │
│         Map (flex: 1)           │  - Title + summary   │
│         Leaflet / satellite     │  - Toggle buttons    │
│         Field polygons          │  - Hint text         │
│         Highlights              │  - VI selector       │
│         Markers                 │  - Field info card   │
│                                 │  - VI chart (340px)  │
│                                 │  - Assessment form   │
└─────────────────────────────────┴──────────────────────┘
```

**Sidebar width:** 380px, fixed, scrollable vertically  
**Map:** fills remaining width, full viewport height  
**Font:** sans-serif

---

## Theme / Colors

| Token | Value | Usage |
|---|---|---|
| `COLOR_BG` | `#1e1e2e` | Sidebar background |
| `COLOR_TEXT` | `#cdd6f4` | Primary text |
| `COLOR_HEADING` | `#cba6f7` | Title |
| `COLOR_MUTED` | `#a6adc8` | Secondary text, hints |
| `COLOR_DIVIDER` | `#313244` | Borders, chart grid |
| `TOGGLE_STYLE` | `#313244` bg | Button off state |
| `TOGGLE_STYLE_ON` | `#45475a` bg | Button on state |
| `UNASSIGNED_COLOR` | `#bbbbbb` | Fields with no block assignment |
| `HIGHLIGHT_COLOR_A` | `red` | Compare mode field A label |
| `HIGHLIGHT_COLOR_B` | `yellow` | Compare mode field B outline |
| `TRACE_COLOR_A` | `#a6e3a1` | VI chart trace for field A |
| `TRACE_COLOR_B` | `#89dceb` | VI chart trace for field B |

---

## Data Sources

All loaded at startup from local files (paths configured via `config.py`):

| File | Contents |
|---|---|
| `{CROP}_drawn_named.geojson` | Field boundary polygons (GeoJSON FeatureCollection, `Name` property = field ID) |
| `{SEASON}_{CROP}_field_veg_index_stats.csv` | VI time-series log: columns `date`, `name`, `ndvi_mean`, `ndvi_std`, `evi_mean`, `evi_std`, `ndre_mean`, `ndre_std`, `cire_mean`, `cire_std`, ... |
| `{CROP}_blocks.csv` | Block/cluster metadata: columns `name`, `block_id`, `cluster` |
| `{CROP}_wwf_map.geojson` | WWF administrative boundaries (optional) |
| `{CROP}_ndvi_z_scores_norm.csv` | Per-field NDVI z-scores, used for Z-Score layer coloring (optional) |
| `{CROP}_whatsapp_markers.geojson` | WhatsApp-exported field pins (optional) |

**Config variables:**
```
CROP      = "shahmeer"
SEASON    = "kharif"
ASSET_DIR = "/path/to/assets"
API_URL   = "http://localhost:3003/api/field-assessment"
LOGS_URL  = "http://localhost:3003/api/field-assessments"
```

---

## Map

**Tile layer:** Esri World Imagery satellite  
`https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}`

**Initial center:** computed as centroid of all field boundaries  
**Initial zoom:** 14

### GeoJSON Layers (stacked in order)

| Layer | Default state | Description |
|---|---|---|
| Field layers (one per color group) | Visible, filled | Fields colored by block ID, grouped by color for performance |
| Z-Score layers (one per quantile bin) | Hidden | Same fields re-colored red→yellow→green by NDVI z-score |
| WWF boundary layer | Hidden | White outline, no fill |
| `highlight-a` | Empty | White outline, weight 4 — selected field A |
| `highlight-b` | Empty | Yellow outline, weight 2.5 — selected field B (compare mode) |
| `submitted-highlights` | Empty | Green outline, weight 4 — fields with `correct` assessment |
| `uncertain-highlights` | Empty | Red outline, weight 4 — fields with `revisit` assessment |
| Markers layer | Hidden | Point markers from WhatsApp KML export |

### Field Layer Styles

**Normal:**
```
fillColor: block color hex
color: "black"
weight: 0.8
fillOpacity: 0.75 (0.5 for unassigned)
```

**Hover:**
```
weight: 2
color: "white"  (or "red" when Outlines Only mode is active)
fillOpacity: 0
```

**Selected (highlight-a):**
```
color: "white"
weight: 4
fillOpacity: 0
```

### Field Colors

Fields are colored by `block_id`. Block colors are derived deterministically from block IDs using a fixed color mapping. Fields with no block assignment use `#bbbbbb`.

### Z-Score Color Scale

10 quantile bins mapped to a red → yellow → green gradient:
- Bin 0 (lowest z-score): `#e74c3c`
- Bin 9 (highest z-score): `#2ecc73`
- Interpolated linearly through yellow at the midpoint

---

## Sidebar

### Header
- **Title:** "VI Analysis" (color: `#cba6f7`)
- **Summary line:** e.g. `"42 assessed · 5 need revisit"` — fetched from backend on load, updated every 60s and after each submit

### Toggle Buttons

All buttons are toggle-style (on/off). Active state has lighter background (`#45475a`).

| Button | Default | Behaviour |
|---|---|---|
| **Fields** | ON | Toggles field polygon fill visibility. When off, all field layers are hidden (data=null) |
| **Outlines only** | OFF | Shows field outlines only (fillOpacity: 0). Hover color changes to red |
| **Z-Score** | OFF | Swaps field layers for z-score colored layers. Disabled if z-score file absent |
| **WWF boundaries** | OFF | Shows/hides WWF boundary GeoJSON overlay |
| **Markers** | OFF | Shows/hides WhatsApp marker pins |
| **Compare** | OFF | Enables compare mode (see below) |
| **Annotations** | OFF | Shows/hides green and red assessment highlight outlines |

**Fields + Z-Score are mutually exclusive** — enabling Z-Score hides field layers, disabling it restores them.

### Hint Text

Single line below buttons. Updates based on mode:
- Default: `"Click a field to view its VI time series."`
- Compare mode, waiting for A: `"Click to set Field A."`
- Compare mode, waiting for B: `"Click to set Field B."`

### VI Selector

Radio buttons inline: **NDVI · EVI · NDRE · CIRE**  
Changing selection re-renders the chart immediately for the current selection.

Y-axis ranges:
```
ndvi: [0, 1]
evi:  [0, 2]
ndre: [0, 1]
cire: [0, 5]
```

### Field Info Card

Appears when a field is selected. Shows:
```
[label: A / B]
[field_id bold]
Block ID {n}  ·  Crop ID {n}
WWF ID: {name or —}
```

In compare mode with two fields selected, shows two cards side by side with a divider.

### VI Chart

Plotly line chart, height 340px, no modebar.

**Background:** `#1e1e2e` (matches sidebar)  
**Grid:** `#313244`  
**Traces:** lines+markers, marker size 4  

- Single field: one trace in `TRACE_COLOR_A` (`#a6e3a1`)
- Compare mode: trace A in `#a6e3a1`, trace B in `#89dceb`
- If no data for a field: annotation text centred in chart

### Assessment Panel

Hidden until a field is selected. White card with subtle shadow.

**Contents:**
1. **"Field Assessment"** heading
2. **Status radio:** `Correct` (value: `"correct"`) / `Incorrect — needs revisit` (value: `"revisit"`)
3. **Notes textarea** (80px height, resizable)
4. **Submit Assessment button** (blue `#4a90d9`, full width)
5. **Submit status message** — success (green) or error (red), shown below button
6. **Last saved card** — shown below submit status if a prior assessment exists in the DB:
   - "LAST SAVED" label (uppercase, muted)
   - Row: timestamp left, status badge right
   - Badge: green `#2d8a4e` for Correct, red `#e74c3c` for Needs revisit
   - Notes text if present

When a field is clicked, the form pre-populates with the last saved status and notes from cache.

---

## Interactions

### Field Click (single mode)

1. Click fires `clickData` on the GeoJSON layer
2. `field_id` extracted from `properties.field_id`
3. `compare-store` updated: `field_a = field_id`, `field_b = null`
4. Sidebar renders: field info card, VI chart, assessment panel
5. `highlight-a` layer set to field's GeoJSON (white thick outline)
6. Assessment form pre-populated from `assessments-cache`

### Compare Mode

1. Click **Compare** — hint changes to "Click to set Field A."
2. First field click → sets `field_a`, hint → "Click to set Field B."
3. Second field click → sets `field_b`
4. Chart shows two traces overlaid
5. Field info shows two cards side by side
6. `highlight-a` (white) on field A, `highlight-b` (yellow) on field B
7. Clicking Compare again resets to single mode, clears field B

### Assessment Submit

1. POST to `API_URL` with `{ field_id, status, notes }`
2. On success:
   - "Saved successfully." shown in green
   - `log-entry` card updated immediately with new timestamp and status
   - `assessments-cache` updated in-place (no re-fetch needed)
   - `submitted-fields` list updated (triggers annotation highlight refresh)
3. On error: error message shown in red

---

## State Management

| Store | Type | Contents |
|---|---|---|
| `compare-store` | in-memory | `{ field_a, field_b, next_slot, compare_on, vi }` |
| `submitted-fields` | in-memory | List of field IDs submitted this session |
| `assessments-cache` | in-memory | Dict `{ field_id: { status, notes, updated_at } }` — populated from backend on load, updated on each submit |

The `assessments-cache` is the key performance mechanism — all field clicks read from this dict rather than making HTTP requests.

---

## Backend API

Express/Node server, default port 3003.

### `POST /api/field-assessment`

Upsert an assessment.

**Request body:**
```json
{ "field_id": "string", "status": "correct|revisit", "notes": "string" }
```

**Response:**
```json
{ "success": true, "data": { "farm_id", "field_id", "status", "notes", "updated_at" } }
```

### `GET /api/field-assessment/:fieldId`

Fetch existing assessment for a single field.

**Response:**
```json
{ "data": { "status", "notes", "updated_at" } }
// or
{ "data": null }  // if no prior assessment
```

### `GET /api/field-assessments`

Fetch all assessments for the farm, ordered by `updated_at` desc.

**Response:**
```json
{ "data": [ { "field_id", "status", "notes", "updated_at" }, ... ] }
```

---

## Annotation Highlights (map outlines)

Driven by `GET /api/field-assessments` — fetched on page load and every 60 seconds.

- Fields with `status == "correct"` → green outline (`#2ecc71`, weight 4)
- Fields with any other status → red outline (`#e74c3c`, weight 4)
- Only visible when **Annotations** toggle is ON
- Summary count always visible regardless of toggle

---

## WhatsApp Markers

If `{CROP}_whatsapp_markers.geojson` exists, its features are rendered as map pins.  
Each marker has:
- Tooltip: feature `Name` property
- Popup: `Name` + stripped HTML description from the `description` property

Toggled via the **Markers** button.

---

## Performance Notes

- Field layers are grouped by color at startup — one GeoJSON layer per unique color rather than one per field
- VI time-series data is pre-grouped by field ID at startup into a dict (`_vi_log_by_field`) — field clicks do a dict lookup, not a DataFrame filter
- Assessments are cached in a client-side store — no HTTP call on field click
- Annotation highlights refresh every 60s (not on every interaction)
