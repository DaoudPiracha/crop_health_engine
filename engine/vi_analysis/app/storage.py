"""
storage.py — Remote asset resolution for the VI Dash app.

When USE_REMOTE_ASSETS=true, downloads all required asset files from the
Express backend (GET /api/assets/:filename) to a local temp directory at
startup and returns that directory path.

The backend proxies the files from Supabase Storage — no Supabase credentials
are needed here. Only BACKEND_URL is required (already in config).

Local dev (USE_REMOTE_ASSETS unset or false): falls back to ASSET_DIR, no
download is performed.

Expected backend layout (flat filenames):
    {CROP}_drawn_named.geojson
    {SEASON}_{CROP}_field_veg_index_stats.csv
    {CROP}_blocks.csv
    {CROP}_wwf_map.geojson          (optional)
    {CROP}_ndvi_z_scores_norm.csv   (optional)
    {CROP}_whatsapp_markers.geojson (optional)
"""
from __future__ import annotations

import os
import tempfile

import requests

REQUIRED_TEMPLATES = [
    "{CROP}_drawn_named.geojson",
    "{SEASON}_{CROP}_field_veg_index_stats.csv",
    "{CROP}_blocks.csv",
]

OPTIONAL_TEMPLATES = [
    "{CROP}_wwf_map.geojson",
    "{CROP}_ndvi_z_scores_norm.csv",
    "{CROP}_whatsapp_markers.geojson",
]


def _expand(template: str, crop: str, season: str) -> str:
    return template.replace("{CROP}", crop).replace("{SEASON}", season)


def resolve_assets(crop: str, season: str, backend_url: str) -> str:
    """
    Download all asset files from the Express backend's /api/assets/:filename
    endpoint into a temp directory.  Returns the temp directory path.

    Raises FileNotFoundError if a required file cannot be fetched.
    """
    tmp_dir = os.path.join(tempfile.gettempdir(), f"kisaan_{crop}")
    os.makedirs(tmp_dir, exist_ok=True)

    base_url = backend_url.rstrip("/")

    for template in REQUIRED_TEMPLATES:
        filename = _expand(template, crop, season)
        dest = os.path.join(tmp_dir, filename)
        resp = requests.get(f"{base_url}/api/assets/{filename}", timeout=60)
        if resp.status_code != 200:
            raise FileNotFoundError(
                f"Required asset not available from backend: {filename} "
                f"(HTTP {resp.status_code})"
            )
        with open(dest, "wb") as f:
            f.write(resp.content)

    for template in OPTIONAL_TEMPLATES:
        filename = _expand(template, crop, season)
        dest = os.path.join(tmp_dir, filename)
        try:
            resp = requests.get(f"{base_url}/api/assets/{filename}", timeout=60)
            if resp.status_code == 200:
                with open(dest, "wb") as f:
                    f.write(resp.content)
        except Exception:
            pass  # optional — skip silently

    return tmp_dir
