"""
config.py — Configuration for the VI Dash app.

Values are read from environment variables. For local development, create a
.env file in the crop_health_scoring/ directory with the values below.
Defaults match the original local setup so existing dev environments work
without any changes.
"""
import os
from dotenv import load_dotenv

load_dotenv()

CROP      = os.environ.get("CROP",   "shahmeer")
SEASON    = os.environ.get("SEASON", "kharif")
ASSET_DIR = os.environ.get("ASSET_DIR", "/Users/daoud/PycharmAssets/shahmeer_farms")

_backend  = os.environ.get("BACKEND_URL", "http://localhost:3003")
API_URL   = f"{_backend}/api/field-assessment"
LOGS_URL  = f"{_backend}/api/field-assessments"
BACKEND_URL = _backend

# Set USE_REMOTE_ASSETS=true in production to download assets from the backend
# instead of reading from ASSET_DIR (which won't exist in a deployed container).
USE_REMOTE_ASSETS = os.environ.get("USE_REMOTE_ASSETS", "false").lower() == "true"
