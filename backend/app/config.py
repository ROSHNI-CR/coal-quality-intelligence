"""
Central configuration — resolves the database path reliably without
requiring environment variables to be set manually.

Priority:
  1. COAL_EIE_DB_PATH environment variable (explicit override)
  2. coal_eie_production_complete.db in the project root (default)
  3. coal_eie.db in the project root (fallback for older filenames)
"""
import os
from pathlib import Path

# backend/app/config.py -> backend/app -> backend -> project_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def get_db_path() -> str:
    # 1. Explicit env var wins
    env = os.environ.get("COAL_EIE_DB_PATH")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = PROJECT_ROOT / env
        if p.exists():
            return str(p)

    # 2. Default production filename
    prod = PROJECT_ROOT / "coal_eie_production_complete.db"
    if prod.exists():
        return str(prod)

    # 3. Legacy filename
    legacy = PROJECT_ROOT / "coal_eie.db"
    if legacy.exists():
        return str(legacy)

    # 4. Return the production path anyway (will give a clear SQLite error)
    return str(prod)

DB_PATH = get_db_path()
