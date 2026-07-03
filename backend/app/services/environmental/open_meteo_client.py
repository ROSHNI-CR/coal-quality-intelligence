"""
Module 2 — Environmental Variable Layer
Open-Meteo Archive API client (PRODUCTION data source).

This is the real production integration. It is intentionally isolated from
synthetic_weather.py so the two code paths can never be silently confused.

Open-Meteo Archive API docs: https://open-meteo.com/en/docs/historical-weather-api

Daily variables requested map onto the blueprint's required environmental
variables as follows:

  Blueprint variable      -> Open-Meteo daily parameter
  ----------------------     ---------------------------------------------
  Temperature              -> temperature_2m_max, temperature_2m_min, temperature_2m_mean
  Relative Humidity        -> relative_humidity_2m_mean/max/min
  Rainfall                 -> precipitation_sum
  Dew Point                -> dew_point_2m_mean (derived by Open-Meteo)
  Wind Speed               -> wind_speed_10m_max  (used as daily representative)
  Wind Gust                -> wind_gusts_10m_max
  Surface Pressure         -> surface_pressure_mean
  Cloud Cover              -> cloud_cover_mean
  Visibility               -> visibility_mean
  Solar Radiation          -> shortwave_radiation_sum
  Weather Code             -> weather_code

Note: Open-Meteo's free daily aggregation endpoint does not expose every one
of these as a "mean/max/min" triple for every variable; where Open-Meteo only
provides a single daily value, that value is mapped to *_mean_* and the
corresponding *_max_*/*_min_* columns are left NULL. NASA POWER is the
designated fallback specifically for solar radiation per the blueprint, and
is stubbed below for future extension.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_PARAMS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "precipitation_sum",
    "dew_point_2m_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "surface_pressure_mean",
    "cloud_cover_mean",
    "visibility_mean",
    "shortwave_radiation_sum",
    "weather_code",
]


@dataclass
class WeatherFetchResult:
    success: bool
    source: str = "open-meteo-archive"
    http_status: Optional[int] = None
    error_message: Optional[str] = None
    daily: Optional[dict] = None  # raw Open-Meteo 'daily' block, lists keyed by param


def fetch_archive_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timeout_seconds: int = 30,
    max_retries: int = 2,
) -> WeatherFetchResult:
    """
    Fetch daily archive weather for a single mine location and date range.

    start_date / end_date: 'YYYY-MM-DD' (inclusive)

    This function never raises on network/API failure — it always returns a
    WeatherFetchResult so the ingestion orchestrator can decide whether to
    fall back to synthetic data. It DOES raise on programmer errors (bad
    lat/lon types, etc.) since those indicate a bug, not an environment
    limitation.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_PARAMS),
        "timezone": "Asia/Kolkata",
    }

    last_error = None
    last_status = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout_seconds)
            last_status = resp.status_code
            if resp.status_code == 200:
                payload = resp.json()
                if "daily" not in payload:
                    return WeatherFetchResult(
                        success=False,
                        http_status=resp.status_code,
                        error_message=f"Malformed response, no 'daily' block: {payload}",
                    )
                return WeatherFetchResult(success=True, http_status=resp.status_code, daily=payload["daily"])
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except requests.exceptions.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))

    return WeatherFetchResult(success=False, http_status=last_status, error_message=last_error)


def is_api_reachable(timeout_seconds: int = 5) -> bool:
    """Lightweight reachability probe used by the ingestion orchestrator
    to decide, once per run, whether to even attempt real API calls
    (avoids burning retry budget per-mine when the sandbox has no egress).

    IMPORTANT: some network setups (e.g. an egress proxy that blocks a host)
    return a normal HTTP response (e.g. 403) for a blocked host rather than
    raising a connection exception. A probe that only catches exceptions
    would incorrectly report such a host as 'reachable'. This function
    therefore requires an actual HTTP 200 to report True."""
    try:
        resp = requests.get(OPEN_METEO_ARCHIVE_URL, params={"latitude": 0, "longitude": 0,
                                                       "start_date": "2025-01-01", "end_date": "2025-01-01",
                                                       "daily": "temperature_2m_max"},
                     timeout=timeout_seconds)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False
