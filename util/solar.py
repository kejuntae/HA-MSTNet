"""Project horizontal irradiance onto a vertical UMAR window."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from astral import Observer
from astral.sun import azimuth as solar_azimuth
from astral.sun import elevation as solar_elevation


# Window azimuth in degrees from the reference PCNN UMAR setup.
UMAR_WINDOW_AZIMUTH_DEGREES = 62.5
UMAR_LATITUDE = 47.403041
UMAR_LONGITUDE = 8.612754
UMAR_TIMEZONE = "Europe/Zurich"


@lru_cache(maxsize=262_144)
def _solar_angles(
    timestamp_nanoseconds: int,
    timezone_name: str,
    latitude: float,
    longitude: float,
) -> tuple[float, float]:
    """Compute and cache solar elevation and azimuth for a UTC timestamp."""
    # Interpret naive timestamps as UTC before converting to Zurich time.
    local_time = pd.Timestamp(
        timestamp_nanoseconds,
        unit="ns",
        tz="UTC",
    ).tz_convert(timezone_name)
    observer = Observer(latitude=latitude, longitude=longitude)
    return (
        float(solar_elevation(observer, local_time.to_pydatetime())),
        float(solar_azimuth(observer, local_time.to_pydatetime())),
    )


def project_horizontal_irradiance(
    horizontal_irradiance: np.ndarray,
    solar_elevation_degrees: np.ndarray,
    solar_azimuth_degrees: np.ndarray,
    window_azimuth_degrees: float = UMAR_WINDOW_AZIMUTH_DEGREES,
) -> np.ndarray:
    """Apply the reference PCNN projection to a vertical window."""
    horizontal = np.asarray(horizontal_irradiance, dtype=float)
    elevation = np.asarray(solar_elevation_degrees, dtype=float).copy()
    azimuth = np.asarray(solar_azimuth_degrees, dtype=float)

    # Clamp low solar elevations to 5 degrees to avoid division by zero.
    elevation[elevation < 5.0] = 5.0
    elevation[elevation > 90.0] = 0.0
    relative_azimuth = np.clip(
        azimuth - float(window_azimuth_degrees),
        0.0,
        180.0,
    )
    elevation_radians = np.deg2rad(elevation)
    projection = (
        np.sin(np.deg2rad(relative_azimuth))
        * np.cos(elevation_radians)
        / np.sin(elevation_radians)
    )
    return horizontal * projection


def horizontal_to_window_irradiance(
    horizontal_irradiance: np.ndarray,
    timestamps: np.ndarray,
    timezone_name: str = UMAR_TIMEZONE,
    longitude: float = UMAR_LONGITUDE,
    latitude: float = UMAR_LATITUDE,
    window_azimuth_degrees: float = UMAR_WINDOW_AZIMUTH_DEGREES,
) -> np.ndarray:
    """Use solar angles at each timestamp to compute window irradiance."""
    horizontal = np.asarray(horizontal_irradiance, dtype=float)
    times = np.asarray(timestamps).astype("datetime64[ns]")
    if horizontal.shape != times.shape:
        raise ValueError('Horizontal irradiance and timestamps must have the same shape.')
    if not np.isfinite(horizontal).all() or np.isnat(times).any():
        raise ValueError('Horizontal irradiance and timestamps must all be valid.')

    # Deduplicate timestamps and cache solar angles for overlapping windows.
    time_numbers = times.astype(np.int64)
    unique_times, inverse = np.unique(time_numbers, return_inverse=True)
    unique_angles = np.array(
        [
            _solar_angles(
                int(timestamp),
                timezone_name,
                float(latitude),
                float(longitude),
            )
            for timestamp in unique_times
        ],
        dtype=float,
    )
    elevation = unique_angles[inverse, 0].reshape(horizontal.shape)
    azimuth = unique_angles[inverse, 1].reshape(horizontal.shape)
    return project_horizontal_irradiance(
        horizontal,
        elevation,
        azimuth,
        window_azimuth_degrees,
    )


def window_irradiance_from_spans(
    horizontal_irradiance: np.ndarray,
    spans: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
    interval_minutes: int,
) -> np.ndarray:
    """Reconstruct step timestamps from window spans and project irradiance."""
    horizontal = np.asarray(horizontal_irradiance, dtype=float)
    if horizontal.ndim != 2 or horizontal.shape[0] != len(spans):
        raise ValueError('RC irradiance must be a two-dimensional array matching the window spans.')
    if not isinstance(interval_minutes, int) or interval_minutes < 1:
        raise ValueError('The RC irradiance sampling interval must be a positive integer in minutes.')

    step_count = horizontal.shape[1]
    offsets = np.arange(step_count) * np.timedelta64(interval_minutes, "m")
    timestamps = np.empty(horizontal.shape, dtype="datetime64[ns]")
    for sample, (start, end) in enumerate(spans):
        start_time = pd.Timestamp(start).to_datetime64().astype("datetime64[ns]")
        generated = start_time + offsets
        expected_end = pd.Timestamp(end).to_datetime64().astype("datetime64[ns]")
        if generated[-1] != expected_end:
            raise ValueError('RC window spans do not match the sampling interval.')
        timestamps[sample] = generated
    return horizontal_to_window_irradiance(horizontal, timestamps)
