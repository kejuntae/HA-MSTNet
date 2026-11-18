"""Shared seven-feature input contract for all thermal models."""

THERMAL_FEATURES = (
    "room_temperature",
    "outdoor_temperature",
    "solar_irradiance",
    "hvac_power",
    "time_sin",
    "time_cos",
    "occupancy",
)
