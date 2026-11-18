"""Evaluate matched native and MPC runs using the five paper metrics."""

import numpy as np
import pandas as pd

from mpc.environment import DEVICE_POWER_POINTS, STEP_SECONDS


def summarize(run, peak):
    trajectory, dense, before, after = run
    days = len(trajectory) * STEP_SECONDS / 86400
    cost = float(np.dot(trajectory["energy_kwh"], trajectory["price"]))
    discomfort = float(after["tdis_tot"]) - float(before["tdis_tot"])
    if not np.isfinite([cost, discomfort]).all():
        raise ValueError("Cost or the official discomfort KPI is nonfinite.")
    states = dense[DEVICE_POWER_POINTS].to_numpy(dtype=float) > 1.0
    switches = np.count_nonzero(states[1:] != states[:-1])
    return {
        "cost_eur_per_day": cost / days,
        "thermal_discomfort_kh": discomfort,
        "hvac_switches_per_day": switches / days,
    }, float(trajectory.loc[peak, "energy_kwh"].sum())


def reduction_pct(baseline, actual):
    """Return a reduction percentage, or NaN for an undefined denominator."""
    return 100 * (baseline - actual) / baseline if baseline > 0 else np.nan


def evaluate(native, controlled):
    """Compare costs and high-price energy only over identical evaluation intervals."""
    reference, comparison = native[0], controlled[0]
    columns = ["time_start", "time_end", "price"]
    if reference.empty or not np.array_equal(reference[columns], comparison[columns]):
        raise ValueError("Native and MPC trajectories must share all timestamps and prices.")
    prices = reference["price"].to_numpy()
    peak = prices == prices.max() if np.unique(prices).size > 1 else np.zeros(len(prices), dtype=bool)
    base, base_peak = summarize(native, peak)
    actual, actual_peak = summarize(controlled, peak)
    return pd.DataFrame([
        {"method": "native", **base, "cost_savings_pct": reduction_pct(base["cost_eur_per_day"], base["cost_eur_per_day"]),
         "load_shift_pct": reduction_pct(base_peak, base_peak)},
        {"method": "ha_mstnet_mpc", **actual,
         "cost_savings_pct": reduction_pct(base["cost_eur_per_day"], actual["cost_eur_per_day"]),
         "load_shift_pct": reduction_pct(base_peak, actual_peak)},
    ])[["method", "cost_eur_per_day", "cost_savings_pct", "thermal_discomfort_kh",
        "load_shift_pct", "hvac_switches_per_day"]]
