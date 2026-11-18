"""Compare native BOPTEST control with HA-MSTNet MPC on one scenario."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mpc.controller import Controller, HISTORY_STEPS
from mpc.environment import Boptest, STEP_SECONDS, feature_row, integrate
from mpc.evaluate import evaluate
from util.artifacts import save_results


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("typical_heat_day", "peak_heat_day")
STEPS_PER_DAY = 86400 // STEP_SECONDS


def advance(client, current, action=None):
    measurements = client.advance(action)
    end = current["time"] + STEP_SECONDS
    if not math.isclose(measurements["time"], end, rel_tol=0, abs_tol=1e-6):
        raise ValueError("BOPTEST did not advance exactly 15 minutes.")
    dense = client.dense(current["time"], end)
    return measurements, dense, integrate(dense)


def run_case(config, controller, directory):
    """Run a fresh instance with native warm-up and record only the evaluation window."""
    directory.mkdir()
    client = Boptest(config["url"])
    try:
        scenario = client.call("PUT", "scenario", {
            "time_period": config["scenario"], "electricity_price": "dynamic",
        })
        start = int(scenario["time_period"]["time"])
        history_seconds = HISTORY_STEPS * STEP_SECONDS
        current = client.call("PUT", "initialize", {
            "start_time": start - history_seconds,
            "warmup_period": 7 * 86400 - history_seconds,
        })
        client.call("PUT", "step", {"step": STEP_SECONDS})
        horizon_seconds = config["horizon_steps"] * STEP_SECONDS
        history = []
        for _ in range(HISTORY_STEPS):
            forecast = client.forecast(horizon_seconds)
            next_state, _, interval = advance(client, current)
            history.append(feature_row(current, forecast, interval["q_mean_w"]))
            current = next_state
        if not math.isclose(current["time"], start, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Warm-up did not end at the scenario start.")
        before = client.call("GET", "kpi")
        previous = float(current["oveHeaPumY_u"])
        bias = 0.0
        rows, dense_parts = [], []
        for _ in range(round(config["days"] * STEPS_PER_DAY)):
            forecast = client.forecast(horizon_seconds)
            if not math.isclose(forecast["time"][0], current["time"], rel_tol=0, abs_tol=1e-6):
                raise ValueError("The forecast does not start at the current decision time.")
            action, solution = None, {"status": "native", "solve_seconds": 0.0}
            if controller is not None:
                inputs = controller.parameters(history, current, forecast, previous, bias)
                action, solution = controller.solve(inputs)
            next_state, dense, interval = advance(client, current, action)
            dense_parts.append(dense)
            temperature = float(next_state["reaTZon_y"] - 273.15)
            price = float(forecast["PriceElectricPowerDynamic"][0])
            if not np.isfinite([temperature, price]).all():
                raise ValueError("Measured temperature or electricity price is nonfinite.")
            row = {
                "time_start": current["time"], "time_end": next_state["time"],
                "temperature_c": temperature, "price": price,
                "comfort_lower_c": forecast["LowerSetp[1]"][1] - 273.15,
                "comfort_upper_c": forecast["UpperSetp[1]"][1] - 273.15,
                "u_command": action, "fallback": controller is not None and action is None,
                "status": solution["status"], "solve_seconds": solution["solve_seconds"],
                **interval,
            }
            if "prediction_c" in solution:
                predicted = solution["prediction_c"][0]
                row["prediction_c"] = predicted + bias
                bias = float(np.clip(
                    (1 - config["bias_alpha"]) * bias
                    + config["bias_alpha"] * (temperature - predicted),
                    -config["bias_limit_c"], config["bias_limit_c"],
                ))
            rows.append(row)
            history.append(feature_row(current, forecast, interval["q_mean_w"]))
            history = history[-HISTORY_STEPS:]
            previous, current = interval["u_mean"], next_state
        after = client.call("GET", "kpi")
    finally:
        client.close()
    trajectory = pd.DataFrame(rows)
    dense = pd.concat(dense_parts, ignore_index=True).drop_duplicates("time", keep="last")
    trajectory.to_csv(directory / "trajectory.csv", index=False)
    dense.to_csv(directory / "dense.csv.gz", index=False)
    save_results(directory / "kpi.json", {"before": before, "after": after})
    return trajectory, dense, before, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "mpc/configs/boptest.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device-model", type=Path, required=True, help="Trusted local joblib file.")
    parser.add_argument("--url")
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--days", type=float)
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("url", "scenario", "days"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config["scenario"] not in SCENARIOS:
        parser.error("Choose typical_heat_day or peak_heat_day.")
    steps = config["days"] * STEPS_PER_DAY
    if not math.isfinite(steps) or steps < 1 or not math.isclose(
            steps, round(steps), rel_tol=0, abs_tol=1e-8):
        parser.error("The duration must be a positive multiple of 15 minutes.")
    if type(config["horizon_steps"]) is not int or config["horizon_steps"] < 1:
        parser.error("horizon_steps must be a positive integer.")
    for key in ("checkpoint", "device_model"):
        if not getattr(args, key).is_file():
            parser.error(f"{key}: file not found: {getattr(args, key)}")
    output = args.output or ROOT / "results/boptest_single_zone" / (
        "mpc_demo_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    torch.set_num_threads(1)
    controller = Controller(config, args.checkpoint, args.device_model)
    output.mkdir(parents=True)
    save_results(output / "config.json", config)
    print("Running native BOPTEST control...", flush=True)
    native = run_case(config, None, output / "native")
    print("Running HA-MSTNet MPC...", flush=True)
    controlled = run_case(config, controller, output / "ha_mstnet_mpc")
    summary = evaluate(native, controlled)
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
