# Minimal BOPTEST MPC example

This example compares native BOPTEST control with HA-MSTNet MPC on the
`bestest_hydronic_heat_pump` test case. It does not require archived experiment
outputs or ship pretrained weights. The default one-day run is a smoke test,
not a reproduction of the paper's multi-seed results.

## Prepare the inputs

1. Start a local BOPTEST service. See the
   [data generation guide](../datasets/Boptest_data_gen/README.md) for setup.
2. Install the dependencies from the repository root:

   ```bash
   python -m pip install -r requirements.txt -r mpc/requirements.txt
   ```

3. Generate the BOPTEST data and fit the heat-pump device model using the
   [data generation guide](../datasets/Boptest_data_gen/README.md).
   The default fitting script writes
   `datasets/Boptest_data_gen/reports/nonlinear_device_model/models.joblib`.
   Use only a trusted local joblib file.
4. Train HA-MSTNet and note the generated `checkpoint.pt` path:

   ```bash
   python main.py train --config configs/experiments/boptest_single_zone.json --model HA-MSTNet --history-steps 48 --horizon-steps 24 --seeds 0 --device cpu --save-weights --no-save-test-plots --plot-every 0
   ```

## Run the comparison

From the repository root, replace `PATH_TO_CHECKPOINT` with the generated path:

```bash
python -m mpc.run_closed_loop --checkpoint PATH_TO_CHECKPOINT --device-model datasets/Boptest_data_gen/reports/nonlinear_device_model/models.joblib
```

Use `--url` to change the service address, `--scenario peak_heat_day` to
change the evaluation period, or `--days 14` for a longer run. Both policies
start from independent BOPTEST instances with identical native warm-up.
The default config is [boptest.json](configs/boptest.json). For a different
forecast horizon, set `horizon_steps` and supply a checkpoint trained with
that horizon and the same 48-step history.

IPOPT minimizes electricity cost, soft comfort penalties, and action
changes. Each 15-minute decision uses measured history, the currently issued
forecasts, and predicted equipment power for the candidate actions. The
controller applies only the first action and releases the override if no
valid plan is accepted.

The run writes a separate measured trajectory, dense samples, and start/end
KPIs for each policy, plus `summary.csv`, under
`results/boptest_single_zone/mpc_demo_*/` by default. Use `--output` to specify
a new destination. The summary reports daily cost, cost savings versus native
control, cumulative thermal discomfort from the official BOPTEST KPI
difference, load shift at the highest observed tariff, and daily HVAC
switches counted from dense device power measurements (>1 W means on).
A saving or load-shift percentage is blank if its baseline is zero; a
constant-price run has no high-price band.
