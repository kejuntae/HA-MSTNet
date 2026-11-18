# BOPTEST single-zone data generation

This directory ships generation scripts, not simulation logs, CSV files,
reports, or fitted models. The data are generated with the
[BESTEST Hydronic Heat Pump test case](https://github.com/ibpsa/project1-boptest/tree/master/testcases/bestest_hydronic_heat_pump).

## Generate the datasets

1. Start a local BOPTEST service following the
   [official deployment instructions](https://github.com/ibpsa/project1-boptest#quick-start-to-deploy-and-use-boptest-on-a-local-computer).
   The commands below assume it is available at `http://127.0.0.1:8000`.
   The reference runs reported BOPTEST `1.0.0-dev`; outputs may vary across versions.
2. Install the generator dependencies and collect the full simulation batch.
   Run these commands from the HA-MSTNet repository root:

   ```bash
   python -m pip install numpy pandas requests
   python datasets/Boptest_data_gen/collect_boptest.py plan
   python datasets/Boptest_data_gen/collect_boptest.py collect --url http://127.0.0.1:8000
   ```

   The collector runs eight independent date blocks (four training, two
   validation, two test). It alternates native control and seeded compressor
   excitation, logs the simulation at five-minute steps, and integrates dense
   power samples into 15-minute records. It also processes each run after
   collection. Raw logs go to `raw/` and per-episode tables to `processed/`.

3. Replace `BATCH_JSON` below with the path to the completed
   `datasets/Boptest_data_gen/reports/batch_*.json` and export:

   ```bash
   python datasets/Boptest_data_gen/export_splits.py --batch BATCH_JSON
   python datasets/Boptest_data_gen/prepare_thermal_data.py
   ```

   The first command creates complete split CSVs and a forecast manifest in
   `datasets/Boptest_data_gen/datasets/complete_15min_2019/`. The second writes
   `train.csv`, `validation.csv`, and `test.csv` to
   `datasets/data_preprocessed/Boptest_single_zone/`, overwriting existing
   files with those names. Use `prepare_thermal_data.py --output-dir PATH`
   to write elsewhere.

The splits contain 9,600 / 1,728 / 1,728 rows, respectively. Independent
episodes and the history and forecast support rows are retained. Construct
windows within each episode and use only `target_origin_allowed=True` rows as
forecast origins. See the [training data format](../data_preprocessed/Boptest_single_zone/README.md).

## Optional: fit a heat pump model for MPC

After exporting the complete splits, install the fitting dependencies and run:

```bash
python -m pip install scikit-learn joblib threadpoolctl
python datasets/Boptest_data_gen/linear_device_model.py
python datasets/Boptest_data_gen/nonlinear_device_model.py
```

The nonlinear fit uses the forecast logs and export manifest to match
outdoor temperature at issue time. The scripts write local fitted models and
reports under `reports/`. Published results may depend on the exact original
simulator and software versions; newly fitted models are not guaranteed to
be byte-identical to historical models.
