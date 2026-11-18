# BOPTEST single-zone data

This repository includes the preprocessed `train.csv`, `validation.csv`, and
`test.csv` for the BESTEST Hydronic Heat Pump case.

Use `configs/experiments/boptest_single_zone.json` with the dataset settings in
`configs/datasets/boptest_single_zone_15min.json`. To generate these CSVs from simulation,
see the [BOPTEST data generation guide](../../Boptest_data_gen/README.md).

## Splits and windows

| Split | CSV rows | Independent episodes | Windows (H=48, P=24, stride=1) |
| --- | ---: | ---: | ---: |
| Train | 9,600 | 4 | 8,832 |
| Validation | 1,728 | 2 | 1,344 |
| Test | 1,728 | 2 | 1,344 |

The configuration sets `split_dataset: false` and reads the three files from
`split_csv_paths`. Do not merge and re-split them: their dates do not form a
single chronological 70/10/20 split. Single-file NEST configurations use
`csv_path` and `split_ratios`.

A window must stay within one `episode_id`, and its last history row must have
`target_origin_allowed=True`. Keep the leading and trailing support rows for
context and labels. The stride is anchored to each episode's original row
positions; filtering invalid origins does not shift it. Longer horizons may
leave some allowed origins without sufficient support.

Fit scaling bounds on training features only, then reuse them for validation
and test. Episode IDs and origin flags are metadata, not model inputs. Future
measured room temperature is used only as the target; forecast inputs replace
it with the last observed room temperature.

## Features and time alignment

| Model feature | CSV column | Meaning |
| --- | --- | --- |
| `room_temperature` | `operative_temperature_c` | Operative temperature (°C) |
| `outdoor_temperature` | `outdoor_temperature_c` | Outdoor temperature (°C) |
| `solar_irradiance` | `solar_ghi_w_m2` | Global horizontal irradiance (W/m²) |
| `hvac_power` | `q_floor_mean_w` | Mean floor heat output over the current 15-minute interval (W) |
| `occupancy` | `occupancy_people` | Scheduled headcount (0 or 5), converted to binary occupancy |
| Time sine and cosine | `Time interval sin`, `Time interval cos` | Time of day on the fixed 2019 model calendar |

The heat input at time `t` drives the transition from `T(t)` to `T(t+15 min)`.
Do not shift it by another row or replace it with electrical power. For the
first training origin (2019-02-09 00:00), the 48 history points start on
2019-02-08 at 12:15 and the 24 targets run from 00:15 to 06:00 on 2019-02-09.

The window builder uses subsequent `operative_temperature_c` rows as labels.
Preprocessing checks that each label matches the preceding row's
`next_operative_temperature_c`. The builder does not append an extra terminal
row; the default H48/P24 setup already has enough support rows for all allowed
origins.

The offline benchmark uses observed future weather, schedules, and heat input.
The [online MPC example](../../../mpc/README.md) uses forecasts available at
decision time and a device model to estimate heat output from candidate actions.

## Run from the project root

```bash
python main.py data --config configs/experiments/boptest_single_zone.json
python main.py data --config configs/experiments/boptest_single_zone.json --model pcnn
python main.py train --config configs/experiments/boptest_single_zone.json \
  --model pcnn --history-steps 48 --horizon-steps 24 \
  --window-stride 1 --epochs 100 --patience 10 --batch-size 32 \
  --seeds 0 --device cuda --plot-every 0 \
  --output-root results/boptest_single_zone
```

Use `--device cpu` if CUDA is unavailable. Without `--model`, `data` checks all
configured models, `train` uses the first, and `compare` runs all of them. Run
`python main.py list-models` to see the current list. PCNN derives signed
heating and cooling from net power in watts; no valve, neighboring temperature,
or HVAC mode columns are required.

The output includes weights when enabled, training history, and
`predictions.npz`. The comparison summary records the split sources and actual
window counts.
