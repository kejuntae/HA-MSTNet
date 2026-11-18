# HA-MSTNet

HA-MSTNet is a physics-guided model for multi-step indoor temperature forecasting.
It combines multiscale history encoding with a thermal branch that uses past
prediction errors to adapt its estimates of persistent disturbances and heat
exchange. This repository provides a common training and evaluation pipeline
for HA-MSTNet and seven baselines, plus an optional closed-loop BOPTEST example.

The eight model IDs are `HA-MSTNet`, `3r2c`, `pcnn`, `apcnn_mlp`, `lstm`,
`mlp`, `modnn`, and `pcicnn`. They use the same seven input features: indoor
temperature, outdoor temperature, solar irradiance, net HVAC thermal power,
time-of-day sine and cosine, and occupancy. Future measured indoor temperature
is a target, not a forecasting input.

## Install

From the repository root, use a Python 3.12 environment and install the core
dependencies:

```bash
python -m pip install -r requirements.txt
```

The training configs default to CUDA. Use `--device cpu` for CPU runs. Data
generation and MPC have additional requirements described below.

## Prepare the data


| Dataset             | Data and configuration                                                                                                                                                                                                                                                                                                                               |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BOPTEST single zone | Uses the BESTEST Hydronic Heat Pump case. The repository includes the pre-split 15-minute `train.csv`, `validation.csv`, and `test.csv` in [`datasets/data_preprocessed/Boptest_single_zone/`](datasets/data_preprocessed/Boptest_single_zone/README.md). To reproduce the inputs from simulation, follow the [BOPTEST generation guide](datasets/Boptest_data_gen/README.md). |
| NEST2024 / UMAR     | Download the four annual CSVs from the [official UMAR release](https://doi.org/10.6084/m9.figshare.24551992.v2) and follow the [NEST preprocessing guide](datasets/nest2024/README.md). It generates separate 15-minute tables for rooms 272, 273, and 274; the occupancy column is a schedule proxy, not measured occupancy.                       |

BOPTEST windows stay within their original episodes and train/validation/test
splits. NEST rooms are modeled separately and split chronologically (70/10/20).
Scaling is fitted on training data only.

## Train and evaluate

```bash
# List models and check the default BOPTEST dataset without training.
python main.py list-models
python main.py data

# Run a short CPU check: one epoch, one seed, and fewer windows.
python main.py train --model HA-MSTNet --epochs 1 --seeds 0 --window-stride 32 --device cpu --plot-every 0 --no-save-weights --no-save-test-plots

# Train HA-MSTNet on BOPTEST with the configured training budget.
python main.py train --model HA-MSTNet --horizon-steps 24 --device auto

# Compare all eight models on NEST room 272 at a 48-step horizon.
python main.py compare --config configs/experiments/umar2024_room272_all.json --horizon-steps 48 --device auto
```

Use `--config` to select a dataset or room and `--horizon-steps` to select
4, 24, 48, or 96 steps. The four [experiment configurations](configs/README.md)
list the model parameters, training settings, and seeds explicitly; the short
CPU check does not reproduce the benchmark.

Runs write metrics, configuration snapshots, and test predictions to
`results/boptest_single_zone/` or `results/NEST2024_three_zone/` by default.
Use `--output-root` to change the destination. Model weights and test forecast
plots are optional; disabling them does not disable result export.

## Optional: run MPC on BOPTEST

Start a BOPTEST service, fit the heat-pump device model, and train a BOPTEST
HA-MSTNet checkpoint with a 48-step history and a 24-step horizon. Follow
the [MPC guide](mpc/README.md) for the additional dependencies and setup.
The example compares native control and HA-MSTNet MPC over the same period.
Replace `PATH_TO_CHECKPOINT` with the path to your trained checkpoint:

```bash
python -m mpc.run_closed_loop --checkpoint PATH_TO_CHECKPOINT --device-model datasets/Boptest_data_gen/reports/nonlinear_device_model/models.joblib
```

Use only a trusted local device-model file. The one-day default is a smoke
test, not a reproduction of the full multi-seed control experiments.

## Repository layout

- [`configs/`](configs/) — data mappings and runnable experiment settings.
- [`datasets/`](datasets/) — data generation and preprocessing guides and scripts.
- [`models/`](models/) — the eight forecasting models and shared components.
- [`exp/`](exp/) and [`util/`](util/) — training, window construction, metrics, and result export.
- [`mpc/`](mpc/) — the optional closed-loop control example.

## License

The original HA-MSTNet code and documentation are released under the
[MIT License](LICENSE). This license does not extend to third-party code
or datasets, including the BOPTEST CSVs and the upstream NEST data.
Consult the applicable upstream terms for those materials.

## Acknowledgments

We thank the maintainers and contributors of the following public repositories
for sharing their work on building thermal modeling and model predictive control:

- [PCNN](https://github.com/Cemempamoi/pcnn)
- [Modularized Neural Network Incorporating Physical Priors for Future Building Energy Modeling](https://github.com/Bugs-Owner/Modularized-Neural-Network-Incorporating-Physical-Priors-for-Future-Building-Energy-Modeling)
- [Adaptive-PCNN](https://github.com/chendong-energy/Adaptive-PCNN)
- [DDMPC](https://github.com/RWTH-EBC/DDMPC)
- [PCNN-Comparison](https://github.com/XuezhengWang/PCNN-Comparison)
