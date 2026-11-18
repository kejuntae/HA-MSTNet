# Experiment configuration

- [`datasets/`](datasets/) defines CSV paths, feature mappings, time alignment, and data splits for BOPTEST and NEST2024 rooms 272, 273, and 274.
- [`experiments/`](experiments/) provides one runnable comparison config per dataset or room. Each config lists all eight models and their explicit `model_params`, along with window sizes, seeds, and training settings. BOPTEST is the CLI default.

The `training` block applies to neural models; 3R2C uses its own system identification parameters in `model_params`. Use `--horizon-steps` to select 4, 24, 48, or 96 steps. Use `--device cpu` if CUDA is unavailable.

```bash
python main.py train --model HA-MSTNet --horizon-steps 24
python main.py compare --config configs/experiments/umar2024_room272_all.json --horizon-steps 48
```
