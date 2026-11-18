"""Fit a linear device model on the training split and evaluate on validation and test."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_FEATURE = 'compressor_mean_readback'
TARGETS = {
    'Q_HVAC': ('q_floor_mean_w', 'a', 'b'),
    'P_HVAC': ('p_electric_total_mean_w', 'c', 'd'),
}


def metrics(actual, predicted):
    """Compute errors in watts; R² is undefined for constant targets."""
    residual = actual - predicted
    total = np.sum((actual - actual.mean()) ** 2)
    return {
        'n_samples': len(actual),
        'r2': float(1 - np.sum(residual ** 2) / total) if total > 0 else None,
        'mae_w': float(np.mean(np.abs(residual))),
        'rmse_w': float(np.sqrt(np.mean(residual ** 2))),
    }


def load_frame(path):
    """Validate required values and time steps without dropping invalid rows."""
    frame = pd.read_csv(path)
    columns = [INPUT_FEATURE] + [item[0] for item in TARGETS.values()]
    if len(frame) < 2 or not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError(f'{path} has too few samples or nonfinite model inputs.')
    if not frame[INPUT_FEATURE].between(-1e-8, 1 + 1e-8).all():
        raise ValueError(f'{path} has compressor modulation outside [0, 1].')
    if not (frame.time_end_s - frame.time_start_s).eq(900).all():
        raise ValueError(f'{path} contains intervals other than 15 minutes.')
    if not frame.target_origin_allowed.isin([True, False]).all():
        raise ValueError(f'{path} has invalid target-origin flags.')
    return frame


def main():
    """Fit ordinary least squares with an intercept and save predictions and metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=ROOT / 'datasets/complete_15min_2019')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'reports/linear_device_model')
    args = parser.parse_args()
    frames = {split: load_frame(args.data_dir / f'{split}.csv')
              for split in ['train', 'validation', 'test']}
    train = frames['train']
    design = np.column_stack([train[INPUT_FEATURE], np.ones(len(train))])
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError('Training modulation is constant; slope and intercept cannot both be identified.')
    coefficients = {}
    for symbol, (column, slope, intercept) in TARGETS.items():
        values, _, _, _ = np.linalg.lstsq(design, train[column].to_numpy(), rcond=None)
        coefficients[symbol] = {slope: float(values[0]), intercept: float(values[1])}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, frame in frames.items():
        prediction = frame[['datetime_start', 'datetime_end', 'episode_id',
                            'episode_row', 'mode', 'data_role',
                            'target_origin_allowed', INPUT_FEATURE]].copy()
        for symbol, (column, slope, intercept) in TARGETS.items():
            params = coefficients[symbol]
            predicted = params[slope] * frame[INPUT_FEATURE].to_numpy() + params[intercept]
            actual = frame[column].to_numpy()
            prediction[column] = actual
            prediction[f'{symbol}_pred_w'] = predicted
            prediction[f'{symbol}_residual_w'] = actual - predicted
            masks = {'all_rows': np.ones(len(frame), dtype=bool),
                     'target_dates_only': frame.target_origin_allowed.to_numpy(dtype=bool)}
            for scope, mask in masks.items():
                if mask.sum() < 2:
                    raise ValueError(f'{split}/{scope} has too few evaluation samples.')
                rows.append({'split': split, 'scope': scope, 'target': symbol,
                             **metrics(actual[mask], predicted[mask])})
        prediction.to_csv(args.output_dir / f'{split}_predictions.csv', index=False)
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / 'metrics.csv', index=False)
    metadata = {
        'method': 'ordinary_least_squares_with_intercept',
        'training_scope': 'all rows of train.csv, including support days',
        'training_samples': len(train),
        'feature_mapping': {'u': INPUT_FEATURE, **{
            symbol: values[0] for symbol, values in TARGETS.items()}},
        'units': {'u': 'dimensionless, 0-1', 'Q_HVAC': 'W', 'P_HVAC': 'W',
                  'a': 'W per unit modulation', 'b': 'W',
                  'c': 'W per unit modulation', 'd': 'W'},
        'coefficients': coefficients,
        'predictions_clipped': False,
        'input_sha256': {f'{split}.csv': hashlib.sha256(
            (args.data_dir / f'{split}.csv').read_bytes()).hexdigest() for split in frames},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / 'model.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    a, b = coefficients['Q_HVAC']['a'], coefficients['Q_HVAC']['b']
    c, d = coefficients['P_HVAC']['c'], coefficients['P_HVAC']['d']
    report = [
        '# Linear device model fitting and evaluation', '',
        f'Q_HVAC = {a:.10f} × u + ({b:.10f}) W',
        f'P_HVAC = {c:.10f} × u + ({d:.10f}) W', '',
        '| Symbol | CSV feature | Definition |', '| --- | --- | --- |',
        '| u | compressor_mean_readback | Mean measured compressor modulation over 15 minutes, dimensionless and in [0, 1]; not valve opening. |',
        '| Q_HVAC | q_floor_mean_w | Mean floor heat output over 15 minutes, from reaQFloHea_y; not condenser heat output. |',
        '| P_HVAC | p_electric_total_mean_w | Mean total HVAC electrical power over 15 minutes: heat pump, circulation pump, and fan. |', '',
        'Q_HVAC is floor-side heat input, which can be stored in the slab. It is not instantaneous net heat input to the room air.', '',
        f'Fit the coefficients once on all {len(train)} rows of train.csv, including history and forecast support days. '
        'Validation and test data are not used for fitting, tuning, or scaling. All operating modes are retained.',
        'Use unweighted ordinary least squares with an intercept, without scaling, forcing a zero intercept, or clipping predictions.'
        'Intercepts and some predictions may therefore be negative. This is a fit artifact, not a claim of negative electricity use.', '',
        '## Metrics', '',
        'all_rows includes the whole CSV; target_dates_only includes only rows with target_origin_allowed=True.'
        'Both scopes use the same fitted coefficients. Training metrics are a fit check only.', '',
        '| Split | Scope | Output | Samples | R² | MAE (W) | RMSE (W) |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        r2 = 'Undefined' if row['r2'] is None else f"{row['r2']:.6f}"
        report.append(f"| {row['split']} | {row['scope']} | {row['target']} | "
                      f"{row['n_samples']} | {r2} | {row['mae_w']:.6f} | {row['rmse_w']:.6f} |")
    report += [
        '', 'MAE = mean(abs(actual - predicted)); RMSE = sqrt(mean((actual - predicted)²)).'
        'R² = 1 - sum((actual - predicted)²) / sum((actual - mean(actual))²). It is dimensionless and may be negative.',
        '', '## Outputs and reproduction', '',
        '- model.json: full-precision coefficients, feature mapping, and input and script hashes.',
        '- metrics.csv: metrics for each split and both evaluation scopes.',
        '- train_predictions.csv / validation_predictions.csv / test_predictions.csv: '
        'per-row observations, predictions, and residuals (actual minus predicted), with timestamps and episode IDs.',
        '- Modulation here is the measured interval average, so this is a conditional fit evaluation.'
        'Future native and fallback actions are not known in advance. No closed-loop MPC was tested.',
        '', 'Run from the case directory:', '', '```bash',
        'conda run -n mpc_env python linear_device_model.py', '```', '',
        'Rerunning overwrites reports and predictions of the same name in the output directory, not the source CSVs.', '',
    ]
    (args.output_dir / 'README.md').write_text('\n'.join(report))
    print(json.dumps(coefficients, indent=2))
    print(results.to_string(index=False))
    print(f'Results directory: {args.output_dir}')


if __name__ == '__main__':
    main()
