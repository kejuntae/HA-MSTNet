"""Fit heat and electrical power as functions of modulation and forecast outdoor temperature."""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

from linear_device_model import INPUT_FEATURE, TARGETS, load_frame, metrics


ROOT = Path(__file__).resolve().parent
FEATURES = [INPUT_FEATURE, 'outdoor_temperature_forecast_c']


def sha256(path):
    """Hash a file to track the exact inputs and code used."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attach_forecasts(frames, manifest_path):
    """Join forecasts by episode, issue time, and valid time without using measured temperatures."""
    manifest = json.loads(manifest_path.read_text())
    forecasts = []
    sources = {}
    for episode in manifest['episodes']:
        path = ROOT / episode['forecast_file']
        sources[episode['episode_id']] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                issued = int(record['issue_time_s'])
                data = record['data']
                if data['time'][0] != issued:
                    raise ValueError('The first forecast valid time does not match the issue time.')
                forecasts.append({'episode_id': episode['episode_id'], 'time_start_s': issued,
                                  'forecast_issue_time_s': issued,
                                  'forecast_valid_time_s': int(data['time'][0]),
                                  'outdoor_temperature_forecast_c': float(data['TDryBul'][0]) - 273.15})
    lookup = pd.DataFrame(forecasts)
    for split, frame in frames.items():
        joined = frame.merge(lookup, on=['episode_id', 'time_start_s'],
                             how='left', validate='one_to_one', sort=False)
        if len(joined) != len(frame) or not np.isfinite(joined[FEATURES].to_numpy()).all():
            raise ValueError(f'{split} has incomplete forecast matches.')
        if not joined.forecast_issue_time_s.eq(joined.time_start_s).all():
            raise ValueError('Forecast issue times are misaligned.')
        frames[split] = joined
    return sources


def build_models(seed):
    """Define candidate models whose scalers are fitted on the training split."""
    models = {}
    for degree in [2, 3]:
        models[f'polynomial_degree_{degree}'] = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            StandardScaler(), LinearRegression())
    models['mlp_hidden32'] = TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(32,), activation='tanh',
                         solver='lbfgs', alpha=0.001, max_iter=3000,
                         max_fun=100000, tol=1e-7, random_state=seed)),
        transformer=StandardScaler())
    return models


def main():
    """Fit heat and electrical power models and report metrics for both evaluation scopes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'datasets/complete_15min_2019')
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'datasets/complete_15min_2019/export_manifest.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/nonlinear_device_model')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    frames = {split: load_frame(args.data_dir / f'{split}.csv')
              for split in ['train', 'validation', 'test']}
    sources = attach_forecasts(frames, args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = frames['train']
    rows, fitted, descriptions = [], {}, {}
    predictions = {split: frame[['datetime_start', 'datetime_end', 'episode_id',
                                 'episode_row', 'mode', 'data_role', 'target_origin_allowed',
                                 *FEATURES, 'forecast_issue_time_s', 'forecast_valid_time_s',
                                 *[v[0] for v in TARGETS.values()]]].copy()
                   for split, frame in frames.items()}
    for target, (column, _, _) in TARGETS.items():
        candidates = build_models(args.seed)
        candidates['linear_u_baseline'] = LinearRegression()
        for name, model in candidates.items():
            columns = [INPUT_FEATURE] if name == 'linear_u_baseline' else FEATURES
            print(f'Fitting {target}/{name} ...', flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with threadpool_limits(limits=1):
                    model.fit(train[columns], train[column])
            key = f'{target}__{name}'
            fitted[key] = model
            info = {'features': columns, 'target': column,
                    'warnings': [str(item.message) for item in caught]}
            if name.startswith('polynomial'):
                poly, scale, regression = model.steps[0][1], model.steps[1][1], model.steps[2][1]
                coefficients = regression.coef_ / scale.scale_
                intercept = regression.intercept_ - np.dot(coefficients, scale.mean_)
                info['polynomial_original_units'] = {
                    'intercept': float(intercept),
                    'terms': dict(zip(poly.get_feature_names_out(columns), coefficients.tolist()))}
            elif name == 'mlp_hidden32':
                info['architecture'] = '2 -> 32 tanh -> 1 linear; independent model per target'
                info['iterations'] = int(model.regressor_.steps[-1][1].n_iter_)
                info['training_loss_scaled'] = float(model.regressor_.steps[-1][1].loss_)
            descriptions[key] = info
            for split, frame in frames.items():
                with threadpool_limits(limits=1):
                    predicted = model.predict(frame[columns])
                if not np.isfinite(predicted).all():
                    raise ValueError(f'{key}/{split} produced nonfinite predictions.')
                predictions[split][key + '_pred_w'] = predicted
                actual = frame[column].to_numpy()
                for scope, mask in [('all_rows', np.ones(len(frame), dtype=bool)),
                                    ('target_dates_only', frame.target_origin_allowed.to_numpy(dtype=bool))]:
                    rows.append({'model': name, 'target': target, 'split': split,
                                 'scope': scope, **metrics(actual[mask], predicted[mask])})
    joblib.dump(fitted, args.output_dir / 'models.joblib')
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / 'metrics.csv', index=False)
    for split, frame in predictions.items():
        frame.to_csv(args.output_dir / f'{split}_predictions.csv', index=False)
    metadata = {'feature_mapping': {'u': INPUT_FEATURE, 'T_out': FEATURES[1],
                                   **{k: v[0] for k, v in TARGETS.items()}},
                'forecast_definition': 'TDryBul[0] - 273.15; issue_time = valid_time = interval start; lead 0 s',
                'forecast_sources': sources, 'models': descriptions, 'seed': args.seed,
                'training_rows': len(train), 'training_scope': 'all train.csv rows including support days',
                'sklearn_version': sklearn.__version__, 'prediction_clipping': False,
                'script_sha256': sha256(Path(__file__)),
                'input_sha256': {split: sha256(args.data_dir / f'{split}.csv') for split in frames}}
    (args.output_dir / 'model_details.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    doc = ['# Two-input device models: polynomials and single-hidden-layer MLP', '',
           'Q_HVAC = f_Q(u, T_out) and P_HVAC = f_P(u, T_out). Fit a separate model for each target.', '',
           '| Symbol | Feature | Units and definition |', '| --- | --- | --- |',
           '| u | compressor_mean_readback | Dimensionless; mean measured compressor modulation over 15 minutes. |',
           '| T_out | outdoor_temperature_forecast_c | °C; data.TDryBul[0] from forecasts.jsonl minus 273.15, not the observed weather column. |',
           '| Q_HVAC | q_floor_mean_w | W; mean floor heat output over 15 minutes. |',
           '| P_HVAC | p_electric_total_mean_w | W; mean total heat pump, pump, and fan electrical power over 15 minutes. |', '',
           '## Forecast alignment', '',
           'Join forecasts by episode_id and time_start_s and take the temperature at the first valid time.'
           'forecast_issue_time_s = forecast_valid_time_s = time_start_s, giving zero lead time.'
           'The prediction CSVs contain these fields. Source data are unchanged. Forecasts may equal observations but come from the forecast API.',
           'This evaluates static conditional predictions for each interval, not fixed-origin 6/12/24-hour forecasts or forecast error by lead time.'
           'For MPC, use values from the forecast issued at the decision time, not forecasts issued later.'
           'Modulation is the measured interval average; native and fallback actions are not known plans.', '',
           '## Models and fitting', '',
           '- polynomial_degree_2: full quadratic polynomial with terms 1, u, T, u², uT, and T².',
           '- polynomial_degree_3: adds u³, u²T, uT², and T³, for ten terms including the intercept.',
           '- mlp_hidden32: two inputs, one 32-unit tanh layer, and one linear output.'
           'Scale inputs and targets on training data and invert outputs to watts. Use L-BFGS with alpha=0.001, max_iter=3000, max_fun=100000, and tol=1e-7.',
           '- linear_u_baseline: refit the original single-input linear model on the same training rows.',
           f'- Fit models and scalers on all {len(train)} training rows only. Fixed seed: {args.seed}.'
           'Model settings are fixed before testing. The MLP results are from one seed, not a multi-seed summary.',
           '- Keep all modes and support days. Do not smooth or clip predictions or force the intercept to zero.', '',
           '## Metrics', '',
           'all_rows includes the whole CSV; target_dates_only uses eligible forecast origins only. Both use the same fitted model.'
           'MAE and RMSE are in watts; R² is dimensionless. Training metrics are a fit check only.', '',
           '| Model | Output | Split | Scope | Samples | R² | MAE (W) | RMSE (W) |',
           '| --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        doc.append(f"| {r['model']} | {r['target']} | {r['split']} | {r['scope']} | {r['n_samples']} | "
                   f"{r['r2']:.6f} | {r['mae_w']:.3f} | {r['rmse_w']:.3f} |")
    doc += ['', '## Fitting warnings', '']
    emitted = [f"- {key}: {warning}" for key, item in descriptions.items() for warning in item['warnings']]
    doc += emitted or ['No training warnings.']
    doc += ['', '## Outputs and reproduction', '',
            '- metrics.csv: metrics for each model, split, and scope.',
            '- model_details.json: feature sources, polynomial coefficients in original units, MLP architecture, warnings, and file hashes.',
            '- models.joblib: fitted models including preprocessing and inverse target transforms. See model_details.json for feature order.',
            '- train_predictions.csv, validation_predictions.csv, and test_predictions.csv: observed targets, forecast inputs, and predictions.',
            '', 'Run from the case directory:', '', '```bash',
            'conda run -n mpc_env python nonlinear_device_model.py', '```', '',
            'Rerunning replaces outputs of the same name without modifying source data or the linear model reports.', '']
    (args.output_dir / 'README.md').write_text('\n'.join(doc))
    print(results[(results.split != 'train') & results.scope.eq('all_rows')].to_string(index=False))
    print(f'Results directory: {args.output_dir}')


if __name__ == '__main__':
    main()
