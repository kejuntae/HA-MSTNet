"""Collect and process the eight BOPTEST single-zone simulation episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import shutil
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DAY = 86400
CASE = 'bestest_hydronic_heat_pump'
ACTIVATE = ['oveHeaPumY_activate', 'oveTSet_activate',
            'ovePum_activate', 'oveFan_activate']
FORECASTS = ['TDryBul', 'HGloHor', 'HDirNor', 'HDifHor', 'Occupancy[1]',
             'InternalGainsCon[1]', 'InternalGainsRad[1]', 'LowerSetp[1]',
             'UpperSetp[1]', 'PriceElectricPowerDynamic']
MAPPING = {
    'operative_temperature_c': ('reaTZon_y', -273.15),
    'outdoor_temperature_c': ('weaSta_reaWeaTDryBul_y', -273.15),
    'supply_temperature_c': ('reaTSup_y', -273.15),
    'return_temperature_c': ('reaTRet_y', -273.15),
    'comfort_lower_c': ('reaTSetHea_y', -273.15),
    'comfort_upper_c': ('reaTSetCoo_y', -273.15),
    'solar_ghi_w_m2': ('weaSta_reaWeaHGloHor_y', 0),
    'q_floor_w': ('reaQFloHea_y', 0),
    'q_condenser_w': ('reaQHeaPumCon_y', 0),
    'p_heat_pump_w': ('reaPHeaPum_y', 0),
    'p_pump_w': ('reaPPumEmi_y', 0),
    'p_fan_w': ('reaPFan_y', 0),
    'compressor_readback': ('oveHeaPumY_u', 0),
}


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    """Write the manifest atomically so partial files are never mistaken for complete ones."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def append_json(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()


def default_config():
    blocks = [('train', 'winter', 31, 63, 0),
              ('train', 'spring', 123, 155, 6),
              ('train', 'autumn', 269, 301, 12),
              ('train', 'year_end', 333, 365, 18),
              ('validation', 'winter', 63, 79, 6),
              ('validation', 'autumn', 301, 317, 18),
              ('prediction_test', 'winter', 79, 95, 12),
              ('prediction_test', 'autumn', 317, 333, 0)]
    return {
        'case_id': CASE, 'base_url': 'http://127.0.0.1:8000',
        'raw_step_s': 300, 'action_step_s': 900,
        'forecast_horizon_s': DAY, 'forecast_interval_s': 900,
        'forecast_points': FORECASTS, 'warmup_days': 7,
        'normal_hours': 24, 'excitation_hours': 24,
        'levels': [0, .2, .4, .6, .8, 1.0],
        'level_weights': [15, 20, 25, 25, 10, 5],
        'hold_hours': [.5, 1, 2, 4], 'hold_weights': [20, 30, 30, 20],
        'max_delta_u': .2, 'envelope_margin_k': 1,
        'recovery_s': 3600, 'guard_lookahead_s': 86400,
        'boundary_probability': .2, 'normal_lower_allowance_k': .25,
        'normal_upper_allowance_k': .25, 'boundary_allowance_k': .75,
        'soft_violation_duration_s': 3600, 'trend_horizon_s': 7200,
        'absolute_temperature_limits_c': [18, 28],
        'excitation_dose_ratio': 1.25,
        'seed_by_split': {
            'train': 2001, 'validation': 3001, 'prediction_test': 4001},
        'protected_days': [[8, 31], [100, 123]],
        'episodes': [dict(id=f'{split}_{name}', split=split, index=i,
                          start_day=a, end_day=b, phase_offset_h=phase)
                     for i, (split, name, a, b, phase) in enumerate(blocks)],
        'calendar': 'NY2019, fixed model time, no DST conversion',
        'power_integration': 'right_rectangle_dense_30s_event_aligned',
        'source_version': 4,
    }


def validate_plan(cfg):
    """Check that episodes do not overlap each other or the protected periods."""
    assert cfg['raw_step_s'] == 300 and cfg['action_step_s'] == 900
    assert cfg['case_id'] == CASE
    intervals = sorted((e['start_day'], e['end_day'], e['id'])
                       for e in cfg['episodes'])
    for i, (a, b, _) in enumerate(intervals):
        assert 0 <= a < a + 9 < b <= 365
        if i:
            assert intervals[i - 1][1] <= a
        for c, d in cfg['protected_days']:
            assert b <= c or a >= d
    rows = []
    for e in cfg['episodes']:
        a, b = e['start_day'], e['end_day']
        rows.append({**e, 'observed_start_s': (a + 7) * DAY,
                     'target_start_s': (a + 8) * DAY,
                     'target_end_s': (b - 1) * DAY,
                     'end_s': b * DAY, 'target_days': b - a - 9,
                     'advance_requests': (b - a - 7) * 288})
    return rows


class Client:
    """Manage a BOPTEST REST instance without retrying failed advance requests."""

    def __init__(self, url, log=None):
        self.url = url.rstrip('/')
        self.session = requests.Session()
        self.testid = None
        self.log = log

    def call(self, method, endpoint, data=None, global_route=False):
        suffix = '' if global_route else '/' + self.testid
        url = self.url + '/' + endpoint + suffix
        readonly = method == 'GET' or endpoint in ('results', 'forecast')
        for attempt in range(3 if readonly else 1):
            started = time.monotonic()
            try:
                r = self.session.request(method, url, json=data,
                                         timeout=(10, 240))
                r.raise_for_status()
                body = r.json()
                if isinstance(body, dict) and body.get('status', 200) != 200:
                    raise RuntimeError(str(body))
                if self.log:
                    append_json(self.log, {'method': method, 'endpoint': endpoint,
                                          'request': data, 'status': r.status_code,
                                          'elapsed_s': time.monotonic() - started})
                return body.get('payload', body) if isinstance(body, dict) else body
            except (requests.RequestException, ValueError):
                if not readonly or attempt == 2:
                    raise
                time.sleep(1)

    def select(self):
        self.testid = self.call('POST', f'testcases/{CASE}/select',
                                global_route=True)['testid']

    def metadata(self):
        result = {k: self.call('GET', k) for k in
                  ['name', 'inputs', 'measurements', 'forecast_points',
                   'step', 'scenario']}
        result['version'] = self.call('GET', 'version', global_route=True)
        assert result['name']['name'] == CASE
        assert set(ACTIVATE + ['oveHeaPumY_u']) <= set(result['inputs'])
        assert set(FORECASTS) <= set(result['forecast_points'])
        for field, unit in [('reaTZon_y', 'K'), ('reaQFloHea_y', 'W'),
                            ('reaPHeaPum_y', 'W')]:
            assert result['measurements'][field]['Unit'] == unit
        return result

    def close(self):
        if self.testid:
            self.call('PUT', 'stop')
            self.testid = None
        self.session.close()


class Policy:
    """Generate rate-limited, multilevel steps on the fixed mixed-control schedule."""

    def __init__(self, cfg, episode, profile):
        self.cfg, self.episode, self.profile = cfg, episode, profile
        seed = cfg['seed_by_split'][episode['split']] * 100 + episode['index']
        self.level_rng, self.hold_rng = random.Random(seed), random.Random(seed + 100000)
        self.target = None
        self.last_target = None
        self.sent = None
        self.hold_start = None
        self.hold_s = None
        self.comfort_since = None
        self.active = False
        self.segment = 0
        self.regime_rng = random.Random(seed + 200000)
        self.regime = 'normal'
        self.temperature_history = []
        self.violation_since = None
        self.normal_dose_history = []
        self.previous_time = None
        self.previous_planned = None
        self.previous_active_u = None
        self.previous_readback = None
        self.dose_budget = None
        self.dose_used = 0.0
        self.dose_closed = False

    def planned(self, t):
        if self.profile == 'baseline':
            return 'normal'
        first = ((self.episode['start_day'] + 8) * DAY
                 + (24 + self.episode['phase_offset_h']) * 3600)
        if t < first:
            return 'normal'
        return 'excitation' if (t - first) % (2 * DAY) < DAY else 'normal'

    def sample(self):
        pairs = [(x, w) for x, w in zip(self.cfg['levels'], self.cfg['level_weights'])
                 if x != self.last_target and (self.profile != 'conservative' or x <= .6)]
        self.regime = ('boundary' if self.regime_rng.random() < self.cfg['boundary_probability']
                       else 'normal')
        self.target = self.level_rng.choices([x for x, _ in pairs],
                                              [w for _, w in pairs])[0]
        cap = 1 if self.target >= .8 else (2 if self.target == 0 else 4)
        if self.profile == 'conservative':
            cap = min(cap, 2)
        holds = [(h, w) for h, w in zip(self.cfg['hold_hours'], self.cfg['hold_weights'])
                 if h <= cap]
        self.hold_s = self.hold_rng.choices([h for h, _ in holds],
                                            [w for _, w in holds])[0] * 3600
        self.hold_start = None
        self.last_target = self.target
        self.segment += 1

    def step(self, t, y, forecast=None):
        lower, upper, temp = (y.get(k, float('nan')) for k in
                              ['reaTSetHea_y', 'reaTSetCoo_y', 'reaTZon_y'])
        valid = all(math.isfinite(v) for v in [lower, upper, temp])
        guard_lower, guard_upper = lower, upper
        if forecast is not None:
            times = np.asarray(forecast['time'])
            mask = (times >= t) & (times <= t + self.cfg['guard_lookahead_s'])
            guard_lower = float(np.max(np.asarray(forecast['LowerSetp[1]'])[mask]))
            guard_upper = float(np.min(np.asarray(forecast['UpperSetp[1]'])[mask]))
        if valid:
            self.temperature_history.append((t, temp))
        self.temperature_history = [(tt, v) for tt, v in self.temperature_history
                                    if tt >= t - 1800]
        slope = 0.0
        if len(self.temperature_history) > 1:
            tt, vv = self.temperature_history[0]
            slope = (temp - vv) / (t - tt)
        projected = temp + slope * self.cfg['trend_horizon_s']
        outside = valid and (temp < lower or temp > upper)
        if outside and self.violation_since is None:
            self.violation_since = t
        elif not outside:
            self.violation_since = None
        prolonged = (self.violation_since is not None and
                     t - self.violation_since >= self.cfg['soft_violation_duration_s'])
        allowance = (self.cfg['boundary_allowance_k'] if self.regime == 'boundary'
                     else self.cfg['normal_upper_allowance_k'])
        lower_allowance = (self.cfg['boundary_allowance_k'] if self.regime == 'boundary'
                           else self.cfg['normal_lower_allowance_k'])
        early_upper = (forecast is not None and valid and
                       max(temp, projected) >= guard_upper + allowance)
        early_lower = (forecast is not None and valid and
                       min(temp, projected) <= guard_lower - lower_allowance)
        # Require recovery near the comfort band; allow brief excursions in boundary mode.
        comfort = (valid and lower <= temp <= upper and
                   guard_lower <= temp <= guard_upper - .25)
        if comfort:
            if self.comfort_since is None:
                self.comfort_since = t
        else:
            self.comfort_since = None
        envelope = valid and lower - 1 <= temp <= upper + 1
        planned = self.planned(t)
        readback = y.get('oveHeaPumY_u', 0.0)
        if self.previous_time is not None:
            elapsed = t - self.previous_time
            if self.previous_planned == 'normal':
                avg = .5 * (self.previous_readback + readback)
                self.normal_dose_history.append((self.previous_time, elapsed, avg * elapsed))
            if self.previous_active_u is not None:
                self.dose_used += self.previous_active_u * elapsed
        self.normal_dose_history = [(tt, dt, dose) for tt, dt, dose in self.normal_dose_history
                                    if tt >= t - DAY]
        if planned == 'excitation' and self.previous_planned != 'excitation':
            observed_s = sum(dt for _, dt, _ in self.normal_dose_history)
            dose = sum(v for _, _, v in self.normal_dose_history)
            self.dose_budget = (dose / observed_s * DAY * self.cfg['excitation_dose_ratio']
                                if observed_s else 0.0)
            self.dose_used = 0.0
            self.dose_closed = False
        dose_exhausted = (planned == 'excitation' and self.dose_budget is not None
                          and (self.dose_closed or self.dose_used >= self.dose_budget - 1e-9))
        reason = ''
        if planned == 'normal':
            mode = 'normal'
        elif not envelope:
            mode, reason = 'fallback', 'invalid_temperature' if not valid else 'envelope'
        elif dose_exhausted:
            mode, reason = 'fallback', 'normal_operation_dose_budget'
        elif prolonged:
            mode, reason = 'fallback', 'violation_duration_60min'
        elif early_upper or early_lower:
            mode, reason = 'fallback', 'anticipatory_upper' if early_upper else 'anticipatory_lower'
        elif not self.active and (self.comfort_since is None or
                                 t - self.comfort_since < self.cfg['recovery_s']):
            mode, reason = 'fallback', 'waiting_comfort_60min'
        elif not self.active and t % 900:
            mode, reason = 'fallback', 'waiting_action_boundary'
        else:
            mode = 'excitation'
        payload = {k: 0 for k in ACTIVATE}
        handover_unknown = False
        if mode != 'excitation':
            self.active = False
            self.target = self.sent = self.hold_start = self.hold_s = None
        else:
            if not self.active:
                self.sent = y.get('oveHeaPumY_u')
                handover_unknown = self.sent is None or not math.isfinite(self.sent)
                if handover_unknown:
                    self.sent = None
                self.active = True
            if t % 900 == 0:
                if self.target is None or (self.hold_start is not None and
                                           t - self.hold_start >= self.hold_s):
                    self.sample()
                if self.sent is None:
                    self.sent = self.target
                else:
                    delta = np.clip(self.target - self.sent, -.2, .2)
                    self.sent = float(np.clip(self.sent + delta, 0, 1))
                if abs(self.sent - self.target) < 1e-9 and self.hold_start is None:
                    self.hold_start = t
            payload['oveHeaPumY_activate'] = 1
            payload['oveHeaPumY_u'] = self.sent
            # End the override if the budget runs out; do not alter the sampled plateau.
            if self.dose_budget is not None and self.dose_used + self.sent * 300 > self.dose_budget:
                mode, reason = 'fallback', 'normal_operation_dose_budget'
                self.dose_closed = True
                payload = {k: 0 for k in ACTIVATE}
                self.active = False
                self.target = self.sent = self.hold_start = self.hold_s = None
        self.previous_time = t
        self.previous_planned = planned
        self.previous_active_u = self.sent if mode == 'excitation' else None
        self.previous_readback = readback
        return payload, {
            'time_start_s': t, 'time_end_s': t + 300,
            'dose_budget_u_seconds': self.dose_budget, 'dose_used_u_seconds': self.dose_used,
            'planned_mode': planned, 'actual_mode': mode, 'fallback_reason': reason,
            'u_requested': self.target, 'u_sent': self.sent,
            'target_level': self.target, 'segment_id': self.segment if self.active else None,
            'planned_hold_s': self.hold_s, 'hold_start_s': self.hold_start,
            'ramping': self.active and self.hold_start is None,
            'remaining_hold_s': (max(0, self.hold_s - (t - self.hold_start))
                                 if self.hold_start is not None else self.hold_s),
            'handover_jump_unknown': handover_unknown, 'guard_lower_k': guard_lower,
            'guard_upper_k': guard_upper, 'excitation_regime': self.regime if self.active else None,
            'temperature_trend_k_per_hour': slope * 3600,
            'projected_temperature_k': projected,
            'comfort_violation_elapsed_s': t - self.violation_since if self.violation_since is not None else 0,
            'payload': payload,
        }


def save_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def collect(cfg, episode, profile='full', pilot_days=None):
    run_id = f"{stamp()}_{episode['id']}_{profile}"
    folder = ROOT / ('pilots' if pilot_days else 'raw') / run_id
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, folder / 'collector_snapshot.py')
    write_json(folder / 'config_snapshot.json', cfg)
    write_json(folder / 'variable_mapping.json', MAPPING)
    start = (episode['start_day'] + 7) * DAY
    end = (episode['end_day'] * DAY if pilot_days is None
           else start + int(pilot_days * DAY))
    assert end <= episode['end_day'] * DAY
    manifest = {'run_id': run_id, 'status': 'running', 'episode': episode,
                'profile': profile, 'pilot': pilot_days is not None,
                'start_s': start, 'end_s': end, 'code_sha256': digest(__file__),
                'config_sha256': digest(folder / 'config_snapshot.json'),
                'created_utc': stamp(), 'python': sys.executable}
    write_json(folder / 'run_manifest.json', manifest)
    begun = time.monotonic()
    files = {}
    client = None
    try:
        for name in ['requests', 'actions', 'forecasts', 'events']:
            files[name] = (folder / f'{name}.jsonl').open('w', encoding='utf-8')
        client = Client(cfg['base_url'], files['requests'])
        client.select()
        manifest['testid'] = client.testid
        write_json(folder / 'run_manifest.json', manifest)
        meta = client.metadata()
        write_json(folder / 'api_points.json', meta)
        client.call('PUT', 'scenario', {'electricity_price': 'dynamic'})
        y = client.call('PUT', 'initialize', {'start_time': start, 'warmup_period': 7 * DAY})
        assert y['time'] == start
        client.call('PUT', 'step', {'step': 300})
        step_readback = client.call('GET', 'step')
        assert (step_readback.get('step') if isinstance(step_readback, dict)
                else step_readback) == 300
        write_json(folder / 'scenario.json', client.call('GET', 'scenario'))
        assert 'oveHeaPumY_u' in y
        fieldnames = list(y)
        files['measurements'] = (folder / 'measurements.csv').open('w', newline='')
        writer = csv.DictWriter(files['measurements'], fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(y)
        policy = Policy(cfg, episode, profile)
        block_start, dense_index = start, 0
        last_mode, last_segment = None, None
        print(f'{run_id}: initialized day {start / DAY:g}', flush=True)
        for t in range(start, end, 300):
            if t % 900 == 0:
                f = client.call('PUT', 'forecast', {
                    'point_names': cfg['forecast_points'],
                    'horizon': min(DAY, 365 * DAY - t), 'interval': 900})
                assert f['time'][0] == t
                append_json(files['forecasts'], {'issue_time_s': t, 'scenario': 'dynamic',
                                                'data': f})
            payload, action = policy.step(t, y, f)
            append_json(files['actions'], action)
            mode, segment = action['actual_mode'], action['segment_id']
            if mode != last_mode or segment != last_segment:
                append_json(files['events'], {'time': t, 'from_mode': last_mode,
                                              'to_mode': mode, 'segment_id': segment,
                                              'reason': action['fallback_reason'],
                                              'target': action['target_level']})
                last_mode, last_segment = mode, segment
            try:
                new_y = client.call('POST', 'advance', payload)
            except Exception:
                # If advance status is uncertain, query and log it without retrying.
                try:
                    recovery = client.call('PUT', 'results', {
                        'point_names': ['reaTZon_y'], 'start_time': t,
                        'final_time': t + 300})
                    write_json(folder / 'uncertain_advance_results.json', recovery)
                except Exception:
                    pass
                raise
            if new_y.get('time') != t + 300:
                raise RuntimeError(f'Unexpected simulation time: {new_y.get("time")}; expected {t + 300}')
            if not all(math.isfinite(new_y[k]) for k in ['reaTZon_y', 'reaTSup_y', 'reaTRet_y']):
                raise RuntimeError('A required temperature is nonfinite. Stopping and keeping the failure record.')
            writer.writerow(new_y)
            temp_c = new_y['reaTZon_y'] - 273.15
            extreme_low, extreme_high = cfg['absolute_temperature_limits_c']
            if not extreme_low <= temp_c <= extreme_high:
                raise RuntimeError(f'Temperature is outside the experimental limits: {temp_c:.3f} ℃')
            files['measurements'].flush()
            # Abort if heating persists after an overtemperature fallback.
            if (action['fallback_reason'] == 'envelope' and
                    y['reaTZon_y'] > y['reaTSetCoo_y'] + 1 and
                    new_y['oveHeaPumY_u'] > .01):
                raise RuntimeError('The compressor did not stop after the high-temperature fallback.')
            y = new_y
            if (t + 300 - block_start >= DAY) or t + 300 == end:
                dense = client.call('PUT', 'results', {
                    'point_names': list(meta['measurements']) + list(meta['inputs']),
                    'start_time': block_start, 'final_time': t + 300})
                if dense['time'][0] != block_start or dense['time'][-1] != t + 300:
                    raise RuntimeError('Dense results do not cover the requested interval.')
                dense_path = folder / f'results_dense_{dense_index:03d}.csv.gz'
                dense_tmp = dense_path.with_name(dense_path.name + '.tmp')
                pd.DataFrame(dense).to_csv(dense_tmp, index=False, compression='gzip')
                dense_tmp.replace(dense_path)
                dense_index += 1
                block_start = t + 300
                manifest['last_time_s'] = block_start
                manifest['elapsed_s'] = time.monotonic() - begun
                write_json(folder / 'run_manifest.json', manifest)
                print(f'{run_id}: day {block_start / DAY:g}, '
                      f'{manifest["elapsed_s"]:.1f}s elapsed', flush=True)
        write_json(folder / 'kpis.json', client.call('GET', 'kpi'))
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest['status'] = 'cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed'
        manifest['error'] = str(exc)
        (folder / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if client:
            try:
                # Stop the dedicated instance and release overrides without advancing time.
                client.close()
                manifest['instance_stopped'] = True
            except Exception as exc:
                manifest['cleanup_error'] = str(exc)
        for f in files.values():
            f.close()
        manifest['elapsed_s'] = time.monotonic() - begun
        manifest['files_sha256'] = {p.name: digest(p) for p in folder.iterdir()
                                    if p.is_file() and p.name != 'run_manifest.json'}
        write_json(folder / 'run_manifest.json', manifest)
    process(folder)
    return folder


def process(folder):
    """Integrate with right-endpoint rectangles without smoothing across control changes."""
    folder = Path(folder)
    manifest = json.loads((folder / 'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed':
        raise ValueError('Only completed collection runs can be processed.')
    y = pd.read_csv(folder / 'measurements.csv')
    actions = pd.read_json(folder / 'actions.jsonl', lines=True, convert_dates=False,
                           precise_float=True, encoding='utf-8')
    dense = pd.concat([pd.read_csv(p) for p in sorted(folder.glob('results_dense_*.csv.gz'))],
                      ignore_index=True)
    dense = dense.drop_duplicates('time', keep='last').sort_values('time')
    dt = dense['time'].diff().to_numpy()
    assert (dt[1:] > 0).all() and (dt[1:] <= 30.001).all()
    assert (y['time'].diff().dropna() == 300).all()
    assert len(actions) + 1 == len(y)
    assert (actions['time_start_s'].to_numpy() == y['time'].to_numpy()[:-1]).all()
    for name in ['reaTZon_y', 'reaQFloHea_y', 'reaQHeaPumCon_y',
                 'reaPHeaPum_y', 'reaPPumEmi_y', 'reaPFan_y', 'oveHeaPumY_u']:
        assert np.isfinite(dense[name]).all(), name
    start, end = manifest['start_s'], manifest['end_s']
    states = pd.DataFrame({'time_s': y['time']})
    for name, (source, offset) in MAPPING.items():
        states[name] = y[source] + offset
    states['p_electric_total_w'] = states[['p_heat_pump_w', 'p_pump_w', 'p_fan_w']].sum(axis=1)
    forecasts = [json.loads(line) for line in
                 (folder / 'forecasts.jsonl').read_text(encoding='utf-8').splitlines()]
    by_issue = {r['issue_time_s']: r['data'] for r in forecasts}
    n = (end - start) // 900
    t_end = dense['time'].to_numpy()[1:]
    # Each dense sample represents the preceding interval; do not interpolate across jumps.
    group = np.floor((t_end - start - 1e-6) / 900).astype(int)
    assert group.min() == 0 and group.max() == n - 1
    duration = np.bincount(group, weights=dt[1:], minlength=n)
    assert np.allclose(duration, 900)
    transitions = pd.DataFrame({'time_start_s': np.arange(start, end, 900),
                                'time_end_s': np.arange(start + 900, end + 900, 900)})
    sample = states.iloc[::3].reset_index(drop=True)
    for col in ['operative_temperature_c', 'outdoor_temperature_c',
                'supply_temperature_c', 'return_temperature_c', 'solar_ghi_w_m2',
                'comfort_lower_c', 'comfort_upper_c']:
        transitions[col] = sample[col].to_numpy()[:-1]
    transitions['next_operative_temperature_c'] = sample['operative_temperature_c'].to_numpy()[1:]
    for target, source in [('q_floor_mean_w', 'reaQFloHea_y'),
                           ('q_condenser_mean_w', 'reaQHeaPumCon_y'),
                           ('p_heat_pump_mean_w', 'reaPHeaPum_y'),
                           ('p_pump_mean_w', 'reaPPumEmi_y'),
                           ('p_fan_mean_w', 'reaPFan_y'),
                           ('compressor_mean_readback', 'oveHeaPumY_u')]:
        transitions[target] = np.bincount(group, weights=dense[source].to_numpy()[1:] * dt[1:], minlength=n) / 900
    transitions['p_electric_total_mean_w'] = transitions[['p_heat_pump_mean_w', 'p_pump_mean_w', 'p_fan_mean_w']].sum(axis=1)
    transitions['electricity_kwh'] = transitions['p_electric_total_mean_w'] * 900 / 3.6e6
    transitions['q_floor_kwh'] = transitions['q_floor_mean_w'] * 900 / 3.6e6
    schedule_mapping = {'occupancy_people': 'Occupancy[1]',
                        'internal_gains_con_w': 'InternalGainsCon[1]',
                        'internal_gains_rad_w': 'InternalGainsRad[1]',
                        'electricity_price_eur_kwh': 'PriceElectricPowerDynamic'}
    for target, source in schedule_mapping.items():
        transitions[target] = [by_issue[int(t)][source][0] for t in transitions['time_start_s']]
    transitions['electricity_cost_eur'] = transitions['electricity_kwh'] * transitions['electricity_price_eur_kwh']
    transitions['planned_mode'] = actions['planned_mode'].to_numpy().reshape(-1, 3)[:, 0]
    for mode in ['normal', 'excitation', 'fallback']:
        transitions[f'{mode}_seconds'] = (actions['actual_mode'].to_numpy().reshape(-1, 3) == mode).sum(axis=1) * 300
    transitions['mode'] = [x[0] if len(set(x)) == 1 else 'mixed'
                           for x in actions['actual_mode'].to_numpy().reshape(-1, 3)]
    episode = manifest['episode']
    transitions['target_origin_allowed'] = ((transitions['time_start_s'] >= (episode['start_day'] + 8) * DAY)
                                            & (transitions['time_start_s'] < (episode['end_day'] - 1) * DAY))
    transitions['split'] = episode['split']
    transitions['episode_id'] = episode['id']
    transitions['temperature_kind'] = 'operative'
    transitions['integration_method'] = 'right_rectangle_dense_max30s'
    # Keep each five-minute action so protection switches are not lost in averages.
    for i in range(3):
        transitions[f'u_sent_substep_{i}'] = actions['u_sent'].to_numpy().reshape(-1, 3)[:, i]
        transitions[f'mode_substep_{i}'] = actions['actual_mode'].to_numpy().reshape(-1, 3)[:, i]
    out = (ROOT / 'processed' / manifest['run_id'] / episode['split'] / episode['id'])
    out.mkdir(parents=True, exist_ok=True)
    states.to_csv(out / 'states_5min.csv', index=False)
    sample.to_csv(out / 'states_15min.csv', index=False)
    transitions.to_csv(out / 'transitions_15min.csv', index=False)
    # Measure actual hold times, including segments cut short by a window boundary or fallback.
    segments = []
    for sid, part in actions.dropna(subset=['segment_id']).groupby('segment_id'):
        hold_part = part[~part['ramping']]
        segments.append({'segment_id': int(sid), 'target': float(part['target_level'].iloc[0]),
                         'planned_hold_s': float(part['planned_hold_s'].iloc[0]),
                         'actual_hold_s': len(hold_part) * 300,
                         'ramp_s': int(part['ramping'].sum()) * 300,
                         'truncated': len(hold_part) * 300 < part['planned_hold_s'].iloc[0]})
    save_csv(out / 'segments.csv', segments)
    coarse_power = y[['reaPHeaPum_y', 'reaPPumEmi_y', 'reaPFan_y']].sum(axis=1)
    dense_energy = float(transitions['electricity_kwh'].sum())
    coarse_energy = float(coarse_power.iloc[1:].sum() * 300 / 3.6e6)
    error = y['reaTZon_y'] - y['reaTSetCoo_y']
    low_error = y['reaTSetHea_y'] - y['reaTZon_y']
    active = actions['actual_mode'].eq('excitation').to_numpy()
    readback = y['oveHeaPumY_u'].to_numpy()[1:]
    tracking_error = float(np.max(np.abs(readback[active] - actions.loc[active, 'u_sent']))) if active.any() else None
    continuous = active[1:] & active[:-1]
    deltas = np.abs(np.diff(actions['u_sent'].to_numpy(dtype=float)))
    assert not continuous.any() or np.nanmax(deltas[continuous]) <= .20000001
    assert tracking_error is None or tracking_error < 1e-6
    assert ((readback >= -1e-8) & (readback <= 1 + 1e-8)).all()
    violation = np.maximum(np.maximum(error.to_numpy()[1:], low_error.to_numpy()[1:]), 0)
    def longest_hours(mask):
        longest = current = 0
        for flag in mask:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        return longest / 12
    target_mask = ((y['time'].to_numpy()[1:] >= (episode['start_day'] + 8) * DAY)
                   & (y['time'].to_numpy()[1:] < (episode['end_day'] - 1) * DAY))
    quality_bands = {}
    for label, mask in [('all_observed', np.ones(len(violation), dtype=bool)),
                        ('target_dates', target_mask)]:
        v = violation[mask]
        quality_bands[label] = {
            'within_comfort_fraction': float(np.mean(v <= 1e-6)),
            'within_0_5k_fraction': float(np.mean(v <= .5)),
            'within_1k_fraction': float(np.mean(v <= 1)),
            'over_1k_hours': float(np.sum(v > 1) / 12),
            'over_2k_hours': float(np.sum(v > 2) / 12),
            'longest_over_1k_hours': longest_hours(v > 1),
        }
    official = json.loads((folder / 'kpis.json').read_text())
    official_energy = official['ener_tot'] * 192
    report = {
        'run_id': manifest['run_id'], 'status': 'passed', 'pilot': manifest['pilot'],
        'split': episode['split'], 'episode_id': episode['id'],
        'observed_days': (end - start) / DAY, 'measurements_5min': len(y),
        'transitions_15min': len(transitions), 'dense_samples': len(dense),
        'dense_max_step_s': float(np.max(dt[1:])),
        'target_intervals': int(transitions['target_origin_allowed'].sum()),
        'comfort_bands': quality_bands,
        'excitation_regime_hours': {str(k): float(v / 12) for k, v in
                                     actions.get('excitation_regime', pd.Series(dtype=str)).value_counts().items()},
        'official_energy_kwh': official_energy,
        'official_energy_relative_error': abs(dense_energy - official_energy) / max(official_energy, 1e-9),
        'actual_mode_hours': {k: float(v * 300 / 3600) for k, v in actions['actual_mode'].value_counts().items()},
        'planned_mode_hours': {k: float(v * 300 / 3600) for k, v in actions['planned_mode'].value_counts().items()},
        'level_hold_hours': {str(k): float(v * 300 / 3600) for k, v in actions.loc[~actions['ramping'], 'target_level'].value_counts().items()},
        'temperature_min_c': float(states['operative_temperature_c'].min()),
        'temperature_max_c': float(states['operative_temperature_c'].max()),
        'supply_min_c': float(states['supply_temperature_c'].min()),
        'supply_max_c': float(states['supply_temperature_c'].max()),
        'comfort_violation_hours_5min': float(((error.iloc[1:] > 0) | (low_error.iloc[1:] > 0)).sum() / 12),
        'envelope_violation_hours_5min': float(((error.iloc[1:] > 1) | (low_error.iloc[1:] > 1)).sum() / 12),
        'max_comfort_violation_k': float(max(0, error.max(), low_error.max())),
        'electricity_kwh_dense': dense_energy, 'electricity_kwh_5min_approx': coarse_energy,
        'coarse_energy_relative_error': abs(coarse_energy - dense_energy) / max(dense_energy, 1e-9),
        'max_override_readback_error': tracking_error,
        'compressor_readback_available_in_normal': True,
        'integration_note': 'Approximate integration uses 30-second right rectangles to avoid smoothing across action changes.',
        'cost_note': 'Multiply interval energy by the tariff at its 15-minute start. Forecasts are retained for auditing.',
        'future_action_note': 'Observed future native and fallback actions are suitable for conditional forecasts only; they are not available online in advance.',
        'processed_directory': str(out.relative_to(ROOT)),
        'processing_code_sha256': digest(__file__),
        'files_sha256': {p.name: digest(p) for p in out.iterdir()
                         if p.is_file() and p.name != 'audit.json'},
    }
    write_json(out / 'audit.json', report)
    write_json(folder / 'processing.json', {'processed_directory': str(out), 'audit': report})
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'metadata', 'collect', 'process', 'audit'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/generation.json')
    parser.add_argument('--episode', default='all')
    parser.add_argument('--profile', choices=['full', 'conservative', 'baseline'], default='full')
    parser.add_argument('--pilot-days', type=float)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--url')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8')) if args.config.exists() else default_config()
    if args.url:
        cfg['base_url'] = args.url
    rows = validate_plan(cfg)
    if args.command == 'plan':
        write_json(args.config, cfg)
        write_json(ROOT / 'configs/split_manifest.json', rows)
        write_json(ROOT / 'configs/variable_mapping.json', MAPPING)
        print(json.dumps({'episodes': len(rows), 'target_days': sum(r['target_days'] for r in rows),
                          'advance_requests': sum(r['advance_requests'] for r in rows),
                          'simulation_days': sum(r['end_day'] - r['start_day'] for r in rows)}, indent=2))
    elif args.command == 'metadata':
        client = Client(cfg['base_url'])
        try:
            client.select()
            write_json(ROOT / 'metadata/api_points.json', client.metadata())
        finally:
            client.close()
    elif args.command in ('process', 'audit'):
        if args.run is None:
            parser.error('--run is required')
        process(args.run)
    else:
        selected = [e for e in cfg['episodes'] if args.episode == 'all' or e['id'] == args.episode]
        if not selected:
            parser.error('unknown episode')
        if args.profile == 'baseline' and not args.pilot_days:
            parser.error('baseline diagnostics require --pilot-days')
        if args.pilot_days and any(e['split'] != 'train' for e in selected):
            parser.error('pilot must use train dates')
        batch = {'created_utc': stamp(), 'status': 'running', 'runs': []}
        batch_path = ROOT / 'reports' / f'batch_{stamp()}.json'
        write_json(batch_path, batch)
        try:
            for episode in selected:
                folder = collect(cfg, episode, args.profile, args.pilot_days)
                batch['runs'].append(str(folder.relative_to(ROOT)))
                write_json(batch_path, batch)
            batch['status'] = 'completed'
        except BaseException as exc:
            batch['status'] = 'cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed'
            batch['error'] = str(exc)
            raise
        finally:
            write_json(batch_path, batch)


if __name__ == '__main__':
    main()
