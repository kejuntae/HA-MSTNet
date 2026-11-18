"""Manage an isolated BOPTEST instance and integrate interval measurements."""

import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

CASE = 'bestest_hydronic_heat_pump'
STEP_SECONDS = 900
DEVICE_POWER_POINTS = ['reaPHeaPum_y', 'reaPPumEmi_y', 'reaPFan_y']
POINTS = ['TDryBul', 'HGloHor', 'Occupancy[1]', 'LowerSetp[1]',
          'UpperSetp[1]', 'PriceElectricPowerDynamic']
DENSE_POINTS = ['reaTZon_y', 'reaQFloHea_y', 'reaPHeaPum_y',
                'reaPPumEmi_y', 'reaPFan_y', 'oveHeaPumY_u',
                'reaTSetHea_y', 'reaTSetCoo_y']


class Boptest:
    """Manage only this instance; never retry a failed advance request."""

    def __init__(self, url):
        self.url = url.rstrip('/')
        self.session = requests.Session()
        # Connect directly to a local service, without system proxies.
        if urlparse(self.url).hostname in ('127.0.0.1', 'localhost', '::1'):
            self.session.trust_env = False
        self.testid = None
        self.testid = self.call('POST', f'testcases/{CASE}/select')['testid']

    def call(self, method, endpoint, data=None):
        suffix = '/' + self.testid if self.testid else ''
        retries = 3 if method == 'GET' or endpoint in ('forecast', 'results') else 1
        for attempt in range(retries):
            try:
                response = self.session.request(
                    method, f'{self.url}/{endpoint}{suffix}', json=data,
                    timeout=(10, 240))
                response.raise_for_status()
                body = response.json()
                if isinstance(body, dict) and body.get('status', 200) != 200:
                    raise RuntimeError(str(body))
                return body.get('payload', body) if isinstance(body, dict) else body
            except (requests.RequestException, ValueError):
                if attempt + 1 == retries:
                    raise
                time.sleep(1)

    def forecast(self, horizon=21600):
        return self.call('PUT', 'forecast', {
            'point_names': POINTS, 'horizon': horizon, 'interval': STEP_SECONDS})

    def advance(self, u=None):
        payload = {name: 0 for name in (
            'oveHeaPumY_activate', 'oveTSet_activate',
            'ovePum_activate', 'oveFan_activate')}
        if u is not None:
            if not np.isfinite(u) or not 0 <= u <= 1:
                raise ValueError('Compressor modulation must be in [0, 1].')
            payload.update(oveHeaPumY_activate=1, oveHeaPumY_u=float(u))
        return self.call('POST', 'advance', payload)

    def dense(self, start, end):
        frame = pd.DataFrame(self.call('PUT', 'results', {
            'point_names': DENSE_POINTS, 'start_time': start, 'final_time': end}))
        frame = frame.drop_duplicates('time', keep='last').sort_values('time')
        if (len(frame) < 2 or not np.isfinite(frame.to_numpy(dtype=float)).all()
                or abs(frame.time.iloc[0] - start) > 1e-5 or abs(frame.time.iloc[-1] - end) > 1e-5):
            raise ValueError('Dense measurements must be finite and cover the full interval.')
        return frame

    def close(self):
        try:
            if self.testid:
                self.call('PUT', 'stop')
                self.testid = None
        finally:
            self.session.close()


def integrate(frame):
    """Integrate dense samples by right rectangles, as in the training data."""
    dt = np.diff(frame.time.to_numpy())
    if not (np.all(dt > 0) and np.max(dt) <= 30.01):
        raise ValueError('Dense measurements must be at most 30 seconds apart.')
    values = frame.iloc[1:]
    p = values[DEVICE_POWER_POINTS].sum(axis=1)
    return {
        'q_mean_w': float(np.dot(values.reaQFloHea_y, dt) / dt.sum()),
        'p_mean_w': float(np.dot(p, dt) / dt.sum()),
        'u_mean': float(np.dot(values.oveHeaPumY_u, dt) / dt.sum()),
        'energy_kwh': float(np.dot(p, dt) / 3.6e6),
    }


def feature_row(y, forecast, q):
    """Build the seven checkpoint inputs, using binary scheduled occupancy."""
    angle = 2 * np.pi * (y['time'] % 86400) / 86400
    return [y['reaTZon_y'] - 273.15,
            y['weaSta_reaWeaTDryBul_y'] - 273.15,
            y['weaSta_reaWeaHGloHor_y'], q,
            np.sin(angle), np.cos(angle), float(forecast['Occupancy[1]'][0] > 0)]
