"""A differentiable HA-MSTNet controller for the BOPTEST heat pump."""

from dataclasses import dataclass
import time

import casadi as ca
import joblib
import numpy as np
import torch

from models.base import ModelContext
from models.registry import canonical_model_name, create_estimator
from mpc.environment import STEP_SECONDS
from util.features import FeatureScaler
from util.model_features import THERMAL_FEATURES

HISTORY_STEPS = 48
DTYPE = torch.float64


def tensor(value):
    """Use CPU double precision for the solver callbacks."""
    return torch.as_tensor(value, dtype=DTYPE)


@dataclass
class Inputs:
    """History, issued forecasts, and feedback available at decision time."""

    past: torch.Tensor
    current: float
    future: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    price: torch.Tensor
    previous: float
    bias: float


class ObjectiveGradient(ca.Callback):
    """Provide the exact objective Jacobian without finite differences."""

    def __init__(self, owner, name):
        ca.Callback.__init__(self)
        self.owner = owner
        self.construct(name, {'enable_fd': False})

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, index):
        return ca.Sparsity.dense(self.owner.horizon if index == 0 else 1, 1)

    def get_sparsity_out(self, index):
        return ca.Sparsity.dense(1, self.owner.horizon)

    def eval(self, arguments):
        """CasADi also passes the objective value; the gradient cache is keyed by action."""
        _, gradient = self.owner.evaluate(arguments[0])
        return [ca.DM(gradient.reshape(1, -1))]


class Objective(ca.Callback):
    """Optimize actions with the history and issued forecasts fixed."""

    def __init__(self, owner, name):
        ca.Callback.__init__(self)
        self.owner = owner
        self.gradients = []
        self.construct(name, {'enable_fd': False})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, index):
        return ca.Sparsity.dense(self.owner.horizon, 1)

    def get_sparsity_out(self, index):
        return ca.Sparsity.scalar()

    def eval(self, arguments):
        value, _ = self.owner.evaluate(arguments[0])
        return [ca.DM(value)]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, input_names, output_names, options):
        gradient = ObjectiveGradient(self.owner, name)
        self.gradients.append(gradient)
        return gradient


class Controller:
    """Optimize bounded compressor plans with CasADi/IPOPT and PyTorch gradients."""

    def __init__(self, config, checkpoint, device_model):
        self.config = config
        self.horizon = config["horizon_steps"]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = saved["metadata"]
        if canonical_model_name(metadata["model"]) != "HA-MSTNet":
            raise ValueError("The MPC example requires a HA-MSTNet checkpoint.")
        if (metadata["history_steps"] != HISTORY_STEPS
                or metadata["horizon_steps"] != self.horizon
                or metadata["interval_minutes"] != 15
                or not metadata["interval_start_inputs"]
                or tuple(metadata["feature_names"]) != THERMAL_FEATURES):
            raise ValueError("Use a BOPTEST checkpoint with H48 and the configured forecast horizon.")

        scale = metadata["scaler"]
        scaler = FeatureScaler(tuple(scale["feature_names"]), scale["minimum"], scale["span"])
        context = ModelContext(THERMAL_FEATURES, HISTORY_STEPS, self.horizon, 15, scaler, True)
        self.minimum = tensor([scaler.minimum[key] for key in THERMAL_FEATURES])
        self.span = tensor([scaler.span[key] for key in THERMAL_FEATURES])
        state = metadata["model_state"]
        estimator = create_estimator("HA-MSTNet", state["options"])
        estimator.load_payload({**state, "state_dict": saved["state_dict"]}, context)
        self.network = estimator.network.double().eval().requires_grad_(False)
        devices = joblib.load(device_model)
        self.devices = {}
        for name in ('Q_HVAC', 'P_HVAC'):
            model = devices[name + '__mlp_hidden32']
            scale, net = (model.regressor_.steps[0][1],
                          model.regressor_.steps[1][1])
            self.devices[name] = tuple(map(tensor, (
                scale.mean_, scale.scale_, net.coefs_[0], net.intercepts_[0],
                net.coefs_[1], net.intercepts_[1],
                model.transformer_.scale_[0], model.transformer_.mean_[0])))
        self.plan = np.full(self.horizon, .5)
        self.inputs = None
        self.cache = None
        self.best = None
        self.evaluations = 0
        self.objective_callback = Objective(self, f'objective_{id(self)}')
        u = ca.MX.sym('u', self.horizon)
        objective = self.objective_callback(u)
        settings = {
            'print_time': False,
            'error_on_fail': False,
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': config['max_iter'],
            'ipopt.max_cpu_time': config['max_cpu_seconds'],
            'ipopt.max_wall_time': config.get('max_wall_seconds', config['max_cpu_seconds']),
            'ipopt.tol': 1e-5,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.acceptable_iter': 3,
            'ipopt.acceptable_obj_change_tol': 1e-5,
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.bound_relax_factor': 0.,
            'ipopt.honor_original_bounds': 'yes',
        }
        self.solver = ca.nlpsol(f'mpc_{id(self)}', 'ipopt',
                               {'x': u, 'f': objective}, settings)
        self.solver_settings = settings

    def device(self, name, u, outdoor):
        """Evaluate the fitted scaler, tanh network, and inverse target transform."""
        mean, scale, w1, b1, w2, b2, sy, my = self.devices[name]
        x = torch.stack((u, outdoor), -1)
        x = (torch.tanh((x - mean) / scale @ w1 + b1) @ w2 + b2).squeeze(-1)
        raw = x * sy + my
        return .5 * (raw + torch.sqrt(raw.square() + 1))

    def parameters(self, history, y, forecast, previous, bias):
        """Check forecast alignment and use only information available at the decision time."""
        n = self.horizon
        times = np.asarray(forecast['time'][:n + 1], dtype=float)
        expected = y['time'] + np.arange(n + 1) * STEP_SECONDS
        if times.shape != expected.shape or not np.allclose(
                times, expected, rtol=0, atol=1e-5):
            raise ValueError('Forecast timestamps do not align with the control intervals.')
        angle = 2 * np.pi * (times[:n] % 86400) / 86400
        future = np.column_stack((
            np.asarray(forecast['TDryBul'][:n]) - 273.15,
            forecast['HGloHor'][:n], np.sin(angle), np.cos(angle),
            np.asarray(forecast['Occupancy[1]'][:n]) > 0))
        values = Inputs(
            tensor(np.asarray(history[-(HISTORY_STEPS - 1):])), y['reaTZon_y'] - 273.15,
            tensor(future),
            tensor(forecast['LowerSetp[1]'][1:n + 1]) - 273.15,
            tensor(forecast['UpperSetp[1]'][1:n + 1]) - 273.15,
            tensor(forecast['PriceElectricPowerDynamic'][:n]), previous, bias)
        for value in vars(values).values():
            if not np.isfinite(np.asarray(value)).all():
                raise ValueError('Controller inputs contain nonfinite values.')
        if values.past.shape != (HISTORY_STEPS - 1, 7) or torch.any(values.lower > values.upper):
            raise ValueError('Invalid history or comfort bounds.')
        return values

    def raw_inputs(self, u, inputs):
        """Build the model window with candidate HVAC power and masked future temperature."""
        q = self.device('Q_HVAC', u, inputs.future[:, 0])
        p = self.device('P_HVAC', u, inputs.future[:, 0]) / 1000
        future = torch.cat((
            torch.full((self.horizon, 1), inputs.current, dtype=DTYPE),
            inputs.future[:, :2], q[:, None], inputs.future[:, 2:]), -1)
        raw = torch.cat((inputs.past, future, future[-1:]), 0)
        return raw, q, p

    def temperature(self, raw):
        scaled = ((raw - self.minimum) / self.span)[None]
        return self.network(scaled)[0] * self.span[0] + self.minimum[0]

    def predict(self, u, inputs):
        """Return uncorrected temperature, heat output, and electrical power."""
        raw, q, p = self.raw_inputs(u, inputs)
        return self.temperature(raw), q, p

    def objective(self, u, inputs):
        """Penalize electricity cost, soft comfort violations, and control movement."""
        temperature, _, p = self.predict(u, inputs)
        temperature = temperature + inputs.bias
        margin = self.config['comfort_margin_c']
        cold = inputs.lower + margin - temperature
        hot = temperature - inputs.upper + margin
        cold = .5 * (cold + torch.sqrt(cold.square() + 1e-4))
        hot = .5 * (hot + torch.sqrt(hot.square() + 1e-4))
        change = u - torch.cat((tensor([inputs.previous]), u[:-1]))
        return (.25 * (inputs.price * p).sum()
                + self.config['comfort_weight'] * .25
                * (cold.square().sum() + hot.square().sum())
                + self.config['move_weight'] * change.square().sum())

    def bind(self, inputs):
        """Reset cached objective values at each decision time."""
        self.inputs = inputs
        self.cache = None
        self.best = None
        self.evaluations = 0

    def evaluate(self, candidate):
        """Evaluate the PyTorch objective and its gradient for CasADi."""
        plan = np.asarray(candidate, dtype=float).reshape(-1)
        if self.inputs is None or plan.shape != (self.horizon,):
            raise ValueError('Missing decision inputs or invalid action shape.')
        if self.cache is not None and np.array_equal(plan, self.cache[0]):
            return self.cache[1], self.cache[2].copy()
        if not np.isfinite(plan).all():
            raise ValueError('IPOPT proposed a nonfinite action.')
        with torch.enable_grad():
            u = tensor(plan.copy()).requires_grad_(True)
            value = self.objective(u, self.inputs)
            gradient, = torch.autograd.grad(value, u)
        value = float(value.detach())
        gradient = gradient.detach().numpy().copy()
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise ValueError('The objective or gradient is nonfinite.')
        self.evaluations += 1
        self.cache = (plan.copy(), value, gradient.copy())
        if np.all(plan >= 0) and np.all(plan <= 1):
            if self.best is None or value < self.best[0]:
                self.best = (value, plan.copy())
        return value, gradient

    def solve(self, inputs):
        """Return the first action of a validated plan, or fall back to native control."""
        started = time.perf_counter()
        initial = self.plan.copy()
        self.bind(inputs)
        status = 'Not_Started'
        converged = False
        iterations = 0
        try:
            initial_cost, _ = self.evaluate(initial)
            result = self.solver(x0=initial, lbx=0., ubx=1.)
            stats = self.solver.stats()
            status = str(stats['return_status'])
            converged = bool(stats['success'])
            iterations = int(stats.get('iter_count', 0))
            candidate = np.asarray(result['x']).reshape(-1)
            if (np.isfinite(candidate).all()
                    and np.all(candidate >= -1e-8) and np.all(candidate <= 1 + 1e-8)):
                self.evaluate(np.clip(candidate, 0, 1))
            limited = status in (
                'Maximum_Iterations_Exceeded', 'Maximum_CpuTime_Exceeded',
                'Maximum_WallTime_Exceeded', 'Search_Direction_Becomes_Too_Small')
            if not (converged or limited) or self.best is None:
                return None, {
                    'status': status, 'solver': 'CasADi+IPOPT',
                    'iterations': iterations,
                    'solve_seconds': time.perf_counter() - started}
            value, plan = self.best
            with torch.no_grad():
                temperature, q, p = self.predict(tensor(plan), inputs)
            if not all(torch.isfinite(v).all() for v in (temperature, q, p)):
                raise ValueError('The selected IPOPT plan produced nonfinite predictions.')
            reused = np.array_equal(plan, initial)
            self.plan = np.r_[plan[1:], plan[-1]]
            return float(plan[0]), {
                'status': status, 'solver': 'CasADi+IPOPT',
                'solver_message': status, 'iterations': iterations,
                'solve_seconds': time.perf_counter() - started,
                'objective': value, 'initial_objective': initial_cost,
                'converged': converged, 'feasible_limited': not converged,
                'warm_start_reused': reused, 'evaluations': self.evaluations,
                'plan_u': plan.tolist(), 'prediction_c': temperature.tolist(),
                'prediction_q_w': q.tolist(), 'prediction_p_kw': p.tolist()}
        except (RuntimeError, ValueError) as error:
            return None, {
                'status': 'IPOPT_Evaluation_Failed', 'solver': 'CasADi+IPOPT',
                'solver_message': str(error), 'iterations': iterations,
                'solve_seconds': time.perf_counter() - started}
