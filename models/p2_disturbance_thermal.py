"""History-aware thermal states and persistent regime mixing for HA-MSTNet."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from models.p2_adaptive_thermal import AdaptiveP2Estimator, AdaptiveP2Forecaster, AdaptiveP2Options
from models.thermal_utils import stable_filter


def varying_filter(equilibrium, retention, initial):
    """Scan a first-order state with bounded retention, without negative powers or Euler steps."""
    rates, values = retention, (1 - retention) * equilibrium
    offset = 1
    while offset < equilibrium.shape[1]:
        right = rates[:, offset:]
        updated = values[:, offset:] + right * values[:, :-offset]
        rates = torch.cat((rates[:, :offset], right * rates[:, :-offset]), 1)
        values = torch.cat((values[:, :offset], updated), 1)
        offset *= 2
    return values + rates * initial[:, None]


@dataclass(frozen=True)
class DisturbanceP2Options(AdaptiveP2Options):
    """Configure the history-aware module and data, history, or control losses."""

    module_variant: str = "regime"
    adaptation_enabled: bool = True
    extra_disturbance_bound_c: float = 2.0
    maximum_exchange_rate: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.module_variant != "regime":
            raise ValueError('Unknown physics module.')
        if type(self.adaptation_enabled) is not bool:
            raise ValueError('adaptation_enabled must be Boolean.')
        for name in ("extra_disturbance_bound_c", "maximum_exchange_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")


class HistoryAwareThermal(nn.Module):
    """Infer persistent disturbances or exchange from causal reference forecast errors."""

    def __init__(self, original, options):
        super().__init__()
        self.base, self.options = original, options
        # Fast memory spans 0.25–4 h; slow memory spans 4–48 h.
        self.memory_time_logits = nn.Parameter(torch.tensor([-1.3862944, -2.3025851]))
        self.memory_gain_logits = nn.Parameter(torch.full((2,), -1.0986123))
        self.exchange_head = nn.Sequential(nn.Linear(6, 8), nn.GELU(), nn.Linear(8, 1))
        nn.init.normal_(self.exchange_head[-1].weight, std=.01)
        nn.init.constant_(self.exchange_head[-1].bias, -3.)
        self.exchange_time_logit = nn.Parameter(torch.tensor(-2.0430739))
        self.regime_head = nn.Sequential(nn.Linear(6, 8), nn.GELU(), nn.Linear(8, 3))
        nn.init.zeros_(self.regime_head[-1].weight)
        nn.init.zeros_(self.regime_head[-1].bias)
        self.regime_background_logits = nn.Parameter(torch.tensor([2., 0., 0.]))
        self.regime_time_logit = nn.Parameter(torch.tensor(-2.0430739))

    @staticmethod
    def persistence_hours(logit):
        """Bound persistence to 0.25–48 h independently of the building time constants."""
        return .25 + 47.75 * logit.sigmoid()

    def reference_history(self, history, context, dt, alignment):
        """Use context-free reference parameters so later measurements cannot affect earlier errors."""
        params = self.base.parameters_from_history(torch.zeros_like(context))
        retention = torch.exp(-dt / params["hours"])
        lag = torch.exp(-dt / params["lag_hours"])
        sources, _ = self.base.sources(history, params)
        drives = sources[:, :-1] if alignment else sources[:, 1:]
        raw = history[:, :-1] if alignment else history[:, 1:]
        filtered = stable_filter(drives, lag, sources[:, 0])
        equilibrium = self.base.equilibrium(raw, filtered, params)
        initial = history[:, 0, :1] + params["offsets"]
        states = stable_filter(equilibrium[..., None], retention, initial[..., None]).squeeze(-1)
        previous = torch.cat((initial[:, None], states[:, :-1]), 1)
        # Anchor to the last measurement, predict the current step, then read its measurement.
        previous = previous + history[:, :-1, :1] - (
            previous * params["weights"][:, None]).sum(-1, keepdim=True)
        forecast = retention[:, None] * previous + (1-retention[:, None]) * equilibrium
        innovation = history[:, 1:, 0] - (forecast * params["weights"][:, None]).sum(-1)
        return innovation

    def memory_features(self, innovation, history, dt):
        """Encode past forecast errors at two time scales and use the temperature difference to infer exchange."""
        fraction = self.memory_time_logits.sigmoid()
        hours = torch.stack((.25 + 3.75*fraction[0], 4 + 44*fraction[1]))
        retention = torch.exp(-dt / hours)[None].expand(len(history), -1)
        clipped = .5 * (innovation / .5).tanh()
        values = clipped[:, :, None, None].expand(-1, -1, 2, 1)
        memory = stable_filter(values, retention, values.new_zeros((len(history), 2, 1)))[:, -1, :, 0]
        temperature_difference = history[:, -1, 1] - history[:, -1, 0]
        features = torch.cat((memory, innovation[:, -1:],
                              innovation.abs().mean(1, keepdim=True),
                              temperature_difference[:, None]/20,
                              (memory[:, :1]*temperature_difference[:, None]).tanh()), -1)
        return memory, hours, features

    def memory_path(self, original, memory, hours, lead):
        gain = 8 * self.memory_gain_logits.sigmoid()
        unbounded = (memory[:, None] * gain * torch.exp(-lead[None, :, None]/hours)).sum(-1)
        bound = self.options.extra_disturbance_bound_c
        disturbance = bound * (unbounded/bound).tanh()
        equilibrium = original["equilibrium_c"] + disturbance[:, :, None]
        states = stable_filter(equilibrium[..., None], original["retention"],
                               original["initial"][..., None]).squeeze(-1)
        return states, equilibrium, disturbance

    def exchange_path(self, original, features, future, lead, dt):
        activity = (features[:, 3:4]/.1).tanh()
        current = self.options.maximum_exchange_rate * self.exchange_head(features).sigmoid() * activity
        hours = self.persistence_hours(self.exchange_time_logit)
        rate = current * torch.exp(-lead[None]/hours)
        intrinsic = 1 / original["hours"][:, None]
        total = intrinsic + rate[:, :, None]
        retention = torch.exp(-dt*total)
        equilibrium = (intrinsic*original["equilibrium_c"]
                       + rate[:, :, None]*future[:, :, 1:2]) / total
        states = varying_filter(equilibrium, retention, original["initial"])
        return states, equilibrium, rate, retention

    def forward(self, history, future, context, dt, alignment):
        original = self.base(history, future, context, dt, alignment)
        if not self.options.adaptation_enabled:
            return original
        innovation = self.reference_history(history, context, dt, alignment)
        lead = torch.arange(1, future.shape[1]+1, device=future.device, dtype=future.dtype)*dt
        diagnostics = dict(innovation_history_c=innovation)
        memory, hours, features = self.memory_features(innovation, history, dt)
        memory_states, memory_equilibrium, disturbance = self.memory_path(original, memory, hours, lead)
        exchange_states, exchange_equilibrium, rate, dynamic_retention = self.exchange_path(
            original, features, future, lead, dt)
        current = self.regime_head(features).softmax(-1)
        background = self.regime_background_logits.softmax(-1)
        persistence = torch.exp(-lead/self.persistence_hours(self.regime_time_logit))
        weights = persistence[None, :, None]*current[:, None] + (1-persistence[None, :, None])*background
        candidates = torch.stack((original["states_c"], memory_states, exchange_states), 2)
        states = (candidates*weights[:, :, :, None]).sum(2)
        equilibrium = (torch.stack((original["equilibrium_c"], memory_equilibrium,
                                     exchange_equilibrium), 2)*weights[:, :, :, None]).sum(2)
        diagnostics.update(
            disturbance_memory_c=memory,
            disturbance_memory_hours=hours[None].expand(len(history), -1),
            disturbance_c=disturbance, exchange_rate_per_hour=rate,
            dynamic_retention=dynamic_retention, regime_weights=weights,
            regime_prediction_c=(candidates*original["weights"][:, None, None]).sum(-1),
        )
        original.update(prediction_c=(states*original["weights"][:, None]).sum(-1),
                        states_c=states, equilibrium_c=equilibrium, **diagnostics)
        return original


class DisturbanceP2Forecaster(AdaptiveP2Forecaster):
    """Add the physical module after initializing shared layers."""

    def __init__(self, context, options):
        super().__init__(context, options)
        # Initialize the new module without changing the shared dropout or sampling RNG.
        with torch.random.fork_rng(devices=[]):
            self.thermal = HistoryAwareThermal(self.thermal, options)
        self.thermal.register_forward_hook(self._capture_physical)

    def forward_components(self, inputs):
        # Cache diagnostics from one physical forward pass.
        result = super().forward_components(inputs)
        result.update(self._physical_diagnostics)
        return result

    def _capture_physical(self, module, arguments, output):
        names = ("innovation_history_c", "disturbance_c", "disturbance_memory_c",
                 "disturbance_memory_hours", "exchange_rate_per_hour", "dynamic_retention",
                 "regime_weights", "regime_prediction_c")
        self._physical_diagnostics = {name: output[name] for name in names if name in output}


class RegimeP2Estimator(AdaptiveP2Estimator):
    """Use the history-aware module and retain compatibility with legacy checkpoints."""

    name = "HA-MSTNet"
    direction = "P2_persistent_mixture_of_shared_physical_regimes"

    option_type = DisturbanceP2Options
    network_type = DisturbanceP2Forecaster
    module_variant = "regime"

    def __init__(self, **parameters):
        if parameters.get("module_variant", self.module_variant) != self.module_variant:
            raise ValueError('The physics module does not match the model identity.')
        super().__init__(**{**parameters, "module_variant": self.module_variant})

    def load_payload(self, payload, context):
        if payload["options"]["module_variant"] != self.module_variant:
            raise ValueError('The checkpoint physics module does not match the model identity.')
        super().load_payload(payload, context)

    def describe_architecture(self):
        result = super().describe_architecture()
        result.update(physical_module_variant=self.module_variant,
                      reference_innovation_parameters="global_reference_no_window_context",
                      adaptation_inputs="observed_history_only",
                      disturbance_unit="equivalent_equilibrium_temperature_c",
                      exchange_rate_unit="per_hour",
                      extra_inputs=False,
                      extra_losses=self.options.loss_profile != "data")
        return result
