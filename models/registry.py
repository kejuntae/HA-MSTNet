"""Model registry and canonical ordering of required input features."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Callable, Mapping

from models.base import BaseEstimator
from util.model_features import THERMAL_FEATURES


# Use this feature order throughout the project.
CANONICAL_FEATURE_ORDER = THERMAL_FEATURES


@dataclass(frozen=True)
class ModelRegistration:
    """Store a model name, required inputs, and a lazy factory."""

    # The factory takes configuration parameters and returns a BaseEstimator.
    name: str
    required_features: tuple[str, ...]
    factory: Callable[[Mapping[str, object]], BaseEstimator]


# Dict insertion order determines the public listing.
_REGISTRY: dict[str, ModelRegistration] = {}
_MODEL_ALIASES = {"physical_patch_thermal_p2_regime": "HA-MSTNet"}


def register_model(registration: ModelRegistration) -> None:
    """Register a unique model name without overwriting an entry."""
    # Prevent multiple implementations from sharing a public name.
    if registration.name in _REGISTRY:
        raise ValueError(f"Model {registration.name} is already registered.")
    # Require canonical feature names before data preparation.
    unknown = [
        name for name in registration.required_features
        if name not in CANONICAL_FEATURE_ORDER
    ]
    if unknown:
        raise ValueError(f"Model has noncanonical features: {', '.join(unknown)}.")
    if registration.required_features != CANONICAL_FEATURE_ORDER:
        raise ValueError('All models must declare the seven input features in canonical order.')
    _REGISTRY[registration.name] = registration


def canonical_model_name(name: str) -> str:
    """Map legacy model identifiers to their public names."""
    return _MODEL_ALIASES.get(name, name)


def get_registration(name: str) -> ModelRegistration:
    """Look up a registration or reject an unknown name."""
    # Report an unknown model as a configuration error.
    try:
        return _REGISTRY[canonical_model_name(name)]
    except KeyError as error:
        raise ValueError(f"Unknown model: {name}.") from error


def list_models() -> tuple[str, ...]:
    """Return model names in public registration order."""
    # Return an immutable view.
    return tuple(_REGISTRY)


def required_feature_union(model_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return the union of required features in canonical order."""
    # Deduplicate first, then restore canonical order.
    required: set[str] = set()
    for model_name in model_names:
        required.update(get_registration(model_name).required_features)
    return tuple(
        name for name in CANONICAL_FEATURE_ORDER if name in required
    )


def create_estimator(
    name: str,
    parameters: Mapping[str, object] | None = None,
) -> BaseEstimator:
    """Construct an estimator through its lazy factory."""
    # An empty parameter dictionary uses constructor defaults.
    return get_registration(name).factory(parameters or {})


def _lazy_factory(
    module_name: str,
    class_name: str,
) -> Callable[[Mapping[str, object]], BaseEstimator]:
    """Defer implementation imports to avoid circular dependencies."""
    # Import the implementation only when creating an estimator.
    def factory(parameters: Mapping[str, object]) -> BaseEstimator:
        module = importlib.import_module(module_name)
        # Accept implementation paths only from the default registrations.
        estimator_class = getattr(module, class_name)
        return estimator_class(**dict(parameters))

    return factory


def _register_defaults() -> None:
    """Register seven baselines and HA-MSTNet."""
    registrations = (
        ("3r2c", "models.rc_3r2c",
         "ThreeR2CEstimator"),
        ("pcnn", "models.pcnn",
         "PCNNEstimator"),
        ("apcnn_mlp", "models.apcnn",
         "APCNNMLPEstimator"),
        ("lstm", "models.lstm",
         "LSTMEstimator"),
        ("mlp", "models.mlp",
         "MLPEstimator"),
        ("modnn", "models.modnn",
         "MODNNEstimator"),
        ("pcicnn", "models.pcicnn",
         "PCICNNEstimator"),
        ("HA-MSTNet", "models.p2_disturbance_thermal",
         "RegimeP2Estimator"),
    )
    for name, module, class_name in registrations:
        register_model(ModelRegistration(
            name, CANONICAL_FEATURE_ORDER, _lazy_factory(module, class_name),
        ))


# Register at import time while keeping implementations lazy.
_register_defaults()
