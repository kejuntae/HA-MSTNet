"""Building thermal models and their registry."""

# Expose only creation and lookup; load implementations on demand.
from models.registry import create_estimator, list_models

__all__ = ["create_estimator", "list_models"]
