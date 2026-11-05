"""
Model catalog loader for chat completion services.

Reads model configurations from a models.yaml file and returns
a dictionary that can be used to configure OpenAIArgs or construct
an OpenAI client directly.
"""

from pathlib import Path
from typing import Dict

import yaml


def load_model_config(model_name: str, models_path: Path = None) -> Dict:
    """Load and validate a model configuration from models.yaml.

    Args:
        model_name: Key identifying the model in models.yaml.
        models_path: Path to models.yaml. Defaults to common/models.yaml
                     (resolved relative to this file's location).

    Returns:
        Model configuration dictionary with at least 'provider' and 'model_name'.

    Raises:
        ValueError: If model_name is not found or required fields are missing.
        FileNotFoundError: If models_path does not exist.
    """
    if models_path is None:
        models_path = Path(__file__).resolve().parent.parent.parent / "models.yaml"

    if not models_path.exists():
        raise FileNotFoundError(f"Model catalog not found: {models_path}")

    with open(models_path, "r") as f:
        catalog = yaml.safe_load(f)

    models = catalog.get("models", {})
    if model_name not in models:
        available = ", ".join(sorted(models.keys()))
        raise ValueError(
            f"Model '{model_name}' not found in {models_path.name}. "
            f"Available: {available}"
        )

    config = models[model_name]

    if "provider" not in config:
        raise ValueError(f"Model '{model_name}' missing required field: provider")
    if "model_name" not in config:
        raise ValueError(f"Model '{model_name}' missing required field: model_name")
    return config


def list_models(models_path: Path = None) -> list:
    """Return a sorted list of model names available in the catalog.

    Args:
        models_path: Path to models.yaml. Defaults to common/models.yaml.

    Returns:
        Sorted list of model name strings.
    """
    if models_path is None:
        models_path = Path(__file__).resolve().parent.parent.parent / "models.yaml"

    if not models_path.exists():
        return []

    with open(models_path, "r") as f:
        catalog = yaml.safe_load(f)

    return sorted(catalog.get("models", {}).keys())
