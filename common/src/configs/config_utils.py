from common.src.configs.configurable_base import ConfigurableBase
from common.src.configs.schema import public_config, runtime_config
import argparse
import os
import tempfile
import yaml


def get_default_config_args() -> argparse.Namespace:
    """
    Get the default configuration argument for the current config path (--config_path).

    Returns:
        An argparse.Namespace object containing the --config_path argument.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="The path to the YAML configuration file."
    )

    return parser.parse_args()


def load_yaml_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    try:
        with open(config_path, "r", encoding="utf-8-sig") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration does not exist: {config_path}") from None
    except (yaml.YAMLError, UnicodeError) as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}" if mark is not None else ""
        raise ValueError(f"Invalid YAML in {config_path}{location}.") from None
    except OSError as exc:
        raise OSError(f"Cannot read configuration {config_path}. {exc.strerror}") from None
    if not isinstance(config, dict) or not config:
        raise ValueError(f"Configuration {config_path} must contain a nonempty YAML mapping.")
    return runtime_config(config)


def save_yaml_config(config: dict, config_path: str):
    """
    Save a YAML configuration file.

    Args:
        config: The configuration dictionary to save.
        config_path: The path to the YAML configuration file.
    """
    reject_plaintext_keys(config)
    config = public_config(config)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", dir=os.path.dirname(os.path.abspath(config_path)),
            prefix=".config-", suffix=".yaml", delete=False,
        ) as file:
            temp_path = file.name
            yaml.dump(config, file, default_flow_style=False,
                      sort_keys=False, allow_unicode=True, width=float("inf"))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, config_path)
    except OSError as exc:
        raise OSError(f"Cannot save configuration {config_path}. {exc.strerror}") from None
    except yaml.YAMLError:
        raise ValueError(f"Cannot serialize configuration for {config_path}.") from None
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def initialize_from_configs(config: dict, config_path: str, config_instances: dict):
    """
    Initialize configurable instances from a configuration dictionary.

    Args:
        config: The configuration dictionary.
        config_path: The path to the configuration file.
        config_instances: A dictionary mapping config keys to ConfigurableBase instances.
    """
    has_error = False

    # For convienience, reconstruct the final config based on the order of the specified instances.
    # This will ensure that the order of the config sections matches the order of the instances.
    final_config = dict()
    
    for key, instance in config_instances.items():
        if not isinstance(instance, ConfigurableBase):
            continue

        instance_error = instance.load_config(config, key)
        has_error = has_error or instance_error
        final_config[key] = config.get(key, dict())

    if has_error:
        # There were errors loading the config
        # Make sure any missing config values are captured in the final config
        for key, value in config.items():
            if key not in final_config:
                final_config[key] = value

        raise ValueError("Invalid configuration. Correct the reported fields before running.")


def reject_plaintext_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"openai_key", "key", "api_key"}:
                secret = child.get("value") if isinstance(child, dict) else child
                if secret:
                    raise ValueError("Plaintext keys require explicit migration to environment references.")
            reject_plaintext_keys(child)
    elif isinstance(value, list):
        for child in value:
            reject_plaintext_keys(child)
