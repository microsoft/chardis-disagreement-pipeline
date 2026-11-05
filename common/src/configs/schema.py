"""Translate public configuration names at the runtime and persistence boundaries."""

import argparse
from copy import deepcopy


SECTION_ALIASES = {
    "llm": "openai",
    "rationales": "rationales_openai",
    "synthesizer": "synthesizer_openai",
}
FIELD_ALIASES = {
    "model": "openai_model",
    "eval_model": "openai_model",
    "endpoint": "openai_endpoint",
    "base_url": "openai_endpoint",
    "api_version": "openai_api_version",
    "key": "openai_key",
    "api_key": "openai_key",
}
PUBLIC_FIELDS = {
    "openai_model": "model",
    "eval_model": "model",
    "openai_endpoint": "endpoint",
    "openai_api_version": "api_version",
    "openai_key": "key",
}
CONNECTION_SECTIONS = frozenset((*SECTION_ALIASES.values(), "evaluation"))


def section_name(name, public=False):
    internal = SECTION_ALIASES.get(name, name)
    if public:
        return {value: key for key, value in SECTION_ALIASES.items()}.get(internal, internal)
    return internal


def values(section):
    """Unwrap field values without interpreting unrelated names or metadata."""
    return {key: entry["value"] if isinstance(entry, dict) and "value" in entry else entry
            for key, entry in section.items()}


def _merge_entry(left, right, label):
    left_value = values({"field": left})["field"]
    right_value = values({"field": right})["field"]
    if type(left_value) is not type(right_value) or left_value != right_value:
        raise ValueError(f"Conflicting aliases for {label}. Keep one spelling or matching values.")
    if isinstance(left, dict) and isinstance(right, dict):
        merged = deepcopy(left)
        for key, value in right.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"Conflicting alias metadata for {label}. Keep one spelling.")
            merged[key] = deepcopy(value)
        return merged
    return deepcopy(left if isinstance(left, dict) else right)


def connection_section(section, *, evaluation=False, public=False):
    """Copy a connection section, retaining wrappers and rejecting alias conflicts."""
    if not isinstance(section, dict):
        raise ValueError("Model connection section must be a mapping.")
    result = {}
    for key, entry in section.items():
        internal = FIELD_ALIASES.get(key, key)
        target = PUBLIC_FIELDS.get(internal, internal) if public else internal
        if evaluation and not public and target == "openai_model":
            target = "eval_model"
        if target in result:
            result[target] = _merge_entry(result[target], entry, PUBLIC_FIELDS.get(internal, internal))
        else:
            result[target] = deepcopy(entry)
    return result


def runtime_config(config):
    """Copy a full configuration into the existing internal representation."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping.")
    result = {}
    for key, section in config.items():
        name = section_name(key)
        converted = (connection_section(section, evaluation=name == "evaluation")
                     if name in CONNECTION_SECTIONS else deepcopy(section))
        if name in result:
            previous = result[name]
            if previous.keys() != converted.keys():
                raise ValueError(f"Conflicting section aliases for {section_name(name, public=True)}.")
            converted = {
                field: _merge_entry(previous[field], entry,
                                    f"{section_name(name, public=True)}.{PUBLIC_FIELDS.get(field, field)}")
                for field, entry in converted.items()
            }
        result[name] = converted
    return result


def public_config(config):
    """Copy a full configuration into the public schema for explicit saves."""
    return {section_name(name, public=True):
            connection_section(section, public=True) if name in CONNECTION_SECTIONS else section
            for name, section in runtime_config(config).items()}


class AliasAction(argparse.Action):
    """Accept repeated equivalent CLI aliases but never silently override them."""

    def __call__(self, parser, namespace, value, option_string=None):
        seen = getattr(namespace, "_connection_options", {})
        if self.dest in seen and seen[self.dest][1] != value:
            previous = seen[self.dest][0]
            raise argparse.ArgumentError(self, f"Conflicting options {previous} and {option_string}.")
        seen[self.dest] = (option_string, value)
        setattr(namespace, "_connection_options", seen)
        setattr(namespace, self.dest, value)
