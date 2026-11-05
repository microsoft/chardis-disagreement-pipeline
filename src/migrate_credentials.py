"""Explicitly replace one plaintext credential after its environment copy is verified."""

import argparse
import hmac
import os

from common.src.configs.config_utils import load_yaml_config, save_yaml_config
from common.src.configs.schema import section_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--binding", action="append", required=True, metavar="SECTION=ENV")
    parser.add_argument("--confirm-removal", action="store_true", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    for binding in args.binding:
        name, separator, variable = binding.partition("=")
        name = section_name(name)
        if not separator or name not in config:
            raise ValueError("Binding must have the form SECTION=ENV.")
        section = config[name]
        matches = [key for key in ("key", "openai_key", "api_key")
                   if isinstance(section.get(key), dict) and section[key].get("value")]
        if len(matches) != 1:
            raise ValueError("Each selected section must have exactly one plaintext credential.")
        secret = section[matches[0]]["value"]
        if not hmac.compare_digest(str(secret), os.environ.get(variable, "")):
            raise ValueError("Provision the named environment variable with the existing key before removal.")
        section[matches[0]]["value"] = ""
        section["api_key_env"] = {"description": "Credential environment reference", "value": variable}
    # Other secret-bearing sections must be migrated together before a safe save.
    remaining = []
    for name, fields in config.items():
        if isinstance(fields, dict):
            for key in ("key", "openai_key", "api_key"):
                item = fields.get(key)
                if isinstance(item, dict) and item.get("value"):
                    remaining.append((name, key, item["value"]))
    if remaining:
        raise ValueError("Supply bindings for every secret-bearing section together. Original unchanged.")
    save_yaml_config(config, args.config)
    print("Credential reference saved. Configure explicit provider, authentication, and destination bindings.")


if __name__ == "__main__":
    main()
