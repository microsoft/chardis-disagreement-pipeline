# Replication configuration index

This directory contains 88 converted research configurations copied from the original release. The source files remain unchanged. All seven dataset families retain their sample, versioned, and legacy configurations.

| Dataset folder | YAML files |
| --- | --- |
| [dices](dices/) | 8 |
| [ghc-real](ghc-real/) | 14 |
| [ghc-syn](ghc-syn/) | 23 |
| [md](md/) | 8 |
| [mhsc](mhsc/) | 4 |
| [pref](pref/) | 8 |
| [yelp](yelp/) | 23 |

## Schema and preserved settings

The public schema follows the [bundled reference](../code-repo-final/configs/disagreement_configs/ghc-syn/ghc-syn_v1.yaml).

| Historical field | Public field |
| --- | --- |
| rationales_openai | rationales |
| synthesizer_openai | synthesizer |
| openai_model and eval_model | model |
| openai_api_version | api_version |
| openai_endpoint and base_url | endpoint |
| openai_key and api_key | key |

Temperatures, seeds, token limits, sampling, threading, reasoning, hypothesis limits, and other experimental settings retain their original values. Extra experimental fields remain present even when absent from the bundled reference.

Nine legacy configs did not contain evaluation settings. No historical evaluation settings were invented for them. The conversion report lists every missing reference field and every extra field.

Public model identifiers `o3` and `deepseek-r1` remain unchanged. Deployment aliases were removed from 223 model fields without inferring replacement model versions. Endpoint, credential destination, key, and legacy scope values are empty. Newly added provider, authentication, environment reference, role, and timeout fields are operational defaults from the reference, not evidence of historical settings.

The [conversion report](../outputs/config_conversion.json) records source and destination checksums, added fields, schema differences, and model fields needing configuration. It contains no removed deployment values.

## Paths and execution limits

Use `code-repo-final` as the working directory when interpreting these archived configs.

Guidelines point to `../replication_prompts/<dataset>/prompt.md`. Future live runs write to `../pipeline_run_outputs/<dataset>/<config>/`, separate from packaged evidence and generated replication comparisons.

Original research datasets are absent, so data paths are empty. Missing external examples remain unresolved. Do not substitute the filtered demonstration datasets for the historical experiment inputs.

These are incomplete configurations for future live experiments. Supply original data and your own model connections before executing them. Offline cached result replication does not load these configs and makes no model requests.

Config version numbers do not establish output run identity. Several output conditions remain blocked pending an explicit mapping. See [REPLICATION.md](../REPLICATION.md) for executable coverage.
