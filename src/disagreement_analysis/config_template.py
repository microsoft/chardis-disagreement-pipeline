"""Config template + scaffolding helpers for the disagreement analysis pipeline.

Mirrors :mod:`src.config_template` but for disagreement analysis:
  - sections are ``rationales``, ``synthesizer``, and
    ``disagreement_explainer``
  - YAMLs live under ``configs/disagreement_configs/<project>/...``
  - prompts live under ``prompts/disagreement_prompts/<project>/...``
  - outputs land under ``outputs/disagreement_outputs/<project>/<v>/...``
"""

import os
import sys


def _add_project_root(marker_filename: str = ".gitignore") -> str | None:
    current = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.isfile(os.path.join(current, marker_filename)):
            break
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent
    if sys.path[0] != current:
        sys.path.insert(0, current)
        return current
    return None


_root_added = _add_project_root()

# Heavy imports (numpy, sklearn-derived clustering, sentence-transformers) are
# pulled in by DisagreementExplainer; defer them until build_empty_config is
# actually called so that anyone introspecting the module (CLI, dashboards)
# doesn't pay the import cost / dep cost up front.
from common.src.openai.openai_args import OpenAIArgs
from common.src.configs.schema import public_config

if _root_added:
    sys.path.remove(_root_added)


def _sections() -> tuple:
    """Return ((section_key, factory), ...) — lazy so heavy deps stay lazy."""
    from src.disagreement_analysis.disagreement_explainer import DisagreementExplainer
    return (
        ("rationales_openai", OpenAIArgs),
        ("synthesizer_openai", OpenAIArgs),
        ("disagreement_explainer", DisagreementExplainer),
    )


# Backward-compat alias for callers that imported SECTIONS directly.
SECTIONS = _sections


def _set(config: dict, section: str, field: str, value) -> None:
    sec = config.setdefault(section, {})
    entry = sec.get(field)
    if isinstance(entry, dict):
        entry["value"] = value
    else:
        sec[field] = {"description": "", "value": value}


def _v_label(version) -> str:
    v = version if version is not None else 1
    s = str(v)
    return s if s.lower().startswith("v") else f"v{v}"


def build_empty_config(
    project_name: str | None = None,
    version: int | str | None = None,
) -> dict:
    """Return a fresh disagreement-analysis config dict with sensible defaults."""
    config: dict = {}

    for section, cls in _sections():
        instance = cls()
        instance.load_config(config, section, required=False)

    # Explicit connection settings for both generation stages.
    # Select models and explicitly bind any different credential destination.
    for section in ("rationales_openai", "synthesizer_openai"):
        _set(config, section, "openai_model", "")
        _set(config, section, "openai_endpoint", "")
        _set(config, section, "credential_endpoint", "")
        _set(config, section, "openai_api_version", "")
        _set(config, section, "scope", "")

    # ── Evaluation defaults (used by validation + heldout LLM-as-judge runs).
    # Required: eval_model, num_iterations.
    connection = OpenAIArgs()
    connection.load_config(config, "evaluation", required=False)
    config["evaluation"].pop("openai_model")
    _set(config, "evaluation", "eval_model", "")
    _set(config, "evaluation", "num_iterations", 3)
    _set(config, "evaluation", "scope", "")
    _set(config, "evaluation", "openai_endpoint", "")
    _set(config, "evaluation", "credential_endpoint", "")
    _set(config, "evaluation", "openai_api_version", "")
    _set(config, "evaluation", "num_threads", 15)
    _set(config, "evaluation", "concepts_per_batch", 10)
    _set(config, "evaluation", "top_hypotheses", 50)

    # ── Project- & version-aware paths ────────────────────────────────────
    if project_name is not None:
        v_label = _v_label(version)
        data_dir   = f"data/{project_name}/"
        guidelines = f"prompts/disagreement_prompts/{project_name}/"
        outputs    = f"outputs/disagreement_outputs/{project_name}/{v_label}/"

        _set(config, "disagreement_explainer", "data_path",          data_dir)
        _set(config, "disagreement_explainer", "data_filename",      "annotated_data.csv")
        _set(config, "disagreement_explainer", "guidelines_path",    guidelines)
        _set(config, "disagreement_explainer", "guidelines_filename","prompt.md")
        _set(config, "disagreement_explainer", "examples_path",      "")
        _set(config, "disagreement_explainer", "output_path",        outputs)

        # Stash project + version so downstream consumers can recover them.
        sec = config.setdefault("disagreement_explainer", {})
        sec["project_name"] = {
            "description": "Project name (set by run_disagreement CLI).",
            "value": project_name,
        }
        sec["version"] = {
            "description": "Version label (set by run_disagreement CLI).",
            "value": v_label,
        }

    for section in ("rationales_openai", "synthesizer_openai", "evaluation"):
        config[section].pop("openai_key", None)
    return public_config(config)


def restamp_version_fields(config: dict, project_name: str, version: int | str) -> dict:
    """Re-apply version-aware fields when promoting a config to a new version.

    Keeps ``data_path`` and ``guidelines_path`` pinned to the canonical
    per-project locations and re-stamps ``output_path`` so each saved version
    writes its own outputs under ``disagreement_outputs/<project>/<v>/`` —
    otherwise newer versions would silently overwrite v1's outputs.
    """
    if not project_name:
        return config
    v_label = _v_label(version)

    sec = config.setdefault("disagreement_explainer", {})
    sec["project_name"] = {
        "description": "Project name (set by run_disagreement CLI).",
        "value": project_name,
    }
    sec["version"] = {
        "description": "Version label (set by run_disagreement CLI).",
        "value": v_label,
    }
    # Always re-stamp output_path so each version writes its own subdir.
    _set(
        config,
        "disagreement_explainer",
        "output_path",
        f"outputs/disagreement_outputs/{project_name}/{v_label}/",
    )
    return config


def ensure_pipeline_directories(config: dict) -> list[str]:
    """Create on disk any directories declared in the config that don't yet exist."""
    created: list[str] = []
    sec = config.get("disagreement_explainer") or {}
    for key in ("data_path", "guidelines_path"):
        entry = sec.get(key)
        if not isinstance(entry, dict):
            continue
        path = entry.get("value")
        if not isinstance(path, str) or not path or path == "<TBD>":
            continue
        target = path if path.endswith(("/", os.sep)) else os.path.dirname(path)
        if not target:
            continue
        os.makedirs(target, exist_ok=True)
        created.append(target)

    project_name = None
    version = None
    pn = sec.get("project_name")
    vn = sec.get("version")
    if isinstance(pn, dict):
        project_name = pn.get("value")
    if isinstance(vn, dict):
        version = vn.get("value")
    if project_name and version:
        out_dir = os.path.join(
            "annotation", "outputs", "disagreement_outputs",
            str(project_name), _v_label(version),
        )
        os.makedirs(out_dir, exist_ok=True)
        created.append(out_dir)
    return created
