# Import standard libraries
import os
import sys


def add_project_root(marker_filename: str = ".gitignore") -> str:
    """Locate the project root and add it to sys.path."""
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


_root_added = add_project_root()

from src.annotator_base import Annotator
from src.preprocessor.preprocessor_base import Preprocessor
from src.prompt_builder.prompt_builder_base import PromptBuilder
from common.src.openai.openai_args import OpenAIArgs
from common.src.configs.schema import public_config

if _root_added:
    sys.path.remove(_root_added)


# Ordered mapping of config sections → factory. Order is preserved in the
# scaffolded YAML file so the user sees `llm` first, then `annotation`, etc.
SECTIONS = (
    ("openai", OpenAIArgs),
    ("annotation", Annotator),
    ("prompt_builder", PromptBuilder),
    ("preprocessor", Preprocessor),
)


def _discover_concepts(project_name: str) -> list:
    """Scan prompts/<project>/concepts/conceptual_definitions/ for
    concept folders and return a list of {name, version} dicts using the
    highest version found in each folder. Returns [] if nothing is found.
    """
    import re
    base = os.path.join("prompts", project_name,
                        "concepts", "conceptual_definitions")
    if not os.path.isdir(base):
        return []
    out = []
    for name in sorted(os.listdir(base)):
        sub = os.path.join(base, name)
        if not os.path.isdir(sub):
            continue
        # Files look like {name}_{version}.md  (e.g. positive_sentiment_1.0.md)
        versions = []
        prefix = name + "_"
        for f in os.listdir(sub):
            if f.startswith(prefix) and f.endswith(".md"):
                v = f[len(prefix):-3]
                if re.match(r"^\d+(\.\d+)*$", v):
                    versions.append(v)
        if not versions:
            continue
        versions.sort(key=lambda s: tuple(int(p) for p in s.split(".")))
        out.append({"name": name, "version": versions[-1]})
    return out


def _set(config: dict, section: str, field: str, value) -> None:
    """Set a value in a {section: {field: {value, description}}} config dict."""
    sec = config.setdefault(section, {})
    entry = sec.get(field)
    if isinstance(entry, dict):
        entry["value"] = value
    else:
        sec[field] = {"description": "", "value": value}


def build_empty_config(
    project_name: str | None = None,
    version: int | str | None = None,
) -> dict:
    """
    Build a fresh config dict containing every ConfigValue field defined by the
    pipeline's configurable classes, with their default values and help text.

    The returned dict matches the {section: {field: {description, value}}}
    layout consumed by the existing orchestrator, with friendlier user-facing
    key names in the llm section (`model`, `endpoint`, …).
    """
    config: dict = {}

    for section, cls in SECTIONS:
        instance = cls()
        # load_config({}, section) populates `config` in-place with the
        # full {description, value} schema for every ConfigValue attribute,
        # using each field's default. We deliberately ignore the returned
        # error flag here — required fields are expected to be empty in a
        # freshly-scaffolded template.
        instance.load_config(config, section, required=False)

    # Explicit annotation connection settings.
    # Select a model and explicitly bind any different credential destination.
    _set(config, "openai", "openai_model", "")
    _set(config, "openai", "openai_endpoint", "")
    _set(config, "openai", "credential_endpoint", "")
    _set(config, "openai", "openai_api_version", "")
    _set(config, "openai", "scope", "")
    _set(config, "annotation", "data_path", "")

    # ── Project- & version-aware paths and topic ──────────────────────────
    if project_name is not None:
        v = version if version is not None else 1
        v_label = f"v{v}" if not str(v).lower().startswith("v") else str(v)
        out_dir    = f"outputs/{project_name}/{v_label}/"
        out_file   = f"{out_dir}{project_name}_{v_label}.csv"
        raw_dir    = f"outputs/{project_name}/{v_label}/raw/"
        topic      = f"{project_name}"
        built_name = f"{project_name}_{v_label}"

        _set(config, "annotation",     "output_path",      out_file)
        _set(config, "annotation",     "raw_output_path",  raw_dir)
        _set(config, "prompt_builder", "annotation_topic", topic)
        _set(config, "prompt_builder", "built_prompt_name", built_name)

        # Auto-discover concepts from prompts/<project>/concepts/
        # conceptual_definitions/. Each subfolder = one simple concept; pick
        # the highest-versioned markdown file inside. This makes a fresh
        # config compile out-of-box if the prompt files already exist.
        discovered = _discover_concepts(project_name)
        if discovered:
            _set(config, "prompt_builder", "concepts", discovered)

        # Stash the project name + version so downstream consumers can recover them.
        config.setdefault("annotation", {})["project_name"] = {
            "description": "Project name (set by the disagreement CLI).",
            "value": project_name,
        }
        config["annotation"]["version"] = {
            "description": "Version label (set by the disagreement CLI).",
            "value": v_label,
        }

    config["openai"].pop("openai_key", None)
    return public_config(config)


def restamp_version_fields(config: dict, project_name: str, version: int | str) -> dict:
    """Re-apply project/version-aware path fields to an existing config.

    Used when promoting a config to a new version (e.g. v1 → v2 after dashboard
    edits): updates output_path, raw_output_path, built_prompt_name, and the
    stored version label so the new YAML doesn't keep pointing at the old
    version's directories.
    """
    if not project_name:
        return config
    v = version if version is not None else 1
    v_label = f"v{v}" if not str(v).lower().startswith("v") else str(v)
    out_dir  = f"outputs/{project_name}/{v_label}/"
    out_file = f"{out_dir}{project_name}_{v_label}.csv"
    raw_dir  = f"outputs/{project_name}/{v_label}/raw/"
    built    = f"{project_name}_{v_label}"

    _set(config, "annotation",     "output_path",      out_file)
    _set(config, "annotation",     "raw_output_path",  raw_dir)
    _set(config, "prompt_builder", "built_prompt_name", built)

    config.setdefault("annotation", {})["project_name"] = {
        "description": "Project name (set by the disagreement CLI).",
        "value": project_name,
    }
    config["annotation"]["version"] = {
        "description": "Version label (set by the disagreement CLI).",
        "value": v_label,
    }
    return config


def stamp_judge_subdir(config: dict, project_name: str, version: int | str, model_short: str) -> dict:
    """Re-stamp output paths into a per-judge subdirectory.

    Used by Panel of Judges: each model in the panel writes to its own
    subfolder (outputs/<project>/<v>/<model_short>/) and gets a
    distinct built_prompt_name so per-judge artifacts don't collide.
    """
    if not project_name or not model_short:
        return config
    v = version if version is not None else 1
    v_label = f"v{v}" if not str(v).lower().startswith("v") else str(v)
    out_dir  = f"outputs/{project_name}/{v_label}/{model_short}/"
    out_file = f"{out_dir}{project_name}_{v_label}__{model_short}.csv"
    raw_dir  = f"outputs/{project_name}/{v_label}/{model_short}/raw/"
    built    = f"{project_name}_{v_label}__{model_short}"

    _set(config, "annotation",     "output_path",      out_file)
    _set(config, "annotation",     "raw_output_path",  raw_dir)
    _set(config, "prompt_builder", "built_prompt_name", built)
    return config


def ensure_pipeline_directories(config: dict) -> list[str]:
    """
    Create on disk any annotation paths declared in the config that don't yet
    exist. Returns the list of directories that were ensured (created or
    already present).

    Paths in the config are interpreted as repo-relative (e.g. ``outputs/...``)
    or absolute.
    """
    created: list[str] = []
    sec = config.get("annotation") or {}
    for key in ("data_path", "output_path", "raw_output_path"):
        entry = sec.get(key)
        if not isinstance(entry, dict):
            continue
        path = entry.get("value")
        if not isinstance(path, str) or not path or path == "<TBD>":
            continue
        # Treat as directory if it ends with a separator, otherwise mkdir its parent.
        target = path if path.endswith(("/", os.sep)) else os.path.dirname(path)
        if not target:
            continue
        os.makedirs(target, exist_ok=True)
        created.append(target)
    return created
