"""Interactive driver for the disagreement analysis pipeline.

Invoked by ``run_disagreement.sh``. Walks the user through:

  1. project-name / version selection
  2. silently scaffolding a config if it doesn't exist
  3. opening the Disagreement Analysis settings dashboard
  4. saving changes as a new auto-incremented version
  5. invoking the disagreement orchestrator in-process

Use ``--skip-setup`` to jump straight to step 5 with the resolved config.
"""

from __future__ import annotations

import argparse
import os
import re
import sys


def _add_project_root() -> str | None:
    current = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.isfile(os.path.join(current, ".gitignore")):
            break
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent
    if current not in sys.path:
        sys.path.insert(0, current)
    return current


_add_project_root()

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table


CONFIGS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "configs", "disagreement_configs")
)


def _version_re(project_name: str) -> re.Pattern[str]:
    return re.compile(
        rf"^(?:{re.escape(project_name)}_)?v(\d+)\.yaml$",
        re.IGNORECASE,
    )


# ── Project / version resolution ────────────────────────────────────────────


def _list_projects() -> list[str]:
    if not os.path.isdir(CONFIGS_ROOT):
        return []
    return sorted(
        name for name in os.listdir(CONFIGS_ROOT)
        if os.path.isdir(os.path.join(CONFIGS_ROOT, name)) and not name.startswith(".")
    )


def _project_dir(project_name: str) -> str:
    return os.path.join(CONFIGS_ROOT, project_name)


def _list_versions(project_name: str) -> list[int]:
    pdir = _project_dir(project_name)
    if not os.path.isdir(pdir):
        return []
    pattern = _version_re(project_name)
    versions: list[int] = []
    for name in os.listdir(pdir):
        m = pattern.match(name)
        if m:
            versions.append(int(m.group(1)))
    return sorted(set(versions))


def _next_version(project_name: str) -> int:
    versions = _list_versions(project_name)
    return (versions[-1] + 1) if versions else 1


def _config_path(project_name: str, version: int) -> str:
    pdir = _project_dir(project_name)
    new_path = os.path.join(pdir, f"{project_name}_v{version}.yaml")
    legacy_path = os.path.join(pdir, f"v{version}.yaml")
    if os.path.isfile(new_path):
        return new_path
    if os.path.isfile(legacy_path):
        return legacy_path
    return new_path


def _print_existing_projects(console: Console) -> None:
    projects = _list_projects()
    if not projects:
        console.print("[dim]No existing projects yet — your input will create a new one.[/dim]")
        return
    table = Table(
        title="Existing disagreement projects",
        title_style="bold cyan", show_header=True, header_style="bold",
        box=None, padding=(0, 2),
    )
    table.add_column("#", style="bold yellow", justify="right")
    table.add_column("Project", style="cyan")
    table.add_column("Versions", style="dim")
    for i, p in enumerate(projects, start=1):
        vs = _list_versions(p)
        vs_str = ", ".join(f"v{v}" for v in vs) if vs else "—"
        table.add_row(str(i), p, vs_str)
    console.print(table)
    console.print(
        "[dim]Type a number to select an existing project, a name to load it, "
        "or a new name to create one.[/dim]"
    )


def _resolve_project_input(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    if raw.isdigit():
        projects = _list_projects()
        idx = int(raw)
        if 1 <= idx <= len(projects):
            return projects[idx - 1]
    return raw


def _print_existing_versions(console: Console, project_name: str) -> None:
    versions = _list_versions(project_name)
    if not versions:
        console.print(
            f"[dim]No versions for [cyan]{project_name}[/cyan] yet — Enter creates v1.[/dim]"
        )
        return
    table = Table(
        title=f"Versions for {project_name}",
        title_style="bold cyan", show_header=True, header_style="bold",
        box=None, padding=(0, 2),
    )
    table.add_column("#", style="bold yellow", justify="right")
    table.add_column("Version", style="cyan")
    for i, v in enumerate(versions, start=1):
        marker = "  [dim](latest)[/dim]" if v == versions[-1] else ""
        table.add_row(str(i), f"v{v}{marker}")
    console.print(table)
    console.print(
        "[dim]Type a number to select, a version like 'v3' or '3', "
        "'latest', or Enter to create new.[/dim]"
    )


def _resolve_version_input(project_name: str, raw: str) -> str:
    raw = raw.strip()
    if raw.isdigit():
        versions = _list_versions(project_name)
        idx = int(raw)
        if 1 <= idx <= len(versions):
            return str(versions[idx - 1])
    return raw


def _resolve_version(project_name: str, requested: str | None) -> tuple[int, bool]:
    if requested in (None, "", "latest"):
        existing = _list_versions(project_name)
        if existing:
            return existing[-1], False
        return 1, True
    if str(requested).lower() == "new":
        return _next_version(project_name), True
    try:
        v = int(str(requested).lstrip("v"))
    except ValueError:
        raise SystemExit(f"Invalid --version value: {requested!r}")
    return v, not os.path.isfile(_config_path(project_name, v))


# ── Main flow ───────────────────────────────────────────────────────────────


def _scaffold_config_if_missing(path: str, project_name: str, version: int) -> bool:
    if os.path.isfile(path):
        return False
    from src.disagreement_analysis.config_template import build_empty_config
    from common.src.configs import config_utils

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg = build_empty_config(project_name=project_name, version=version)
    config_utils.save_yaml_config(cfg, path)
    return True


def _wait_for_install(console: Console) -> None:
    flag = os.environ.get("CHARDIS_INSTALL_DONE")
    log = os.environ.get("CHARDIS_INSTALL_LOG", "/tmp/chardis_disagreement_install.log")
    if not flag:
        return
    import time
    with console.status("[cyan]Finishing dependency install...[/cyan]"):
        deadline = time.time() + 600
        while time.time() < deadline:
            if os.path.isfile(flag):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Dependency installation timed out. See {log}")
    with open(flag, encoding="utf-8") as file:
        status = file.read().strip()
    if status != "0":
        raise RuntimeError(f"Dependency installation failed or returned an invalid status. See {log}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="run_disagreement")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip the dashboard and run without confirmation. Supply --project to skip its prompt.")
    parser.add_argument("--project", help="Project name (skip the prompt).")
    parser.add_argument("--version", help="Version number, 'latest', or 'new'.")
    parser.add_argument("--allow-preview-root", action="append", default=[],
                        help="Explicitly allow a directory for this dashboard launch.")
    args = parser.parse_args()

    console = Console()

    # ── Step 1: project + version ────────────────────────────────────────
    project_name = args.project
    while not project_name:
        raw = Prompt.ask(
            "[bold cyan]Project name[/bold cyan] [dim](type '?' to list existing, or pick a number)[/dim]"
        ).strip()
        if raw in ("?", "ls", "list"):
            _print_existing_projects(console)
            continue
        if not raw:
            console.print("[yellow]Project name cannot be empty.[/yellow]")
            continue
        resolved = _resolve_project_input(raw)
        if resolved is None:
            continue
        if raw.isdigit() and resolved == raw:
            console.print(f"[yellow]No project at index {raw}. Type '?' to list.[/yellow]")
            continue
        if raw.isdigit():
            console.print(f"[dim]→ Selected [cyan]{resolved}[/cyan][/dim]")
        project_name = resolved

    version_input = args.version
    if version_input is None and not args.skip_setup:
        existing = _list_versions(project_name)
        if existing:
            hint = f"latest = v{existing[-1]} · Enter = create new · '?' to list"
        else:
            hint = "Enter = create v1 · '?' to list"
        while True:
            version_input = Prompt.ask(
                f"[bold cyan]Version[/bold cyan] (number, 'latest', or 'new')  [dim]{hint}[/dim]",
                default="new", show_default=False,
            )
            stripped = str(version_input).strip()
            if stripped in ("?", "ls", "list"):
                _print_existing_versions(console, project_name)
                continue
            if stripped.isdigit():
                resolved = _resolve_version_input(project_name, stripped)
                if resolved == stripped:
                    versions = _list_versions(project_name)
                    if versions:
                        console.print(
                            f"[yellow]No version at index {stripped}. Type '?' to list.[/yellow]"
                        )
                        continue
                else:
                    console.print(f"[dim]→ Selected v{resolved}[/dim]")
                    version_input = resolved
            break

    version, _is_new = _resolve_version(project_name, version_input)
    cfg_path = _config_path(project_name, version)

    # ── Step 2: install gate ─────────────────────────────────────────────
    _wait_for_install(console)

    # Now safe to import heavy modules.
    from src.disagreement_analysis import disagreement_orchestrator
    from src.disagreement_analysis.config_template import (
        build_empty_config,
        ensure_pipeline_directories,
        restamp_version_fields,
    )
    from src.dashboard.disagreement_dashboard import run_dashboard
    from common.src.configs import config_utils
    from common.src.configs.schema import runtime_config

    created = _scaffold_config_if_missing(cfg_path, project_name, version)

    # ── Step 3: dashboard (skippable) ────────────────────────────────────
    if not args.skip_setup:
        if created:
            prompt_msg = (
                f"Project [cyan]{project_name}[/cyan] (v{version}) not found. "
                f"Create new project and open Disagreement Analysis dashboard?"
            )
        else:
            prompt_msg = (
                f"Open Disagreement Analysis settings dashboard for "
                f"[cyan]{project_name}[/cyan] (v{version})?"
            )

        if Confirm.ask(prompt_msg, default=True, show_default=False):
            cfg = config_utils.load_yaml_config(cfg_path)
            # Backfill any new ConfigValue fields added since this YAML was first written.
            defaults = runtime_config(build_empty_config(project_name=project_name, version=version))
            for sec, fields in defaults.items():
                if not isinstance(fields, dict):
                    continue
                cfg.setdefault(sec, {})
                _section_values(cfg, sec)
                for fkey, fmeta in fields.items():
                    if fkey not in cfg[sec] and isinstance(fmeta, dict) and "value" in fmeta:
                        cfg[sec][fkey] = fmeta

            def save_dashboard_config(updated: dict) -> None:
                nonlocal cfg_path, version
                new_version = _next_version(project_name)
                new_path = _config_path(project_name, new_version)
                restamp_version_fields(updated, project_name, new_version)
                _validate_save_config(updated)
                label = Prompt.ask(
                    f"[bold]Save as new version v{new_version}.[/bold] "
                    f"Optional short note (blank = none)",
                    default="",
                )
                if label:
                    updated.setdefault("disagreement_explainer", {})["version_note"] = {
                        "description": "Free-text note describing this saved version.",
                        "value": label,
                    }
                config_utils.save_yaml_config(updated, new_path)
                cfg_path = new_path
                version = new_version
                console.print(f"[green]✓ Saved {cfg_path}[/green]")

            result = run_dashboard(
                cfg, project_name, str(version), on_save=save_dashboard_config,
                approved_roots=tuple(args.allow_preview_root),
            )
            if result.cancelled:
                console.print("Run cancelled. No pipeline stages were launched.")
                return
            if not result.dirty:
                console.print("[dim]No changes — keeping current version.[/dim]")
            if result.action != "run":
                console.print("Settings saved. No pipeline stages were launched.")
                return
        else:
            console.print("Dashboard declined. Use --skip-setup for an explicit batch run.")
            return

    # ── Step 4: hand off to the orchestrator ────────────────────────────
    final_cfg = config_utils.load_yaml_config(cfg_path)
    _validate_run_config(final_cfg)
    stages = _stage_plan(final_cfg)
    _print_run_summary(console, final_cfg, cfg_path, project_name, version, stages)
    ensured = ensure_pipeline_directories(final_cfg)
    if ensured:
        for d in ensured:
            console.print(f"[dim]✓ Ensured directory: {d}[/dim]")

    disagreement_orchestrator.run(cfg_path)

    # ── Step 5/6: hypothesis evaluation on validation + held-out splits ────
    try:
        _run_eval_pipeline(console, cfg_path, project_name, str(version))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Evaluation pipeline failed: {exc}[/red]")
        raise
    _print_output_summary(console, final_cfg, stages)


def _resolve_run_dir(output_path: str) -> str:
    """Resolve the explainer's output_path the same way disagreement_explainer does."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if not output_path:
        raise ValueError("disagreement_explainer.output_path is empty in config.")
    if os.path.isabs(output_path):
        return output_path
    return os.path.normpath(os.path.join(repo_root, output_path))


def _eval_settings_from_cfg(cfg: dict) -> dict:
    """Pull the ``evaluation`` section into a flat dict of values."""
    from common.src.configs.schema import runtime_config
    cfg = runtime_config(cfg)
    sec = cfg.get("evaluation", {}) or {}
    out: dict = {}
    for k, meta in sec.items():
        if isinstance(meta, dict) and "value" in meta:
            out[k] = meta["value"]
        else:
            out[k] = meta
    return out


def _section_values(cfg: dict, name: str) -> dict:
    from common.src.configs.schema import PUBLIC_FIELDS, runtime_config, section_name
    cfg = runtime_config(cfg)
    name = section_name(name)
    label = section_name(name, public=True)
    section = cfg.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"Configuration section {label} must be a mapping.")
    values = {}
    for field, entry in section.items():
        if isinstance(entry, dict) and "value" in entry:
            values[field] = entry["value"]
        elif name == "evaluation" and not isinstance(entry, dict):
            values[field] = entry
        else:
            raise ValueError(f"Configuration field {label}.{PUBLIC_FIELDS.get(field, field)} needs a value entry.")
    return values


def _stage_plan(cfg: dict) -> dict[str, tuple[str, str, str]]:
    from common.src.utils import artifacts
    de = _section_values(cfg, "disagreement_explainer")
    ev = _section_values(cfg, "evaluation")
    run_dir = _resolve_run_dir(de.get("output_path", ""))
    explanations = os.path.join(run_dir, "disagreement_explanations.json")
    discovery_reuse = artifacts.reusable(explanations, artifacts.discovery_identity(cfg)) if os.path.exists(explanations) else False
    stages = {
        "Discovery": (
            "reuse" if discovery_reuse else "run",
            explanations,
            "Existing hypotheses skip discovery" if os.path.isfile(explanations) else "Generate hypotheses",
        ),
    }
    from common.src.utils.artifacts import model_directory
    model = model_directory(str(ev.get("eval_model") or "unconfigured"))
    for stage, folder, flag in (
        ("Validation", "validation", "CHARDIS_SKIP_VALIDATION"),
        ("Heldout", "heldout", "CHARDIS_SKIP_HELDOUT"),
    ):
        guard = os.path.join(run_dir, folder, model, "overall_predictiveness_logit.csv")
        if not model:
            action, reason = "skip", "No evaluation model configured"
        elif os.environ.get(flag, "").lower() in ("1", "true", "yes"):
            action, reason = "skip", f"{flag} is set"
        elif os.path.isfile(guard):
            artifacts.reusable(guard, artifacts.evaluation_identity(cfg, folder))
            action, reason = "reuse", "Verified ranking skips this stage"
        else:
            action, reason = "run", "Evaluate hypotheses"
        stages[stage] = (action, guard, reason)
    significant = os.path.join(run_dir, "validation", model, "significant_disagreement_explanations.csv")
    if stages["Heldout"][0] == "run":
        if stages["Validation"][0] != "run" and not os.path.isfile(significant):
            stages["Heldout"] = ("skip", stages["Heldout"][1], f"Missing prerequisite {significant}")
        else:
            stages["Heldout"] = (
                "run", stages["Heldout"][1],
                f"Requires validation annotations and significant hypotheses at {significant}",
            )
    return stages


def _validate_save_config(cfg: dict) -> None:
    from common.src.configs.config_utils import reject_plaintext_keys
    from common.src.openai.connections import endpoint, normalized, validate
    from common.src.configs.schema import section_name
    reject_plaintext_keys(cfg)
    for name in ("rationales_openai", "synthesizer_openai", "evaluation", "disagreement_explainer"):
        values = _section_values(cfg, name)
        name = section_name(name, public=True)
        if name != "disagreement_explainer":
            values = normalized(values)
            if values.get("provider") not in {"openai", "compatible", "local"}:
                raise ValueError(f"{name} requires an explicit supported provider.")
            if values.get("auth_mode") not in {"api-key", "none"}:
                raise ValueError(f"{name} requires explicit auth_mode.")
            if values.get("openai_endpoint"):
                endpoint(values["openai_endpoint"])
            if values.get("credential_endpoint"):
                endpoint(values["credential_endpoint"])
            probe = dict(values)
            probe["openai_model"] = values.get("openai_model") or "incomplete"
            probe["openai_endpoint"] = values.get("openai_endpoint") or (
                "http://127.0.0.1" if values["provider"] == "local" else "https://example.invalid")
            probe["credential_endpoint"] = probe["openai_endpoint"]
            probe["credential_provider"] = values["provider"]
            probe["api_key_env"] = values.get("api_key_env") or "INCOMPLETE"
            probe["scope"] = values.get("scope") or "incomplete"
            probe["openai_api_version"] = values.get("openai_api_version") or "incomplete"
            validate(probe, live=False)
        for field in ("num_iterations", "num_threads", "concepts_per_batch", "top_hypotheses", "example_size"):
            if field in values and (type(values[field]) is not int or values[field] <= 0):
                raise ValueError(f"{name}.{field} must be a positive integer.")


def _validate_run_config(cfg: dict) -> None:
    from common.src.openai.openai_args import OpenAIArgs
    from src.disagreement_analysis.disagreement_explainer import DisagreementExplainer

    _validate_save_config(cfg)
    de = _section_values(cfg, "disagreement_explainer")
    ev = _section_values(cfg, "evaluation")
    for name in ("rationales_openai", "synthesizer_openai"):
        _section_values(cfg, name)
    stages = _stage_plan(cfg)
    explainer = DisagreementExplainer()

    def require_file(field: str, path: str) -> None:
        if not path:
            raise ValueError(f"Set {field} to a readable input file.")
        try:
            with open(path, "rb"):
                pass
        except OSError as exc:
            raise ValueError(f"Cannot read {field} at {path}. {exc.strerror}") from None

    data = explainer._resolve_combined_path(de.get("data_path"), de.get("data_filename", "annotated_data.csv"))
    require_file("disagreement_explainer.data_path", data)
    run_dir = _resolve_run_dir(de.get("output_path", ""))
    if os.path.exists(run_dir) and not os.path.isdir(run_dir):
        raise ValueError(f"Output path must be a directory. {run_dir}")

    def require_connection(name: str, values: dict) -> None:
        from common.src.openai.connections import normalized, validate
        from common.src.configs.schema import section_name
        try:
            validate(normalized(values), live=True)
        except ValueError as exc:
            raise ValueError(f"{section_name(name, public=True)}: {exc}") from None

    if stages["Discovery"][0] == "run":
        guidelines = explainer._resolve_combined_path(
            de.get("guidelines_path"), de.get("guidelines_filename", "prompt.md"),
        )
        require_file("disagreement_explainer.guidelines_path", guidelines)
        if de.get("examples_path"):
            require_file("disagreement_explainer.examples_path", de["examples_path"])
        for field, default in (("example_size", 5), ("num_annotations", -1), ("random_state", 42), ("num_threads", 10)):
            value = de.get(field, default)
            if type(value) is not int or (field == "example_size" and value <= 0):
                raise ValueError(f"Invalid disagreement_explainer.{field}. Expected an integer"
                                 + (" greater than zero." if field == "example_size" else "."))
        for field in ("more_than_2", "train_on_test"):
            if type(de.get(field, False)) is not bool:
                raise ValueError(f"disagreement_explainer.{field} must be true or false.")
        for name in ("rationales_openai", "synthesizer_openai"):
            require_connection(name, _section_values(cfg, name))
    if any(stages[name][0] == "run" for name in ("Validation", "Heldout")):
        require_connection("evaluation", ev)
        for field in ("concepts_per_batch", "top_hypotheses", "heldout_top_hypotheses", "num_iterations", "num_threads"):
            value = ev.get(field)
            if value not in (None, "", 0):
                try:
                    valid = not isinstance(value, bool) and int(value) > 0 and str(int(value)) == str(value)
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    raise ValueError(f"evaluation.{field} must be a positive integer.")
        if ev.get("top_ks") is not None:
            values = ev["top_ks"]
            if not isinstance(values, (list, tuple)):
                raise ValueError("evaluation.top_ks must be a list of integers.")
            for value in values:
                try:
                    valid = not isinstance(value, bool) and str(int(value)) == str(value)
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    raise ValueError("evaluation.top_ks must be a list of integers.")
        if stages["Discovery"][0] == "reuse":
            folders = [("validation", "validation.csv", "validation_annotate.csv")]
            if stages["Heldout"][0] == "run":
                folders.append(("heldout", "test.csv", "test_annotate.csv"))
            for folder, source, judge in folders:
                require_file(f"{folder} source data", os.path.join(run_dir, folder, source))
                if stages["Validation" if folder == "validation" else "Heldout"][0] == "run":
                    require_file(f"{folder} judge data", os.path.join(run_dir, folder, judge))


def _print_run_summary(console: Console, cfg: dict, cfg_path: str, project: str,
                       version: int, stages: dict[str, tuple[str, str, str]]) -> None:
    from common.src.configs.schema import section_name
    de = _section_values(cfg, "disagreement_explainer")
    console.print(f"\nRun {project} v{version}", markup=False)
    console.print(f"Configuration  {os.path.abspath(cfg_path)}", markup=False)
    for name in ("rationales_openai", "synthesizer_openai"):
        values = _section_values(cfg, name)
        console.print(f"{section_name(name, public=True)}  {values.get('openai_model', '(unset)')}", markup=False)
    console.print(f"Evaluation model  {_section_values(cfg, 'evaluation').get('eval_model') or '(unset)'}", markup=False)
    from src.disagreement_analysis.disagreement_explainer import DisagreementExplainer
    explainer = DisagreementExplainer()
    for field, filename in (("data_path", "data_filename"), ("guidelines_path", "guidelines_filename")):
        default = "annotated_data.csv" if field == "data_path" else "prompt.md"
        path = explainer._resolve_combined_path(de.get(field), de.get(filename, default))
        console.print(f"{field}  {os.path.abspath(path) if path else '(unset)'}", markup=False)
    console.print(f"Outputs  {_resolve_run_dir(de.get('output_path', ''))}", markup=False)
    if de.get("train_on_test"):
        console.print("WARNING  Debug mode reuses data across splits. Results are not independent heldout evaluation.")
    for name, (action, artifact, reason) in stages.items():
        console.print(f"{name}  {action.upper()}  {reason}\n  {artifact}", markup=False)
    console.print("Existing judge response caches may be reused. No cached files will be deleted.")


def _print_output_summary(console: Console, cfg: dict,
                          stages: dict[str, tuple[str, str, str]]) -> None:
    console.print("\nOutput files")
    for name, (action, path, reason) in stages.items():
        if os.path.isfile(path):
            state = "Available" if action == "run" else "Existing"
        else:
            state = "Not produced"
        console.print(f"{name}  {state}  {path}", markup=False)
        if action == "skip":
            console.print(f"  Skipped  {reason}", markup=False)
    run_dir = _resolve_run_dir(_section_values(cfg, "disagreement_explainer").get("output_path", ""))
    from common.src.utils.artifacts import model_directory
    model = model_directory(str(_section_values(cfg, "evaluation").get("eval_model") or "unconfigured"))
    for label, path in (
        ("Validation hypotheses", os.path.join(run_dir, "validation", model, "significant_disagreement_explanations.csv")),
        ("Heldout min aggregation", os.path.join(run_dir, "heldout", model, "min", "overall_predictiveness_logit.csv")),
        ("Metadata", os.path.join(run_dir, "metadata.json")),
        ("Run log", os.path.join(run_dir, "run_log.txt")),
        ("Setup log", os.environ.get("CHARDIS_INSTALL_LOG", "")),
    ):
        if path:
            state = "Available" if os.path.isfile(path) else "Absent"
            console.print(f"{label}  {state}  {path}", markup=False)
    console.print("Rows labeled 'all (validation set only)' are not heldout results.")


def _run_eval_pipeline(console, cfg_path: str, project: str, v_label: str) -> None:
    from common.src.configs.config_utils import load_yaml_config
    from common.src.utils.artifacts import writer
    cfg = load_yaml_config(cfg_path)
    root = _resolve_run_dir(_section_values(cfg, "disagreement_explainer")["output_path"])
    with writer(root):
        _run_eval_pipeline_unlocked(console, cfg_path, project, v_label)


def _run_eval_pipeline_unlocked(console, cfg_path: str, project: str, v_label: str) -> None:
    """Stages [5] (validation) and [6] (held-out) — port of run_pref.sh."""
    import math

    from common.src.configs import config_utils
    from src.disagreement_analysis.prepare_disagreement_eval_config import (
        prepare_eval_config,
    )
    from src import annotation_orchestrator
    from src.disagreement_analysis import evaluate as eval_mod
    from common.src.utils import artifacts
    from pathlib import Path

    cfg = config_utils.load_yaml_config(cfg_path) or {}
    de_section = cfg.get("disagreement_explainer", {}) or {}
    output_path = de_section.get("output_path", {})
    if isinstance(output_path, dict):
        output_path = output_path.get("value", "")
    run_dir = _resolve_run_dir(output_path)

    explanations_file = os.path.join(run_dir, "disagreement_explanations.json")
    if not os.path.isfile(explanations_file):
        raise ValueError("Evaluation requires completed discovery artifacts.")
    artifacts.reusable(explanations_file, artifacts.discovery_identity(cfg))

    eval_settings = _eval_settings_from_cfg(cfg)
    if not eval_settings.get("eval_model"):
        raise ValueError("Set evaluation.model before evaluation.")

    run_name = project  # legacy: annotation_topic / concept name prefix uses the project name
    eval_model = artifacts.model_directory(str(eval_settings["eval_model"]))
    concepts_per_batch = int(eval_settings.get("concepts_per_batch", 10) or 10)
    top_hypotheses = int(eval_settings.get("top_hypotheses", 50) or 50)
    # Optional separate cap for the held-out stage. When unset, falls back to
    # `top_hypotheses` so older configs (v1-v16) behave identically. When set
    # (e.g. v17), allows validation to label top_hypotheses (in JSON order) and
    # held-out to label only heldout_top_hypotheses (top-N by validation F1).
    heldout_top_hypotheses_raw = eval_settings.get("heldout_top_hypotheses", None)
    if heldout_top_hypotheses_raw in (None, "", 0):
        heldout_top_hypotheses = top_hypotheses
    else:
        heldout_top_hypotheses = int(heldout_top_hypotheses_raw)
    val_max_batches = max(1, int(math.ceil(top_hypotheses / concepts_per_batch)))
    heldout_max_batches = max(1, int(math.ceil(heldout_top_hypotheses / concepts_per_batch)))

    # Held-out logits sweep configuration. `evaluation.top_ks` is a list of
    # k values for top-k feature-subset logits (in addition to all_significant
    # / all_significant_corrected which are always emitted via the -1, -2
    # sentinels). Falls back to the legacy default if missing so older configs
    # behave unchanged.
    raw_top_ks = eval_settings.get("top_ks", None)
    if raw_top_ks is None:
        held_out_top_ks: tuple = (20, 10, 5, -1, -2)
    else:
        # Always append -1, -2 so all_significant + all_significant_corrected
        # rows still appear in overall_predictiveness_logit.csv.
        held_out_top_ks = tuple(int(x) for x in raw_top_ks) + (-1, -2)

    eval_yaml_root = os.path.join(
        os.path.dirname(cfg_path), "eval", v_label
    )
    os.makedirs(eval_yaml_root, exist_ok=True)

    versioned_prompt_dir = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        "prompts",
        "disagreement_prompts",
        project,
        v_label,
    )
    versioned_prompt_dir = os.path.abspath(versioned_prompt_dir)
    versioned_prompt_dir_held_out = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "prompts",
            "disagreement_prompts",
            project,
            f"{v_label}_held_out",
        )
    )

    # ---- Stage [5]: validation ---------------------------------------------
    validation_dir = os.path.join(run_dir, "validation")
    validation_guard = os.path.join(validation_dir, eval_model, "overall_predictiveness_logit.csv")
    skip_validation = os.environ.get("CHARDIS_SKIP_VALIDATION", "").lower() in ("1", "true", "yes")
    if skip_validation:
        if os.environ.get("CHARDIS_SKIP_HELDOUT", "").lower() not in ("1", "true", "yes"):
            if not artifacts.reusable(validation_guard, artifacts.evaluation_identity(cfg, "validation")):
                raise ValueError("Skipping validation requires verified validation results for heldout evaluation.")
        console.print(
            "[yellow][5] CHARDIS_SKIP_VALIDATION is set — skipping validation stage.[/yellow]"
        )
        console.print(
            f"  [dim]Held-out stage will read significant_disagreement_explanations.csv "
            f"from {os.path.join(validation_dir, eval_model)} (must exist).[/dim]"
        )
    elif os.path.isfile(validation_guard):
        artifacts.reusable(validation_guard, artifacts.evaluation_identity(cfg, "validation"))
        console.print(
            f"[dim][5] Validation ranking already exists at {validation_guard} — skipping.[/dim]"
        )
    else:
        console.print(f"[bold cyan][5] Ranking hypotheses on validation set...[/bold cyan]")
        for i in range(1, val_max_batches + 1):
            eval_run_name = f"eval_{i}"
            eval_config_file = os.path.join(eval_yaml_root, f"{eval_run_name}.yaml")
            console.print(f"  • Preparing eval batch {i}/{val_max_batches} → {eval_config_file}")

            out = prepare_eval_config(
                out_path=eval_config_file,
                project=project,
                v_label=v_label,
                run_name=run_name,
                eval_run_name=eval_run_name,
                eval_settings=eval_settings,
                dataset="validation_annotate.csv",
                batch=i,
                outputs_dir=validation_dir,
                outputs_file=explanations_file,
                significant_hypotheses_file=None,
                held_out=False,
            )
            if out is None:
                console.print(f"  [dim]No more batches needed — stopping at batch {i}.[/dim]")
                break

            console.print(f"  • Annotating batch {i}/{val_max_batches}")
            annotation_orchestrator.run(eval_config_file)

        console.print("[bold cyan][5] Running evaluate.py on validation outputs...[/bold cyan]")
        eval_mod.main(
            held_out=False,
            outputs_dir=validation_dir,
            run_name=run_name,
            eval_model=eval_model,
            prompt_path=versioned_prompt_dir,
        )
        artifacts.seal(validation_guard, artifacts.evaluation_identity(cfg, "validation"),
                       [p for p in Path(validation_dir).rglob("*") if p.is_file()
                        and p.suffix in {".csv", ".json"} and not p.name.endswith(".manifest.json")])
        console.print("[green]✓ Validation ranking complete.[/green]")

    # ---- Stage [6]: held-out -----------------------------------------------
    if os.environ.get("CHARDIS_SKIP_HELDOUT", "").lower() in ("1", "true", "yes"):
        console.print(
            "[yellow][6] CHARDIS_SKIP_HELDOUT is set — skipping held-out stage.[/yellow]"
        )
        return
    held_out_dir = os.path.join(run_dir, "heldout")
    held_out_guard = os.path.join(held_out_dir, eval_model, "overall_predictiveness_logit.csv")
    significant_file = os.path.join(
        validation_dir, eval_model, "significant_disagreement_explanations.csv"
    )
    if os.path.isfile(held_out_guard):
        artifacts.reusable(held_out_guard, artifacts.evaluation_identity(cfg, "heldout"))
        console.print(
            f"[dim][6] Held-out ranking already exists at {held_out_guard} — skipping.[/dim]"
        )
        return
    if not os.path.isfile(significant_file):
        raise ValueError("Heldout evaluation requires completed validation rankings.")

    console.print(f"[bold cyan][6] Ranking significant hypotheses on held-out set...[/bold cyan]")
    eval_yaml_root_ho = os.path.join(os.path.dirname(cfg_path), "eval", f"{v_label}_held_out")
    os.makedirs(eval_yaml_root_ho, exist_ok=True)

    for i in range(1, heldout_max_batches + 1):
        eval_run_name = f"eval_{i}"
        eval_config_file = os.path.join(eval_yaml_root_ho, f"{eval_run_name}.yaml")
        console.print(f"  • Preparing held-out eval batch {i}/{heldout_max_batches} → {eval_config_file}")

        out = prepare_eval_config(
            out_path=eval_config_file,
            project=project,
            v_label=v_label,
            run_name=run_name,
            eval_run_name=eval_run_name,
            eval_settings=eval_settings,
            dataset="test_annotate.csv",
            batch=i,
            outputs_dir=held_out_dir,
            outputs_file=explanations_file,
            significant_hypotheses_file=significant_file,
            held_out=True,
        )
        if out is None:
            console.print(f"  [dim]No more held-out batches needed — stopping at batch {i}.[/dim]")
            break

        console.print(f"  • Annotating held-out batch {i}/{heldout_max_batches}")
        annotation_orchestrator.run(eval_config_file)

    console.print("[bold cyan][6] Running evaluate.py on held-out outputs...[/bold cyan]")
    eval_mod.main(
        held_out=True,
        outputs_dir=held_out_dir,
        run_name=run_name,
        eval_model=eval_model,
        prompt_path=versioned_prompt_dir_held_out,
        top_ks=held_out_top_ks,
    )
    artifacts.seal(held_out_guard, artifacts.evaluation_identity(cfg, "heldout"),
                   [p for p in Path(held_out_dir).rglob("*") if p.is_file()
                    and p.suffix in {".csv", ".json"} and not p.name.endswith(".manifest.json")])
    console.print("[green]✓ Held-out ranking complete.[/green]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        Console(stderr=True).print(str(exc), markup=False)
        sys.exit(1)
