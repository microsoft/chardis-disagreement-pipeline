"""
Prepares the per-batch eval YAML for the disagreement-analysis evaluation
stages (validation + heldout). Both an in-process API and a backwards-compatible
CLI are exposed.

Prompts are written under
``prompts/disagreement_prompts/<project>/<v_label>/...`` so each
disagreement-run version owns its own concept files (vs. the legacy
``prompts/<run_name>/...`` layout).
"""

import argparse
import os
import json
import math
from typing import Any, Dict, List, Optional
import pandas as pd

from common.src.configs.schema import AliasAction, connection_section, runtime_config, values


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _must_get(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing key '{key}' in YAML at {ctx}")
    return d[key]


def _set_value(cfg: Dict[str, Any], section: str, field: str, value: Any) -> None:
    """Expects structure: cfg[section][field]["value"] = ..."""
    sec = _must_get(cfg, section, f"root['{section}']")
    fld = _must_get(sec, field, f"root['{section}']['{field}']")
    if not isinstance(fld, dict):
        raise TypeError(f"Expected dict at root['{section}']['{field}'], got {type(fld)}")
    fld["value"] = value


def _set_value_either(
    cfg: Dict[str, Any],
    candidates: List[List[str]],
    value: Any,
) -> None:
    """Sets cfg[path...][-1]["value"] for the first matching path."""
    last_err = None
    for path in candidates:
        try:
            if len(path) == 1:
                fld = _must_get(cfg, path[0], f"root['{path[0]}']")
                if not isinstance(fld, dict):
                    raise TypeError(f"Expected dict at root['{path[0]}'], got {type(fld)}")
                fld["value"] = value
                return
            elif len(path) == 2:
                _set_value(cfg, path[0], path[1], value)
                return
            else:
                cur: Any = cfg
                ctx = "root"
                for k in path:
                    if not isinstance(cur, dict):
                        raise TypeError(f"Expected dict while walking {path}; at {ctx} got {type(cur)}")
                    cur = _must_get(cur, k, f"{ctx}['{k}']")
                    ctx = f"{ctx}['{k}']"
                if not isinstance(cur, dict):
                    raise TypeError(f"Expected dict at {ctx}, got {type(cur)}")
                cur["value"] = value
                return
        except Exception as e:
            last_err = e
            continue
    raise KeyError(f"Could not find any candidate path to set value. Last error: {last_err}")


def _load_explanations(explanations: str) -> List[str]:
    """Reads a JSON file and returns list[str] explanations."""
    if not os.path.exists(explanations):
        raise FileNotFoundError(f"explanations not found: {explanations}")

    with open(explanations, "r", encoding="utf-8-sig") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        explanations_ls = obj
    elif isinstance(obj, dict) and "explanations" in obj and isinstance(obj["explanations"], list):
        explanations_ls = obj["explanations"]
    else:
        raise ValueError(
            f"Unsupported explanations JSON structure. Expected list[str] or dict with key 'explanations'. "
            f"Got: {type(obj)}"
        )

    for idx, x in enumerate(explanations_ls):
        if not isinstance(x, str):
            raise TypeError(f"explanations[{idx}] is not a string: {type(x)}")
        if not x.strip():
            raise ValueError(f"explanations[{idx}] is empty/whitespace")

    return explanations_ls


def _concepts_for_batch(run_name: str, batch: int, n_total: int, batch_size: int = 10) -> List[Dict[str, Any]]:
    """Produces the concepts slice for the given batch based on n_total."""
    if n_total < 1:
        raise ValueError(f"n_total must be >= 1, got {n_total}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    num_batches = int(math.ceil(n_total / batch_size))
    if batch < 1 or batch > num_batches:
        raise ValueError(f"--batch must be in [1,{num_batches}], got {batch}")

    start = (batch - 1) * batch_size + 1
    end = min(batch * batch_size, n_total)
    return [{"name": f"{run_name}_{i}", "version": "0.0"} for i in range(start, end + 1)]


def _safe_unlink(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"WARNING: failed to delete {path!r}: {e}")


def _load_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing prompt file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_inputs_context(base_dir) -> str:
    path = os.path.join(base_dir, "prompts", "DEFAULTS", "disagreement_explainer", "inputs_context", "inputs_context_0.0.md")
    return _load_file(path)


def _versioned_prompts_root(base_dir: str, project: str, v_label: str, held_out: bool) -> str:
    """Compute the absolute prompts directory for a disagreement run version.

    Layout:
      prompts/disagreement_prompts/<project>/<v_label>/
      prompts/disagreement_prompts/<project>/<v_label>_held_out/  (when held_out=True)
    """
    suffix = f"{v_label}_held_out" if held_out else v_label
    return os.path.join(base_dir, "prompts", "disagreement_prompts", project, suffix)


def _annotation_topic(project: str, v_label: str, held_out: bool) -> str:
    """Topic string passed to PromptBuilder.

    Mirrors the on-disk layout: prompt_builder joins ``prompts/`` +
    topic + ``concepts/...`` so this needs to match _versioned_prompts_root.
    """
    suffix = f"{v_label}_held_out" if held_out else v_label
    return os.path.join("disagreement_prompts", project, suffix)


def prep_for_evaluation(
    base_dir: str,
    project: str,
    v_label: str,
    run_name: str,
    explanations: List[str],
    held_out: bool,
) -> str:
    """Write inputs_context + per-explanation concept markdown files.

    Args:
        base_dir: absolute path to the repository root
        project: project name (e.g. ``dices``)
        v_label: version label (e.g. ``v3``)
        run_name: used as the prefix for generated concept names
        explanations: list of non-empty strings; each becomes a concept file
        held_out: route to the ``<v_label>_held_out`` sibling dir when True

    Returns the absolute prompts root that was populated.
    """
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError(f"run_name must be a non-empty string. Got: {run_name!r}")

    prompts_root = _versioned_prompts_root(base_dir, project, v_label, held_out)
    base_inputs_dir = os.path.join(prompts_root, "inputs_context")
    base_concepts_dir = os.path.join(prompts_root, "concepts", "conceptual_definitions")

    os.makedirs(base_inputs_dir, exist_ok=True)
    os.makedirs(base_concepts_dir, exist_ok=True)

    inputs_context = _load_inputs_context(base_dir)
    inputs_context_path = os.path.join(base_inputs_dir, "inputs_context_0.0.md")
    with open(inputs_context_path, "w", encoding="utf-8") as f:
        f.write(inputs_context)

    for idx, text in enumerate(explanations, start=1):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Item at index {idx-1} is not a non-empty string: {text!r}")
        new_name = f"{run_name}_{idx}"
        md_content = f"## Conceptual Category: {new_name}\n### Label Type: boolean\n{text}"
        concept_dir = os.path.join(base_concepts_dir, new_name)
        os.makedirs(concept_dir, exist_ok=True)
        md_path = os.path.join(concept_dir, f"{new_name}_0.0.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

    return prompts_root


def _seed_eval_config_yaml(
    out_path: str,
    project: str,
    v_label: str,
) -> Dict[str, Any]:
    """Build a fresh annotation-orchestrator-compatible YAML dict.

    The completed configuration is saved only after explicit evaluation
    connection settings have been applied.
    """
    # Lazy import to avoid pulling annotation deps when only the CLI is used.
    from src.config_template import build_empty_config

    return runtime_config(build_empty_config(project_name=project, version=v_label))


def prepare_eval_config(
    *,
    out_path: str,
    project: str,
    v_label: str,
    run_name: str,
    eval_run_name: str,
    eval_settings: Dict[str, Any],
    dataset: str,
    batch: int,
    outputs_dir: str,
    outputs_file: str,
    significant_hypotheses_file: Optional[str] = None,
    held_out: bool = False,
) -> Optional[str]:
    """In-process driver. Returns the eval YAML path on success, or ``None``
    if ``batch`` is past the last needed batch (caller should stop looping).
    """
    eval_settings = values(connection_section(eval_settings, evaluation=True))
    explanations = _load_explanations(outputs_file)

    if held_out:
        if not significant_hypotheses_file or not os.path.isfile(significant_hypotheses_file):
            raise FileNotFoundError(
                f"--held_out requires significant_hypotheses_file; missing: {significant_hypotheses_file!r}"
            )
        # Use the CSV's row order directly. The validation stage writes
        # significant_disagreement_explanations.csv sorted by (significant, f1,
        # precision, recall) and the JSON list is built from the same ordering,
        # so consuming this CSV preserves F1-rank order. The held-out prompts
        # will then be re-numbered _1, _2, ... in the *same* order as the JSON
        # list — which is what evaluate.prepare_data assumes when it positionally
        # renames the held-out columns to match `significant_hypotheses`.
        sig_hypo_df = pd.read_csv(significant_hypotheses_file)
        explanations = sig_hypo_df["hypothesis"].tolist()

    if held_out:
        # Heldout uses its own cap if set, else falls back to top_hypotheses
        # (preserves v1-v16 behavior where the same cap applied to both stages).
        heldout_top_n_raw = eval_settings.get("heldout_top_hypotheses", None)
        if heldout_top_n_raw in (None, "", 0):
            top_n = int(eval_settings.get("top_hypotheses", 50) or 50)
        else:
            top_n = int(heldout_top_n_raw)
    else:
        top_n = int(eval_settings.get("top_hypotheses", 50) or 50)
    explanations = explanations[:top_n]
    if not explanations:
        return None

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

    prep_for_evaluation(
        base_dir=base_dir,
        project=project,
        v_label=v_label,
        run_name=run_name,
        explanations=explanations,
        held_out=held_out,
    )

    n_total = len(explanations)
    batch_size = int(eval_settings.get("concepts_per_batch", 10) or 10)
    num_batches = int(math.ceil(n_total / batch_size))
    if batch > num_batches:
        return None

    # Seed a fresh eval YAML using the annotation pipeline's defaults.
    cfg = _seed_eval_config_yaml(out_path, project=project, v_label=v_label)

    eval_model = str(eval_settings.get("eval_model") or "").strip()
    if not eval_model:
        raise ValueError("evaluation.model is required")
    num_iterations = int(eval_settings.get("num_iterations", 3) or 3)
    num_threads = int(eval_settings.get("num_threads", 15) or 15)

    _set_value_either(cfg, [["openai", "openai_model"], ["openai", "model"]], eval_model)
    from common.src.openai.connections import normalized, validate
    connection = normalized(eval_settings)
    validate(connection, live=False)
    from common.src.openai.openai_args import OpenAIArgs
    fields = vars(OpenAIArgs())
    for field, value in connection.items():
        if field in fields:
            cfg["openai"][field] = {"description": "", "value": value}

    eval_dir = os.path.join(outputs_dir, eval_run_name)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_dir, "raw"), exist_ok=True)

    _set_value_either(cfg, [["annotation", "data_path"]], os.path.join(outputs_dir, dataset))
    _set_value_either(cfg, [["annotation", "output_path"]], os.path.join(eval_dir, "eval.csv"))
    _set_value_either(cfg, [["annotation", "batch_size"]], 1)
    _set_value_either(cfg, [["annotation", "num_threads"]], num_threads)
    _set_value_either(cfg, [["annotation", "num_iterations"]], num_iterations)
    _set_value_either(cfg, [["annotation", "raw_output_path"]], os.path.join(eval_dir, "raw"))

    topic = _annotation_topic(project, v_label, held_out)
    _set_value_either(cfg, [["prompt_builder", "built_prompt_name"]], eval_run_name)
    _set_value_either(cfg, [["prompt_builder", "annotation_topic"]], topic)
    _set_value_either(cfg, [["prompt_builder", "inputs_context"]], "inputs_context_0.0.md")

    concepts_list = _concepts_for_batch(run_name, batch, n_total=n_total, batch_size=batch_size)
    _set_value_either(cfg, [["prompt_builder", "concepts"]], concepts_list)
    _set_value_either(cfg, [["prompt_builder", "examples"]], False)
    # Per-annotation diagnostics: drop confidence + evidence span for speed.
    # For DeepSeek (a reasoning model), also drop the explicit reasoning field
    # in output (the model still does internal reasoning).
    is_reasoning_model = "deepseek" in eval_model.lower() or eval_model.lower().startswith(("o1", "o3"))
    # Allow eval_settings.prompt_reasoning to override the default. When omitted,
    # fall back to the historical behavior (request reasoning for non-reasoning
    # models). Setting it to False explicitly disables the per-concept reasoning
    # field in the annotation prompt regardless of model.
    prompt_reasoning_setting = eval_settings.get("prompt_reasoning")
    if prompt_reasoning_setting is None:
        prompt_reasoning = not is_reasoning_model
    else:
        prompt_reasoning = bool(prompt_reasoning_setting)
    _set_value_either(cfg, [["prompt_builder", "reasoning"]], prompt_reasoning)
    _set_value_either(cfg, [["prompt_builder", "reasoning_global"]], False)
    _set_value_either(cfg, [["prompt_builder", "confidence"]], False)
    _set_value_either(cfg, [["prompt_builder", "confidence_global"]], False)
    _set_value_either(cfg, [["prompt_builder", "evidence_span"]], False)
    _set_value_either(cfg, [["prompt_builder", "evidence_span_global"]], False)

    _ensure_dir_for_file(out_path)
    from common.src.configs.config_utils import save_yaml_config
    save_yaml_config(cfg, out_path)

    print(f"Wrote eval YAML → {out_path}")
    print(f"Explanations: {n_total} (num_batches={num_batches}, batch_size={batch_size})")
    print(f"Batch {batch}: concepts {concepts_list[0]['name']} .. {concepts_list[-1]['name']}")
    print(f"Eval outputs dir: {eval_dir}")
    return out_path


def main() -> None:
    """Backwards-compatible CLI. New callers should use ``prepare_eval_config``."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", dest="out_path", required=True, help="Output YAML path")
    p.add_argument("--dataset", type=str, required=True, help="Eval dataset filename")
    p.add_argument("--batch", type=int, required=True, help="1-indexed batch number")
    p.add_argument("--project", required=True, help="Disagreement project name (e.g. dices)")
    p.add_argument("--version", required=True, help="Disagreement run version label (e.g. v3)")
    p.add_argument("--run_name", required=True, help="Base run name (used for concept names)")
    p.add_argument("--eval_run_name", required=True, help="Eval run name (output folder + built_prompt_name)")
    p.add_argument("--model", "--eval_model", "--openai_model", dest="eval_model",
                   action=AliasAction, metavar="MODEL", required=True, help="Model identifier for evaluation")
    p.add_argument("--num_iterations", type=int, default=3)
    p.add_argument("--scope", default="")
    p.add_argument("--provider", required=True, choices=["openai", "compatible", "local"])
    p.add_argument("--auth_mode", required=True, choices=["api-key", "none"])
    p.add_argument("--api_key_env", default="")
    p.add_argument("--credential_endpoint", default="")
    p.add_argument("--credential_provider", default="")
    p.add_argument("--message_role", default="system", choices=["system", "developer"])
    p.add_argument("--api_version", "--openai_api_version", dest="openai_api_version",
                   action=AliasAction, metavar="API_VERSION", default="")
    p.add_argument("--endpoint", "--openai_endpoint", "--base_url", dest="openai_endpoint",
                   action=AliasAction, metavar="ENDPOINT", default="")
    p.add_argument("--num_threads", type=int, default=15)
    p.add_argument("--concepts_per_batch", type=int, default=10)
    p.add_argument("--top_hypotheses", type=int, default=50)
    p.add_argument("--outputs_dir", required=True)
    p.add_argument("--outputs_file", required=True)
    p.add_argument("--significant_hypotheses_file", required=False)
    p.add_argument("--held_out", action="store_true")
    args = p.parse_args()

    eval_settings = {
        "eval_model": args.eval_model,
        "num_iterations": args.num_iterations,
        "scope": args.scope,
        "provider": args.provider,
        "auth_mode": args.auth_mode,
        "api_key_env": args.api_key_env,
        "credential_endpoint": args.credential_endpoint,
        "credential_provider": args.credential_provider,
        "message_role": args.message_role,
        "openai_endpoint": args.openai_endpoint,
        "openai_api_version": args.openai_api_version,
        "num_threads": args.num_threads,
        "concepts_per_batch": args.concepts_per_batch,
        "top_hypotheses": args.top_hypotheses,
    }

    out = prepare_eval_config(
        out_path=args.out_path,
        project=args.project,
        v_label=args.version,
        run_name=args.run_name,
        eval_run_name=args.eval_run_name,
        eval_settings=eval_settings,
        dataset=args.dataset,
        batch=args.batch,
        outputs_dir=args.outputs_dir,
        outputs_file=args.outputs_file,
        significant_hypotheses_file=args.significant_hypotheses_file,
        held_out=args.held_out,
    )
    if out is None:
        print(f"Stopping: batch {args.batch} is past the last needed batch.")
        _safe_unlink(args.out_path)
        return


if __name__ == "__main__":
    main()