"""Disagreement Analysis settings dashboard.

A slim, web-based settings panel for the disagreement analysis pipeline.
Renders one tab ("Settings") with the key knobs from the disagreement YAML
config and stubs out additional tabs as "Coming soon".

Returns the saved configuration, change state, and cancellation outcome.
"""

from __future__ import annotations

import copy
import hmac
import secrets
import json
import math
import os
import socket
import sys
import threading
import time
import webbrowser
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass
from typing import Callable
from common.src.configs.schema import PUBLIC_FIELDS, runtime_config, section_name


# ── Model catalog helpers (used by the model deployment dropdown) ───────────

# Path to the shared model catalog yaml.
_MODELS_YAML_PATH = "common/models.yaml"

# Provider grouping for the model dropdown. Keys are display labels; values
# are case-insensitive substrings that map a catalog key into that group.
# Order is preserved.
_MODEL_PROVIDER_GROUPS: list[tuple[str, list[str]]] = [
    ("OpenAI GPT",        ["gpt-4", "gpt-5"]),
    ("OpenAI Reasoning",  ["o1", "o3", "o4"]),
    ("DeepSeek",          ["deepseek"]),
    ("xAI",               ["grok"]),
    ("Meta",              ["llama"]),
    ("Mistral",           ["mistral"]),
]


def _is_supported_model(catalog_key: str) -> bool:
    """Return True iff ``openai_args.get_model_messages`` can map this model
    to a chat role. Anything else would crash at runtime, so we shouldn't
    expose it in the dashboard."""
    k = (catalog_key or "").lower()
    return (
        any(tag in k for tag in ("gpt-4o", "4o", "4.1")) or
        any(tag in k for tag in ("o1", "o3", "gpt-5")) or
        ("deepseek" in k)
    )


def _load_model_catalog() -> list[tuple[str, str]]:
    """Return [(catalog_key, deployment_name), ...] from common/models.yaml.
    Only returns models the judge runner can actually invoke. Returns an
    empty list if the file isn't readable."""
    try:
        import yaml
        with open(_MODELS_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return []
    models = data.get("models", {}) or {}
    out: list[tuple[str, str]] = []
    for key, meta in models.items():
        if not isinstance(meta, dict):
            continue
        if not _is_supported_model(str(key)):
            continue
        deployment = str(meta.get("model_name") or key)
        out.append((str(key), deployment))
    return out


def _grouped_model_catalog() -> list[tuple[str, list[tuple[str, str]]]]:
    """Group the model catalog by provider. Entries that don't match any
    configured prefix land in 'Other'. Empty groups are dropped."""
    catalog = _load_model_catalog()
    buckets: list[list[tuple[str, str]]] = [[] for _ in _MODEL_PROVIDER_GROUPS]
    other: list[tuple[str, str]] = []
    for key, dep in catalog:
        k = key.lower()
        matched = False
        for idx, (_, prefixes) in enumerate(_MODEL_PROVIDER_GROUPS):
            if any(p in k for p in prefixes):
                buckets[idx].append((key, dep))
                matched = True
                break
        if not matched:
            other.append((key, dep))
    grouped: list[tuple[str, list[tuple[str, str]]]] = [
        (label, items)
        for (label, _), items in zip(_MODEL_PROVIDER_GROUPS, buckets)
        if items
    ]
    if other:
        grouped.append(("Other", other))
    return grouped


def _model_options_html(selected: str) -> str:
    """Render <option> tags grouped by provider, pre-selecting `selected`."""
    try:
        groups = _grouped_model_catalog()
    except Exception:
        groups = []

    if not groups:
        # Fall back to a free-text-only experience: emit just the current value.
        s = escape(str(selected or ""))
        return f'<option value="{s}" selected>{s or "(custom)"}</option>'

    out = [f'<option value="{escape(str(selected or ""))}" selected>{escape(str(selected) or "(current)")}</option>']
    seen = {str(selected or "")}
    for label, items in groups:
        out.append(f'<optgroup label="{escape(label)}">')
        for key, deployment in items:
            if deployment in seen:
                continue
            seen.add(deployment)
            out.append(
                f'<option value="{escape(deployment)}">{escape(key)} — '
                f'<span class="dim">{escape(deployment)}</span></option>'
            )
        out.append("</optgroup>")
    return "\n".join(out)


# ── Tiny form helpers ────────────────────────────────────────────────────────

def _val(section: dict, field: str, default=""):
    entry = section.get(field) if isinstance(section, dict) else None
    if isinstance(entry, dict):
        return entry.get("value", default)
    return default


def _coerce(raw: str, current):
    """Coerce a form value back to the type of the current config value."""
    if isinstance(current, bool):
        if raw.lower() not in ("true", "false", "on", "off", "1", "0", "yes", "no"):
            raise ValueError("Expected true or false.")
        return raw.lower() in ("true", "on", "1", "yes")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError("Expected an integer.") from None
    if isinstance(current, float):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError("Expected a finite number.") from None
        if not math.isfinite(value):
            raise ValueError("Expected a finite number.")
        return value
    return raw


def _apply_form(config: dict, submitted: dict[str, str]) -> None:
    if not isinstance(submitted, dict):
        raise ValueError("Settings must be an object.")
    converted = runtime_config(config)
    for key, raw in submitted.items():
        if "__" not in key:
            raise ValueError("Unknown settings field.")
        section, field = key.split("__", 1)
        if field in {"key", "openai_key", "api_key", "credential_endpoint", "credential_provider"}:
            raise ValueError("Credential bindings must be changed in the local configuration.")
        sec = converted.get(section)
        if not isinstance(sec, dict):
            raise ValueError("Unknown settings section.")
        meta = sec.get(field)
        if not isinstance(meta, dict) or "value" not in meta:
            raise ValueError("Unknown settings field.")
        label = f"{section_name(section, public=True)}.{PUBLIC_FIELDS.get(field, field)}"
        if not isinstance(raw, str):
            raise ValueError(f"Invalid value for {label}.")
        try:
            meta["value"] = _coerce(raw, meta["value"])
        except ValueError as exc:
            raise ValueError(f"{label}  {exc}") from None
    config.clear()
    config.update(converted)


# ── HTML page ────────────────────────────────────────────────────────────────

_PAGE_CSS = """
<style>
  :root {
    --background: #FAF9F8;
    --foreground: #242424;
    --card: #FFFFFF;
    --secondary: #F5F5F5;
    --border: #EDEBE9;
    --border-hover: #C8C6C4;
    --text-secondary: #616161;
    --text-tertiary: #A19F9D;
    --blue: #0078D4;
    --blue-glow: rgba(0,120,212,0.08);
    --blue-glow-strong: rgba(0,120,212,0.18);
    --teal: #038387;
    --teal-glow: rgba(3,131,135,0.08);
    --purple: #8764B8;
    --purple-glow: rgba(135,100,184,0.08);
    --orange: #D83B01;
    --orange-glow: rgba(216,59,1,0.08);
    --green: #107C10;
    --warn: #B88300;
    --warn-glow: rgba(184,131,0,0.10);
    --warn-strong: #F8D56B;
    --radius: 6px;
    --font-sans: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'Cascadia Code', Consolas, 'Courier New', monospace;
  }
  * { box-sizing: border-box; }
  body {
    font-family: var(--font-sans);
    background: var(--background);
    color: var(--foreground);
    margin: 0;
    padding: 0 0 96px;
    font-size: 14px;
    line-height: 1.5;
  }
  header {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 18px 32px 16px;
  }
  .header-inner {
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .header-inner h1 { font-size: 20px; margin: 0; font-weight: 600; }
  .header-inner h1 .v { color: var(--text-secondary); font-weight: 400; margin-left: 4px; }
  .header-inner .meta { color: var(--text-secondary); font-size: 13px; }
  main { max-width: 1200px; margin: 28px auto; padding: 0 32px; }

  /* ── Section card ── */
  .section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 16px;
    overflow: hidden;
  }
  .section.required { border-left: 3px solid var(--blue); }
  .section.required.teal { border-left-color: var(--teal); }
  .section-head {
    padding: 14px 20px 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
  }
  .section-head h2 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .section-head .icon {
    width: 26px; height: 26px;
    border-radius: 5px;
    display: grid; place-items: center;
    font-size: 14px;
    background: var(--blue-glow);
    color: var(--blue);
  }
  .section-head .icon.teal   { background: var(--teal-glow);   color: var(--teal); }
  .section-head .icon.purple { background: var(--purple-glow); color: var(--purple); }
  .section-head .icon.orange { background: var(--orange-glow); color: var(--orange); }
  .section-head .desc {
    color: var(--text-secondary);
    font-size: 12px;
    flex: 1;
    margin-left: 14px;
  }
  .req-pill {
    font-family: var(--font-mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: var(--blue-glow);
    color: var(--blue);
    padding: 3px 8px;
    border-radius: 10px;
    font-weight: 600;
  }
  .section.required.teal .req-pill { background: var(--teal-glow); color: var(--teal); }
  .section-body { padding: 18px 20px 20px; }

  /* ── Field grid ── */
  .row {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px 20px;
  }
  .row.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .row.cols-1 { grid-template-columns: 1fr; }
  @media (max-width: 720px) {
    .row, .row.cols-3 { grid-template-columns: 1fr; }
  }
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field .label-mono {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-tertiary);
  }
  .field input[type="text"],
  .field input[type="number"],
  .field select {
    width: 100%;
    padding: 7px 10px;
    border: 1px solid var(--border-hover);
    border-radius: 4px;
    background: var(--card);
    font-family: inherit;
    font-size: 13px;
    color: var(--foreground);
  }
  .field input:focus,
  .field select:focus {
    outline: none;
    border-color: var(--blue);
    box-shadow: 0 0 0 2px var(--blue-glow-strong);
  }
  .field .hint {
    margin: 4px 0 0;
    font-size: 11px;
    color: var(--text-tertiary);
  }
  .field.toggle {
    flex-direction: row;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--secondary);
  }
  .field.toggle input[type="checkbox"] { transform: scale(1.1); margin: 0; }
  .field.toggle .label-mono { color: var(--foreground); margin: 0; }

  /* ── Footer / actions ── */
  .footer-bar {
    position: fixed; left: 0; right: 0; bottom: 0;
    background: rgba(255,255,255,0.96);
    backdrop-filter: blur(10px);
    border-top: 1px solid var(--border);
    padding: 14px 32px;
    display: flex; justify-content: flex-end; gap: 10px;
    z-index: 50;
  }
  .btn {
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 18px;
    border-radius: 4px;
    border: 1px solid var(--border-hover);
    background: var(--card);
    color: var(--foreground);
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .btn:hover { border-color: var(--blue); color: var(--blue); }
  .btn.primary {
    background: var(--blue);
    color: white;
    border-color: var(--blue);
  }
  .btn.primary:hover { background: #106EBE; color: white; }
  .btn[disabled] { opacity: 0.4; cursor: not-allowed; }

  /* ── Sub-tabs ── */
  .pane-tabs {
    display: flex;
    gap: 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 18px;
  }
  .pane-tab {
    background: transparent;
    border: 0;
    border-bottom: 2px solid transparent;
    padding: 10px 16px;
    font-family: inherit;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    cursor: pointer;
    margin-bottom: -1px;
  }
  .pane-tab:hover { color: var(--foreground); }
  .pane-tab.active {
    color: var(--foreground);
    border-bottom-color: var(--teal);
  }
  .pane { display: none; }
  .pane.active { display: block; }

  /* ── Confirm rows ── */
  .confirm-row {
    margin-top: 8px;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .confirm-row .btn.confirmed {
    background: #1F4D2E !important;
    color: #B8E6C5 !important;
    border-color: #2E7048 !important;
  }
  .confirm-state { font-size: 12px; color: var(--text-tertiary); }
  .confirm-state.ok { color: var(--green); }

  /* ── Read-only guidelines preview ── */
  .prompt-preview {
    background: var(--secondary);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px 16px;
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 1.55;
    color: var(--foreground);
    white-space: pre-wrap;
    word-wrap: break-word;
    max-height: 480px;
    overflow: auto;
  }
  .prompt-preview.empty {
    color: var(--text-tertiary);
    font-style: italic;
    font-family: var(--font-sans);
  }
  .prompt-preview-meta {
    font-size: 11px;
    color: var(--text-tertiary);
    margin: 0 0 8px;
    font-family: var(--font-mono);
  }

  /* ── File-status banner (data + guidelines existence checks) ── */
  .file-status {
    margin-top: 12px;
    padding: 10px 14px;
    border-radius: 4px;
    border: 1px solid var(--border);
    background: var(--secondary);
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    color: var(--text-secondary);
  }
  .file-status .dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--text-tertiary);
    flex-shrink: 0;
  }
  .file-status.ok {
    background: rgba(16,124,16,0.06);
    border-color: rgba(16,124,16,0.30);
    color: var(--green);
  }
  .file-status.ok .dot { background: var(--green); }
  .file-status.missing {
    background: rgba(216,59,1,0.06);
    border-color: rgba(216,59,1,0.30);
    color: var(--orange);
  }
  .file-status.missing .dot { background: var(--orange); }
  .file-status.checking .dot {
    animation: stac-pulse 1.2s ease-in-out infinite;
  }
  @keyframes stac-pulse {
    0%, 100% { opacity: 0.4; }
    50%      { opacity: 1; }
  }
  .file-status .msg { word-break: break-all; }

  /* ── Data preview (first rows + annotator detection) ── */
  .data-preview { margin-top: 14px; }
  .data-preview .preview-meta {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-tertiary);
    margin: 0 0 6px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .data-preview .table-scroll {
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: auto;
    max-height: 220px;
    background: var(--secondary);
  }
  .data-preview table {
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 11px;
    white-space: nowrap;
    width: max-content;
    min-width: 100%;
  }
  .data-preview th, .data-preview td {
    padding: 6px 10px;
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    text-align: left;
    vertical-align: top;
    max-width: 360px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .data-preview th {
    position: sticky; top: 0;
    background: var(--background);
    color: var(--text-secondary);
    font-weight: 600;
  }
  .data-preview th.annotator-col {
    color: var(--blue);
    background: var(--blue-glow);
  }
  .data-preview .empty {
    padding: 10px 14px;
    color: var(--text-tertiary);
    font-style: italic;
    font-family: var(--font-sans);
    font-size: 12px;
  }
  .annotator-alert {
    margin-top: 10px;
    padding: 8px 12px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 11px;
    border: 1px solid var(--border);
    background: var(--secondary);
    color: var(--text-secondary);
  }
  .annotator-alert.warn {
    color: #a4262c;
    background: rgba(216,59,1,0.06);
    border-color: rgba(164,38,44,0.45);
    font-weight: 600;
  }

  /* ── Chunk summary card ── */
  .chunk-summary {
    margin-top: 14px;
    padding: 10px 14px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--secondary);
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-secondary);
    display: flex;
    flex-wrap: wrap;
    gap: 6px 18px;
  }
  .chunk-summary .stat { display: inline-flex; gap: 6px; align-items: baseline; }
  .chunk-summary .stat .k {
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.4px;
    font-size: 10px;
  }
  .chunk-summary .stat .v {
    color: var(--foreground);
    font-weight: 600;
  }
  .chunk-summary.empty { color: var(--text-tertiary); font-style: italic; }

  /* ── Advanced settings collapsible ── */
  .advanced-settings {
    margin-top: 18px;
    border-top: 1px dashed var(--border);
    padding-top: 14px;
  }
  .advanced-settings > summary {
    cursor: pointer;
    list-style: none;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--secondary);
    display: inline-flex;
    align-items: center;
    gap: 8px;
    user-select: none;
  }
  .advanced-settings > summary::-webkit-details-marker { display: none; }
  .advanced-settings > summary::before {
    content: "▸";
    font-family: var(--font-sans);
    font-size: 10px;
    transition: transform 0.15s ease;
  }
  .advanced-settings[open] > summary::before { transform: rotate(90deg); }
  .advanced-settings > summary:hover { color: var(--blue); border-color: var(--blue); }
  .advanced-body { margin-top: 14px; }

  /* ── Saved overlay ── */
  .overlay {
    position: fixed; inset: 0;
    background: rgba(250,249,248,0.96);
    display: none;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 14px;
    z-index: 100;
    color: var(--foreground);
  }
  .overlay.active { display: flex; }
  .overlay .check {
    width: 56px; height: 56px;
    border-radius: 50%;
    background: var(--blue-glow);
    color: var(--blue);
    display: grid; place-items: center;
    font-size: 30px;
  }
  .overlay .msg { font-size: 14px; color: var(--text-secondary); }

  /* About this version */
  .about-version-wrap { max-width: 1200px; margin: 16px auto 0; padding: 0 32px; }
  .about-version-wrap .section { margin-bottom: 0; }
  .about-version-wrap textarea {
    width: 100%; padding: 8px 10px; border: 1px solid var(--border);
    border-radius: var(--radius); font-family: var(--font-sans); font-size: 13px;
    resize: vertical; min-height: 60px; background: var(--card); color: var(--foreground);
  }
  .about-version-wrap textarea:focus {
    outline: none; border-color: var(--blue);
    box-shadow: 0 0 0 3px var(--blue-glow);
  }
  .model-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .model-pills .pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 4px 10px 4px 8px;
    background: var(--secondary); border: 1px solid var(--border);
    border-radius: 999px; font-size: 12px;
  }
  .model-pills .pill-label {
    text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--text-secondary); font-weight: 600; font-size: 10.5px;
  }
  .model-pills .pill-value {
    font-family: var(--font-mono); color: var(--foreground);
  }
</style>
"""


def _field(section: str, field: str, label: str, value, help_text: str = "",
           kind: str = "text", choices: list[str] | None = None) -> str:
    """Render a single labeled form field."""
    name = f"{section}__{field}"
    if kind == "bool":
        checked = "checked" if value else ""
        return (
            f'<label class="field toggle" for="{name}">'
            f'<input type="checkbox" id="{name}" name="{name}" value="true" {checked}>'
            f'<span class="label-mono">{escape(label)}</span>'
            f'</label>'
        )

    label_html = f'<span class="label-mono">{escape(label)}</span>'
    if kind == "number":
        ctrl = f'<input type="number" step="any" id="{name}" name="{name}" value="{escape(str(value))}">'
    elif kind == "select" and choices:
        opts = "".join(
            f'<option value="{escape(c)}"{" selected" if str(c)==str(value) else ""}>{escape(c)}</option>'
            for c in choices
        )
        ctrl = f'<select id="{name}" name="{name}">{opts}</select>'
    elif kind == "model":
        ctrl = f'<select id="{name}" name="{name}">{_model_options_html(str(value or ""))}</select>'
    else:
        ctrl = f'<input type="text" id="{name}" name="{name}" value="{escape(str(value))}">'

    hint_html = f'<p class="hint">{escape(help_text)}</p>' if help_text else ""
    return f'<div class="field">{label_html}{ctrl}{hint_html}</div>'


def _section(title: str, desc: str, body_html: str, *,
             icon: str = "⚙", icon_class: str = "",
             required: bool = False, required_class: str = "") -> str:
    cls_parts = ["section"]
    if required:
        cls_parts.append("required")
    if required_class:
        cls_parts.append(required_class)
    cls = " ".join(cls_parts)
    icon_cls = f"icon {icon_class}".strip()
    pill = '<span class="req-pill">Required</span>' if required else ""
    return (
        f'<section class="{cls}">'
        f'<div class="section-head">'
        f'<h2><span class="{icon_cls}">{icon}</span>{escape(title)}</h2>'
        f'<span class="desc">{escape(desc)}</span>'
        f'{pill}'
        f'</div>'
        f'<div class="section-body">{body_html}</div>'
        f'</section>'
    )


def _generic_field_kind(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return "text"


# Fields rendered explicitly in the curated Advanced order. Anything else still
# present in the section is appended at the bottom as a generic field.
_ADVANCED_LLM_FIELDS_ORDER = (
    "openai_endpoint", "openai_api_version", "scope", "provider",
    "temperature", "top_p", "max_tokens", "max_completion_tokens",
    "max_retries", "min_retry_delay", "max_retry_delay",
    "frequency_penalty", "presence_penalty",
    "logprobs", "top_logprobs", "stream", "with_raw_response",
    "openai_key",
)

_LLM_FIELD_LABELS = {
    "openai_endpoint":        "Endpoint",
    "openai_api_version":     "API version",
    "scope":                  "Auth scope",
    "provider":               "Provider",
    "temperature":            "Temperature",
    "top_p":                  "Top-p",
    "max_tokens":             "Max tokens",
    "max_completion_tokens":  "Max completion tokens",
    "max_retries":            "Max retries",
    "min_retry_delay":        "Min retry delay (s)",
    "max_retry_delay":        "Max retry delay (s)",
    "frequency_penalty":      "Frequency penalty",
    "presence_penalty":       "Presence penalty",
    "logprobs":               "Log probs",
    "top_logprobs":           "Top log probs",
    "stream":                 "Stream",
    "with_raw_response":      "Include raw response",
    "openai_key":             "API key (override)",
}

_LLM_FIELD_HELP = {
    "openai_endpoint":     "Base URL of the selected service.",
    "openai_api_version":  "Legacy compatibility field. Not sent by current adapters.",
    "scope":               "Legacy compatibility field. Current adapters do not acquire OAuth tokens.",
    "temperature":         "0–2. Lower = more deterministic.",
    "max_tokens":          "Cap per response. -1 = no cap.",
    "max_retries":         "Retry attempts on transient errors.",
}


def _openai_section_html(title: str, section_key: str, sec: dict,
                         icon_class: str = "") -> str:
    # Only the model deployment is shown up front. Everything else goes into a
    # collapsible "Advanced settings" dropdown.
    head = (
        '<div class="row cols-1">'
        + _field(section_key, "openai_model", "Model",
                 _val(sec, "openai_model"),
                 "Pick from the catalog or type a custom value.", kind="model")
        + '</div>'
    )

    # Render advanced fields in curated order, then append any leftovers.
    rendered: list[str] = []
    seen = {"openai_model"}
    for fname in _ADVANCED_LLM_FIELDS_ORDER:
        if fname in {"openai_key", "key", "api_key", "credential_endpoint", "credential_provider"}:
            seen.add(fname)
            continue
        if fname not in sec:
            continue
        seen.add(fname)
        current = _val(sec, fname, "")
        kind = _generic_field_kind(current)
        label = _LLM_FIELD_LABELS.get(fname, fname.replace("_", " ").capitalize())
        help_text = _LLM_FIELD_HELP.get(fname, "")
        rendered.append(_field(section_key, fname, label, current, help_text, kind=kind))

    for fname, meta in sec.items():
        if fname in {"openai_key", "key", "api_key", "credential_endpoint", "credential_provider"}:
            continue
        if fname in seen:
            continue
        if not isinstance(meta, dict) or "value" not in meta:
            continue
        current = meta.get("value")
        kind = _generic_field_kind(current)
        label = _LLM_FIELD_LABELS.get(fname, fname.replace("_", " ").capitalize())
        help_text = _LLM_FIELD_HELP.get(fname, str(meta.get("description", "") or ""))
        rendered.append(_field(section_key, fname, label, current, help_text, kind=kind))

    # Group fields into rows of 2 to keep the layout tidy.
    grid_rows: list[str] = []
    for i in range(0, len(rendered), 2):
        chunk = "".join(rendered[i:i + 2])
        style = ' style="margin-top:14px;"' if i > 0 else ""
        grid_rows.append(f'<div class="row"{style}>{chunk}</div>')

    advanced_html = (
        '<details class="advanced-settings">'
        '<summary>Advanced settings</summary>'
        '<div class="advanced-body">'
        + "".join(grid_rows)
        + '</div>'
        '</details>'
    )

    body = head + advanced_html
    return _section(title, "Provider and environment credential reference for this stage.",
                    body, icon="◆", icon_class=icon_class,
                    required=True, required_class=icon_class)


def _locations_section_html(de: dict) -> str:
    body = (
        '<div class="row" style="grid-template-columns: 2fr 1fr;">'
        + _field("disagreement_explainer", "data_path", "Data directory",
                 _val(de, "data_path"),
                 "Folder containing the annotated CSV (or full path to a CSV).")
        + _field("disagreement_explainer", "data_filename", "Data filename",
                 _val(de, "data_filename", "annotated_data.csv"),
                 "CSV filename inside Data directory.")
        + '</div>'
        + '<div id="data-file-status" class="file-status checking" '
          'data-dir-input="disagreement_explainer__data_path" '
          'data-name-input="disagreement_explainer__data_filename">'
          '  <span class="dot"></span>'
          '  <span class="msg">Checking data file…</span>'
          '</div>'
        + '<div id="data-preview" class="data-preview"></div>'
        + '<div id="annotator-alert" class="annotator-alert" style="display:none;"></div>'
        + '<div class="row" style="grid-template-columns: 2fr 1fr; margin-top:14px;">'
        + _field("disagreement_explainer", "guidelines_path", "Guidelines directory",
                 _val(de, "guidelines_path"),
                 "Folder containing the guidelines markdown (or full path to a .md).")
        + _field("disagreement_explainer", "guidelines_filename", "Guidelines filename",
                 _val(de, "guidelines_filename", "prompt.md"),
                 "Markdown filename inside Guidelines directory.")
        + '</div>'
        + '<div id="guidelines-file-status" class="file-status checking" '
          'data-dir-input="disagreement_explainer__guidelines_path" '
          'data-name-input="disagreement_explainer__guidelines_filename" '
          'data-kind="guidelines">'
          '  <span class="dot"></span>'
          '  <span class="msg">Checking guidelines file…</span>'
          '</div>'
        + '<div class="row" style="grid-template-columns: 1fr; margin-top:14px;">'
        + _field("disagreement_explainer", "output_path", "Outputs directory",
                 _val(de, "output_path"),
                 "Folder for preprocessing artifacts, raw outputs, metadata.json, and final disagreement_explanations.json. Created if missing.")
        + '</div>'
        + '<div id="output-dir-status" class="file-status checking" '
          'data-dir-input="disagreement_explainer__output_path" '
          'data-kind="outputs">'
          '  <span class="dot"></span>'
          '  <span class="msg">Checking outputs directory…</span>'
          '</div>'
    )
    return _section("Locations", "Where to find the inputs the disagreement pipeline reads.",
                    body, icon="📁", required=True)


def _pipeline_section_html(de: dict) -> str:
    body = (
        '<div class="row cols-3">'
        + _field("disagreement_explainer", "example_size", "Examples per chunk",
                 _val(de, "example_size", 5),
                 "Disagreeing examples grouped per LLM chunk.", kind="number")
        + _field("disagreement_explainer", "num_annotations", "Num annotations",
                 _val(de, "num_annotations", -1),
                 "Cap on items processed. -1 = use all.", kind="number")
        + _field("disagreement_explainer", "num_threads", "Threads",
                 _val(de, "num_threads", 10),
                 "Parallel worker threads.", kind="number")
        + '</div>'
        + '<div id="chunk-summary" class="chunk-summary empty">'
          'Waiting for data file to estimate chunking…'
          '</div>'
        + '<div class="row cols-3" style="margin-top:14px;">'
        + _field("disagreement_explainer", "random_state", "Random seed",
                 _val(de, "random_state", 42),
                 "Seed for sampling and chunking. Split seed remains fixed at 42.", kind="number")
        + _field("disagreement_explainer", "more_than_2", "More than 2 annotators",
                 _val(de, "more_than_2", False), kind="bool")
        + _field("disagreement_explainer", "train_on_test", "Train on test (debug)",
                 _val(de, "train_on_test", False),
                 "Reuses data across all splits. Not independent heldout evaluation.", kind="bool")
        + '</div>'
    )
    return _section("Pipeline", "Sampling, parallelism, and annotator-handling knobs.",
                    body, icon="⚙", icon_class="purple")


def _repo_base_dir() -> str:
    """Absolute path to the repository root."""
    try:
        from src.disagreement_analysis.disagreement_explainer import (
            DisagreementExplainer,
        )
        return DisagreementExplainer.BASE_DIR
    except Exception:
        return os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir))


def _resolve_under_repo(path: str) -> str:
    """Resolve a config path (which may be repo-relative or absolute) to an absolute path.

    Config files store paths like ``data/...`` or ``prompts/...`` which are
    relative to the repo root. They may also be absolute. This normalizes both.
    """
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    base = _repo_base_dir()
    return os.path.normpath(os.path.join(base, path))


def _approved_preview_path(path: str, roots: tuple[str, ...], kind: str = "dir") -> str:
    """Canonicalize and authorize a preview path before inspecting it."""
    if not path:
        raise ValueError("Preview path is empty.")
    candidate = os.path.realpath(path)
    for root in roots:
        if candidate == root:
            checked = root
        elif candidate.startswith(os.path.join(root, "")):
            checked = candidate
        else:
            continue
        relative = os.path.relpath(checked, root)
        if relative != "." and any(part.startswith(".") for part in relative.split(os.sep)):
            raise ValueError("Hidden files cannot be previewed.")
        suffix = os.path.splitext(checked)[1].lower()
        if kind == "csv" and suffix != ".csv":
            raise ValueError("Dataset preview requires a CSV file.")
        if kind == "file" and suffix not in {".csv", ".md"}:
            raise ValueError("Only CSV datasets and Markdown guidelines may be inspected.")
        return checked
    raise ValueError("Path is outside the approved preview roots.")


def _combine_dir_and_filename(
    directory: str, filename: str, roots: tuple[str, ...] | None = None,
) -> str:
    """Combine a config directory + filename into a single resolved absolute path.

    Mirrors DisagreementExplainer._resolve_combined_path semantics, then
    resolves to an approved absolute path before any filesystem checks.
    """
    d = (directory or "").strip()
    f = (filename or "").strip()
    if not d:
        return ""
    if roots is None:
        roots = (os.path.realpath(_repo_base_dir()),)
    abs_d = _approved_preview_path(_resolve_under_repo(d), roots)
    if os.path.isfile(abs_d):
        return abs_d
    if f and (d.endswith(("/", os.sep)) or os.path.isdir(abs_d)
              or not os.path.splitext(d)[1]):
        return _approved_preview_path(os.path.join(abs_d, f), roots)
    return abs_d


def _read_data_info(full_path: str) -> dict:
    """Inspect a CSV at ``full_path`` and return preview info for the dashboard.

    Returns: {exists, full_path, error?, rows, columns, annotator_cols,
              more_than_2, sample_rows (list of dicts, up to 2)}.
    Annotator columns are inferred as any column other than 'unique_id' and 'text'
    (matching DisagreementExplainer's own assumption).
    """
    payload: dict = {
        "exists": False,
        "full_path": full_path,
        "rows": 0,
        "columns": [],
        "annotator_cols": [],
        "more_than_2": False,
        "sample_rows": [],
    }
    if not full_path or not os.path.isfile(full_path):
        return payload
    payload["exists"] = True
    try:
        import pandas as pd  # local import; pandas already a runtime dep
    except Exception as e:  # pragma: no cover - defensive
        payload["error"] = f"pandas unavailable: {e}"
        return payload
    try:
        # Read full CSV to get an accurate row count (datasets are small <100k rows).
        df = pd.read_csv(full_path, low_memory=False)
    except Exception as e:
        payload["error"] = f"Could not read CSV: {e}"
        return payload

    cols = [str(c) for c in df.columns]
    annotator_cols = [c for c in cols if c not in {"unique_id", "text"}]
    head = df.head(2).fillna("").astype(str)
    sample_rows = head.to_dict(orient="records")

    # Unique label values across all annotator columns. The preprocessor uses
    # this to compute `num_combinations = len(labels)**2` per label-pair, which
    # in turn drives chunk sizing. Treat empty/whitespace as missing.
    label_values: set = set()
    for c in annotator_cols:
        vals = df[c].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        label_values.update(vals.unique().tolist())
    num_label_types = len(label_values)

    payload.update({
        "rows": int(len(df)),
        "columns": cols,
        "annotator_cols": annotator_cols,
        "more_than_2": len(annotator_cols) > 2,
        "num_label_types": int(num_label_types),
        "sample_rows": sample_rows,
    })
    return payload


# Cache of expensive preprocess() preview runs keyed by
# All preprocessing inputs participate in the bounded cache.
_PREVIEW_CACHE: dict = {}
_PREVIEW_CACHE_MAX = 8


def _preview_preprocess(
    full_path: str, example_size: int, more_than_2: bool,
    num_annotations: int = -1, random_state: int = 42, train_on_test: bool = False,
    force: bool = False,
) -> dict:
    """Run the real Preprocessor against ``full_path`` (in a temp dir) and return
    the populated ``stats`` dict. This is the same code path the run uses, so the
    chunk count / kept-pair / disagreement-loss numbers are exact rather than
    estimated. Cached by file identity and all preprocessing settings.
    """
    out: dict = {"ok": False}
    if not full_path or not os.path.isfile(full_path):
        out["error"] = "data file not found"
        return out
    try:
        ex = int(example_size)
    except Exception:
        out["error"] = "invalid example_size"
        return out
    if ex <= 0:
        out["error"] = "example_size must be > 0"
        return out

    try:
        mtime = os.path.getmtime(full_path)
    except OSError as e:
        out["error"] = f"stat failed: {e}"
        return out
    cache_key = (full_path, mtime, ex, bool(more_than_2), num_annotations, random_state, train_on_test)
    if force:
        _PREVIEW_CACHE.pop(cache_key, None)
    if cache_key in _PREVIEW_CACHE:
        return _PREVIEW_CACHE[cache_key]

    import contextlib
    import io
    import tempfile

    try:
        import pandas as pd
    except Exception as e:
        out["error"] = f"pandas unavailable: {e}"
        return out

    # Import the Preprocessor lazily so importing this dashboard module doesn't
    # pull in scikit-learn/krippendorff for callers that don't need a preview.
    try:
        from src.disagreement_analysis.preprocessor_disagreement import (
            Preprocessor,
        )
        from src.disagreement_analysis.disagreement_explainer import DisagreementExplainer
    except Exception as e:
        out["error"] = f"preprocessor import failed: {e}"
        return out

    try:
        explainer = DisagreementExplainer()
        explainer.data_path = full_path
        explainer.data_filename = ""
        explainer.num_annotations = num_annotations
        df = explainer.get_annotation_data()
    except Exception as e:
        out["error"] = f"could not read CSV: {e}"
        return out

    pp = Preprocessor()
    with tempfile.TemporaryDirectory(prefix="chardis_dash_preview_") as td:
        tp = os.path.join(td, "train")
        vp = os.path.join(td, "val")
        sp = os.path.join(td, "test")
        try:
            # Suppress preprocessor's verbose stdout so we don't spam the dashboard log.
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                pp.preprocess(
                    df,
                    example_size=ex,
                    more_than_2=bool(more_than_2),
                    num_annotations=num_annotations,
                    random_state=random_state,
                    train_path=tp,
                    validation_path=vp,
                    test_path=sp,
                    train_on_test=train_on_test,
                )
        except Exception as e:
            out["error"] = f"preprocess failed: {e.__class__.__name__}: {e}"
            return out

        # Read the actual split sizes from the CSVs the preprocessor just wrote.
        try:
            train_rows = len(pd.read_csv(os.path.join(tp, "train.csv")))
            val_rows = len(pd.read_csv(os.path.join(vp, "validation.csv")))
            test_rows = len(pd.read_csv(os.path.join(sp, "test.csv")))
        except (OSError, ValueError) as exc:
            out["error"] = f"Could not read preview split counts. {exc}"
            return out

    stats = dict(pp.stats)
    stats.update({
        "ok": True,
        "example_size": ex,
        "more_than_2": bool(more_than_2),
        "num_annotations": num_annotations,
        "random_state": random_state,
        "train_on_test": train_on_test,
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "test_rows": test_rows,
    })

    # Bound the cache.
    if len(_PREVIEW_CACHE) >= _PREVIEW_CACHE_MAX:
        # drop an arbitrary entry (FIFO-ish)
        try:
            _PREVIEW_CACHE.pop(next(iter(_PREVIEW_CACHE)))
        except StopIteration:
            pass
    _PREVIEW_CACHE[cache_key] = stats
    return stats


def _guidelines_preview_html(de: dict, roots: tuple[str, ...] | None = None) -> str:
    """Render a read-only preview of the guidelines markdown referenced by the config."""
    rel_dir = str(_val(de, "guidelines_path", "") or "").strip()
    rel_name = str(_val(de, "guidelines_filename", "") or "").strip()
    display = (rel_dir + (rel_name if (rel_dir.endswith(("/", os.sep)) or not rel_name)
                          else ("/" + rel_name)) if rel_dir else "")
    if not rel_dir:
        body = '<div class="prompt-preview empty">No guidelines path set yet — fill it in under Run Setup → Locations.</div>'
        meta = ""
    else:
        if roots is None:
            roots = (os.path.realpath(_repo_base_dir()),)
        abs_path = _combine_dir_and_filename(rel_dir, rel_name, roots)
        if abs_path:
            abs_path = _approved_preview_path(abs_path, roots, "file")
        if abs_path and os.path.isfile(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    text = f.read()
                if not text.strip():
                    body = f'<div class="prompt-preview empty">{escape(display)} is empty.</div>'
                else:
                    body = f'<div class="prompt-preview">{escape(text)}</div>'
            except Exception as e:
                body = (
                    f'<div class="prompt-preview empty">Could not read '
                    f'{escape(display)}: {escape(str(e))}</div>'
                )
        else:
            body = (
                f'<div class="prompt-preview empty">Guidelines file not found at '
                f'{escape(abs_path or display)}.</div>'
            )
        meta = f'<p class="prompt-preview-meta">📄 {escape(display)} · read-only</p>'

    body_html = meta + body
    return _section(
        "Guidelines (read-only)",
        "Preview of the annotation guidelines fed into the Rationales LLM.",
        body_html,
        icon="📖",
        icon_class="teal",
    )


def _evaluation_section_html(ev: dict) -> str:
    """Required eval settings: model + iterations. Rest goes in Advanced."""
    head = (
        '<div class="row">'
        + _field("evaluation", "eval_model", "Evaluation model",
                 _val(ev, "eval_model", ""),
                 "Pick from the catalog or type a custom value.", kind="model")
        + _field("evaluation", "num_iterations", "Iterations per item",
                 _val(ev, "num_iterations", 3),
                 "How many times each item is annotated for inter-iteration agreement.",
                 kind="number")
        + '</div>'
    )

    # Advanced fields, rendered in pairs.
    advanced_fields: list[str] = []
    advanced_specs = (
        ("provider",           "Provider",            "openai, compatible, or local", "text"),
        ("auth_mode",          "Authentication",      "api-key, or none for literal loopback endpoints", "text"),
        ("api_key_env",        "Credential variable", "Environment variable name, never its value", "text"),
        ("message_role",       "Instruction role",    "system or developer", "text"),
        ("scope",              "Auth scope",          "Legacy compatibility field. Current adapters do not acquire OAuth tokens.", "text"),
        ("openai_endpoint",    "Endpoint",            "Base URL of the selected service.",        "text"),
        ("openai_api_version", "API version",         "Legacy compatibility field. Not sent by current adapters.", "text"),
        ("num_threads",        "Threads",             "Parallel worker threads per batch.",      "number"),
        ("concepts_per_batch", "Concepts per batch",  "Hypotheses bundled into a single multi-concept prompt. Item batch size is fixed at 1.", "number"),
        ("top_hypotheses",     "Top hypotheses cap",  "Max hypotheses to evaluate (top N by frequency).", "number"),
    )
    for fname, label, help_text, kind in advanced_specs:
        default = _val(ev, fname, "")
        advanced_fields.append(
            _field("evaluation", fname, label, default, help_text, kind=kind)
        )
    grid_rows: list[str] = []
    for i in range(0, len(advanced_fields), 2):
        chunk = "".join(advanced_fields[i:i + 2])
        style = ' style="margin-top:14px;"' if i > 0 else ""
        grid_rows.append(f'<div class="row"{style}>{chunk}</div>')
    advanced_html = (
        '<details class="advanced-settings">'
        '<summary>Advanced settings</summary>'
        '<div class="advanced-body">'
        + "".join(grid_rows)
        + '</div>'
        '</details>'
    )

    body = head + advanced_html
    return _section(
        "Annotation (Eval)",
        "LLM-as-judge settings used for the validation + heldout evaluation stages.",
        body, icon="◆", icon_class="orange",
        required=True, required_class="orange",
    )


def _about_version_section_html(config: dict) -> str:
    """Top-of-page section with free-text version notes + live model summary.

    The textarea persists into ``disagreement_explainer.version_note`` (the
    same field the CLI's "save as new version" prompt writes). The model
    pills mirror the current Rationales / Synthesizer / Annotator (Eval)
    selections and are updated live by JS as those selects change.
    """
    de = config.get("disagreement_explainer", {}) or {}
    note_meta = de.get("version_note") or {}
    note_value = note_meta.get("value") if isinstance(note_meta, dict) else ""
    note_value = "" if note_value is None else str(note_value)

    rat_model = ((config.get("rationales_openai") or {}).get("openai_model") or {}).get("value", "") or ""
    syn_model = ((config.get("synthesizer_openai") or {}).get("openai_model") or {}).get("value", "") or ""
    ann_model = ((config.get("evaluation") or {}).get("eval_model") or {}).get("value", "") or ""

    note_name = "disagreement_explainer__version_note"

    body = f"""
<div class="field">
  <span class="label-mono">Notes</span>
  <textarea id="{note_name}" name="{note_name}" rows="3"
            placeholder="What's different about this version? Why are you saving it?">{escape(note_value)}</textarea>
  <p class="hint">Stored in <code>disagreement_explainer.version_note</code>. Useful for spotting differences when comparing past versions.</p>
</div>
<div class="model-pills" aria-live="polite">
  <span class="pill"><span class="pill-label">Rationales</span><span class="pill-value" id="about-rat-model">{escape(rat_model or "—")}</span></span>
  <span class="pill"><span class="pill-label">Synthesizer</span><span class="pill-value" id="about-syn-model">{escape(syn_model or "—")}</span></span>
  <span class="pill"><span class="pill-label">Annotator (Eval)</span><span class="pill-value" id="about-ann-model">{escape(ann_model or "—")}</span></span>
</div>
"""
    return _section(
        "About this version",
        "Free-text notes plus a live summary of the LLMs in this configuration.",
        body,
        icon="ⓘ",
        icon_class="blue",
    )


def _ensure_version_note_field(config: dict) -> None:
    """Pre-seed ``disagreement_explainer.version_note`` so _apply_form accepts it."""
    sec = config.setdefault("disagreement_explainer", {})
    if not isinstance(sec, dict):
        return
    if "version_note" not in sec or not isinstance(sec.get("version_note"), dict):
        sec["version_note"] = {
            "description": "Free-text note describing this saved version.",
            "value": "",
        }


def _eval_setup_pane_html(config: dict) -> str:
    ev = config.get("evaluation", {}) or {}
    return _evaluation_section_html(ev)


def _method_setup_pane_html(config: dict, roots: tuple[str, ...] | None = None) -> str:
    de = config.get("disagreement_explainer", {}) or {}
    rat = config.get("rationales_openai", {}) or {}
    syn = config.get("synthesizer_openai", {}) or {}
    return (
        _openai_section_html("Rationales LLM", "rationales_openai", rat)
        + _openai_section_html("Synthesizer LLM", "synthesizer_openai", syn,
                               icon_class="teal")
        + _guidelines_preview_html(de, roots)
    )


def _run_setup_pane_html(config: dict) -> str:
    de = config.get("disagreement_explainer", {}) or {}
    return (
        _locations_section_html(de)
        + _pipeline_section_html(de)
    )


def _settings_tab_html(config: dict) -> str:
    """Legacy single-pane renderer; kept for backwards compatibility."""
    return _method_setup_pane_html(config) + _run_setup_pane_html(config)


def _render_page(
    config: dict, project_name: str, version: str, roots: tuple[str, ...] | None = None,
) -> str:
    config = runtime_config(config)
    _ensure_version_note_field(config)
    about_html = _about_version_section_html(config)
    method_html = _method_setup_pane_html(config, roots)
    run_html = _run_setup_pane_html(config)
    eval_html = _eval_setup_pane_html(config)
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Disagreement Analysis · {escape(project_name)} · v{escape(str(version))}</title>
{_PAGE_CSS}
</head><body>
<header>
  <div class="header-inner">
    <h1>Disagreement Analysis Settings <span class="v">{escape(project_name)} · v{escape(str(version))}</span></h1>
  </div>
</header>
<main>
  <div class="pane-tabs">
    <button type="button" class="pane-tab active" data-pane-target="method">Method Setup</button>
    <button type="button" class="pane-tab"        data-pane-target="run">Run Setup</button>
    <button type="button" class="pane-tab"        data-pane-target="eval">Annotation (Eval)</button>
  </div>
  <form id="settings-form">
    <section class="about-version-wrap">
      {about_html}
    </section>
    <div class="pane active" data-pane="method">
      <div class="confirm-row">
        <button type="button" class="btn primary" data-confirm="method">Confirm Method Setup</button>
        <span class="confirm-state" id="confirm-method-state"></span>
      </div>
      {method_html}
    </div>
    <div class="pane" data-pane="run">
      <div class="confirm-row">
        <button type="button" class="btn primary" data-confirm="run">Confirm Run Setup</button>
        <span class="confirm-state" id="confirm-run-state"></span>
      </div>
      {run_html}
    </div>
    <div class="pane" data-pane="eval">
      <div class="confirm-row">
        <button type="button" class="btn primary" data-confirm="eval">Confirm Annotation (Eval)</button>
        <span class="confirm-state" id="confirm-eval-state"></span>
      </div>
      {eval_html}
    </div>
  </form>
</main>
<div class="footer-bar">
  <span id="save-error" role="alert"></span>
  <button type="button" class="btn" id="btn-cancel" onclick="cancelChanges()">Cancel</button>
  <button type="button" class="btn primary" id="btn-save" disabled
          title="Confirm Method Setup, Run Setup, and Annotation (Eval) first"
          onclick="saveChanges()">Confirm Disagreement Settings</button>
</div>
<div class="overlay" id="overlay">
  <div class="check">✓</div>
  <div class="msg">Settings saved. You can close this tab.</div>
</div>
<script>
// ── Sub-tab switching ──────────────────────────────────────────────────────
document.querySelectorAll(".pane-tab").forEach(tab => {{
  tab.addEventListener("click", () => {{
    const target = tab.dataset.paneTarget;
    document.querySelectorAll(".pane-tab").forEach(t =>
      t.classList.toggle("active", t === tab));
    document.querySelectorAll(".pane").forEach(p =>
      p.classList.toggle("active", p.dataset.pane === target));
  }});
}});

// ── Per-pane confirm tracking ──────────────────────────────────────────────
const confirmState = {{ method: false, run: false, eval: false }};
const saveBtn = document.getElementById("btn-save");
function refreshSaveBtn() {{
  const ok = confirmState.method && confirmState.run && confirmState.eval;
  saveBtn.disabled = !ok;
  if (ok) saveBtn.removeAttribute("title");
  else    saveBtn.setAttribute("title", "Confirm Method Setup, Run Setup, and Annotation (Eval) first");
}}
function _label(key) {{
  if (key === "method") return "Method Setup";
  if (key === "run")    return "Run Setup";
  if (key === "eval")   return "Annotation (Eval)";
  return key;
}}
function markConfirmed(key) {{
  const btn = document.querySelector('button[data-confirm="' + key + '"]');
  if (btn) {{
    btn.classList.add("confirmed");
    btn.textContent = "✓ " + _label(key) + " confirmed";
  }}
  const stateEl = document.getElementById("confirm-" + key + "-state");
  if (stateEl) {{ stateEl.classList.add("ok"); stateEl.textContent = "Locked in. Edit any field in this pane to re-confirm."; }}
  refreshSaveBtn();
}}
function invalidateConfirm(key) {{
  if (!confirmState[key]) return;
  confirmState[key] = false;
  const btn = document.querySelector('button[data-confirm="' + key + '"]');
  if (btn) {{
    btn.classList.remove("confirmed");
    btn.textContent = "Confirm " + _label(key);
  }}
  const stateEl = document.getElementById("confirm-" + key + "-state");
  if (stateEl) {{ stateEl.classList.remove("ok"); stateEl.textContent = ""; }}
  refreshSaveBtn();
}}
document.querySelectorAll('button[data-confirm]').forEach(btn => {{
  btn.addEventListener("click", () => {{
    const key = btn.dataset.confirm;
    confirmState[key] = true;
    markConfirmed(key);
  }});
}});
// Editing any field within a pane invalidates only that pane's confirm.
document.querySelectorAll(".pane").forEach(pane => {{
  const key = pane.dataset.pane;
  const handler = () => invalidateConfirm(key);
  pane.addEventListener("input", handler);
  pane.addEventListener("change", handler);
}});
refreshSaveBtn();

// ── Live file-status banners (data + guidelines + outputs dir) ─────────────
function _wireFileStatus(elId) {{
  const status = document.getElementById(elId);
  if (!status) return;
  const dirEl  = document.getElementById(status.dataset.dirInput);
  const nameEl = status.dataset.nameInput
    ? document.getElementById(status.dataset.nameInput) : null;
  if (!dirEl) return;
  const kind = status.dataset.kind || "data";
  const dot = status.querySelector(".dot");
  const msg = status.querySelector(".msg");
  let timer = null;

  async function check() {{
    const dir  = dirEl.value || "";
    const name = nameEl ? (nameEl.value || "") : "";
    status.classList.remove("ok", "missing");
    status.classList.add("checking");
    if (kind === "outputs") {{
      msg.textContent = "Checking outputs directory…";
    }} else if (kind === "guidelines") {{
      msg.textContent = "Checking guidelines file…";
    }} else {{
      msg.textContent = "Checking data file…";
    }}
    try {{
      const url = "/check-file?dir=" + encodeURIComponent(dir)
                + "&name=" + encodeURIComponent(name)
                + (kind === "outputs" ? "&kind=dir" : "");
      const r = await fetch(url);
      const data = await r.json();
      status.classList.remove("checking");
      if (kind === "outputs") {{
        if (!dir) {{
          status.classList.add("missing");
          msg.textContent = "⚠ Outputs directory not set.";
        }} else if (data.is_file) {{
          status.classList.add("missing");
          msg.textContent = "⚠ Path points to a file, not a directory: " + data.full_path;
        }} else if (data.exists) {{
          status.classList.add("ok");
          msg.textContent = "✓ Outputs directory ready: " + data.full_path;
        }} else {{
          status.classList.add("ok");
          msg.textContent = "↳ Outputs directory will be created at: " + data.full_path;
        }}
      }} else {{
        const noun = (kind === "guidelines" ? "Guidelines file" : "Data file");
        if (data.exists) {{
          status.classList.add("ok");
          msg.textContent = "✓ " + noun + " found at " + data.full_path;
        }} else {{
          status.classList.add("missing");
          msg.textContent = "⚠ " + noun + " does not exist at " + (data.full_path || "(empty path)");
        }}
      }}
    }} catch (err) {{
      status.classList.remove("checking");
      status.classList.add("missing");
      msg.textContent = "⚠ Could not check: " + err;
    }}
  }}
  function debounced() {{ clearTimeout(timer); timer = setTimeout(check, 250); }}
  dirEl.addEventListener("input", debounced);
  if (nameEl) nameEl.addEventListener("input", debounced);
  check();
}}
_wireFileStatus("data-file-status");
_wireFileStatus("guidelines-file-status");
_wireFileStatus("output-dir-status");

// ── Data preview + annotator detection + chunk-size summary ────────────────
const _previewEl  = document.getElementById("data-preview");
const _alertEl    = document.getElementById("annotator-alert");
const _chunkEl    = document.getElementById("chunk-summary");
const _dirInput   = document.getElementById("disagreement_explainer__data_path");
const _nameInput  = document.getElementById("disagreement_explainer__data_filename");
const _exSizeInput = document.getElementById("disagreement_explainer__example_size");
const _moreThan2El = document.getElementById("disagreement_explainer__more_than_2");
const _numAnnotationsEl = document.getElementById("disagreement_explainer__num_annotations");
const _randomStateEl = document.getElementById("disagreement_explainer__random_state");
const _trainOnTestEl = document.getElementById("disagreement_explainer__train_on_test");

let _lastDataInfo = null;

function _esc(s) {{
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}}

function _renderPreview(info) {{
  if (!_previewEl) return;
  if (!info || !info.exists) {{
    _previewEl.innerHTML = '';
    return;
  }}
  if (info.error) {{
    _previewEl.innerHTML = '<p class="preview-meta">First rows preview</p>'
      + '<div class="empty">' + _esc(info.error) + '</div>';
    return;
  }}
  const cols = info.columns || [];
  const annSet = new Set(info.annotator_cols || []);
  const rows = info.sample_rows || [];
  const head = '<tr>' + cols.map(c => {{
    const cls = annSet.has(c) ? ' class="annotator-col"' : '';
    return '<th' + cls + '>' + _esc(c) + '</th>';
  }}).join('') + '</tr>';
  const body = rows.length === 0
    ? '<tr><td colspan="' + cols.length + '" style="font-style:italic;color:var(--text-tertiary)">(no rows)</td></tr>'
    : rows.map(r => '<tr>' + cols.map(c => '<td>' + _esc(r[c]) + '</td>').join('') + '</tr>').join('');
  const meta = 'First ' + rows.length + ' of ' + info.rows + ' rows · '
    + cols.length + ' columns (' + (info.annotator_cols.length) + ' annotator) · scroll →';
  _previewEl.innerHTML =
    '<p class="preview-meta">' + _esc(meta) + '</p>'
    + '<div class="table-scroll"><table><thead>' + head + '</thead>'
    + '<tbody>' + body + '</tbody></table></div>';
}}

function _renderAnnotatorAlert(info) {{
  if (!_alertEl) return;
  if (!info || !info.exists) {{ _alertEl.style.display = 'none'; return; }}
  const cols = info.annotator_cols || [];
  const n = cols.length;
  _alertEl.style.display = '';
  if (Boolean(_moreThan2El && _moreThan2El.checked) !== info.more_than_2) {{
    _alertEl.classList.add('warn');
    _alertEl.textContent = '⚠ Detected ' + n +
      ' annotator columns. Check the selected "More than 2 annotators" setting.';
  }} else {{
    _alertEl.classList.remove('warn');
    _alertEl.textContent = '✓ Detected ' + n + ' annotator column' +
      (n === 1 ? '' : 's') + (n ? ' (' + cols.join(', ') + ')' : '') +
      '. The selected annotator mode matches the data.';
  }}
}}

function _chunkRefreshFooter() {{
  // Renders a footer row with a "Refresh" button. When the current ex_size /
  // more_than_2 don't match the values that produced the last preview, also
  // shows a "stale" hint to nudge the user to click it.
  const stale = _isPreviewStale();
  const busy = _previewBusy;
  const label = busy ? '⟳ Refreshing…' : '↻ Refresh';
  const btn = '<button type="button" id="chunk-refresh-btn" class="btn btn-ghost"'
            + (busy ? ' disabled' : '')
            + ' style="font-size:12px;padding:4px 10px;">' + label + '</button>';
  const hint = stale
    ? '<span style="color:#b58900;font-size:12px;">Settings changed — click Refresh to recompute.</span>'
    : '<span style="color:#888;font-size:12px;">Re-run preprocessor preview.</span>';
  return '<div style="display:flex;align-items:center;gap:10px;margin-top:6px;grid-column:1/-1;">'
       + btn + hint + '</div>';
}}

function _wireChunkRefreshBtn() {{
  const b = document.getElementById('chunk-refresh-btn');
  if (b && !b._wired) {{
    b._wired = true;
    b.addEventListener('click', function() {{ _refreshDataInfo({{force: true}}); }});
  }}
}}

function _renderChunkSummary(info) {{
  if (!_chunkEl) return;
  if (_isPreviewStale()) {{
    _chunkEl.innerHTML = '<span>Preview is out of date for the selected settings.</span>' + _chunkRefreshFooter();
    _wireChunkRefreshBtn();
    return;
  }}
  if (info && info.error) {{
    _chunkEl.innerHTML = '<span>' + _esc(info.error) + '</span>' + _chunkRefreshFooter();
    _wireChunkRefreshBtn();
    return;
  }}
  const ex = parseInt((_exSizeInput && _exSizeInput.value) || '0', 10) || 0;
  if (!ex) {{
    _chunkEl.classList.add('empty');
    _chunkEl.textContent = 'Set "Examples per chunk" to see chunking info.';
    return;
  }}

  // Same formula as preprocessor.create_multiple_stratified_samples for chunk size:
  //   num_combinations = L²;  sample_per_class = ex // L²
  //   chunk base = sample_per_class * L²  + up to 6 extras (3 disagree + 3 agree)
  const L = info && info.preview_stats && info.preview_stats.ok
    ? info.preview_stats.num_label_types : (info && info.exists ? (info.num_label_types || 0) : 0);
  const combos = L > 0 ? L * L : 0;
  const spc = combos > 0 ? Math.floor(ex / combos) : 0;
  const chunkBase = combos > 0 ? spc * combos : ex;
  const chunkSize = chunkBase + 6;
  const baseLabel = combos > 0
    ? (chunkBase + ' stratified (' + L + ' labels → ' + combos + ' bins × ' + spc + ') + ≤6 extras')
    : (ex + ' stratified + ≤6 extras (assumes balanced labels)');

  // Hard error: with too few examples, the preprocessor can't form a single bin.
  if (combos > 0 && chunkBase === 0) {{
    _chunkEl.classList.remove('empty');
    _chunkEl.innerHTML =
      '<span class="stat"><span class="k">⚠ chunk size is 0</span><span class="v">' +
      ex + ' examples / ' + combos + ' label-pair bins rounds to 0 — bump examples per chunk to at least ' + combos + '.</span></span>';
    return;
  }}

  // Need data file to compute anything beyond chunk size.
  if (!info || !info.exists) {{
    _chunkEl.classList.remove('empty');
    _chunkEl.innerHTML =
      '<span class="stat"><span class="k">Chunk size</span><span class="v">≤ ' + chunkSize + ' rows</span></span>'
      + '<span class="stat"><span class="k">Per chunk</span><span class="v">' + baseLabel + '</span></span>'
      + '<span class="stat"><span class="k">Total chunks</span><span class="v">— (need data file)</span></span>'
      + _chunkRefreshFooter();
    _wireChunkRefreshBtn();
    return;
  }}

  const ps = info.preview_stats;
  if (ps && ps.ok) {{
    // Exact numbers from a real Preprocessor run.
    const numChunks    = ps.num_chunks || 0;
    const totalRows    = ps.total_chunk_rows || 0;
    const keptPairs    = ps.kept_cluster_pairs || 0;
    const candPairs    = ps.candidate_cluster_pairs || 0;
    const trainRows    = ps.train_rows || 0;
    const valRows      = ps.validation_rows || 0;
    const testRows     = ps.test_rows || 0;
    const pairsNote    = ps.more_than_2
      ? (' (across ' + keptPairs + ' kept / ' + candPairs + ' candidate cluster pairs)')
      : '';
    _chunkEl.classList.remove('empty');
    _chunkEl.innerHTML =
      '<span class="stat"><span class="k">Chunk size</span><span class="v">' +
        (numChunks ? Math.round(totalRows / numChunks) : chunkSize) +
        ' rows avg (≤ ' + chunkSize + ')</span></span>'
      + '<span class="stat"><span class="k">Per chunk</span><span class="v">' + baseLabel + '</span></span>'
      + '<span class="stat"><span class="k">Total chunks</span><span class="v">' + numChunks +
        ' (' + totalRows + ' rows total)' + pairsNote + '</span></span>'
      + '<span class="stat"><span class="k">From</span><span class="v">' + trainRows +
        ' train / ' + valRows + ' val / ' + testRows + ' test · ' + ps.input_rows + ' processed · ' + L + ' labels</span></span>'
      + (ps.train_on_test ? '<strong>Debug mode reuses data across splits. Not independent heldout evaluation.</strong>' : '')
      + _chunkRefreshFooter();
    _wireChunkRefreshBtn();
    return;
  }}

  const errLine = (ps && ps.error)
    ? ('<span class="stat"><span class="k">Preview unavailable</span><span class="v">' + _esc(ps.error) + '</span></span>')
    : '<span class="stat"><span class="k">…</span><span class="v">computing exact chunk count</span></span>';
  _chunkEl.classList.remove('empty');
  _chunkEl.innerHTML = errLine + _chunkRefreshFooter();
  _wireChunkRefreshBtn();
}}

async function _refreshDataInfo(opts) {{
  if (!_dirInput) return;
  opts = opts || {{}};
  const dir  = _dirInput.value || "";
  const name = _nameInput ? (_nameInput.value || "") : "";
  // Pass example_size + more_than_2 so the server can run an actual preprocess preview.
  const ex = (_exSizeInput && _exSizeInput.value) || "";
  const m2 = (_moreThan2El && _moreThan2El.checked) ? "true" : "false";
  const settings = _previewSettings();
  const requestId = ++_previewRequestId;
  if (opts.force) {{
    _previewBusy = true;
    _renderChunkSummary(_lastDataInfo);
  }}
  try {{
    const url = "/data-info?dir=" + encodeURIComponent(dir)
              + "&name=" + encodeURIComponent(name)
              + (ex ? ("&example_size=" + encodeURIComponent(ex)) : "")
              + "&more_than_2=" + m2
              + "&num_annotations=" + encodeURIComponent(settings.num_annotations)
              + "&random_state=" + encodeURIComponent(settings.random_state)
              + "&train_on_test=" + settings.train_on_test
              + (opts.force ? "&force=true" : "");
    const r = await fetch(url);
    if (!r.ok) throw new Error("Could not load preview");
    const info = await r.json();
    if (requestId !== _previewRequestId) return;
    _lastDataInfo = info;
    if (info.preview_stats && info.preview_stats.ok) {{
      settings.more_than_2 = String(info.preview_stats.more_than_2);
    }}
    _lastPreviewKey = JSON.stringify(settings);
    _previewBusy = false;
    _renderPreview(info);
    _renderAnnotatorAlert(info);
    _renderChunkSummary(info);
  }} catch (err) {{
    if (requestId !== _previewRequestId) return;
    _lastPreviewKey = JSON.stringify(settings);
    _lastDataInfo = {{error: "Could not load preview. " + err}};
    _previewBusy = false;
    if (_previewEl) _previewEl.innerHTML = '';
    if (_alertEl) _alertEl.style.display = 'none';
    _renderChunkSummary(_lastDataInfo);
  }}
}}
let _lastPreviewKey = "";
let _previewBusy = false;
let _previewRequestId = 0;
function _previewSettings() {{
  return {{
    dir: (_dirInput && _dirInput.value) || "",
    name: (_nameInput && _nameInput.value) || "",
    example_size: (_exSizeInput && _exSizeInput.value) || "",
    more_than_2: String(Boolean(_moreThan2El && _moreThan2El.checked)),
    num_annotations: _numAnnotationsEl ? _numAnnotationsEl.value : "-1",
    random_state: _randomStateEl ? _randomStateEl.value : "42",
    train_on_test: String(Boolean(_trainOnTestEl && _trainOnTestEl.checked))
  }};
}}
function _isPreviewStale() {{
  return _lastPreviewKey !== "" && _lastPreviewKey !== JSON.stringify(_previewSettings());
}}
let _dataInfoTimer = null;
function _debouncedDataInfo() {{
  clearTimeout(_dataInfoTimer);
  _dataInfoTimer = setTimeout(_refreshDataInfo, 300);
}}
function _markStale() {{
  // Don't re-run preview automatically — just re-render so the refresh button
  // appears with the "stale" warning.
  if (_lastDataInfo) {{
    _renderAnnotatorAlert(_lastDataInfo);
    _renderChunkSummary(_lastDataInfo);
  }}
}}
if (_dirInput)  _dirInput.addEventListener("input", () => {{ _markStale(); _debouncedDataInfo(); }});
if (_nameInput) _nameInput.addEventListener("input", () => {{ _markStale(); _debouncedDataInfo(); }});
// Don't auto-rerun on ex_size / more_than_2 changes — just flag stale and let
// the user click "Refresh" under the chunk summary.
if (_exSizeInput) _exSizeInput.addEventListener("input", _markStale);
if (_moreThan2El) _moreThan2El.addEventListener("change", _markStale);
if (_numAnnotationsEl) _numAnnotationsEl.addEventListener("input", _markStale);
if (_randomStateEl) _randomStateEl.addEventListener("input", _markStale);
if (_trainOnTestEl) _trainOnTestEl.addEventListener("change", _markStale);
_refreshDataInfo();

// ── About this version: live model summary ─────────────────────────────────
(function bindAboutVersionPills() {{
  const map = [
    {{ inputId: "rationales_openai__openai_model",  pillId: "about-rat-model" }},
    {{ inputId: "synthesizer_openai__openai_model", pillId: "about-syn-model" }},
    {{ inputId: "evaluation__eval_model",           pillId: "about-ann-model" }},
  ];
  for (const {{inputId, pillId}} of map) {{
    const input = document.getElementById(inputId);
    const pill  = document.getElementById(pillId);
    if (!input || !pill) continue;
    const update = () => {{ pill.textContent = (input.value || "").trim() || "—"; }};
    input.addEventListener("change", update);
    input.addEventListener("input",  update);
    update();
  }}
}})();

// ── Save / cancel ──────────────────────────────────────────────────────────
function collect() {{
  const out = {{}};
  const form = document.getElementById('settings-form');
  for (const el of form.elements) {{
    if (!el.name) continue;
    if (el.type === 'checkbox') out[el.name] = el.checked ? 'true' : 'false';
    else out[el.name] = el.value;
  }}
  return out;
}}
async function saveChanges() {{
  if (saveBtn.disabled) return;
  if (!document.getElementById('settings-form').reportValidity()) return;
  await submitAction('/save', collect(), "Settings saved. You can close this tab.");
}}
async function cancelChanges() {{
  await submitAction('/cancel', {{}}, "Run cancelled. You can close this tab.");
}}
async function runChanges() {{
  if (!document.getElementById('settings-form').reportValidity()) return;
  await submitAction('/run', collect(), "Run requested. Check the terminal.");
}}
async function submitAction(path, payload, message) {{
  const errorEl = document.getElementById('save-error');
  const cancelBtn = document.getElementById('btn-cancel');
  errorEl.textContent = path === '/save'
    ? "Saving settings. Follow any prompts in the terminal." : "";
  saveBtn.disabled = true;
  cancelBtn.disabled = true;
  try {{
    const response = await fetch(path, {{
      method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload)
    }});
    if (!response.ok) {{
      const error = await response.json();
      throw new Error(error.error || "Could not save settings.");
    }}
    document.querySelector('#overlay .msg').textContent = message;
    document.getElementById('overlay').classList.add('active');
  }} catch (err) {{
    errorEl.textContent = String(err);
    cancelBtn.disabled = false;
    refreshSaveBtn();
  }}
}}
</script>
</body></html>
"""


# ── Server plumbing ──────────────────────────────────────────────────────────

@dataclass
class DashboardResult:
    config: dict
    dirty: bool = False
    cancelled: bool = False
    action: str = "save"


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def run_dashboard(
    config: dict, project_name: str, version: str,
    on_save: Callable[[dict], None] | None = None,
    approved_roots: tuple[str, ...] = (),
) -> DashboardResult:
    """Open the disagreement analysis settings dashboard.

    Keep the dashboard open if validation or the save callback fails.
    """
    working = runtime_config(config)
    _ensure_version_note_field(working)
    original = copy.deepcopy(working)
    token = secrets.token_urlsafe(32)
    roots = tuple(os.path.realpath(p) for p in (_repo_base_dir(), *approved_roots))
    de = working.get("disagreement_explainer", {})
    guidelines = _combine_dir_and_filename(_val(de, "guidelines_path", ""),
                                          _val(de, "guidelines_filename", "prompt.md"), roots)
    if guidelines:
        _approved_preview_path(guidelines, roots, "file")
    page_html = _render_page(working, project_name, version, roots)
    page_html = page_html.replace(
        "<script>", "<script>\n"
        f"const launchToken = {json.dumps(token)};\n"
        "const originalFetch = window.fetch.bind(window);\n"
        "window.fetch = (url, options={}) => {"
        "if (new URL(url, location.href).origin !== location.origin) "
        "return Promise.reject(new Error('External requests are not allowed'));"
        "return originalFetch(url, {...options, "
        "headers:{...options.headers, 'Authorization':'Bearer '+launchToken}});};\n", 1)
    page_html = page_html.replace("</form>", '<button type="button" onclick="runChanges()">Run</button></form>', 1)
    result = DashboardResult(original)
    done = threading.Event()
    action_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a, **_k):
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if not self.authorized():
                return
            if self.path in ("/", "/index.html"):
                bootstrap = """<!doctype html><meta charset="utf-8"><p>Opening settings...</p>
<script>
const token = location.hash.slice(1);
history.replaceState(null, '', '/');
fetch('/app', {headers:{Authorization:'Bearer '+token}}).then(async response => {
  if (!response.ok) throw new Error('Launch authorization required');
  const html = await response.text(); document.open(); document.write(html); document.close();
}).catch(() => {document.body.textContent='Launch this dashboard from the CLI.';});
</script>"""
                self._send(200, bootstrap.encode(), "text/html; charset=utf-8")
            elif self.path == "/app":
                self._send(200, page_html.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path.startswith("/check-file"):
                from urllib.parse import urlparse, parse_qs
                qs = parse_qs(urlparse(self.path).query)
                directory = (qs.get("dir") or [""])[0]
                filename  = (qs.get("name") or [""])[0]
                kind      = (qs.get("kind") or [""])[0]
                try:
                    if kind == "dir":
                        full = _approved_preview_path(_resolve_under_repo(directory), roots) if directory else ""
                    else:
                        full = _approved_preview_path(
                            _combine_dir_and_filename(directory, filename, roots), roots, "file",
                        )
                except ValueError:
                    self._send(403, b'{"error":"Preview path not approved."}', "application/json")
                    return
                if kind == "dir":
                    # Directory existence check (outputs folder). Treat existing
                    # files as a problem; non-existent dirs are reported so the
                    # UI can tell the user it'll be created.
                    payload = {
                        "dir": directory,
                        "name": filename,
                        "full_path": full,
                        "kind": "dir",
                        "exists": bool(full) and os.path.isdir(full),
                        "is_file": bool(full) and os.path.isfile(full),
                    }
                else:
                    payload = {
                        "dir": directory,
                        "name": filename,
                        "full_path": full,
                        "exists": bool(full) and os.path.isfile(full),
                    }
                self._send(200, json.dumps(payload).encode("utf-8"),
                           "application/json")
            elif self.path.startswith("/data-info"):
                from urllib.parse import urlparse, parse_qs
                qs = parse_qs(urlparse(self.path).query)
                directory = (qs.get("dir") or [""])[0]
                filename  = (qs.get("name") or [""])[0]
                try:
                    full = _approved_preview_path(
                        _combine_dir_and_filename(directory, filename, roots), roots, "csv",
                    )
                except ValueError:
                    self._send(403, b'{"error":"Preview path not approved."}', "application/json")
                    return
                payload = _read_data_info(full)
                # Optional: actual preprocessing preview when example_size is provided.
                ex_raw = (qs.get("example_size") or [""])[0]
                if ex_raw and payload.get("exists"):
                    try:
                        ex_int = int(ex_raw)
                        m2_raw = (qs.get("more_than_2") or [""])[0].strip().lower()
                        # Default: auto-detect from the data when client doesn't send it.
                        if m2_raw in {"true", "1", "yes"}:
                            m2 = True
                        elif m2_raw in {"false", "0", "no"}:
                            m2 = False
                        else:
                            m2 = bool(payload.get("more_than_2"))
                        payload["preview_stats"] = _preview_preprocess(
                            full, ex_int, m2,
                            num_annotations=int((qs.get("num_annotations") or ["-1"])[0]),
                            random_state=int((qs.get("random_state") or ["42"])[0]),
                            train_on_test=_coerce((qs.get("train_on_test") or ["false"])[0], False),
                            force=_coerce((qs.get("force") or ["false"])[0], False),
                        )
                    except ValueError:
                        payload["preview_stats"] = {"ok": False, "error": "Invalid preview settings."}
                self._send(200, json.dumps(payload).encode("utf-8"),
                           "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if not self.authorized(mutation=True):
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 <= length <= 1024 * 1024:
                    raise ValueError
            except ValueError:
                self._send(400, b'{"error":"Invalid request size."}', "application/json")
                return
            raw = self.rfile.read(length) if length else b""
            with action_lock:
                if done.is_set():
                    self._send(409, b'{"error":"Dashboard already closed."}', "application/json")
                    return
                if self.path in ("/save", "/run"):
                    candidate = copy.deepcopy(working)
                    try:
                        submitted = json.loads(raw.decode("utf-8"))
                        _apply_form(candidate, submitted)
                    except (ValueError, UnicodeError) as exc:
                        message = "Invalid settings JSON." if isinstance(exc, (json.JSONDecodeError, UnicodeError)) else str(exc)
                        self._send(400, json.dumps({"error": message}).encode("utf-8"), "application/json")
                        return
                    dirty = candidate != original
                    try:
                        if dirty and on_save is not None:
                            on_save(candidate)
                    except (OSError, ValueError) as exc:
                        self._send(400, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                        return
                    result.config = candidate
                    result.dirty = dirty
                    result.action = "run" if self.path == "/run" else "save"
                    done.set()
                    self._send(204, b"", "text/plain")
                elif self.path == "/cancel":
                    result.cancelled = True
                    result.action = "cancel"
                    done.set()
                    self._send(204, b"", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")

        def authorized(self, mutation=False):
            host = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != host:
                self._send(403, b'{"error":"Invalid host."}', "application/json")
                return False
            public = not mutation and self.path in ("/", "/index.html")
            if not public and not hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + token):
                self._send(403, b'{"error":"Launch authorization required."}', "application/json")
                return False
            if mutation and (self.headers.get("Origin") != f"http://{host}"
                             or self.headers.get_content_type() != "application/json"):
                self._send(403, b'{"error":"Invalid mutation origin or content type."}', "application/json")
                return False
            return True

    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/#{token}"
    print("\nOpening the private dashboard in your browser.")
    try:
        webbrowser.open(url, new=2)
    except Exception:
        pass

    try:
        done.wait()
        time.sleep(0.4)
    finally:
        server.shutdown()
        server.server_close()

    return result
