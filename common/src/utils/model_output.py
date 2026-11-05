"""Strict, provider independent response contracts."""

import json


def string_list(raw):
    try:
        result = json.loads(raw)
    except (ValueError, TypeError):
        raise ValueError("Model must return a JSON array of nonempty strings.") from None
    if not isinstance(result, list) or not result or any(
            not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError("Model must return a JSON array of nonempty strings.")
    return [item.strip() for item in result]


def annotations(raw, records, specs):
    try:
        rows = json.loads(raw)
    except (ValueError, TypeError):
        raise ValueError("Judge response is not a JSON array.") from None
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Judge response must contain annotation objects.")
    expected_ids = {str(record["unique_id"]) for record in records}
    if len(expected_ids) != len(records):
        raise ValueError("Input IDs collide after serialization.")
    simple = {s["name"]: s for s in specs if s["kind"] == "simple"}
    nested = {s["name"]: s for s in specs if s["kind"] == "nested_simple"}
    expected = {(uid, name) for uid in expected_ids for name in (nested or {None: None})}
    seen = set()

    def check(value, spec):
        kind = spec["label_type"]
        if kind == "boolean":
            valid = type(value) is int and value in (0, 1)
        elif kind == "integer":
            valid = type(value) is int and value >= 0 and (
                spec.get("int_scale_max") is None or value <= spec["int_scale_max"])
        elif kind == "categorical":
            valid = isinstance(value, str) and value in spec.get("categories", [])
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(f"Invalid annotation value for {spec['name']}.")

    for row in rows:
        key = (str(row.get("unique_id")), row.get("nested"))
        if key not in expected or key in seen:
            raise ValueError("Judge response has duplicate, missing, or foreign identities.")
        seen.add(key)
        required = {"unique_id", *simple}
        if nested:
            required.update(("nested", "value"))
            check(row.get("value"), nested[row["nested"]])
        if not required.issubset(row):
            raise ValueError("Judge response is missing required concepts.")
        for name, spec in simple.items():
            check(row[name], spec)
        for name, value in row.items():
            if name in required:
                continue
            suffix = name.rsplit("_", 1)[-1]
            prefix = name.rsplit("_", 1)[0] if "_" in name else None
            if prefix is not None and prefix not in {*simple, "value"}:
                raise ValueError("Judge response has an unexpected field.")
            if suffix in {"Reasoning", "EvidenceSpan"} and isinstance(value, str):
                continue
            if suffix == "Confidence" and type(value) is int and 1 <= value <= 5:
                continue
            raise ValueError("Judge response has an invalid diagnostic field.")
    if seen != expected:
        raise ValueError("Judge response does not cover every input and concept.")
    return rows
