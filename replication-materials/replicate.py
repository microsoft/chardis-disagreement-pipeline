"""Recompute supported validation, heldout and corpus results from local evidence."""

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "replication_results"
METRIC_FIELDS = ("f1", "precision", "recall", "p_value_raw", "n00", "n01", "n10", "n11")


def contained_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path escapes the replication package: {relative}")
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(relative):
    return json.loads(contained_path(relative).read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict):
        return {key: clean(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(child) for child in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(relative, value):
    path = contained_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(relative, frame):
    path = contained_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g")


def load_asset(record):
    path = contained_path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Evidence checksum mismatch: {record['path']}")
    if path.name.endswith(".csv.gz"):
        frame = pd.read_csv(path, float_precision="round_trip", dtype=record["dtypes"])
    elif path.suffix == ".parquet":
        frame = pd.read_parquet(path, engine="pyarrow")
    else:
        raise ValueError(f"Unsupported evidence format: {path.name}")
    if list(frame.columns) != record["columns"] or len(frame) != record["rows"]:
        raise ValueError(f"Evidence shape mismatch: {record['path']}")
    return frame


def binary_metrics(target, judgments):
    target, judgments = np.asarray(target), np.asarray(judgments)
    if target.ndim != 1 or judgments.shape != target.shape or not len(target):
        raise ValueError("Targets and judgments must be nonempty aligned vectors")
    if not np.isin(target, [0, 1]).all() or not np.isin(judgments, [0, 1]).all():
        raise ValueError("Missing or nonbinary judgments cannot be silently imputed")
    table = np.array([
        [np.count_nonzero((judgments == h) & (target == y)) for y in (0, 1)]
        for h in (0, 1)
    ])
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        p_value, test = float("nan"), "degenerate"
    else:
        _, p_value, _, expected = chi2_contingency(table, correction=False)
        test = "chi2"
        if expected.min() < 5:
            _, p_value = fisher_exact(table, alternative="two-sided")
            test = "fisher_exact"
    return {
        "f1": f1_score(target, judgments, average="macro", zero_division=0),
        "precision": precision_score(target, judgments, average="macro", zero_division=0),
        "recall": recall_score(target, judgments, average="macro", zero_division=0),
        "p_value_raw": float(p_value), "test": test,
        **dict(zip(("n00", "n01", "n10", "n11"), table.ravel().tolist())),
    }


def numeric_match(actual, expected, decimals):
    if actual is None or expected is None:
        return actual is expected
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        return False
    if decimals == 0:
        return actual == expected
    return f"{actual:.{decimals}f}" == f"{expected:.{decimals}f}"


def verify_run(run, catalog, assets, alpha):
    targets = assets[run["targets"]]
    judgments = assets[run["judgments"]]
    exported = assets[run["metrics"]]
    expected_ids = [item["id"] for item in catalog]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError(f"Duplicate hypothesis identity in {run['id']}")
    if list(judgments.columns) != ["row", *expected_ids]:
        raise ValueError(f"Judgment columns are not aligned in {run['id']}")
    if (len(targets) != run["rows"] or len(catalog) != run["hypotheses"]
            or len(exported) != len(catalog)
            or not np.array_equal(targets["row"], judgments["row"])
            or not np.array_equal(targets["row"], np.arange(len(targets)))
            or not np.array_equal(exported["hypothesis"], np.arange(1, len(catalog) + 1))):
        raise ValueError(f"Row or hypothesis alignment mismatch in {run['id']}")
    metrics, checks = [], []
    for ordinal, hypothesis in enumerate(catalog):
        values = binary_metrics(targets.observed_disagreement, judgments[hypothesis["id"]])
        row = {
            "hypothesis_id": hypothesis["id"], "source_id": hypothesis["source_id"],
            "hypothesis": hypothesis["text"], "source_order": ordinal,
            **values, "significant": bool(values["p_value_raw"] <= alpha),
        }
        metrics.append(row)
        for field in METRIC_FIELDS:
            actual, expected = values[field], exported.iloc[ordinal][field]
            matches = (actual == expected if field.startswith("n") else
                       np.isclose(actual, expected, atol=1e-14, rtol=1e-10, equal_nan=True))
            checks.append({
                "run": run["id"], "hypothesis_id": hypothesis["id"], "metric": field,
                "actual": actual, "exported": expected, "status": "match" if matches else "mismatch",
            })
    frame = pd.DataFrame(metrics)
    ranked = frame[frame.significant].sort_values(
        ["f1", "precision", "recall", "source_order"], ascending=[False, False, False, True]
    ).reset_index(drop=True)
    return frame, ranked, checks


def bootstrap_f1(target, predictions, iterations=2000, seed=42):
    binary_metrics(target, predictions)
    target, predictions = np.asarray(target), np.asarray(predictions)
    rng = np.random.default_rng(seed)
    codes = 2 * target + predictions
    scores = []
    for start in range(0, iterations, 64):
        samples = codes[rng.choice(len(codes), size=(min(64, iterations - start), len(codes)),
                                   replace=True)]
        tn, fp, fn, tp = [(samples == k).sum(axis=1) for k in range(4)]
        denominators = np.column_stack((2 * tn + fp + fn, 2 * tp + fp + fn))
        numerator = 2 * np.column_stack((tn, tp))
        values = np.divide(numerator, denominators, out=np.zeros_like(numerator, dtype=float),
                           where=denominators != 0)
        # sklearn's macro average includes only labels present in truth or predictions.
        scores.extend(values.sum(axis=1) / (denominators != 0).sum(axis=1))
    low, high = np.percentile(scores, [2.5, 97.5])
    return {"macro_f1": f1_score(target, predictions, average="macro", zero_division=0),
            "f1_ci_low": float(low), "f1_ci_high": float(high),
            "ci_half_width": float((high - low) / 2)}


def choose_threshold(validation_scores):
    if set(validation_scores) != {0.25, 0.5, 0.75}:
        raise ValueError("Threshold selection requires all three validation scores")
    if not all(math.isfinite(value) for value in validation_scores.values()):
        raise ValueError("Invalid validation threshold score")
    best = max(validation_scores.values())
    tied = [t for t in (0.5, 0.25, 0.75) if validation_scores[t] == best]
    return tied[0], tied


def overall_results(extension, runs, catalogs, ranked, assets):
    results, diagnostics, checks = {}, [], []
    by_id = {r["id"]: r for r in runs}
    for condition in extension["conditions"]:
        run_id = condition["run"]
        run = by_id[run_id]
        if condition["status"] == "blocked":
            results[run_id] = dict(condition)
            continue
        if condition["selection"] == "reference":
            features = [h["id"] for h in catalogs[run_id]]
        elif condition["selection"] == "top_3":
            features = ranked[run_id].head(3).hypothesis_id.tolist()
        elif condition["selection"] == "all_significant":
            features = ranked[run_id].hypothesis_id.tolist()
        else:
            raise ValueError("Unknown classifier feature selection")
        if condition["status"] == "convention":
            if features or (run["dataset"], run["method"]) != ("pref", "hypogenic"):
                raise ValueError("Empty hypothesis convention applied outside its authorized scope")
            results[run_id] = dict(condition, macro_f1=0.0, ci_half_width=0.0,
                                   features=[], threshold=None)
            continue
        train = assets[run["judgments"]]
        train_target = assets[run["targets"]].observed_disagreement.to_numpy()
        test = assets[condition["heldout_judgments"]]
        targets = assets[condition["heldout_targets"]]
        if (len(test) != condition["heldout_rows"]
                or not np.array_equal(test.row, targets.row)
                or not np.array_equal(test.row, np.arange(len(test)))):
            raise ValueError(f"Heldout row alignment mismatch {run_id}")
        test_target = targets.observed_disagreement.to_numpy()
        if len(np.unique(train_target)) != 2:
            raise ValueError(f"Classifier requires both validation target classes {run_id}")
        cache = {}

        def fit(feature_list):
            key = tuple(feature_list)
            if not key or len(set(key)) != len(key):
                raise ValueError(f"Empty or duplicated classifier features {run_id}")
            if key in cache:
                return cache[key]
            for frame, y in [(train, train_target), (test, test_target)]:
                for feature in key:
                    binary_metrics(y, frame[feature].to_numpy())
            model = LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs", dual=False, tol=1e-4,
                fit_intercept=True, intercept_scaling=1.0, class_weight="balanced",
                max_iter=2000, random_state=None, warm_start=False,
            )
            model.fit(train[list(key)].to_numpy(), train_target)
            if np.any(model.n_iter_ >= model.max_iter):
                raise ValueError(f"Logistic regression did not converge {run_id}")
            val_probability = model.predict_proba(train[list(key)].to_numpy())[:, 1]
            test_probability = model.predict_proba(test[list(key)].to_numpy())[:, 1]
            thresholds = {}
            for threshold in (0.5, 0.25, 0.75):
                thresholds[threshold] = {
                    "validation_f1": f1_score(train_target, val_probability >= threshold,
                                               average="macro", zero_division=0),
                    **bootstrap_f1(test_target, (test_probability >= threshold).astype("int8")),
                }
            cache[key] = thresholds
            return thresholds

        primary = fit(features)
        threshold, ties = choose_threshold({t: r["validation_f1"] for t, r in primary.items()})
        results[run_id] = {
            "run": run_id, "dataset": run["dataset"], "method": run["method"],
            "generator": run["generator"], "judge": run["judge"], "split": "heldout",
            "status": "computed", "selection": condition["selection"], "features": features,
            "source_features": [next(h["source_id"] for h in catalogs[run_id] if h["id"] == f)
                                for f in features],
            "threshold": threshold, "tied_thresholds": ties, **primary[threshold],
        }
        for t, row in primary.items():
            diagnostics.append(dict(row, run=run_id, threshold=t, features=features,
                                    selection="primary", selected=t == threshold))
        for archived in condition["archived_metrics"]:
            feature_list = archived["features"]
            t = archived["threshold"]
            actual = fit(feature_list)[t]
            diagnostics.append(dict(actual, run=run_id, threshold=t, features=feature_list,
                                    selection="archived", selected=False))
            for field in ("f1", "f1_ci_low", "f1_ci_high"):
                value = actual["macro_f1" if field == "f1" else field]
                checks.append({
                    "run": run_id, "metric": field, "threshold": t, "features": feature_list,
                    "actual": value, "exported": archived[field],
                    "status": "match" if np.isclose(value, archived[field], atol=1e-12,
                                                    rtol=1e-10) else "mismatch",
                })
    return results, diagnostics, checks


def corpus_results(extension, assets):
    results = {}
    for dataset, record in extension["corpora"].items():
        frequency = assets[record["frequency"]]
        counts = frequency[[c for c in frequency if c.startswith("label_")]].to_numpy()
        weights = frequency["items"].to_numpy()
        if (counts < 0).any() or (weights <= 0).any():
            raise ValueError(f"Invalid corpus frequency counts {dataset}")
        total = counts.sum(axis=1)
        classes = (counts > 0).sum(axis=1)
        probability = np.divide(counts, total[:, None], out=np.zeros_like(counts, dtype=float),
                                where=total[:, None] > 0)
        logp = np.log(probability, out=np.zeros_like(probability), where=probability > 0)
        denominator = np.log(np.maximum(classes, 1))
        entropy = np.divide(-(probability * logp).sum(axis=1), denominator,
                            out=np.zeros(len(counts)), where=denominator > 0)
        prevalences = {}
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            disagree = np.where(total <= 2, classes > 1, entropy >= threshold)
            prevalences[str(threshold)] = float(100 * weights[disagree].sum() / weights.sum())
        disagreements = np.where(total <= 2, classes > 1, entropy >= 0.9)
        annotators = assets[record["annotators"]]
        pairs = assets[record["pairs"]]
        contexts = assets[record["contexts"]]
        pair_disagreement = pairs.disagreements * pairs.pairs
        numerator = pair_disagreement[pairs.overlap < 50].sum()
        denominator = pair_disagreement.sum()
        if int((total * weights).sum()) != int(annotators.annotations.sum()):
            raise ValueError(f"Annotation sufficient statistics disagree {dataset}")
        results[dataset] = {
            "corpus_rows": int(weights.sum()),
            "annotators": int((annotators.annotations > 0).sum()),
            "mean_annotators": float(np.average(total, weights=weights)),
            "context_examples": int((contexts.occurrences * contexts["items"]).sum()),
            "disagreements": int(weights[disagreements].sum()),
            "disagreement_pct": prevalences["0.9"],
            "pruned_pct": float(100 * numerator / denominator) if denominator else 0.0,
            "prevalence_pct": prevalences,
            "label_totals": dict(zip(record["labels"], (counts * weights[:, None]).sum(axis=0).tolist())),
            **{name + "_rows": int(frequency.loc[frequency.split == code, "items"].sum())
               for name, code in record["split_codes"].items()},
        }
    return results


def compare_cells(registry, runs, frames, ranked, overall=None, corpora=None):
    results = []
    lookup = {(run["dataset"], run["method"], run["judge"]): run for run in runs}
    for index, cell in enumerate(registry["cells"], 1):
        row = dict(cell, comparison_id=f"numeric-{index}")
        if cell["status"] == "convention":
            row.update(actual=overall[cell["run"]][cell["metric"]],
                       reason="User-approved empty hypothesis set zero, not measured classifier performance")
            results.append(row)
            continue
        if cell["status"] != "supported":
            row["actual"] = None
            results.append(row)
            continue
        run = lookup.get((cell["dataset"], cell["method"], cell["judge"]))
        if cell.get("evidence") == "corpus":
            actual = corpora[cell["dataset"]][cell["metric"]]
            if cell["metric"] == "prevalence_pct":
                actual = actual[str(cell["entropy_threshold"])]
        elif cell["metric"] in ("macro_f1", "ci_half_width"):
            result = overall[cell["run"]]
            actual = result[cell["metric"]]
            row.update(threshold=result["threshold"], features=result["features"])
        elif cell["metric"] == "hypothesis_f1":
            actual = float(frames[cell["run"]].set_index("hypothesis_id").loc[cell["hypothesis_id"], "f1"])
        elif cell["metric"] == "breadth":
            actual = len(ranked[run["id"]])
        elif cell["metric"] == "validation_rows":
            actual = run["rows"]
        else:
            raise ValueError(f"Supported metric has no computation: {cell['metric']}")
        row["actual"] = actual
        row["status"] = "match" if numeric_match(actual, cell["expected"], cell["decimals"]) else "mismatch"
        if run and cell.get("evidence") != "corpus":
            row["run"] = run["id"]
        results.append(row)
    return results


def compare_lists(registry, ranked, catalogs):
    comparisons = []
    table_rows = {}
    for listing in registry["hypothesis_lists"]:
        if listing["status"] != "supported":
            comparisons.append({
                "paper_item": listing["paper_item"], "dataset": listing["dataset"],
                "method": listing["method"], "generator": listing["generator"],
                "metric": "hypothesis_list", "status": listing["status"],
                "reason": listing["reason"], "expected": listing["expected_count"],
                "actual": None,
            })
            continue
        run_id = listing["run"]
        actual = ranked[run_id]
        if listing["selection"] == "top":
            actual = actual.head(listing["limit"])
        elif listing["selection"] != "significant":
            raise ValueError("Unknown hypothesis selection rule")
        expected = listing["rows"]
        comparisons.append({
            "paper_item": listing["paper_item"], "dataset": listing["dataset"],
            "method": listing["method"], "generator": listing["generator"],
            "metric": "hypothesis_count", "expected": len(expected), "actual": len(actual),
            "status": "match" if len(expected) == len(actual) else "mismatch", "run": run_id,
        })
        for position in range(max(len(actual), len(expected))):
            found = actual.iloc[position].hypothesis_id if position < len(actual) else None
            wanted = expected[position]["hypothesis_id"] if position < len(expected) else None
            comparisons.append({
                "paper_item": listing["paper_item"], "dataset": listing["dataset"],
                "method": listing["method"], "generator": listing["generator"],
                "metric": "hypothesis_order", "row": position + 1,
                "expected": wanted, "actual": found,
                "status": "match" if found == wanted else "mismatch", "run": run_id,
                "mapping_basis": expected[position]["mapping_basis"] if position < len(expected) else "",
            })
        for position, row in enumerate(actual.itertuples(), 1):
            table_rows.setdefault(listing["paper_item"], []).append({
                "dataset": listing["dataset"], "method": listing["method"],
                "generator": listing["generator"], "judge": "o3", "split": "validation",
                "rank": position, "hypothesis_id": row.hypothesis_id,
                "hypothesis": row.hypothesis, "recomputed_f1": row.f1,
                "p_value_raw": row.p_value_raw,
            })
    return comparisons, table_rows


def markdown_table(frame):
    def display(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(display(value) for value in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines) + "\n"


def render_results(registry, numeric, lists, table_rows, export_checks, manifest):
    comparisons = numeric + lists
    write_json("replication_results/comparisons.json", comparisons)
    write_csv("replication_results/comparisons.csv", pd.DataFrame(comparisons))
    write_csv("replication_results/export_checks.csv", pd.DataFrame(export_checks))
    write_json("replication_results/coverage.json", registry["coverage"])
    coverage = {item["paper_item"]: item for item in registry["coverage"]}
    for table in range(1, 20):
        label = f"Table {table}"
        rows = [row for row in numeric if row["paper_item"] == label]
        lines = [f"# {label}", "", coverage[label]["reason"], ""]
        if rows:
            frame = pd.DataFrame(rows)
            columns = [key for key in ["dataset", "method", "judge", "metric", "row",
                                       "entropy_threshold", "expected", "actual", "status"]
                       if key in frame.columns]
            write_csv(f"replication_results/table_{table:02d}_numbers.csv", frame)
            lines.append(markdown_table(frame[columns]))
        if label in table_rows:
            frame = pd.DataFrame(table_rows[label])
            write_csv(f"replication_results/table_{table:02d}_hypotheses.csv", frame)
            lines.append(markdown_table(frame))
            lines.append("F1 values in these list exports are recomputed validation diagnostics. "
                         "The appendix list tables do not print F1 values.\n")
        blocked = [row for row in lists if row["paper_item"] == label and row["status"] == "blocked"]
        for row in blocked:
            lines.append(f"{row['method']} with {row['generator']} remains blocked. {row['reason']}\n")
        contained_path(f"replication_results/table_{table:02d}.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    for figure in (1, 2):
        item = coverage[f"Figure {figure}"]
        contained_path(f"replication_results/figure_{figure:02d}.md").write_text(
            f"# Figure {figure}\n\n{item['reason']}.\n", encoding="utf-8"
        )
    numeric_counts = pd.Series([row["status"] for row in numeric]).value_counts().to_dict()
    list_counts = pd.Series([row["status"] for row in lists]).value_counts().to_dict()
    export_mismatches = [row for row in export_checks if row["status"] == "mismatch"]
    paper_mismatches = [row for row in comparisons if row["status"] == "mismatch"]
    blocked = any(row["status"] == "blocked" for row in comparisons) or bool(manifest["blocked_mappings"])
    exit_code = 1 if paper_mismatches or export_mismatches else 2 if blocked else 0
    summary = {
        "coverage": "partial", "exit_code": exit_code,
        "runs": len(manifest["runs"]), "hypotheses": sum(r["hypotheses"] for r in manifest["runs"]),
        "numeric_comparisons": numeric_counts, "hypothesis_list_comparisons": list_counts,
        "export_checks": len(export_checks), "export_mismatches": len(export_mismatches),
        "classifier_conditions": manifest["classifier_counts"],
        "paper_mismatches": paper_mismatches,
        "blocked_mappings": manifest["blocked_mappings"],
        "configuration_model_fields_needing_user_configuration": manifest["configuration_model_review_required"],
        "source_paper_sha256": registry["paper_sha256"],
        "script_sha256": sha256(Path(__file__)),
        "manifest_sha256": sha256(contained_path("outputs/manifest.json")),
        "paper_registry_sha256": sha256(contained_path("outputs/paper_registry.json")),
        "expanded_manifest_sha256": sha256(contained_path("outputs/expanded/manifest.json")),
        "expanded_paper_registry_sha256": sha256(contained_path("outputs/expanded/paper_registry.json")),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["pandas", "numpy", "scipy", "scikit-learn", "pyarrow"]},
    }
    write_json("replication_results/summary.json", summary)
    lines = [
        "# Replication comparison summary", "",
        "This is partial validation, heldout and corpus result replication. "
        "Convention values are counted separately from measured matches.", "",
        f"Processed {summary['runs']} runs and {summary['hypotheses']} hypotheses.",
        f"Checked {len(export_checks)} stored metric values with {len(export_mismatches)} mismatches.", "",
        "## Paper comparisons", "",
        markdown_table(pd.DataFrame([
            {"kind": kind, "status": status, "count": count}
            for kind, counts in [("numeric", numeric_counts), ("hypothesis list", list_counts)]
            for status, count in counts.items()
        ])),
        "## Mismatches", "",
    ]
    for row in paper_mismatches:
        lines.append(
            f"{row['paper_item']} {row['dataset']} {row['method']} "
            f"{row.get('judge', '')} {row['metric']} "
            f"reports {row['expected']} and recomputes to {row['actual']}.\n"
        )
    if not paper_mismatches:
        lines.append("No supported paper comparison mismatched.\n")
    lines.extend([
        "## Limits", "",
        "Missing ModernBERT predictions, manual semantic recall labels, DeepSeek oracle provenance, "
        "and the random baseline protocol block complete Tables 1 through 3 coverage. "
        "Transfer, additional discovery experiments, manual correctness and cost records remain unavailable.",
        "",
        "See comparisons.csv for every registered numeric or list comparison and coverage.json "
        "for all tables and figures. A match for a list uses the explicitly documented editorial "
        "identity mapping where the paper shortened a hypothesis.",
        "",
        f"The process exits with code {exit_code}. Code 1 flags measured mismatches. "
        "Code 2 flags execution errors or unresolved required mappings when there are no measured mismatches.",
    ])
    contained_path("replication_results/SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Recomputed {summary['runs']} validation runs and {summary['hypotheses']} hypotheses.")
    print(f"Stored metric mismatches {len(export_mismatches)} of {len(export_checks)} checks.")
    print(f"Paper mismatches {len(paper_mismatches)}. Coverage remains partial.")
    for row in paper_mismatches:
        print(f"{row['paper_item']} {row['dataset']} {row['method']} "
              f"{row.get('judge', '')} {row['metric']} "
              f"paper {row['expected']} computed {row['actual']}")
    print("Reports written to replication-materials/replication_results.")
    return exit_code


def main():
    manifest = load_json("outputs/manifest.json")
    extension = load_json(manifest["extension_manifest"])
    registry = load_json(manifest["paper_registry"])
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported manifest version")
    if extension["format_version"] != 1:
        raise ValueError("Unsupported expanded manifest version")
    manifest["runs"].extend(extension["runs"])
    manifest["hypotheses"].update(extension["hypotheses"])
    if set(manifest["assets"]) & set(extension["assets"]):
        raise ValueError("Expanded assets overwrite original evidence")
    manifest["assets"].update(extension["assets"])
    manifest["blocked_mappings"] = [
        row for row in manifest["blocked_mappings"] if row["dataset"] == "ghc-real"
    ]
    if sha256(contained_path(registry["source_paper"])) != registry["paper_sha256"]:
        raise ValueError("The paper differs from the version used for the comparison registry")
    contained_path("replication_results").mkdir(parents=True, exist_ok=True)
    assets = {name: load_asset(record) for name, record in manifest["assets"].items()}
    frames, ranked, export_checks = {}, {}, []
    for run in manifest["runs"]:
        if run["status"] != "supported":
            raise ValueError("Unresolved run included in executable evidence")
        frame, ordering, checks = verify_run(
            run, manifest["hypotheses"][run["id"]], assets, registry["alpha"]
        )
        frames[run["id"]], ranked[run["id"]] = frame, ordering
        export_checks.extend(checks)
        write_csv(f"replication_results/runs/{run['id']}/hypothesis_metrics.csv", frame)
    overall, diagnostics, overall_checks = overall_results(
        extension, manifest["runs"], manifest["hypotheses"], ranked, assets)
    manifest["classifier_counts"] = pd.Series(
        [result["status"] for result in overall.values()]).value_counts().to_dict()
    corpora = corpus_results(extension, assets)
    write_json("replication_results/overall_metrics.json", overall)
    write_csv("replication_results/threshold_diagnostics.csv", pd.DataFrame(diagnostics))
    write_csv("replication_results/overall_export_checks.csv", pd.DataFrame(overall_checks))
    write_json("replication_results/corpus_statistics.json", corpora)
    split_checks = []
    conditions = {c["run"]: c for c in extension["conditions"]}
    for cell in registry["cells"]:
        if cell["paper_item"] != "Table 1":
            continue
        condition = conditions[cell["run"]]
        target = assets[condition["heldout_targets"]].observed_disagreement
        feature = assets[condition["heldout_judgments"]][cell["hypothesis_id"]]
        heldout = binary_metrics(target, feature)["f1"]
        validation = float(frames[cell["run"]].set_index("hypothesis_id")
                           .loc[cell["hypothesis_id"], "f1"])
        split_checks.append(dict(cell, validation_f1=validation, heldout_f1=heldout,
                                 heldout_matches=numeric_match(heldout, cell["expected"], cell["decimals"])))
    write_csv("replication_results/table_01_split_diagnostics.csv", pd.DataFrame(split_checks))
    numeric = compare_cells(registry, manifest["runs"], frames, ranked, overall, corpora)
    lists, table_rows = compare_lists(registry, ranked, manifest["hypotheses"])
    return render_results(registry, numeric, lists, table_rows,
                          export_checks + overall_checks, manifest)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Replication failed: {error}", file=sys.stderr)
        sys.exit(2)
