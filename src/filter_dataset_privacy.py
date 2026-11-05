"""Explicit, offline whole-record privacy filtering for bundled sample datasets."""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unicodedata


SCHEMAS = {
    "ghc-syn": ("unique_id", "text", "Group_Hate_Judge_Response",
                "Group_Individual_Hate_Judge_Response"),
    "yelp": ("unique_id", "text", "A1", "A2"),
}
ENTITY_CATEGORIES = {
    "PERSON": "name", "ORG": "organization", "GPE": "location",
    "LOC": "location", "FAC": "location",
}
PATTERNS = {
    "email": re.compile(
        r"\b[\w.+-]+\s*(?:@|\[\s*at\s*\]|\(\s*at\s*\)|\s+at\s+)"
        r"\s*[\w-]+(?:\s*(?:\.|\[\s*dot\s*\]|\(\s*dot\s*\)|\s+dot\s+)"
        r"\s*[\w-]+)+\b", re.I),
    "phone": re.compile(
        r"(?<!\w)(?:\+?\d{1,3}[\s().-]*)?"
        r"(?:\(\d{2,4}\)|\d{3})[\s.-]*\d{3}[\s.-]*\d{4}(?!\w)"
        r"|(?<!\w)\d{3}[-.]\d{4}(?!\w)"
        r"|(?<!\w)\+\d[\d\s().-]{6,}\d(?!\w)"
        r"|\b(?:call|phone|tel|text|fax)\s*(?:me|us)?\s*[:=]?\s*"
        r"\+?\d[\d\s().-]{5,}\d\b", re.I),
    "handle": re.compile(r"(?<!\w)@[\w.-]+|\b(?:username|user\s*id|handle)\s*[:=]\s*\S+", re.I),
    # Any link can lead back to a source profile, including shortened URLs.
    "link": re.compile(
        r"\b(?:https?://|www\.)\S+|\b[\w-]+\.[a-z]{2,24}(?:\b|/)", re.I),
    "address": re.compile(
        r"\b\d{1,6}\s+(?:[\w.'-]+\s+){0,5}"
        r"(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|court|ct|"
        r"boulevard|blvd|way|place|pl|parkway|pkwy|terrace|highway|hwy)\b"
        r"|\bP\.?\s*O\.?\s*Box\s+\d+\b"
        r"|\b(?:address|apt|apartment|suite)\s*[#:=]?\s*\d+\b", re.I),
    "name": re.compile(
        r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+"
        r"|\b(?:named|called|server|waiter|waitress|bartender|manager)"
        r"\s+(?:was\s+|is\s+)?[A-Z][a-z]+\b"
        r"|\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"),
    "identifier": re.compile(r"(?<!\w)\d{3}-\d{2}-\d{4}(?!\w)|(?<!\w)\d{13,19}(?!\w)"),
    "contact_hint": re.compile(r"\b(?:phone|mobile|contact|address)\b.{0,30}\d", re.I),
}


class PrivacyError(Exception):
    """An actionable failure message that contains no source text."""


@dataclass
class Dataset:
    project: str
    path: Path
    digest: str
    rows: list[dict[str, str]]


def load_dataset(root, project):
    path = Path(root) / "data" / project / "annotated_data.csv"
    try:
        raw = path.read_bytes()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        if tuple(reader.fieldnames or ()) != SCHEMAS[project]:
            raise PrivacyError(f"{project}: unexpected CSV schema.")
        rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        raise PrivacyError(f"{project}: cannot read a valid UTF-8 CSV.") from None
    ids = set()
    for row in rows:
        if (None in row or any(value is None for value in row.values())
                or not row["text"].strip()
                or not re.fullmatch(r"[0-9]+", row["unique_id"])
                or any(not row[key].strip() for key in SCHEMAS[project][2:])):
            raise PrivacyError(f"{project}: invalid or incomplete record.")
        identifier = int(row["unique_id"])
        if identifier in ids:
            raise PrivacyError(f"{project}: duplicate numeric identifiers.")
        ids.add(identifier)
    if not rows:
        raise PrivacyError(f"{project}: dataset is empty.")
    return Dataset(project, path, hashlib.sha256(raw).hexdigest(), rows)


def load_detector():
    try:
        import spacy
        import en_core_web_sm
    except ImportError:
        raise PrivacyError("Install requirements-privacy.txt in the optional privacy environment.") from None
    if spacy.__version__ != "3.8.7" or en_core_web_sm.__version__ != "3.8.0":
        raise PrivacyError("Detector versions differ from requirements-privacy.txt.")
    try:
        detector = en_core_web_sm.load(disable=["parser", "lemmatizer"])
    except (OSError, ValueError):
        raise PrivacyError("Cannot load the installed local entity model.") from None
    if not {"ner", "tagger", "attribute_ruler"}.issubset(detector.pipe_names):
        raise PrivacyError("The installed detector lacks entity or proper noun recognition.")
    return detector


def filter_rows(dataset, detector):
    kept = []
    counts = Counter()
    texts = ["".join(char for char in unicodedata.normalize("NFKC", row["text"])
                     if unicodedata.category(char) != "Cf") for row in dataset.rows]
    processed = 0
    try:
        for row, text, doc in zip(dataset.rows, texts, detector.pipe(texts, batch_size=64)):
            categories = {key for key, pattern in PATTERNS.items() if pattern.search(text)}
            categories.update(ENTITY_CATEGORIES[entity.label_] for entity in doc.ents
                              if entity.label_ in ENTITY_CATEGORIES)
            if any(token.pos_ == "PROPN" for token in doc):
                categories.add("possible_name")
            if categories:
                counts.update(categories)
            else:
                kept.append(row)
            processed += 1
    except Exception:
        # Detector exceptions can include source text. Fail closed without echoing it.
        raise PrivacyError(f"{dataset.project}: local detection failed; no data was written.") from None
    if processed != len(dataset.rows):
        raise PrivacyError(f"{dataset.project}: detector did not process every record.")
    return kept, dict(sorted(counts.items()))


def summary(dataset, kept, categories):
    a, b = SCHEMAS[dataset.project][2:]
    return {
        "project": dataset.project, "input": len(dataset.rows),
        "removed": len(dataset.rows) - len(kept), "retained": len(kept),
        "removal_categories": categories,
        "retained_agreement": sum(row[a] == row[b] for row in kept),
        "retained_disagreement": sum(row[a] != row[b] for row in kept),
        "retained_labels": {col: dict(sorted(Counter(row[col] for row in kept).items()))
                            for col in (a, b)},
    }


def validate_preprocessing(candidate_root, config_root):
    import pandas as pd
    import yaml
    from src.disagreement_analysis.preprocessor_disagreement import Preprocessor

    with tempfile.TemporaryDirectory(prefix="privacy-preprocess-") as work:
        for project in SCHEMAS:
            dataset = load_dataset(candidate_root, project)
            config_path = Path(config_root) / "configs" / "disagreement_configs" / project / f"{project}_v1.yaml"
            with config_path.open() as handle:
                config = yaml.safe_load(handle)["disagreement_explainer"]
            options = {key: config[key]["value"] for key in (
                "example_size", "more_than_2", "num_annotations", "random_state", "train_on_test")}
            if options["train_on_test"]:
                raise PrivacyError(f"{project}: independent splits are required.")
            paths = {name: str(Path(work) / project / name)
                     for name in ("train", "validation", "test")}
            processor = Preprocessor()
            chunks = processor.preprocess(
                pd.DataFrame(dataset.rows), **options,
                train_path=paths["train"], validation_path=paths["validation"],
                test_path=paths["test"])
            if not chunks or not any(len(chunk) for group in chunks for chunk in group):
                raise PrivacyError(f"{project}: filtering leaves no usable example chunks.")
            for name, directory in paths.items():
                frame = pd.read_csv(Path(directory) / f"{name}.csv", dtype=str)
                a, b = SCHEMAS[project][2:]
                if frame.empty or set((frame[a] != frame[b]).tolist()) != {False, True}:
                    raise PrivacyError(f"{project}: filtered split lacks agreement coverage.")
                processor.get_unique_labels(frame, a, b)


def check_preprocessing(candidate_root, config_root, runtime_python):
    result = subprocess.run(
        [str(runtime_python), "-m", "src.filter_dataset_privacy",
         "--validate-staged", str(candidate_root), "--root", str(config_root)],
        cwd=config_root, capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False,
    )
    if result.returncode:
        raise PrivacyError(
            "Offline preprocessing validation failed. Check runtime dependencies and "
            "retained label/split coverage. No source text is included in this error.")


def apply_filtered(datasets, retained, validate):
    if not datasets or len(datasets) != len(retained):
        raise PrivacyError("Every dataset must have a corresponding filtered result.")
    changed = []
    with tempfile.TemporaryDirectory(prefix=".privacy-filter-", dir=datasets[0].path.parents[2]) as work:
        root = Path(work)
        try:
            for dataset, rows in zip(datasets, retained):
                if not rows:
                    raise PrivacyError(f"{dataset.project}: no records survive filtering.")
                original = iter(dataset.rows)
                for row in rows:
                    if not any(row == source for source in original):
                        raise PrivacyError(f"{dataset.project}: retained values or ordering changed.")
                target = root / "data" / dataset.project / "annotated_data.csv"
                target.parent.mkdir(parents=True)
                with target.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=SCHEMAS[dataset.project])
                    writer.writeheader()
                    writer.writerows(rows)
                    handle.flush()
                    os.fsync(handle.fileno())
                if load_dataset(root, dataset.project).rows != rows:
                    raise PrivacyError(f"{dataset.project}: staged CSV round trip failed.")
            validate(root)
            for dataset in datasets:
                if hashlib.sha256(dataset.path.read_bytes()).hexdigest() != dataset.digest:
                    raise PrivacyError(f"{dataset.project}: source changed since scanning.")
            for dataset in datasets:
                if hashlib.sha256(dataset.path.read_bytes()).hexdigest() != dataset.digest:
                    raise PrivacyError(f"{dataset.project}: source changed before replacement.")
                os.replace(root / "data" / dataset.project / "annotated_data.csv", dataset.path)
                changed.append(dataset.project)
        except (OSError, PrivacyError) as error:
            message = str(error) if isinstance(error, PrivacyError) else "Cannot stage or replace dataset files."
            state = ", ".join(changed) if changed else "none"
            raise PrivacyError(f"{message} Files replaced: {state}.") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--validate-staged", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--runtime-python", type=Path, help="Existing pipeline Python for offline validation.")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.validate_staged:
        try:
            validate_preprocessing(args.validate_staged, root)
        except Exception:
            print("Offline preprocessing validation failed.", file=sys.stderr)
            return 2
        return 0
    try:
        datasets = [load_dataset(root, project) for project in SCHEMAS]
        detector = load_detector()
        retained = []
        for dataset in datasets:
            kept, categories = filter_rows(dataset, detector)
            retained.append(kept)
            print(json.dumps(summary(dataset, kept, categories), sort_keys=True))
        removed = sum(len(dataset.rows) - len(rows) for dataset, rows in zip(datasets, retained))
        if args.check:
            return 1 if removed else 0
        runtime = (args.runtime_python or root / ".venv" / "bin" / "python").absolute()
        validate = lambda candidate: check_preprocessing(candidate, root, runtime)
        apply_filtered(datasets, retained, validate)
        for dataset, expected in zip(datasets, retained):
            saved = load_dataset(root, dataset.project)
            if saved.rows != expected or filter_rows(saved, detector)[0] != expected:
                raise PrivacyError("Persisted filtering verification failed; files have already been replaced.")
        validate(root)
        print("Filtered files saved and verified. No original backups were retained.")
        return 0
    except (PrivacyError, OSError) as error:
        message = str(error) if isinstance(error, PrivacyError) else "Local file or runtime access failed."
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
