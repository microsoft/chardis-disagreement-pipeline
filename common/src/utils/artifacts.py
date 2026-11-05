"""Content-addressed artifact manifests and atomic run records."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[3]
VERSION = 1


def model_directory(model):
    from urllib.parse import quote
    if not isinstance(model, str) or not model.strip() or model in {".", ".."}:
        raise ValueError("Set a nonempty model identifier.")
    return quote(model, safe="-_.")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_config(value):
    if isinstance(value, dict):
        return {key: safe_config(child) for key, child in value.items()
                if key not in {"key", "openai_key", "api_key"}}
    if isinstance(value, list):
        return [safe_config(child) for child in value]
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(safe_config(value), sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def identity(settings, files=(), stage="annotation"):
    sources = {
        "discovery": ["src/disagreement_analysis/disagreement_explainer.py",
                      "src/disagreement_analysis/disagreement_orchestrator.py",
                      "src/disagreement_analysis/preprocessor_disagreement.py", "src/run_estimator.py"],
        "annotation": ["src/annotator_base.py", "src/annotation_orchestrator.py",
                       "src/preprocessor", "src/prompt_builder", "src/run_estimator.py"],
        "evaluation": ["src/disagreement_analysis/evaluate.py",
                       "src/disagreement_analysis/prepare_disagreement_eval_config.py",
                       "src/run_disagreement_cli.py", "src/annotator_base.py",
                       "src/annotation_orchestrator.py", "src/preprocessor", "src/prompt_builder"],
    }[stage]
    implementations = []
    for name in ["common/src", *sources]:
        path = ROOT / name
        implementations.extend(path.rglob("*.py") if path.is_dir() else [path] if path.is_file() else [])
    return fingerprint({
        "version": VERSION, "stage": stage, "settings": settings,
        "files": {str(Path(path).resolve()): digest(path) for path in files},
        "implementation": {str(p.relative_to(ROOT)): digest(p)
                           for p in sorted(set(implementations))},
        "locks": {p.name: digest(p) for p in sorted((ROOT / "locks").glob("py311-*.txt"))},
    })


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".artifact-", delete=False) as file:
            temporary = file.name
            json.dump(value, file, ensure_ascii=False, allow_nan=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def manifest_path(path):
    return Path(str(path) + ".manifest.json")


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", dir=path.parent,
                                         prefix=".artifact-", delete=False) as file:
            temporary = file.name
            frame.to_csv(file, index=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def reusable(path, expected):
    path = Path(path)
    manifest = manifest_path(path)
    if not path.exists() and not manifest.exists():
        return False
    if not path.is_file() or not manifest.is_file():
        raise ValueError(f"Unverified or incomplete artifact {path}. Import legacy artifacts explicitly or select a new run.")
    try:
        record = json.loads(manifest.read_text())
        valid = (isinstance(record, dict) and record["version"] == VERSION and record["identity"] == expected
                 and record["status"] == "complete" and record["outputs"]
                 and isinstance(record["outputs"], dict)
                 and str(path.resolve()) in record["outputs"]
                 and all(Path(name).is_file() and digest(name) == checksum
                         for name, checksum in record["outputs"].items()))
    except (ValueError, KeyError, TypeError, OSError):
        valid = False
    if not valid:
        raise ValueError(f"Artifact provenance does not match: {path}. Original files were preserved.")
    return True


def seal(path, expected, companions=(), imported=None):
    paths = {Path(path).resolve(), *(Path(p).resolve() for p in companions)}
    record = {"version": VERSION, "identity": expected, "status": "complete",
              "outputs": {str(p): digest(p) for p in sorted(paths)}}
    if imported:
        record["legacy_import"] = imported
    write_json(manifest_path(path), record)


@contextmanager
def writer(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Run directory already has a writer lock: {lock}. Inspect before removing a stale lock.") from None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink()


def discovery_identity(cfg):
    from common.src.configs.schema import runtime_config
    from common.src.openai.connections import normalized
    from src.disagreement_analysis.disagreement_explainer import DisagreementExplainer
    cfg = runtime_config(cfg)
    de = normalized(cfg["disagreement_explainer"])
    explainer = DisagreementExplainer()
    files = [
        explainer._resolve_combined_path(de["data_path"], de.get("data_filename", "")),
        explainer._resolve_combined_path(de["guidelines_path"], de.get("guidelines_filename", "")),
    ]
    if de.get("examples_path"):
        files.append(de["examples_path"])
    files.extend(sorted((ROOT / "prompts/DEFAULTS/disagreement_explainer").rglob("*.md")))
    return identity({name: normalized(cfg[name]) for name in (
        "disagreement_explainer", "rationales_openai", "synthesizer_openai")}, files, stage="discovery")


def evaluation_identity(cfg, folder):
    from common.src.configs.schema import runtime_config
    from common.src.openai.connections import normalized
    cfg = runtime_config(cfg)
    de = normalized(cfg["disagreement_explainer"])
    root = Path(de["output_path"])
    if not root.is_absolute():
        root = ROOT / root
    files = [root / "disagreement_explanations.json"]
    if folder == "validation":
        files.extend([root / folder / "validation.csv", root / folder / "validation_annotate.csv"])
    else:
        model = model_directory(normalized(cfg["evaluation"])["openai_model"])
        files.extend([root / folder / "test.csv", root / folder / "test_annotate.csv"])
        files.extend(sorted((root / "validation" / model).rglob("*.csv")))
        files.extend(sorted((root / "validation" / model).glob("significant*.json")))
    files.extend(sorted((ROOT / "prompts/DEFAULTS").rglob("*.md")))
    return identity({"stage": folder, "evaluation": normalized(cfg["evaluation"])}, files, stage="evaluation")
