"""Import complete discovery artifacts with explicitly declared provenance."""

import argparse
import json
from pathlib import Path

import pandas as pd

from common.src.configs.config_utils import load_yaml_config
from common.src.openai.connections import normalized
from common.src.utils import artifacts, model_output


def import_discovery(config_path, confirm=False):
    if not confirm:
        raise ValueError("Explicit confirmation of the input/config association is required.")
    cfg = load_yaml_config(config_path)
    de = normalized(cfg["disagreement_explainer"])
    root = Path(de["output_path"])
    if not root.is_absolute():
        root = artifacts.ROOT / root
    output = root / "disagreement_explanations.json"
    with artifacts.writer(root):
        if artifacts.manifest_path(output).exists():
            raise ValueError("A manifest already exists. Import will not overwrite it.")
        model_output.string_list(output.read_text(encoding="utf-8-sig"))
        companions = [root / "train.csv", root / "validation/validation.csv",
                      root / "validation/validation_annotate.csv", root / "heldout/test.csv",
                      root / "heldout/test_annotate.csv", root / "metadata.json"]
        frames = [pd.read_csv(p, dtype={"unique_id": str}) for p in companions[:5]]
        for frame in frames:
            if "unique_id" not in frame or frame["unique_id"].duplicated().any() or frame.empty:
                raise ValueError("Legacy splits must contain unique, nonempty input identities.")
        from src.disagreement_analysis.disagreement_explainer import DisagreementExplainer
        source = pd.read_csv(DisagreementExplainer()._resolve_combined_path(
            de["data_path"], de.get("data_filename", "")), dtype={"unique_id": str})
        if source["unique_id"].isna().any() or source["unique_id"].duplicated().any():
            raise ValueError("Source identities must be present and unique.")
        source = source.set_index("unique_id")
        for index in (0, 1, 3):
            frame = frames[index].set_index("unique_id")
            if not set(frame.index).issubset(source.index) or not set(source.columns).issubset(frame.columns):
                raise ValueError("Legacy splits do not match the declared source schema and identities.")
            actual = frame[list(source.columns)].fillna("").astype(str)
            expected = source.loc[frame.index].fillna("").astype(str)
            if not actual.equals(expected):
                raise ValueError("Legacy split content differs from the declared source.")
        for original, judge in ((1, 2), (3, 4)):
            if "observed_disagreement" not in frames[original] or "text_to_annotate" not in frames[judge]:
                raise ValueError("Legacy validation and heldout files lack required evaluation columns.")
            left = frames[original].set_index("unique_id")["text"].sort_index()
            right = frames[judge].set_index("unique_id")["text_to_annotate"].sort_index()
            if not left.equals(right):
                raise ValueError("Legacy judge texts differ from their split.")
        ids = [set(frame["unique_id"].astype(str)) for frame in frames]
        if ids[1] != ids[2] or ids[3] != ids[4]:
            raise ValueError("Legacy judge inputs do not match split identities.")
        if not de.get("train_on_test") and any(ids[a] & ids[b] for a, b in ((0, 1), (0, 3), (1, 3))):
            raise ValueError("Legacy splits overlap.")
        if not isinstance(json.loads(companions[-1].read_text()), dict):
            raise ValueError("Legacy metadata must be an object.")
        artifacts.seal(output, artifacts.discovery_identity(cfg), companions,
                       imported={"association": "user_confirmed", "historical_model_revision": "unknown",
                                 "original_files": "preserved", "source_config": str(Path(config_path).resolve())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--confirm-association", action="store_true")
    args = parser.parse_args()
    import_discovery(args.config, args.confirm_association)
    print("Discovery imported with historical provenance marked unknown. Evaluation must run with verified new annotations.")


if __name__ == "__main__":
    main()
