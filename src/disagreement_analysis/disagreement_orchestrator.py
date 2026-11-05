# Import standard libraries
import time
import os
import sys

def add_project_root(marker_filename: str = ".gitignore") -> str:
    """
    Locate the project root by searching upward for a marker file 
    (default = '.gitignore'), and temporarily add it to sys.path. 
    This code finds the root of the repo and adds it to Python's 
    import search path so that all modules in the project 
    can be imported reliably from anywhere.

    Returns:
        str: The path to the project root if added to sys.path, None otherwise.
    """
    current = os.path.abspath(os.path.dirname(__file__))

    # Walk up until the marker file is found
    while True:
        marker = os.path.join(current, marker_filename)
        
        if os.path.isfile(marker):
            break

        parent = os.path.dirname(current)
        
        if parent == current:  # reached filesystem root
            return None
        
        current = parent

    # Add root to sys.path if not already there
    if sys.path[0] != current:
        sys.path.insert(0, current)
        return current

    return None

# Add project root so shared modules can be imported
_root_added = add_project_root()

# Import shared modules
from src.disagreement_analysis.disagreement_explainer import DisagreementExplainer
from common.src.configs import config_utils
from common.src.openai.openai_args import OpenAIArgs
from common.src.utils import time_utils, type_utils

# Remove the temporary path to avoid polluting the import namespace
if _root_added:
    sys.path.remove(_root_added)

def _derive_run_name(config_path: str) -> str:
    """Derive a stable <project>/<version> run name from the config path.

    Configs live under ``configs/disagreement_configs/<project>/<file>.yaml``.
    The run name encodes both project and version so outputs land at
    ``outputs/disagreement_outputs/<project>/<version>/...``.
    """
    abs_path = os.path.abspath(config_path)
    version = os.path.basename(abs_path).replace(".yaml", "")
    project = os.path.basename(os.path.dirname(abs_path))
    if project and project not in {"disagreement_configs", "configs"}:
        return os.path.join(project, version)
    return version


def run(config_path: str | None = None):
    """
    Main execution entry point for the disagreement analysis workflow.

    Args:
        config_path: Optional explicit YAML config path. When omitted, the path
            is parsed from ``--config_path`` on ``sys.argv`` (legacy CLI use).

    1. Resolve YAML config path
    2. Load the YAML config
    3. Build component instances based on user-specified parameters
    4. Run the disagreement analysis process and log timing/status updates
    """

    # 1. Resolve config path (CLI fallback for backward compat)
    if config_path is None:
        args = config_utils.get_default_config_args()
        config_path = args.config_path

    # 2. Load the YAML config into a dictionary that holds settings for:
    #    - OpenAI API configuration for rationales generation and synthesis
    #    - DisagreementExplainer settings
    config = config_utils.load_yaml_config(config_path)
    from common.src.utils import artifacts
    from pathlib import Path
    from common.src.openai.connections import normalized
    values = normalized(config["disagreement_explainer"])
    run_dir = Path(values["output_path"])
    if not run_dir.is_absolute():
        run_dir = Path(DisagreementExplainer.BASE_DIR) / run_dir
    with artifacts.writer(run_dir):
        return _run_locked(config_path, config, run_dir)


def _run_locked(config_path, config, run_dir):
    from common.src.utils import artifacts
    output = run_dir / "disagreement_explanations.json"
    expected = artifacts.discovery_identity(config)
    if artifacts.reusable(output, expected):
        print("Reusing verified discovery artifacts.")
        return
    run_name = _derive_run_name(config_path)

    # 3. Build instance of classes dynamically from config
    disagreement_explainer = DisagreementExplainer()
    rationales_openai_args = OpenAIArgs()
    synthesizer_openai_args = OpenAIArgs()

    config_utils.initialize_from_configs(
        config,
        config_path,
        {
            "disagreement_explainer": disagreement_explainer,
            "rationales_openai": rationales_openai_args,
            "synthesizer_openai": synthesizer_openai_args,
        }
    )

    # 4. Run the disagreement analysis process
    start_time = time.perf_counter()

    # Log start of disagreement analysis process
    print(f"\n{time_utils.get_time_and_elapsed_status(start_time)}: Starting disagreement analysis process...")

    # Execute main process
    disagreement_explainer.execute_disagreement_process(
        rationales_openai_args=rationales_openai_args,
        synthesizer_openai_args=synthesizer_openai_args,
        run_name=run_name
    )
    artifacts.seal(output, expected, [
        run_dir / "train.csv", run_dir / "validation/validation.csv",
        run_dir / "validation/validation_annotate.csv", run_dir / "heldout/test.csv",
        run_dir / "heldout/test_annotate.csv", run_dir / "metadata.json",
    ])

    # Log completion of the disagreement analysis process
    print(f"{time_utils.get_time_and_elapsed_status(start_time)}: Completed disagreement analysis process!\n")


if __name__ == "__main__":
    try:
        run()
    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print("\nUnexpected error occurred:")
        print(f"    {type(e).__name__}: {e}")
        sys.exit(1)