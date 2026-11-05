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
from src.annotator_base import Annotator
from src.preprocessor.preprocessor_base import Preprocessor
from src.prompt_builder.prompt_builder_base import PromptBuilder
from common.src.configs import config_utils
from common.src.openai.openai_args import OpenAIArgs
from common.src.utils import time_utils, type_utils

# Remove the temporary path to avoid polluting the import namespace
if _root_added:
    sys.path.remove(_root_added)


def get_custom_prompt_builder(config: dict):
    """
    Retrieves a custom PromptBuilder instance based on the configuration. The method will dynamically
    try to instantiate an instance of the specified class that derives from PromptBuilder. If the specified
    class cannot be found or instantiated, it falls back to the default PromptBuilder.

    The configuration should have a prompt_builder.type.value property in the following format: src.prompt_builder.<module>|<ClassName>

    Args:
        config (dict): Configuration dictionary containing prompt builder settings.

    Returns:
        PromptBuilder: An instance of the specified PromptBuilder subclass or the default PromptBuilder.
    """
    prompt_builder_config = config.setdefault("prompt_builder", dict())

    # If the prompt builder type is not specified, use the default PromptBuilder
    prompt_builder_type = prompt_builder_config.setdefault(
        "type",
        {
            "description": "Prompt builder to use during annotations. Expected format is src.prompt_builder.<module>|<ClassName>.",
            "value": "src.prompt_builder.prompt_builder_base|PromptBuilder"
        }
    )

    module_parts = prompt_builder_type.get("value", "").split("|")

    if len(module_parts) != 2:
        raise ValueError("prompt_builder.type must have the form module|ClassName.")

    module = module_parts[0]
    class_name = module_parts[1]

    try:
        prompt_builder = type_utils.create_instance(module, class_name)

        if not isinstance(prompt_builder, PromptBuilder):
            raise TypeError("Configured prompt builder has an incompatible type.")
    except Exception:
        raise ValueError("Cannot initialize the configured prompt builder.") from None

    return prompt_builder


def get_custom_preprocessor(config):
    """
    Retrieves a custom Preprocessor instance based on the configuration. The method will dynamically
    try to instantiate an instance of the specified class that derives from Preprocessor. If the specified
    class cannot be found or instantiated, it falls back to the default Preprocessor.

    The configuration should have a preprocessor.type.value property in the following format: src.preprocessor.<module>|<ClassName>

    Args:
        config (dict): Configuration dictionary containing prompt builder settings.

    Returns:
        Preprocessor: An instance of the specified Preprocessor subclass or the default Preprocessor.
    """
    preprocessor_config = config.setdefault("preprocessor", dict())

    # If the preprocessor type is not specified, use the default Preprocessor
    preprocessor_type = preprocessor_config.setdefault(
        "type",
        {
            "description": "Preprocessor to use during annotations. Expected format is src.preprocessor.<module>|<ClassName>.",
            "value": "src.preprocessor.preprocessor_base|Preprocessor"
        }
    )

    module_parts = preprocessor_type.get("value", "").split("|")

    if len(module_parts) != 2:
        raise ValueError("preprocessor.type must have the form module|ClassName.")

    module = module_parts[0]
    class_name = module_parts[1]

    try:
        preprocessor = type_utils.create_instance(module, class_name)

        if not isinstance(preprocessor, Preprocessor):
            raise TypeError("Configured preprocessor has an incompatible type.")
    except Exception:
        raise ValueError("Cannot initialize the configured preprocessor.") from None

    return preprocessor


def expand_annotation_paths(config: dict, run_name: str) -> dict:
    """
    Expands {run_name} placeholders in annotation paths and persists run_name.
    Handles YAML schema where each field is a dict containing 'value'.
    """
    annotation = config.setdefault("annotation", {})

    # Persist run_name for downstream consumers (optional but handy)
    if "run_name" not in annotation:
        annotation["run_name"] = {
            "description": "The name of this annotation run, typically derived from the YAML filename.",
            "value": run_name
        }
    else:
        # If annotation.run_name exists, normalize to the provided run_name
        if isinstance(annotation["run_name"], dict):
            annotation["run_name"]["value"] = run_name
        else:
            annotation["run_name"] = {"description": "Run name", "value": run_name}

    def expand_value_field(field_dict):
        """Expand {run_name} inside a dict with a 'value' key."""
        if isinstance(field_dict, dict):
            val = field_dict.get("value")
            if isinstance(val, str):
                field_dict["value"] = val.format(run_name=run_name)

    # Expand known path fields that have nested 'value'
    for key in ["data_path", "output_path", "raw_output_path"]:
        if key in annotation:
            expand_value_field(annotation[key])

    return config

def run(config_path: str | None = None):
    """
    Main execution entry point for the annotation workflow.

    Args:
        config_path: Optional explicit YAML config path. When omitted (e.g. the
            module is run as a script) the path is parsed from --config_path on
            sys.argv, preserving the historical CLI behavior.

    Steps:
        1. Resolve the YAML config path
        2. Load the YAML config
        3. Build component instances based on user-specified parameters
        4. Load any custom extension modules specified in the config
        5. Run the annotation process and log timing/status updates
    """

    # 1. Resolve config path (CLI fallback for backward compat)
    if config_path is None:
        args = config_utils.get_default_config_args()
        config_path = args.config_path

    # 2. Load the YAML config into a dictionary that holds settings for:
    #    - OpenAI API configuration
    #    - Annotator behavior and runtime options
    #    - Prompt-building rules
    #    - Preprocessing requirements
    config = config_utils.load_yaml_config(config_path)
    run_name = os.path.basename(config_path).replace(".yaml", "")
    config = expand_annotation_paths(config, run_name)

    # 3. Build instance of classes dynamically from config
    openai_args = OpenAIArgs()
    annotator = Annotator()
    prompt_builder = get_custom_prompt_builder(config)
    preprocessor = get_custom_preprocessor(config)

    config_utils.initialize_from_configs(
        config,
        config_path,
        {
            "openai": openai_args,
            "annotation": annotator,
            "prompt_builder": prompt_builder,
            "preprocessor": preprocessor
        }
    )

    # 5. Run the annotation process
    start_time = time.perf_counter()

    # Skip annotators that do not have a valid data path (i.e., not configured by the user)
    if annotator.data_path is None or len(annotator.data_path) == 0:
        raise ValueError("Annotation data_path is required.")

    # Log start of annotation process
    print(f"\n{time_utils.get_time_and_elapsed_status(start_time)}: Starting annotation process...")

    # Execute the annotation logic and store results
    from common.src.utils.artifacts import writer
    with writer(os.path.dirname(annotator.output_path) or "."):
        try:
            annotator.execute_annotation_process(openai_args, prompt_builder, preprocessor, run_name)
        finally:
            if annotator.client is not None:
                annotator.client.close()

    # Log completion of the annotation process
    print(f"{time_utils.get_time_and_elapsed_status(start_time)}: DONE!\n")


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
