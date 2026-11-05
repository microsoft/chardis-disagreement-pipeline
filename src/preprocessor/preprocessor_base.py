import numpy as np
import os
import pandas as pd
import sys
import warnings


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
from common.src.configs.configurable_base import ConfigurableBase

# Remove the temporary path to avoid polluting the import namespace
if _root_added:
    sys.path.remove(_root_added)

class Preprocessor(ConfigurableBase):
    """
    Preprocess and validate annotation input datasets.

    Responsibilities:
    -----------------
    Ensure required columns exist:
        - unique_id (nonempty string or integer, unique after serialization)
        - text_to_annotate (must be str)

    Optional columns:
        - grounding
        - context

    Construct `llm_query`:
        Format:
        [unique_id]: <unique_id> 
        [grounding]: <grounding> (optional)
        [context]: <context> (optional)
        [text_to_annotate]: <text_to_annotate>

    Diagnostics:
        - Report distribution of LLM query lengths
        - Detect non-English or unusual characters that could cause issues
        - Validate that `llm_query` is not empty for any row
    """

    REQUIRED_COLUMNS = ["unique_id", "text_to_annotate"]
    OPTIONAL_COLUMNS = ["grounding", "context"]

    def __init__(self):
        super().__init__()

    def preprocess(self,
                   df: pd.DataFrame,
                   sanity_check: bool = False,
                   num_annotations: int = -1,
                   random_state: int = -1,
                   batch_size: int = 1) -> pd.DataFrame:
        """
        Full preprocessing pipeline.

        Sampling behavior:
        ---------------------------------------
        1. sanity_check = True → return first batch_size rows (handled by annotator)
        2. num_annotations > 0 → randomly sample n rows
        3. random_state controls reproducibility of sampling.
        """
        print(f"\n[Preprocessor] Loaded dataset with {len(df)} rows.")

        # ---- 1. OPTIONAL SANITY CHECK ----
        if sanity_check:
            print(
                f"[Preprocessor] Sanity-check mode enabled: returning first {batch_size} row(s).")
            df = df.head(batch_size)

        # ---- 2. OPTIONAL SAMPLING ----
        if isinstance(num_annotations, int) and num_annotations > 0:
            if num_annotations > len(df):
                raise ValueError(
                    f"num_annotations={num_annotations} exceeds dataset size ({len(df)})."
                )

            if random_state < 0:
                random_state = None

            print(
                f"[Preprocessor] Sampling {num_annotations} rows (random_state={random_state})")
            df = df.sample(n=num_annotations,
                           random_state=random_state).reset_index(drop=True)

        df = df.copy()

        Preprocessor._check_required_columns(df)
        Preprocessor._validate_unique_id(df)
        Preprocessor._validate_text_columns(df)
        Preprocessor._warn_missing_optional_columns(df)

        df = Preprocessor._build_llm_query(df)

        Preprocessor._diagnose_llm_query(df)
        # Preprocessor._check_unusual_characters(df)
        Preprocessor._validate_llm_query_not_empty(df)

        print("[Preprocessor] Preprocessing complete.\n")
        return df

    # ------------------------------------------------------------
    # Column Validation
    # ------------------------------------------------------------
    def _check_required_columns(df: pd.DataFrame):
        missing = [
            c for c in Preprocessor.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Dataset is missing required column(s): {missing}. "
                f"The dataset must contain columns: {Preprocessor.REQUIRED_COLUMNS}"
            )

    def _warn_missing_optional_columns(df: pd.DataFrame):
        # Optional columns being absent is expected; no warning needed.
        return

    def _validate_unique_id(df: pd.DataFrame):
        values = df["unique_id"]
        if values.isna().any() or not values.map(
                lambda value: isinstance(value, (str, int, np.integer))
                and not isinstance(value, (bool, np.bool_)) and bool(str(value).strip())).all():
            raise ValueError("unique_id values must be nonempty strings or integers.")
        if values.astype(str).duplicated().any():
            raise ValueError("Every row must have a unique serialized unique_id.")

    def _validate_text_columns(df: pd.DataFrame):
        
        # Columns to process = only those that already exist
        target_cols = [c for c in ["text_to_annotate"] + Preprocessor.OPTIONAL_COLUMNS if c in df.columns]

        for col in target_cols:
            if col in df.columns:
                # Check type safety before coercion
                if not df[col].map(lambda x: isinstance(x, (str, type(None)))).all():
                    raise ValueError(
                        f"Column '{col}' must contain only strings or NaN values."
                    )
                df[col] = df[col].astype(str).fillna("").str.strip()
            else:
                df[col] = ""

    # ------------------------------------------------------------
    # Build LLM Query
    # ------------------------------------------------------------
    def _build_llm_query(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build an llm_query block that conditionally includes
        optional fields like grounding and context only if present.
        """

        # Always include unique_id
        query_parts = [
            "[unique_id]: " + df["unique_id"].astype(str)
        ]

        # Include grounding only if the column exists
        if "grounding" in df.columns:
            query_parts.append("[grounding]: " + df["grounding"])

        # Include context only if the column exists
        if "context" in df.columns:
            query_parts.append("[context]: " + df["context"])

        # Always include text_to_annotate
        query_parts.append("[text to annotate]: " + df["text_to_annotate"])

        # Combine into a single text block per row
        df["llm_query"] = (
            pd.concat(query_parts, axis=1)
            .apply(lambda row: "\n".join(row), axis=1)
        )

        return df

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    def _diagnose_llm_query(df: pd.DataFrame):
        lengths = df["llm_query"].str.split().str.len()
        median = int(np.median(lengths))
        p90 = int(np.percentile(lengths, 90))
        max_len = int(lengths.max())

        print("\nLLM Query Length Diagnostics:")
        print(f"Median length: {median} words")
        print(f"90th percentile: {p90} words")
        print(f"Max length: {max_len} words\n")

        if max_len > 20000:
            warnings.warn(
                f"Extremely long llm_query detected (max = {max_len} words).\n"
                f"This may cause truncation or model failures.",
                UserWarning
            )

    #    # def _check_unusual_characters(df: pd.DataFrame):

    #     # Allow unicode letters, emoji, punctuation
    #     allowed_pattern = re.compile(
    #         r"[^\w\s.,!?;:'\"()\[\]\-–—/@#*&%+=\u2019\u2013\u2014"
    #         r"\U0001F300-\U0001FAFF]"  # emoji range
    #     )

    #     df["has_unusual_chars"] = df["llm_query"].apply(
    #         lambda x: bool(allowed_pattern.search(x))
    #     )

    #     if df["has_unusual_chars"].any():
    #         issues = df[df["has_unusual_chars"]].head(5)[["unique_id", "llm_query"]]
    #         print("\nWarning: Suspicious characters detected in some rows.")
    #         print("Example problematic rows:")
    #         print(issues.to_string(index=False))

    def _validate_llm_query_not_empty(df):
        if "llm_query" not in df.columns:
            raise ValueError(
                "Expected 'llm_query' column not found in the dataset.")

        # Identify problematic rows
        bad_rows = df[
            df["llm_query"].isna() |
            (df["llm_query"].astype(str).str.strip() == "")
        ]

        if not bad_rows.empty:
            examples = bad_rows.head(10)[["unique_id", "llm_query"]]

            raise ValueError(
                "Some rows have empty or missing `llm_query` values.\n"
                "Each annotation entry must contain a valid combined text block.\n\n"
                f"Example problematic rows (up to 10):\n{examples.to_string(index=False)}"
            )

        return True
