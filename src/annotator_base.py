# Import standard libraries
from datetime import datetime, timezone
import json
import math
import pandas as pd
import re
import time
from importlib.metadata import metadata
import os
import sys
import warnings
from openai import OpenAI
from common.src.utils import artifacts, model_output


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
from common.src.utils import json_utils, thread_utils, time_utils
from common.src.openai.openai_args import OpenAIArgs
from common.src.configs.configurable_base import ConfigurableBase
from common.src.configs.config_value import ConfigValue

# Remove the temporary path to avoid polluting the import namespace
if _root_added:
    sys.path.remove(_root_added)


class Annotator(ConfigurableBase):
    """
    Base class for code that annotates a dataset with labels for a given concept.
    Manages core parameters and provides methods for executing the annotation process.
    """
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_NUM_ITERATIONS = 1
    DEFAULT_NUM_THREADS = 15
    DEFAULT_NUM_ANNOTATIONS = -1

    DEFAULT_SANITY_CHECK = False
    DEFAULT_RANDOM_STATE = -1

    def __init__(self):
        """
        Initialize core annotation parameters and runtime settings.

        Args:
            data_path (str): Input dataset path.
            output_path (str): Directory to save annotated outputs.
            batch_size (int, optional): Samples per API call.
            num_iterations (int, optional): Number of annotation rounds.
            num_threads (int, optional): Threads for parallel execution.
            num_annotations (int, optional): Annotations per sample.
            raw_output_path (str, optional): Path for raw or intermediate outputs.
            sanity_check (bool, optional): If true, use only the first rows of the dataset.
            random_state (int, optional): Random seed for sampling.
        """
        self.data_path = ConfigValue(
            None,
            True,
            "Directory (or full file path) containing the CSV to annotate.")

        self.data_filename = ConfigValue(
            "data.csv",
            False,
            "CSV filename inside data_path. Ignored if data_path already points to a file (default = data.csv).")

        self.output_path = ConfigValue(
            None,
            True,
            "The path of the output file to save the annotations to.")

        self.batch_size = ConfigValue(
            Annotator.DEFAULT_BATCH_SIZE,
            False,
            f"The number of texts to annotate in each API call (default = {Annotator.DEFAULT_BATCH_SIZE}).")

        self.num_iterations = ConfigValue(
            Annotator.DEFAULT_NUM_ITERATIONS,
            False,
            f"The number of times to annotate each text (default = {Annotator.DEFAULT_NUM_ITERATIONS}).")

        self.aggregation = ConfigValue(
            "majority",
            False,
            "When num_iterations > 1, how to aggregate boolean/integer labels across "
            "iterations: 'majority' (mode), 'average' (mean), or 'max'. Categorical "
            "labels are always aggregated by mode; string labels and reasoning/evidence "
            "spans are combined into a list. Confidence scores are averaged.")

        self.judge_panel = ConfigValue(
            [],
            False,
            "Optional Panel of Judges: list of model deployment names. When two or "
            "more are supplied, the CLI will fan out the run into one identical "
            "config per model, each writing to its own per-model output subdirectory.")

        self.panel_aggregation = ConfigValue(
            "majority",
            False,
            "When the Panel of Judges is enabled, how to aggregate boolean/integer "
            "labels across judges: 'majority' (mode), 'average' (mean), or 'max'. "
            "Categorical labels are aggregated by mode; string labels and "
            "reasoning/evidence spans are combined into a list. Confidence scores "
            "are averaged.")

        self.num_threads = ConfigValue(
            Annotator.DEFAULT_NUM_THREADS,
            False,
            f"The number of threads to use for parallel processing (default = {Annotator.DEFAULT_NUM_THREADS}).")

        self.num_annotations = ConfigValue(
            Annotator.DEFAULT_NUM_ANNOTATIONS,
            False,
            f"The number of annotations to process from the input data. Use -1 to process all data (default = {Annotator.DEFAULT_NUM_ANNOTATIONS}).")

        self.raw_output_path = ConfigValue(
            "",
            False,
            help="The path where raw annotations should be saved.")

        self.sanity_check = ConfigValue(
            Annotator.DEFAULT_SANITY_CHECK,
            False,
            "If true, use only the first rows of the dataset."
        )

        self.random_state = ConfigValue(
            Annotator.DEFAULT_RANDOM_STATE,
            False,
            "Random seed for sampling."
        )

        self.client = None
        from threading import Lock
        self._client_lock = Lock()

    def _get_openai_client(self, openai_args: OpenAIArgs) -> OpenAI:
        with self._client_lock:
            if self.client is None:
                self.client = openai_args.get_client()

        return self.client

    def _resolve_data_file(self) -> str:
        """Combine data_path + data_filename into the actual CSV path to read.

        If data_path already points to a file (or doesn't end in a separator
        and has a recognised extension), it's used as-is. Otherwise data_path
        is treated as a directory and data_filename is appended.
        """
        path = str(self.data_path or "")
        filename = str(getattr(self, "data_filename", "") or "")
        if not path:
            return path
        if os.path.isfile(path):
            return path
        if not filename or os.path.isdir(path) or path.endswith(("/", os.sep)):
            return os.path.join(path, filename) if filename else path
        # Path looks like a file (has an extension) — use it directly.
        if os.path.splitext(path)[1]:
            return path
        return os.path.join(path, filename) if filename else path

    def _get_annotation_data(self) -> pd.DataFrame:
        text_to_annotate = pd.read_csv(self._resolve_data_file(), encoding="utf-8-sig", dtype={"unique_id": str})

        if self.num_annotations > 0 and self.num_annotations < len(text_to_annotate):
            text_to_annotate = text_to_annotate.head(self.num_annotations)

        return text_to_annotate

    def _get_annotations_task(self,
                              text_to_annotate: pd.DataFrame,
                              range_tuple: tuple,
                              iteration: int,
                              openai_args: OpenAIArgs,
                              system_prompt,
                              run_name: str
                              ) -> pd.DataFrame:
        """
        Process a batch of text annotations using the configured OpenAI model.

        This function handles batching, prompt construction, API calls, JSON parsing, 
        and error recovery for a specific subset of the annotation data.

        Workflow:
            1. Check whether raw output for this batch already exists and, if so, load it.
            2. Prepare model input (slice the data, read the prompt, build user/system messages).
            3. Call the OpenAI API with retry logic for transient errors (429/503).
            4. Parse and, if needed, clean the model response into valid JSON.
            5. Attach logprobs to the output when requested and available.
            6. Save the raw model output for reproducibility/caching.

        Args:
            text_to_annotate (pd.DataFrame): The full dataset to annotate.
            range_tuple (tuple): Start and end row indices defining this batch.
            iteration (int): The current annotation iteration number.
            openai_args (OpenAIArgs): Configuration object for OpenAI API calls.

        Returns:
            pd.DataFrame: The annotations for this batch, or an empty DataFrame on error.
        """

        (start_row, end_row) = range_tuple
        records = text_to_annotate.iloc[start_row:end_row].to_dict("records")
        expected = artifacts.identity({
            "records": records, "prompt": system_prompt, "connection": vars(openai_args),
            "iteration": iteration, "concepts": self._concept_specs,
            "hypotheses": self._untrusted_concepts,
        })

        raw_output_file = None
        llm_query = text_to_annotate["llm_query"][start_row:end_row].str.cat(
            sep=" ")

        # --- Step 1: Check if the batch was already processed ---
        try:
            if self.raw_output_path is not None and len(self.raw_output_path) > 0:

                raw_output_file = os.path.join(
                    self.raw_output_path, f"{run_name}_{start_row}_{end_row}_{iteration}.json")

                # If raw output already exists, load it to skip reprocessing
                if artifacts.reusable(raw_output_file, expected):
                    with open(raw_output_file, "r", encoding="utf-8-sig") as file:
                        output_dict = json.load(file)

                    output_dict = model_output.annotations(
                        json.dumps(output_dict), records, self._concept_specs)
                    output_df = pd.DataFrame(output_dict)

                    # add metadata columns with None values
                    output_df = self._align_output_with_llm_query(
                        output_df, llm_query, records)
                    output_df["run_id"] = run_name
                    output_df["model_name"] = None
                    output_df["annotation_timestamp"] = None
                    output_df["latency_seconds"] = None
                    output_df["prompt_tokens"] = None
                    output_df["completion_tokens"] = None
                    output_df["total_tokens"] = None

                    return output_df

                # if not cached → create folder now so future writes work
                os.makedirs(os.path.dirname(raw_output_file), exist_ok=True)

        except (OSError, ValueError):
            raise

        # --- Step 2: Prepare prompt and model input ---

        # prepare the messages for the model
        user_content = json.dumps({
            "untrusted_records": [{k: v for k, v in row.items() if k != "llm_query"} for row in records],
            "untrusted_hypotheses": getattr(self, "_untrusted_concepts", []),
        }, ensure_ascii=False)
        system_prompt = system_prompt

        # Build the message format required by the model
        messages = openai_args.get_model_messages(
            system_prompt=system_prompt, user_content=user_content)

        # --- Step 3: Execute annotation request with retry logic ---
        for _attempt in range(1):
            json_str = ""
            try:
                # Send the annotation request to OpenAI
                start_time_batch = time.perf_counter()
                client = self._get_openai_client(openai_args)
                chat_completion = openai_args.get_api_chat_completion(
                    client, messages=messages)
                output = chat_completion.choices[0].message.content

                # --- Step 4: Parse model output into structured JSON ---
                output_df, output_dict = self._parse_and_format_model_output(
                    output=output,
                    llm_query=llm_query,
                    chat_completion=chat_completion,
                    start_time_batch=start_time_batch,
                    run_name=run_name,
                    openai_config=openai_args,
                    input_records=records,
                )
                # Cache output
                if raw_output_file is not None:
                    artifacts.write_json(raw_output_file, output_dict)
                    artifacts.seal(raw_output_file, expected)
                return output_df

            except Exception as e:
                from common.src.openai.connections import RequestFailure
                if isinstance(e, RequestFailure):
                    raise
                raise RuntimeError(f"Annotation batch {start_row}:{end_row} failed ({type(e).__name__}).") from None

    def _parse_and_format_model_output(
        self,
        output: str,
        llm_query: str,
        chat_completion,
        start_time_batch: float,
        run_name: str,
        openai_config,
        input_records,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Clean, parse, and structure model output into a DataFrame, 
        enrich it with metadata, and attach metadata to text_to_annotate.

        Args:
            output (str): raw model response text
            llm_query (str): combined LLM query for this batch
            chat_completion: OpenAI completion object
            start_time_batch (float): timestamp when request started
            run_name (str): unique run ID
            openai_config: OpenAIArgs object for logprobs

        Returns:
            pd.DataFrame: structured annotations for this batch
            dict: cleaned JSON (for caching)
        """

        # -----------------------
        # 1. Parse JSON
        # -----------------------
        raw = output

        output_dict = model_output.annotations(raw, input_records, self._concept_specs)

        # Convert to DataFrame
        output_df = pd.DataFrame(output_dict)

        # Align output rows with input text instances
        output_df = self._align_output_with_llm_query(output_df, llm_query, input_records)

        # -----------------------
        # 2. Metadata extraction
        # -----------------------
        model_name = getattr(chat_completion, "model", None)
        created = getattr(chat_completion, "created", None)
        timestamp_iso = (
            datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
            if isinstance(created, (int, float)) else None
        )
        latency = time.perf_counter() - start_time_batch
        usage = getattr(chat_completion, "usage", None)

        # Attach run-level + call-level metadata as columns on the output batch
        output_df["run_id"] = run_name
        output_df["model_name"] = model_name
        output_df["annotation_timestamp"] = timestamp_iso
        output_df["latency_seconds"] = latency

        if usage:
            output_df["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
            output_df["completion_tokens"] = getattr(
                usage, "completion_tokens", None)
            output_df["total_tokens"] = getattr(usage, "total_tokens", None)
        else:
            output_df["prompt_tokens"] = None
            output_df["completion_tokens"] = None
            output_df["total_tokens"] = None

        # -----------------------
        # 3. Add logprobs if needed
        # -----------------------
        if getattr(openai_config, "logprobs", False):
            try:
                logprobs = openai_config.get_logprobs(chat_completion)
            except Exception as e:
                print("error with logprobs:", e)
                logprobs = None
            output_df["logprobs"] = logprobs

        return output_df, output_dict

    def _align_output_with_llm_query(self, output_df: pd.DataFrame, llm_query: str, input_records=None) -> pd.DataFrame:
        """
        Align each row of output_df with the correct input text instance
        extracted from the combined llm_query string.

        Steps:
        -------
        1. Split llm_query on "[unique_id]:" occurrences.
        2. Extract the unique_id for each chunk.
        3. Build a mapping unique_id → text block.
        4. Merge onto output_df using the unique_id column.

        Args:
            output_df: Model-generated annotations for one batch.
                    Must contain a column named "unique_id".
            llm_query: The combined block of all input LLM query text for the batch.

        Returns:
            output_df with a new column:
                - input_text: the correct slice of llm_query for that unique_id
        """

        # ------------------------------------------------------------
        # Step 1: Split on "[unique_id]:"
        # ------------------------------------------------------------
        if input_records is None:
            raise ValueError("Original input records are required for identity alignment.")
        mapping = {str(row["unique_id"]): row for row in input_records}
        output_df = output_df.copy()
        output_df["input_text"] = [mapping[str(uid)].get("llm_query", mapping[str(uid)].get("text_to_annotate"))
                                  for uid in output_df["unique_id"]]
        output_df["unique_id"] = [mapping[str(uid)]["unique_id"] for uid in output_df["unique_id"]]
        return output_df

    def _compute_run_metadata(
            self,
            original_df,
            annotated_df,
            model_name: str,
            run_id: str,
            start_time: float,
            end_time: float) -> dict:
        """
        Compute and return metadata about the annotation run.
        """
        # ------------------------------------------------------------
        # (1) Run identifiers and timing
        # ------------------------------------------------------------
        total_runtime_sec = end_time - start_time

        first_ts = annotated_df["annotation_timestamp"].iloc[0]
        last_ts = annotated_df["annotation_timestamp"].iloc[-1]

        # ------------------------------------------------------------
        # (2) Check missing unique_ids
        # ------------------------------------------------------------
        orig_ids = set(original_df["unique_id"].astype(str))
        got_ids = set(annotated_df["unique_id"].astype(str))

        missing_ids = sorted(list(orig_ids - got_ids))
        num_missing = len(missing_ids)

        # ------------------------------------------------------------
        # (3) Token-level metadata (must be stored at the unique-id level)
        # ------------------------------------------------------------
        # Expect columns: prompt_tokens, completion_tokens, total_tokens
        # If not all present, handle gracefully
        token_cols = ["prompt_tokens", "completion_tokens", "total_tokens"]
        available_token_cols = [
            c for c in token_cols if c in annotated_df.columns]

        token_summary = {}
        if available_token_cols:
            token_summary = (
                annotated_df
                .groupby("unique_id")[available_token_cols]
                .sum()
                .agg(["sum", "mean"])
                .to_dict()
            )

        # ------------------------------------------------------------
        # (4) Latency (per unique_id)
        # ------------------------------------------------------------
        if "latency_seconds" in annotated_df.columns:
            latency_summary = (
                annotated_df.groupby("unique_id")["latency_seconds"]
                .mean()
                .agg(["mean", "max", "min", "count"])
                .to_dict()
            )
        else:
            latency_summary = {}

        # Build final metadata dictionary
        meta = {
            "run_id": run_id,
            "model_name": model_name,
            "total_runtime_seconds": total_runtime_sec,
            "annotation_timestamp_start": str(first_ts),
            "annotation_timestamp_end": str(last_ts),
            "num_unique_ids_expected": len(orig_ids),
            "num_unique_ids_received": len(got_ids),
            "num_missing_unique_ids": num_missing,
            "missing_unique_ids": missing_ids,
            "token_metadata": token_summary,
            "latency_metadata": latency_summary,
            "cost_metadata": self._compute_cost_metadata(annotated_df, model_name),
        }

        return meta

    def _aggregate_iterations(
        self,
        long_df: "pd.DataFrame",
        prompt_builder,
        method: str,
        group_col: str = "iteration",
    ) -> tuple["pd.DataFrame", dict]:
        """Aggregate a long-format multi-iteration df to one row per unique_id.

        ``group_col`` names the column that distinguishes the stacked groups
        (e.g. ``"iteration"`` for multi-iteration aggregation, ``"judge"`` for
        Panel-of-Judges aggregation). It is dropped from the output.

        Aggregation rules:
          - boolean / integer concept columns: use `method` ('majority'|'average'|'max').
          - categorical concept columns: mode (modal label).
          - str concept columns + *_Reasoning + *_EvidenceSpan: combine into list[str].
          - *_Confidence: mean.
          - prompt_tokens / completion_tokens / total_tokens / latency_seconds: sum (mean for latency).
          - everything else: take the value from the first group.

        Returns (aggregated_df, consistency_metadata) where consistency_metadata
        has per-concept percent-agreement (% of unique_ids where ALL groups
        produced the same primary label).
        """
        import pandas as pd
        from collections import Counter

        if "unique_id" not in long_df.columns:
            return long_df.copy(), {}

        # Build {concept_name: label_type} from prompt_builder._concept_specs
        concept_types: dict[str, str] = {}
        specs = getattr(prompt_builder, "_concept_specs", None) or []
        for s in specs:
            if s.get("kind") in ("simple", "nested_simple") and s.get("label_type"):
                concept_types[str(s["name"])] = str(s["label_type"]).lower()

        method = (method or "majority").lower()
        if method not in ("majority", "average", "max"):
            method = "majority"

        def _mode(series):
            vals = [v for v in series.tolist() if v is not None and not (isinstance(v, float) and pd.isna(v))]
            if not vals:
                return None
            return Counter(vals).most_common(1)[0][0]

        def _mean(series):
            try:
                return pd.to_numeric(series, errors="coerce").dropna().mean()
            except Exception:
                return None

        def _max(series):
            try:
                return pd.to_numeric(series, errors="coerce").dropna().max()
            except Exception:
                return None

        def _list_strs(series):
            return [str(v) for v in series.tolist()
                    if v is not None and not (isinstance(v, float) and pd.isna(v))]

        all_cols = list(long_df.columns)
        rows = []
        consistency: dict[str, dict] = {}

        for concept in concept_types:
            consistency[concept] = {"label_type": concept_types[concept],
                                    "agreement_count": 0, "total": 0,
                                    "percent_agreement": None}

        grouped = long_df.groupby("unique_id", sort=True)
        for uid, g in grouped:
            row: dict = {"unique_id": uid}
            for col in all_cols:
                if col in ("unique_id", group_col, "iteration"):
                    continue
                series = g[col]

                # Concept primary column (matches a known concept name exactly)
                if col in concept_types:
                    lt = concept_types[col]
                    if lt in ("boolean", "integer"):
                        if method == "average":
                            row[col] = _mean(series)
                        elif method == "max":
                            row[col] = _max(series)
                        else:  # majority
                            row[col] = _mode(series)
                    elif lt == "categorical":
                        row[col] = _mode(series)
                    elif lt == "str":
                        row[col] = _list_strs(series)
                    else:
                        row[col] = _mode(series)

                    # Track exact-match agreement on the raw per-iteration values
                    raw_vals = [v for v in series.tolist()
                                if v is not None and not (isinstance(v, float) and pd.isna(v))]
                    if raw_vals:
                        consistency[col]["total"] += 1
                        if len(set(raw_vals)) == 1:
                            consistency[col]["agreement_count"] += 1
                        # Per-row consistency: % of iterations matching the modal value
                        modal = Counter(raw_vals).most_common(1)[0][0]
                        match = sum(1 for v in raw_vals if v == modal)
                        row[f"{col}_consistency"] = round(100.0 * match / len(raw_vals), 2)
                    else:
                        row[f"{col}_consistency"] = None
                    continue

                # Auxiliary per-concept columns
                if col.endswith("_Reasoning") or col.endswith("_EvidenceSpan"):
                    row[col] = _list_strs(series)
                elif col.endswith("_Confidence"):
                    row[col] = _mean(series)
                elif col in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    try:
                        row[col] = pd.to_numeric(series, errors="coerce").dropna().sum()
                    except Exception:
                        row[col] = series.iloc[0]
                elif col == "latency_seconds":
                    row[col] = _mean(series)
                else:
                    # Pass-through fields (text, metadata, timestamps, etc.)
                    row[col] = series.iloc[0]
            rows.append(row)

        agg_df = pd.DataFrame(rows)

        # Finalise consistency percentages
        for concept, info in consistency.items():
            tot = info["total"]
            info["percent_agreement"] = round(100.0 * info["agreement_count"] / tot, 2) if tot else None

        return agg_df, consistency

    @staticmethod
    def aggregate_panel_outputs(
        per_judge_paths: list[tuple[str, str]],
        output_csv_path: str,
        prompt_builder,
        method: str,
    ) -> tuple["pd.DataFrame", dict]:
        """Combine per-judge annotation CSVs into one aggregated CSV.

        ``per_judge_paths`` is a list of ``(judge_label, csv_path)`` pairs.
        The aggregation logic mirrors :py:meth:`_aggregate_iterations`: each
        judge's CSV is stacked vertically with a ``judge`` column, then the
        same per-column rules are applied (boolean/integer use ``method``;
        categorical → mode; str / *_Reasoning / *_EvidenceSpan → list;
        *_Confidence → mean; tokens summed; latency averaged).

        Writes the aggregated CSV to ``output_csv_path`` and a sidecar
        ``metadata.json`` next to it. Returns ``(agg_df, consistency_metadata)``.
        """
        import os, json
        import pandas as pd

        frames = []
        loaded_judges: list[str] = []
        for judge_label, path in per_judge_paths:
            if not path or not os.path.isfile(path):
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            df["judge"] = str(judge_label)
            frames.append(df)
            loaded_judges.append(str(judge_label))

        if not frames:
            return pd.DataFrame(), {}

        long_df = pd.concat(frames, ignore_index=True, sort=False)

        # Re-use the instance-level aggregator. We only need the type-handling
        # behaviour, not any instance state, so a temporary stub is enough.
        stub = Annotator.__new__(Annotator)
        agg_df, consistency = stub._aggregate_iterations(
            long_df, prompt_builder, method, group_col="judge"
        )

        os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
        agg_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

        meta_path = os.path.join(os.path.dirname(output_csv_path), "panel_metadata.json")
        meta = {
            "panel_judges": loaded_judges,
            "num_judges": len(loaded_judges),
            "panel_aggregation": method,
            "consistency_metadata": consistency,
            "per_judge_csvs": [{"judge": j, "path": p} for j, p in per_judge_paths
                               if os.path.isfile(p)],
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)
        except Exception:
            pass

        return agg_df, consistency

    def _compute_cost_metadata(self, annotated_df, model_name: str) -> dict:
        """Compute USD cost from observed token usage and model pricing."""
        try:
            from src.run_estimator import (
                MODEL_PRICING, MODEL_REASONING_MULT, resolve_model_key,
            )
        except ImportError:
            return {"cost_total_usd": None, "reason": "Pricing unavailable"}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            prompt_tokens = int(annotated_df["prompt_tokens"].fillna(0).sum()) \
                if "prompt_tokens" in annotated_df.columns else 0
            completion_tokens = int(annotated_df["completion_tokens"].fillna(0).sum()) \
                if "completion_tokens" in annotated_df.columns else 0

        key = resolve_model_key(model_name or "")
        pricing = MODEL_PRICING.get(key)
        if pricing is None:
            return {
                "cost_total_usd": None,
                "model_resolved": key,
                "prompt_tokens_total": prompt_tokens,
                "completion_tokens_total": completion_tokens,
                "warning": f"No pricing entry for model '{key}'",
            }
        if annotated_df[["prompt_tokens", "completion_tokens"]].isna().any().any():
            return {"cost_total_usd": None, "reason": "Cached or missing token usage"}

        in_price, out_price, _cached = pricing
        # Completion tokens in API usage already include reasoning tokens for
        # reasoning models, so no extra multiplier is needed here.
        cost_input = (prompt_tokens / 1_000_000.0) * in_price
        cost_output = (completion_tokens / 1_000_000.0) * out_price
        cost_total = cost_input + cost_output

        def _fmt(c: float) -> str:
            if c < 0.01:
                return f"${c:.4f}"
            if c < 1.0:
                return f"${c:.3f}"
            return f"${c:.2f}"

        return {
            "model_resolved": key,
            "prompt_tokens_total": prompt_tokens,
            "completion_tokens_total": completion_tokens,
            "input_price_per_1m_usd": in_price,
            "output_price_per_1m_usd": out_price,
            "cost_input_usd": round(cost_input, 6),
            "cost_output_usd": round(cost_output, 6),
            "cost_total_usd": round(cost_total, 6),
            "cost_display": _fmt(cost_total),
            "reasoning_multiplier_assumed": MODEL_REASONING_MULT.get(key, 1.0),
        }

    def execute_annotation_process(
        self,
        openai_args: OpenAIArgs,
        prompt_builder,
        preprocessor,
        run_name: str,
    ):
        """
        Run the full annotation pipeline: load data, generate annotations, and save results.

        The function:
            1. Load input data to annotate from `data_path`
            2. Run annotation process for each iteration
            3. Annotate batches in parallel using threads
            4. Combine results from all batches.
            5. Save all annotations to disk in `output_path`

        Args:
            openai_args (OpenAIArgs): Configuration for OpenAI API calls, including model and parameters.
        """
        start_time = time.perf_counter()
        for field in ("batch_size", "num_iterations"):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer.")
        if not self.raw_output_path:
            self.raw_output_path = os.path.join(os.path.dirname(self.output_path), "raw")

        # --- Step 1: Load input data to annotate from `data_path` ---
        print(f"\n{time_utils.get_time_and_elapsed_status(start_time)}: Getting annotation data from {self.data_path}...")
        text_to_annotate = self._get_annotation_data()
        print(f"{time_utils.get_time_and_elapsed_status(start_time)}: DONE!")

        # --- Step 2: Preprocess the data ---
        text_to_annotate = preprocessor.preprocess(text_to_annotate,
                                                   sanity_check=self.sanity_check,
                                                   num_annotations=self.num_annotations,
                                                   random_state=self.random_state,
                                                   batch_size=self.batch_size)

        # Initialize an empty DataFrame to hold annotation results across all iterations
        out = pd.DataFrame()

        # Determine how many batches to process based on total data size and batch size
        num_rows = len(text_to_annotate)
        if not num_rows:
            raise ValueError("Annotation input contains no records.")
        num_batches = math.ceil(num_rows/self.batch_size)

        # Ensure the number of threads has correctly been specified
        if self.num_threads < 1:
            self.num_threads = Annotator.get_default_num_threads()

        # Validate prompt concepts and build base prompt
        prompt_builder.validate_concepts()
        self._concept_specs = prompt_builder._concept_specs
        prompt = prompt_builder.build_prompt(text_to_annotate)
        self._untrusted_concepts = getattr(prompt_builder, "_untrusted_concepts", [])
        final_identity = artifacts.identity({
            "records": text_to_annotate.to_dict("records"), "prompt": prompt,
            "hypotheses": self._untrusted_concepts, "connection": vars(openai_args),
            "iterations": self.num_iterations, "aggregation": self.aggregation,
        })
        if artifacts.reusable(self.output_path, final_identity):
            return
        status_path = str(self.output_path) + ".status.json"
        artifacts.write_json(status_path, {"status": "in_progress", "identity": final_identity})

        # Track per-iteration outputs so we can save them individually and
        # then aggregate across iterations after the loop.
        iteration_dfs: list[pd.DataFrame] = []

        # --- Step 3: Run annotation process for each iteration ---
        for iteration in range(self.num_iterations):
            print(f"\n{time_utils.get_time_and_elapsed_status(start_time)}: Beginning Round " +
                  str(iteration + 1) + " of " + str(self.num_iterations))

            # Define start and end indices for each batch
            range_tuples = [
                (i * self.batch_size, min((i + 1) * self.batch_size, num_rows))
                for i in range(num_batches)
            ]

            # --- Step 4: Annotate batches in parallel using threads ---
            def batch_outcome(rt):
                try:
                    return rt[0], self._get_annotations_task(
                        text_to_annotate, rt, iteration, openai_args, prompt, run_name), None
                except Exception as exc:
                    from common.src.openai.connections import RequestFailure
                    return rt[0], None, {"start": rt[0], "end": rt[1],
                                       "iteration": iteration, "error_type": type(exc).__name__,
                                       "http_status": exc.status_code if isinstance(exc, RequestFailure) else None}

            outcomes = thread_utils.execute_func_in_parallel_with_status(
                range_tuples,
                batch_outcome,
                self.num_threads,
                "Getting Annotations"
            )
            failures = [error for _, _, error in outcomes if error]
            if failures:
                artifacts.write_json(status_path, {"status": "incomplete", "identity": final_identity,
                                                   "failures": failures})
                raise RuntimeError(f"{len(failures)} annotation batches failed. Successful raw batches were preserved.")
            result_dfs = [frame for _, frame, _ in sorted(outcomes, key=lambda item: item[0])]

            # --- Step 5: Combine results from all batches ---
            print(
                f"{time_utils.get_time_and_elapsed_status(start_time)}: Add results to the final output...")
            iter_df = pd.DataFrame()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                for text_df_out in result_dfs:
                    # Always tag with iteration so per-iteration files have provenance
                    text_df_out['iteration'] = iteration + 1
                    iter_df = pd.concat([iter_df, text_df_out])
                    out = pd.concat([out, text_df_out])
            iteration_dfs.append(iter_df)

            print(f"{time_utils.get_time_and_elapsed_status(start_time)}: DONE!")

        # --- Step 6: Save annotations to disk ---
        print(f"\n{time_utils.get_time_and_elapsed_status(start_time)}: Saving annotation data to {self.output_path}...")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # If more than one iteration, save each iteration to its own CSV next
        # to the final aggregated output, then build the aggregated wide df.
        consistency_metadata: dict = {}
        if self.num_iterations > 1:
            output_dir = os.path.dirname(self.output_path)
            base, ext = os.path.splitext(os.path.basename(self.output_path))
            ext = ext or ".csv"
            for idx, iter_df in enumerate(iteration_dfs, start=1):
                iter_path = os.path.join(output_dir, f"{base}_iter{idx}{ext}")
                artifacts.write_csv(iter_path, iter_df)
                print(f"  Saved iteration {idx} -> {iter_path}")

            aggregated, consistency_metadata = self._aggregate_iterations(
                out, prompt_builder, str(self.aggregation or "majority").lower()
            )
            artifacts.write_csv(self.output_path, aggregated)
            print(f"  Saved aggregated -> {self.output_path}")
        else:
            # Single-iteration: drop the iteration column we added so the file
            # looks the same as before, and save as the final output.
            single = out.drop(columns=["iteration"], errors="ignore")
            artifacts.write_csv(self.output_path, single)

        end_time = time.perf_counter()

        metadata = self._compute_run_metadata(
            original_df=text_to_annotate,
            annotated_df=out,
            model_name=openai_args.openai_model,
            run_id=run_name,
            start_time=start_time,
            end_time=end_time
        )
        if consistency_metadata:
            metadata["consistency_metadata"] = consistency_metadata
            metadata["aggregation_method"] = str(self.aggregation or "majority").lower()
            metadata["num_iterations"] = int(self.num_iterations)

        # Optionally save metadata to disk
        output_dir = os.path.dirname(self.output_path)
        meta_out = os.path.join(output_dir, f"{run_name}_metadata.json")
        artifacts.write_json(meta_out, metadata)

        print(f"Saved metadata to: {meta_out}")
        artifacts.write_json(status_path, {"status": "complete", "identity": final_identity})
        from pathlib import Path
        artifacts.seal(self.output_path, final_identity, [
            status_path, meta_out, *Path(self.output_path).parent.glob(Path(self.output_path).stem + "_iter*.csv")
        ])