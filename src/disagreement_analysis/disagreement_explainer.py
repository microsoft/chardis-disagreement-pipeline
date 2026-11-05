# Import standard libraries
from datetime import datetime, timezone
import json
import threading
from typing import List
import pandas as pd
import time
import os
import sys
import concurrent.futures as cf
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Heavy ML imports (pdc_dp_means, sentence_transformers, sklearn.preprocessing)
# are only needed at runtime by specific methods. Defer them so
# config-only consumers (CLI, dashboard, config_template) can import this
# module without paying the dep cost.

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
from src.disagreement_analysis.preprocessor_disagreement import Preprocessor
from common.src.utils import json_utils, thread_utils, time_utils
from common.src.utils import artifacts, model_output
from common.src.openai.openai_args import OpenAIArgs
from common.src.configs.configurable_base import ConfigurableBase
from common.src.configs.config_value import ConfigValue

# Remove the temporary path to avoid polluting the import namespace
if _root_added:
    sys.path.remove(_root_added)
    
# -------------------------------
# DisagreementExplainer
# -------------------------------

class DisagreementExplainer(ConfigurableBase):
    """
    Main process for explaining disagreements between two annotation labels.
    """
    DEFAULT_NUM_ANNOTATIONS = -1
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_TRAIN_ON_TEST = False
    BASE_DIR = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir))
    DEFAULT_NUM_EXAMPLES = 5
    DEFAULT_MORE_THAN_2 = False
    DEFAULT_NUM_THREADS = 10
    EMBEDDING_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"

    def __init__(self):
        """    
        Initialize core parameters and runtime settings.
        
        Args:
            Data and directories:
            - data_path (str): path to CSV for input data
            - guidelines_path (str): path to concept specific annotation guidelines
            - examples_path (str): path to in-context examples
        
            Disagreement settings:
            - example_size (int): number of in-context examples to provide
            - train_on_test (bool): whether to train on test data
            - more_than_2 (bool): whether there are more than 2 annotators
            
            Runtime settings:
            - num_annotations (int): number of annotations to process (-1 = all)
            - random_state (int): random seed for reproducibility
            - num_threads (int, optional): Threads for parallel execution.
        """
        self.data_path = ConfigValue(
            None,
            True,
            "Directory (or full path) of the CSV with input data.")

        self.data_filename = ConfigValue(
            "annotated_data.csv",
            False,
            "CSV filename inside data_path. Ignored if data_path already points at a file.")

        self.guidelines_path = ConfigValue(
            None,
            True,
            "Directory (or full path) of the annotation guidelines markdown.")

        self.guidelines_filename = ConfigValue(
            "prompt.md",
            False,
            "Markdown filename inside guidelines_path. Ignored if guidelines_path already points at a file.")

        self.examples_path = ConfigValue(
            None,
            True,
            "The path to in-context examples.")

        self.output_path = ConfigValue(
            None,
            False,
            "Directory where preprocessing artifacts, raw outputs, metadata.json, and final disagreement_explanations.json are written. Created if missing.")

        # base_dir is always the repo root. Not user-configurable.
        self.base_dir = DisagreementExplainer.BASE_DIR
        
        self.example_size = ConfigValue(
            DisagreementExplainer.DEFAULT_NUM_EXAMPLES,
            True,
            "The number of in-context examples to provide.")
        
        self.train_on_test = ConfigValue(
            DisagreementExplainer.DEFAULT_TRAIN_ON_TEST,
            False,
            "Whether to train on test data.")
        
        self.more_than_2 = ConfigValue(
            DisagreementExplainer.DEFAULT_MORE_THAN_2, 
            False,
            "Whether there are more than 2 annotators.")
        
        self.num_annotations = ConfigValue(
            DisagreementExplainer.DEFAULT_NUM_ANNOTATIONS,
            False,
            f"The number of annotations to process from the input data. Use -1 to process all data (default = {DisagreementExplainer.DEFAULT_NUM_ANNOTATIONS}).")

        self.random_state = ConfigValue(
            DisagreementExplainer.DEFAULT_RANDOM_STATE,
            False,
            "Random seed for sampling."
        )
        
        self.num_threads = ConfigValue(
            DisagreementExplainer.DEFAULT_NUM_THREADS,
            False,
            f"The number of threads to use for parallel processing (default = {DisagreementExplainer.DEFAULT_NUM_THREADS})."
        )

    def _resolve_combined_path(self, directory, filename) -> str:
        """Combine a directory + filename into a path.

        Mirrors annotation.Annotator._resolve_data_file: if `directory` already
        points at an existing file, return it; otherwise join with `filename`.
        Empty `filename` falls through to `directory` for backwards compatibility
        with configs that stored a single full path.
        """
        d = (str(directory) if directory is not None else "").strip()
        f = (str(filename) if filename is not None else "").strip()
        if not d:
            return ""
        if os.path.isfile(d):
            return d
        if f and (d.endswith(("/", os.sep)) or os.path.isdir(d)
                  or not os.path.splitext(d)[1]):
            return os.path.join(d, f)
        return d

    def resolved_data_path(self) -> str:
        return self._resolve_combined_path(self.data_path, getattr(self, "data_filename", ""))

    def resolved_guidelines_path(self) -> str:
        return self._resolve_combined_path(self.guidelines_path, getattr(self, "guidelines_filename", ""))

    def get_annotation_data(self) -> pd.DataFrame:
        text_to_annotate = pd.read_csv(self.resolved_data_path(), encoding="utf-8-sig")

        if self.num_annotations > 0 and self.num_annotations < len(text_to_annotate):
            text_to_annotate = text_to_annotate.head(self.num_annotations)

        return text_to_annotate
    
    def get_openai_client(self, openai_args:OpenAIArgs):
        return openai_args.get_client()

    # ── Token / cost tracking ────────────────────────────────────────────
    def _reset_token_usage(self) -> None:
        """(Re)initialize the per-stage token usage accumulators for one run."""
        self._token_lock = threading.Lock()
        self._token_usage = {
            "rationales": {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "cached_prompt_tokens": 0,
                "total_tokens": 0,
            },
            "synthesizer": {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "cached_prompt_tokens": 0,
                "total_tokens": 0,
            },
        }

    def _record_usage(self, stage: str, chat_completion) -> None:
        """Add one chat_completion's usage to the per-stage counters (thread-safe).

        Reads ``usage.prompt_tokens`` / ``completion_tokens`` / ``total_tokens``
        and, when present, the reasoning-tokens detail and prompt-cache hit count.
        Silently drops the call if usage is missing (e.g. mocked responses).
        """
        if stage not in getattr(self, "_token_usage", {}) or chat_completion is None:
            return
        usage = getattr(chat_completion, "usage", None)
        if usage is None:
            return
        prompt_t  = int(getattr(usage, "prompt_tokens", 0) or 0)
        compl_t   = int(getattr(usage, "completion_tokens", 0) or 0)
        total_t   = int(getattr(usage, "total_tokens", 0) or (prompt_t + compl_t))
        # Reasoning tokens (o-series, gpt-5 with reasoning) are billed as output.
        reasoning_t = 0
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_t = int(getattr(details, "reasoning_tokens", 0) or 0)
        # Cached input tokens (prompt-cache hits) are billed at the discounted rate.
        cached_t = 0
        pdetails = getattr(usage, "prompt_tokens_details", None)
        if pdetails is not None:
            cached_t = int(getattr(pdetails, "cached_tokens", 0) or 0)

        with self._token_lock:
            bucket = self._token_usage[stage]
            bucket["calls"]                 += 1
            bucket["prompt_tokens"]         += prompt_t
            bucket["completion_tokens"]     += compl_t
            bucket["reasoning_tokens"]      += reasoning_t
            bucket["cached_prompt_tokens"]  += cached_t
            bucket["total_tokens"]          += total_t

    def _compute_cost(self, stage_usage: dict, model_name: str) -> dict:
        """Convert a stage's token totals into a USD cost using ``run_estimator``'s
        pricing table. Cached prompt tokens are charged at the discounted rate;
        completion tokens (including reasoning) at the output rate.
        """
        try:
            # Re-use the same pricing table the cost estimator UI uses.
            from src.run_estimator import (
                MODEL_PRICING, DEFAULT_PRICING, resolve_model_key,
            )
        except ImportError:
            return {"input_usd": None, "output_usd": None, "total_usd": None,
                    "model_resolved": None, "model_input": model_name, "reason": "Pricing unavailable"}
        key = resolve_model_key(model_name)
        if key not in MODEL_PRICING:
            return {"input_usd": None, "output_usd": None, "total_usd": None,
                    "model_resolved": key, "reason": "Model pricing unavailable"}
        in_p, out_p, cached_p = MODEL_PRICING[key]
        prompt_t  = int(stage_usage.get("prompt_tokens", 0))
        cached_t  = int(stage_usage.get("cached_prompt_tokens", 0))
        compl_t   = int(stage_usage.get("completion_tokens", 0))
        # Cached portion comes out of prompt_tokens (per OpenAI usage semantics).
        uncached_t = max(0, prompt_t - cached_t)
        in_cost     = (uncached_t / 1_000_000.0) * in_p + (cached_t / 1_000_000.0) * cached_p
        out_cost    = (compl_t / 1_000_000.0) * out_p
        return {
            "input_usd":  round(in_cost, 6),
            "output_usd": round(out_cost, 6),
            "total_usd":  round(in_cost + out_cost, 6),
            "model_input": model_name,
            "model_resolved": key,
            "input_price_per_mtok":        in_p,
            "output_price_per_mtok":       out_p,
            "cached_input_price_per_mtok": cached_p,
        }

    def _summarize_token_usage(
        self,
        rationales_model: str,
        synthesizer_model: str,
    ) -> dict:
        """Build the ``token_usage`` block for metadata.json (totals + per-stage)."""
        usage = getattr(self, "_token_usage", {})
        rationales = dict(usage.get("rationales", {}))
        synthesizer = dict(usage.get("synthesizer", {}))
        r_cost = self._compute_cost(rationales, rationales_model)
        s_cost = self._compute_cost(synthesizer, synthesizer_model)
        def combined(field):
            return (round(r_cost[field] + s_cost[field], 6)
                    if r_cost[field] is not None and s_cost[field] is not None else None)
        total = {
            "calls":                rationales.get("calls", 0)               + synthesizer.get("calls", 0),
            "prompt_tokens":        rationales.get("prompt_tokens", 0)       + synthesizer.get("prompt_tokens", 0),
            "completion_tokens":    rationales.get("completion_tokens", 0)   + synthesizer.get("completion_tokens", 0),
            "reasoning_tokens":     rationales.get("reasoning_tokens", 0)    + synthesizer.get("reasoning_tokens", 0),
            "cached_prompt_tokens": rationales.get("cached_prompt_tokens", 0)+ synthesizer.get("cached_prompt_tokens", 0),
            "total_tokens":         rationales.get("total_tokens", 0)        + synthesizer.get("total_tokens", 0),
            "input_usd":            combined("input_usd"),
            "output_usd":           combined("output_usd"),
            "total_usd":            combined("total_usd"),
        }
        return {
            "rationales":  {**rationales,  "cost": r_cost},
            "synthesizer": {**synthesizer, "cost": s_cost},
            "total": total,
        }

    def _load_file(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing prompt file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _write_run_metadata(
        self,
        run_dir: str,
        stats: dict,
        run_name: str,
        token_usage: dict | None = None,
    ) -> None:
        """Write a metadata.json file describing the run's preprocessing outputs.

        Combines the train/val/test split sizes (read off the CSVs the preprocessor
        just wrote) with the corpus + chunking diagnostics tracked by
        :class:`Preprocessor.stats`. When ``token_usage`` is provided it also
        records per-stage prompt/completion token counts and USD cost estimates.
        """
        def _csv_rows(path: str) -> int:
            """Count CSV records (not physical lines), so embedded newlines in
            quoted fields don't inflate the count."""
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Required split is missing: {path}")
            return int(pd.read_csv(path, low_memory=False, usecols=[0]).shape[0])

        train_csv = os.path.join(run_dir, "train.csv")
        val_csv   = os.path.join(run_dir, "validation", "validation.csv")
        test_csv  = os.path.join(run_dir, "heldout", "test.csv")

        meta = {
            "embedding_revision": self.EMBEDDING_REVISION,
            "run_name": run_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_path": self.resolved_data_path(),
            "splits": {
                "train_size":      _csv_rows(train_csv),
                "validation_size": _csv_rows(val_csv),
                "test_size":       _csv_rows(test_csv),
            },
            "chunking": {
                "examples_per_chunk":   int(self.example_size),
                "num_chunks":           int(stats.get("num_chunks", 0)),
                "total_chunk_rows":     int(stats.get("total_chunk_rows", 0)),
            },
            "annotators": {
                "num_annotators":            int(stats.get("num_annotators", 0)),
                "num_label_types":           int(stats.get("num_label_types", 0)),
                "label_values":              stats.get("label_values", []),
                "avg_annotators_per_text":   round(float(stats.get("avg_annotators_per_text", 0.0)), 3),
                "more_than_2_mode":          bool(self.more_than_2),
            },
            "disagreement": {
                "total_disagreements":         int(stats.get("total_disagreements", 0)),
                "disagreement_pct":            round(float(stats.get("disagreement_pct", 0.0)), 2),
                "train_disagreements":         int(stats.get("train_disagreements", 0)),
                "holdout_disagreements":       int(stats.get("holdout_disagreements", 0)),
                "disagreement_cases_kept":     int(stats.get("disagreement_cases_kept", 0)),
                "disagreement_cases_lost":     int(stats.get("disagreement_cases_lost", 0)),
                "disagreement_cases_lost_pct": round(float(stats.get("disagreement_cases_lost_pct", 0.0)), 2),
            },
            "annotator_pairs": {
                "min_pair_overlap":              int(stats.get("min_pair_overlap", 0)),
                "raw_annotator_pairs":           int(stats.get("raw_annotator_pairs", 0)),
                "raw_annotator_pairs_kept":      int(stats.get("raw_annotator_pairs_kept", 0)),
                "raw_annotator_pairs_pruned":    int(stats.get("raw_annotator_pairs_pruned", 0)),
                "annotator_pairs_pruned_pct":    round(float(stats.get("annotator_pairs_pruned_pct", 0.0)), 2),
                "total_pairwise_disagreements":  int(stats.get("total_pairwise_disagreements", 0)),
                "pruned_pairwise_disagreements": int(stats.get("pruned_pairwise_disagreements", 0)),
                "pruned_pair_disagreement_pct":  round(float(stats.get("pruned_pair_disagreement_pct", 0.0)), 2),
                "candidate_cluster_pairs":       int(stats.get("candidate_cluster_pairs", 0)),
                "kept_cluster_pairs":            int(stats.get("kept_cluster_pairs", 0)),
                "pruned_cluster_pairs":          int(stats.get("pruned_cluster_pairs", 0)),
                "cluster_pairs_pruned_pct":      round(float(stats.get("cluster_pairs_pruned_pct", 0.0)), 2),
            },
            "input_rows": int(stats.get("input_rows", 0)),
        }
        if token_usage is not None:
            meta["token_usage"] = token_usage

        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, "metadata.json")
        artifacts.write_json(out_path, meta)
        print(f"[DisagreementExplainer] Wrote run metadata → {out_path}")
    
    def _load_rationales(self):
        path = os.path.join(self.base_dir, "prompts", "DEFAULTS", "disagreement_explainer", "rationales", "rationales_1.0.md")
        return self._load_file(path)

    def _load_synthesizer(self) -> str:
        path = os.path.join(self.base_dir, "prompts", "DEFAULTS", "disagreement_explainer", "synthesizer", "synthesizer_1.0.md")
        return self._load_file(path)

    def get_rationales(self, 
                         results,
                         rationales_template: str,
                         guidelines: str,
                         df,
                         openai_config:OpenAIArgs,
                         label: str,
                         client,
                         annotator_id):
        # Generate rationales for one annotator
        prompt = rationales_template + "\n" + guidelines
        records = [{"text": row["text"], "label": row[label]} for _, row in df.iterrows()]
        messages = openai_config.get_model_messages(
            system_prompt=prompt, user_content=json.dumps({"untrusted_labeled_texts": records}, ensure_ascii=False))
        chat_completion = openai_config.get_api_chat_completion(client, messages=messages)
        self._record_usage("rationales", chat_completion)
        rationales = model_output.string_list(chat_completion.choices[0].message.content)
        rationales_label = annotator_id +"_rationales"
        results[rationales_label].append(rationales)

        return results, rationales

    def synthesize_rationales(self, 
                                chunks, 
                                rationales_template: str, 
                                synthesizer_template: str, 
                                guidelines: str, 
                                raw_dir: Path,
                                rationales_openai_args:OpenAIArgs,
                                synthesizer_openai_args:OpenAIArgs):
        from contextlib import closing
        with closing(self.get_openai_client(rationales_openai_args)) as rationale_client, \
                closing(self.get_openai_client(synthesizer_openai_args)) as synthesis_client:
            return self._synthesize_rationales(
                chunks, rationales_template, synthesizer_template, guidelines, raw_dir,
                rationales_openai_args, synthesizer_openai_args, rationale_client, synthesis_client)

    def _synthesize_rationales(self, chunks, rationales_template, synthesizer_template,
                              guidelines, raw_dir, rationales_openai_args, synthesizer_openai_args,
                              rationale_client, synthesis_client):
        annotator_cols = [c for c in chunks[0].columns if c not in {"unique_id", "text"}]
        annotator1 = annotator_cols[0]
        annotator2 = annotator_cols[1]
        
        results = {f"{annotator1}_rationales": [], f"{annotator2}_rationales": [], 'explanations': [], 'in_context_examples': []}
        
        # for each annotator chunk, get rationales
        for sample_df in tqdm(chunks):
            results['in_context_examples'].append(list(sample_df["text"]))

            results, A1_rationales = self.get_rationales(results,
                                                           rationales_template,
                                                           guidelines,
                                                           sample_df,
                                                           rationales_openai_args,
                                                           annotator1,
                                                           rationale_client,
                                                           annotator1)

            results, A2_rationales = self.get_rationales(results,
                                                           rationales_template,
                                                           guidelines,
                                                           sample_df,
                                                           rationales_openai_args,
                                                           annotator2,
                                                           rationale_client,
                                                           annotator2)
            
            # Synthesize disagreement explanations from rationales for both annotators
            synthesizer_prompt = synthesizer_template + "\nAnnotation guidelines:\n" + guidelines
            messages = synthesizer_openai_args.get_model_messages(
                system_prompt=synthesizer_prompt,
                user_content=json.dumps({"untrusted_rationales": [A1_rationales, A2_rationales]}, ensure_ascii=False))
            chat_completion = synthesizer_openai_args.get_api_chat_completion(synthesis_client, messages=messages)
            self._record_usage("synthesizer", chat_completion)
            disagreement_explanations = model_output.string_list(chat_completion.choices[0].message.content)
            results['explanations'].append(disagreement_explanations)

            # Cache output
            if raw_dir is not None:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                timestamp = str(timestamp)
                raw_output_path = os.path.join(raw_dir, f"chunk_{timestamp}_results.json")
                with open(raw_output_path, "w", encoding="utf-8-sig") as file:
                    json.dump(results, file)
            
        return results
        
    def cosine_similarity(self, v1, v2):
        """Compute the cosine similarity between two vectors."""
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return np.dot(v1, v2) / (norm_v1 * norm_v2)

    def find_closest_examples(self, n_clusters, cluster_centers, embeddings, cluster_predictions, mcs):
        closest_examples = {}
        cluster_sizes = {}        
        
        for cluster in range(n_clusters):
            if cluster not in cluster_predictions: continue
            cluster_indices = np.where(cluster_predictions == cluster)[0]
            size_of_cluster = len(cluster_indices)
            cluster_embeddings = embeddings[cluster_indices]
            
            similarities = [
                self.cosine_similarity(embedding, cluster_centers[cluster]) 
                for embedding in cluster_embeddings
            ]
            closest_index_in_cluster = np.argmax(similarities)

            closest_examples[cluster] = mcs[cluster_indices[closest_index_in_cluster]]
            cluster_sizes[cluster] = size_of_cluster

        return closest_examples, cluster_sizes

    def cluster_explanations(self, results, model, raw_dir=None):
        # Lazy heavy deps — only loaded when clustering actually runs.
        from sklearn.preprocessing import normalize
        from pdc_dp_means import DPMeans

        explanations = results['explanations']
        
        # Union of explanations 
        union_of_explanations = []
        for e_set in explanations:
            if type(e_set) == list:
                union_of_explanations += [
                    d.strip('- ').strip() 
                    for d in e_set if d.strip() and len(d.strip('- ').strip().split()) > 2]
            else:
                union_of_explanations += [
                    d.strip('- ').strip() 
                    for d in e_set.split('\n') 
                    if d.strip() and len(d.strip('- ').strip().split()) > 2]

        # Embed explanations using sentence-transformers
        embeddings = model.encode(union_of_explanations)
        embeddings = embeddings.astype(np.float64)
        embeddings = normalize(embeddings, norm='l2')

        # cluster embeddings
        # small delta results in many small clusters, each with high internal semantic similarity
        dpmeans = DPMeans(n_init=1, delta=0.3) 
        dpmeans.fit(embeddings)
        cluster_predictions = dpmeans.predict(embeddings)
        cluster_centers = dpmeans.cluster_centers_
        n_clusters = dpmeans.n_clusters

        # find closest examples
        cluster_representatives, cluster_sizes = self.find_closest_examples(n_clusters, 
                                                                            cluster_centers,
                                                                            embeddings, 
                                                                            cluster_predictions,
                                                                            union_of_explanations)

        results['cluster_predictions'] = cluster_predictions.tolist()
        results['all_deduplicated_explanations'] = cluster_representatives
        results['cluster_sizes'] = cluster_sizes    
        
        top_cluster_indices = sorted(
            cluster_sizes.keys(), 
            key=lambda x: cluster_sizes[x], 
            reverse=True
        )

        # create a dict to hold cluster representative and cluster members
        clusters = {}
        for i in top_cluster_indices:
            cluster_member_indices = np.where(np.array(cluster_predictions) == i)[0]
            clusters[i] = {
                'size': cluster_sizes[i],
                'representative': cluster_representatives[i],
                'members': [union_of_explanations[j] for j in cluster_member_indices]
            }
        results['clusters'] = clusters
        
        # Cache output
        if raw_dir is not None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            timestamp = str(timestamp)
            raw_output_path = os.path.join(raw_dir, f"chunk_{timestamp}_clusters.json")
            with open(raw_output_path, "w", encoding="utf-8-sig") as file:
                json.dump(results, file)
                
        explanations_by_frequency = []
        for c in results['clusters'].values():
            explanations_by_frequency.append(c['representative'])
                
        return explanations_by_frequency

    def aggregate_results(self, all_results: list[dict]) -> dict:
        agg = {
            "explanations": [],
            "in_context_examples": [],
        }

        for r in all_results:
            if "in_context_examples" in r:
                agg["in_context_examples"].extend(r["in_context_examples"])
            if "explanations" in r:
                agg["explanations"].extend(r["explanations"])
        return agg

    def execute_disagreement_process(
        self,
        rationales_openai_args: OpenAIArgs,
        synthesizer_openai_args: OpenAIArgs,
        run_name: str
        ):
        """Executes the disagreement explanation process.

        Args:
            rationales_openai_args (OpenAIArgs): Configuration for OpenAI API.
            synthesizer_openai_args (OpenAIArgs): Configuration for OpenAI API.
            run_name (str): Name of the current run.

        Returns:
            list: Clustered explanations sorted by frequency.
        """

        start_time = time.perf_counter()

        # Initialize per-run token usage accumulator (thread-safe).
        self._reset_token_usage()

        ###################################################
        # --- Step 1: Load input data from `data_path` ---#
        ###################################################
        print(f"\n{time_utils.get_time_and_elapsed_status(start_time)}: Getting annotation data from {self.resolved_data_path()}...")
        
        # Main function to get the annotated data
        annotated_data = self.get_annotation_data()
        
        print(f"{time_utils.get_time_and_elapsed_status(start_time)}: DONE!")
        
        ################################################################################################
        # --- Step 2: Preprocess the data: Train-Test Split, Annotator Selection, Chunking the data ---#
        ################################################################################################
        print(f"\n{time_utils.get_time_and_elapsed_status(start_time)}: Preprocessing the data...")
        
        # Define output paths. Prefer the user-configured output_path; fall back
        # to the legacy <base_dir>/outputs/disagreement_outputs/<run> location.
        if not isinstance(run_name, str):
            run_name = str(run_name)
        configured_output = (self.output_path or "").strip() if isinstance(self.output_path, str) else ""
        if configured_output:
            if os.path.isabs(configured_output):
                run_dir = configured_output
            else:
                run_dir = os.path.normpath(os.path.join(self.base_dir, configured_output))
        else:
            run_dir = os.path.join(self.base_dir, "outputs", "disagreement_outputs", run_name)

        os.makedirs(run_dir, exist_ok=True)
        output_path = os.path.join(run_dir, "disagreement_explanations.json")
        raw_dir = os.path.join(run_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        train_path = run_dir
        validation_path = os.path.join(train_path, "validation")
        test_path = os.path.join(train_path, "heldout")

        # Skip-guard: if a curated/previous disagreement_explanations.json already
        # exists at the resolved run_dir, skip preprocessing + LLM stages entirely
        # to avoid (a) burning rationalizer/synthesizer tokens on an already-
        # completed run and (b) overwriting a user-curated explanations file.
        # Downstream eval consumes validation.csv / test.csv + the explanations
        # JSON directly, so no further work in this method is required.
        if os.path.isfile(output_path):
            raise ValueError("Existing discovery output requires provenance verification through the orchestrator.")

        # Main function to preprocess the data
        preprocessor = Preprocessor()
        chunks_ls = preprocessor.preprocess(df=annotated_data,
                                              example_size=self.example_size,
                                              more_than_2=self.more_than_2,
                                              num_annotations=self.num_annotations,
                                              random_state=self.random_state,
                                              train_path=train_path,
                                              validation_path=validation_path,
                                              test_path=test_path,
                                              train_on_test=self.train_on_test)
        if not chunks_ls:
            raise ValueError("Preprocessing produced no usable chunks. Check annotator mode and sampling settings.")

        # Persist a run metadata.json with corpus + chunk diagnostics.
        try:
            self._write_run_metadata(
                run_dir=train_path,
                stats=preprocessor.stats,
                run_name=run_name,
            )
        except (OSError, ValueError):
            raise

        print(f"{time_utils.get_time_and_elapsed_status(start_time)}: DONE!")
        
        ####################################################################
        # --- Step 3: Generate Hypotheses: Rationalizer and Synthesizer ---#
        ####################################################################
        print(f"\n{time_utils.get_time_and_elapsed_status(start_time)}: Generating disagreement explanations...")
        
        # Load prompts and guidelines
        rationales_template = self._load_rationales()
        synthesizer_template = self._load_synthesizer()
        guidelines = self._load_file(self.resolved_guidelines_path())
        ex_path = (str(self.examples_path).strip() if self.examples_path is not None else "")
        if ex_path:
            examples = self._load_file(ex_path)
            guidelines += "\n\n" + examples
        
        # Store each chunk for debugging purposes
        for i, chunk in enumerate(chunks_ls):
            for sub_df in chunk:
                sub_df.to_csv(os.path.join(raw_dir, f"chunk_{i}_data.csv"), index=False, encoding="utf-8-sig")
        
        # Main function to get the explanations for each chunk in parallel
        explanations_ls = []
        explanations_ls = thread_utils.execute_func_in_parallel_with_status(
            chunks_ls,
            lambda chunks: self.synthesize_rationales(chunks,
                                                        rationales_template,
                                                        synthesizer_template,
                                                        guidelines,
                                                        raw_dir,
                                                        rationales_openai_args,
                                                        synthesizer_openai_args
                                                        ),
            self.num_threads,
            "Getting disagreement explanations for each chunk"
            )
        
        explanations = self.aggregate_results(all_results=explanations_ls)
        if not explanations.get("explanations"):
            raise ValueError("Discovery returned no hypotheses.")
    
        print(f"{time_utils.get_time_and_elapsed_status(start_time)}: DONE!")

        # Re-write metadata.json now that LLM token counts are populated.
        try:
            r_model = getattr(rationales_openai_args, "openai_model", "") or ""
            s_model = getattr(synthesizer_openai_args, "openai_model", "") or ""
            token_summary = self._summarize_token_usage(r_model, s_model)
            self._write_run_metadata(
                run_dir=train_path,
                stats=preprocessor.stats,
                run_name=run_name,
                token_usage=token_summary,
            )
            tot = token_summary.get("total", {})
            print(
                "[DisagreementExplainer] LLM usage: "
                f"{tot.get('calls', 0)} calls · "
                f"{tot.get('prompt_tokens', 0)} prompt + "
                f"{tot.get('completion_tokens', 0)} completion tokens · "
                f"estimated cost: {tot.get('total_usd') if tot.get('total_usd') is not None else 'unknown'}"
            )
        except (OSError, ValueError):
            raise
        
        ################################################################
        # --- Step 4: Cluster and rank the explanations by frequency ---
        ################################################################
        print(f"\n{time_utils.get_time_and_elapsed_status(start_time)}: Clustering the disagreement explanations...")
        
        # Cluster explanations using embeddings (lazy import so config-only
        # consumers don't pay the sentence-transformers / torch cost).
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', revision=self.EMBEDDING_REVISION)
        clustered_explanations_by_frequency = self.cluster_explanations(explanations, model, raw_dir=raw_dir)
        print(f"{time_utils.get_time_and_elapsed_status(start_time)}: DONE!")
        
        # Save the final explanations to output_path
        if output_path is not None:
            artifacts.write_json(output_path, clustered_explanations_by_frequency)