import os
import glob
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import sys
import re
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from scipy.stats import fisher_exact, chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.multitest import multipletests

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

#############################################################
# Preprocessing Functions 
#############################################################

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def _as_binary(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors="raise")
    x = x.fillna(0)
    vals = set(pd.unique(pd.Series(x).dropna()))
    if not vals.issubset({0, 1}):
        raise ValueError(f"Non-binary values found in {series.name!r}: {vals}")
    return x.astype(int).to_numpy()

def _load_explanations(explanations: str):
    """
    Reads a JSON file and returns list[str] explanations.

    Supports either:
    - top-level list[str]
    - top-level dict with key "explanations": list[str]
    """
    if not os.path.exists(explanations):
        raise FileNotFoundError(f"explanations not found: {explanations}")

    with open(explanations, "r", encoding="utf-8-sig") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        explanations_ls = obj
    elif isinstance(obj, dict) and "explanations" in obj and isinstance(obj["explanations"], list):
        explanations_ls = obj["explanations"]
    else:
        raise ValueError(
            f"Unsupported explanations JSON structure. Expected list[str] or dict with key 'explanations'. "
            f"Got: {type(obj)}"
        )

    # Validate
    for idx, x in enumerate(explanations_ls):
        if not isinstance(x, str):
            raise TypeError(f"explanations[{idx}] is not a string: {type(x)}")
        if not x.strip():
            raise ValueError(f"explanations[{idx}] is empty/whitespace")

    return explanations_ls

def _read_hypothesis_text(prompt_path: str, hypothesis_name: str) -> str:
    """
    Reads:
      prompts/{run_name}/concepts/conceptual_definitions/{hypothesis_name}/{hypothesis_name}_0.0.md
    Returns empty string if missing.
    """
    path = Path(prompt_path) / "concepts" / "conceptual_definitions" / hypothesis_name / f"{hypothesis_name}_0.0.md"
    try:
        text = path.read_text(encoding="utf-8").strip()
        match = re.split(r"\bboolean\b", text, flags=re.IGNORECASE, maxsplit=1)
        if len(match) == 2:
            text = match[1].strip()
        
        return text
    
    except FileNotFoundError:
        return ""
    except Exception as e:
        # You may prefer to raise here; returning "" keeps the pipeline running.
        return f"[ERROR reading {path}: {type(e).__name__}: {e}]"
    
def _aggregate_iter_files(eval_dir: str, mode: str) -> pd.DataFrame | None:
    """Re-aggregate per-iteration eval files for a single eval_* batch dir.

    Loads all ``eval_iter*.csv`` files (one per annotator iteration) and
    collapses them per ``unique_id`` using either ``mode`` aggregation:

      * ``"majority"`` — per-row mode of the iteration values (ties broken by
        rounding the mean upward, so ties on a binary column resolve to 1).
      * ``"max"``      — per-row max (1 if any iteration says 1).
      * ``"min"``      — per-row min (1 only if every iteration says 1; binary AND).

    Returns a wide dataframe (one row per unique_id, one column per concept)
    or ``None`` when no per-iteration files exist (caller falls back to the
    annotator's own aggregated ``eval.csv``).
    """
    iter_paths = sorted(glob.glob(os.path.join(eval_dir, "eval_iter*.csv")))
    if not iter_paths:
        return None

    dropped = {
        "input_text", "prompt_tokens", "completion_tokens", "total_tokens",
        "latency_seconds", "model_name", "annotation_timestamp", "run_id", "iteration",
    }
    iter_dfs = []
    for p in iter_paths:
        df = load_csv(p)
        if "unique_id" not in df.columns:
            raise ValueError(f"`unique_id` column missing from {p}")
        df = df.drop(columns=[c for c in df.columns if c in dropped], errors="ignore")
        iter_dfs.append(df)

    long_df = pd.concat(iter_dfs, ignore_index=True)
    long_df = long_df.loc[:, ~long_df.columns.str.contains("Reasoning")]

    # Coerce all non-key columns to numeric (NaN on parse failures) so
    # max/min comparisons don't crash on stringified labels (e.g. "0"/"1").
    for c in long_df.columns:
        if c == "unique_id":
            continue
        long_df[c] = pd.to_numeric(long_df[c], errors="coerce")

    if mode == "max":
        agg_df = long_df.groupby("unique_id", as_index=False).max(numeric_only=False)
    elif mode == "min":
        agg_df = long_df.groupby("unique_id", as_index=False).min(numeric_only=False)
    elif mode == "majority":
        # Per-column per-id mode; round-the-mean tiebreak preserves "any positive" leaning.
        def _majority(series: pd.Series):
            vals = series.dropna()
            if vals.empty:
                return np.nan
            try:
                return int(round(float(vals.mean())))
            except (TypeError, ValueError):
                m = vals.mode()
                return m.iloc[0] if not m.empty else np.nan
        agg_df = long_df.groupby("unique_id", as_index=False).agg(_majority)
    else:
        raise ValueError(f"Unsupported aggregation mode: {mode!r}")

    return agg_df


def prepare_data(
    path: str,
    eval_model: str,
    outputs_dir: str,
    held_out: bool,
    aggregation: str | None = None,
) -> pd.DataFrame:
    
    base_df = load_csv(path)

    if "unique_id" not in base_df.columns:
        raise ValueError("`unique_id` column missing from eval dataset")

    # -------------------------
    # Recognize eval_n folders
    # -------------------------
    eval_dirs = sorted(
        d for d in glob.glob(os.path.join(outputs_dir, "eval_*"))
        if os.path.isdir(d)
    )

    if not eval_dirs:
        raise ValueError(f"No eval_* directories found in {outputs_dir}")

    # -------------------------
    # Merge eval.csv files
    # -------------------------
    combined_df = base_df.copy()

    for eval_dir in eval_dirs:
        eval_csv_path = os.path.join(eval_dir, "eval.csv")
        if aggregation is not None:
            agg_df = _aggregate_iter_files(eval_dir, aggregation)
            if agg_df is not None:
                eval_df = agg_df
            else:
                # Fallback: no per-iter files (probably num_iterations=1).
                eval_df = load_csv(eval_csv_path)
        else:
            eval_df = load_csv(eval_csv_path)

        if "unique_id" not in eval_df.columns:
            raise ValueError(
                f"`unique_id` column missing in eval data for {eval_dir}"
            )

        # drop columns
        eval_df = eval_df.drop(
            columns=[
                'input_text',
                'prompt_tokens',
                'completion_tokens',
                'total_tokens',
                'latency_seconds',
                'model_name',
                'annotation_timestamp',
                'run_id',
            ],
            errors='ignore',
        )

        # Drop auxiliary per-concept suffix columns (Reasoning, consistency,
        # Confidence, EvidenceSpan). These are noise for predictiveness and
        # the LLM occasionally emits malformed keys (e.g. "<project>_Reasoning"
        # without a hypothesis number) which would collide across batches.
        aux_suffixes = ("_Reasoning", "_consistency", "_Confidence",
                        "_EvidenceSpan", "_Evidence_Span")
        eval_df = eval_df.drop(
            columns=[c for c in eval_df.columns
                     if c != "unique_id" and any(c.endswith(s) for s in aux_suffixes)],
            errors="ignore",
        )

        # If columns collide across batches it usually means either:
        #   (a) the prompted concept set overlaps between adjacent batches
        #       (e.g. eval_1 prompts ghc-real_1..11 and eval_2 prompts
        #       ghc-real_11..20 — both produce a "ghc-real_11" column), or
        #   (b) the LLM hallucinated concept names outside the prompted set.
        # In case (a), one batch typically has dense data for the shared col
        # while the other has it almost entirely NaN (the LLM only emits as
        # many booleans as it "expects"). Resolve both cases by keeping, for
        # each shared column, the version with MORE non-null values — and
        # filling any remaining NaNs from the other side using unique_id
        # alignment. This is loss-less: nothing real is dropped.
        overlapping_cols = (set(combined_df.columns) & set(eval_df.columns)) - {"unique_id"}
        if overlapping_cols:
            shared_eval = eval_df[["unique_id", *sorted(overlapping_cols)]]
            shared_combined = combined_df[["unique_id", *sorted(overlapping_cols)]]
            for col in sorted(overlapping_cols):
                left_n = shared_combined[col].notna().sum()
                right_n = shared_eval[col].notna().sum()
                # If the incoming batch has strictly more annotated cells,
                # promote it as the base; otherwise keep the existing one.
                # Either way, use the other side to fill remaining NaNs.
                merged = shared_combined[["unique_id", col]].merge(
                    shared_eval[["unique_id", col]],
                    on="unique_id", how="left", suffixes=("_keep", "_other"),
                )
                base = merged[f"{col}_keep" if left_n >= right_n else f"{col}_other"]
                fill = merged[f"{col}_other" if left_n >= right_n else f"{col}_keep"]
                combined_df = combined_df.drop(columns=[col])
                combined_df = combined_df.merge(
                    pd.DataFrame({"unique_id": merged["unique_id"], col: base.fillna(fill)}),
                    on="unique_id", how="left", validate="one_to_one",
                )
                print(
                    f"  [info] resolved overlapping column {col!r} from "
                    f"{eval_csv_path}: kept side with more data "
                    f"(existing non-null={left_n}, incoming={right_n}); "
                    f"final non-null={combined_df[col].notna().sum()}"
                )
            eval_df = eval_df.drop(columns=list(overlapping_cols), errors="ignore")

        combined_df = combined_df.merge(
            eval_df,
            on="unique_id",
            how="left",
            validate="one_to_one",
        )
    
    # Change column names to align with original significant hypotheses
    if held_out:
        outputs_dir_parent = os.path.dirname(outputs_dir)
        val_dir = os.path.join(outputs_dir_parent, "validation", eval_model)
        output_file = os.path.join(val_dir, "significant_disagreement_explanations.json")
        
        # Load significant hypotheses from JSON if held out analysis
        significant_hypotheses = _load_explanations(output_file)
        
        #combined_df = combined_df.drop(columns=["md-o3-o4_9"], errors="ignore")
        # print(combined_df.columns.tolist())
        # print(len(combined_df.columns.tolist()))
        # print("Renaming columns...")
        # print(len(significant_hypotheses))
        # print(f"Renaming columns to: {significant_hypotheses}")
        
        # Find the split point (keeps all data and ordering unchanged)
        idx = combined_df.columns.get_loc("observed_disagreement")

        # The held-out eval CSVs use generic per-batch concept names
        # (e.g. dices_1, dices_2, ...) and each hypothesis comes with auxiliary
        # suffix columns (e.g. "<name>_Reasoning", "<name>_consistency",
        # "<name>_Confidence", "<name>_EvidenceSpan"). Drop those aux columns
        # so what remains is one column per hypothesis lining up 1:1 with
        # significant_hypotheses (in batch order), then rename to the original
        # significant hypothesis names.
        left_cols = list(combined_df.columns[: idx + 1])
        right_cols = list(combined_df.columns[idx + 1:])
        aux_suffixes = ("_Reasoning", "_consistency", "_Confidence",
                        "_EvidenceSpan", "_Evidence_Span")
        right_cols_kept = [c for c in right_cols if not any(c.endswith(s) for s in aux_suffixes)]

        # Additionally, restrict to columns that actually look like
        # hypothesis identifiers (i.e. share the project prefix used in
        # significant_hypotheses and end in `_<digits>`). This prevents
        # stray columns from leaking into the ordinal rename — examples
        # observed in the wild: a residual `text` column from the input
        # data, or a malformed bare project name (`ghc-syn`, `md`)
        # emitted by the LLM as a top-level key without a hypothesis
        # number. Both would otherwise pass the aux-suffix filter and
        # corrupt the ordered rename below.
        if significant_hypotheses:
            sample = next(iter(significant_hypotheses))
            project_prefix = sample.rsplit("_", 1)[0] + "_"
            hyp_pattern = re.compile(rf"^{re.escape(project_prefix)}\d+$")
            stray = [c for c in right_cols_kept if not hyp_pattern.match(c)]
            if stray:
                print(
                    f"  [warn] dropping {len(stray)} non-hypothesis columns from "
                    f"held-out merged eval data: {stray[:8]}"
                    f"{'...' if len(stray) > 8 else ''}"
                )
                right_cols_kept = [c for c in right_cols_kept if hyp_pattern.match(c)]

        right_cols_dropped = [c for c in right_cols if c not in right_cols_kept]
        if right_cols_dropped:
            combined_df = combined_df.drop(columns=right_cols_dropped)

        # Held-out annotations may have been capped via evaluation.top_hypotheses
        # (e.g. only the top 15 of 26 significant hypotheses were annotated).
        # In that case the held-out merged eval data has fewer hypothesis
        # columns than the validation JSON lists. Since both the JSON and the
        # held-out prompt-numbering (`<run>_1`, `<run>_2`, ...) are derived
        # from the SAME F1-rank-sorted significant_df (see
        # `disagreement_predictiveness_suite` ~line 781), truncating the JSON
        # to the first N entries gives a positionally-correct rename target.
        if len(right_cols_kept) < len(significant_hypotheses):
            print(
                f"  [info] held-out: {len(right_cols_kept)} annotated hypothesis "
                f"columns < {len(significant_hypotheses)} significant from validation "
                f"— renaming to first {len(right_cols_kept)} (F1-rank order from validation)."
            )
            significant_hypotheses = significant_hypotheses[: len(right_cols_kept)]
        elif len(right_cols_kept) != len(significant_hypotheses):
            raise ValueError(
                f"Held-out rename mismatch: {len(right_cols_kept)} hypothesis columns "
                f"in merged eval data vs {len(significant_hypotheses)} significant hypotheses. "
                f"merged hypothesis cols: {right_cols_kept[:10]}{'...' if len(right_cols_kept) > 10 else ''}; "
                f"significant: {list(significant_hypotheses)[:10]}{'...' if len(significant_hypotheses) > 10 else ''}"
            )

        # Rename hypothesis columns by ORDER to the original significant names
        combined_df.columns = left_cols + list(significant_hypotheses)

    # -------------------------
    # Write output
    # -------------------------
    sub = aggregation if aggregation else ""
    outputs_dir_eff = os.path.join(outputs_dir, eval_model, sub) if sub else os.path.join(outputs_dir, eval_model)
    os.makedirs(outputs_dir_eff, exist_ok=True)
    out_path = os.path.join(outputs_dir_eff, "eval_dataset.csv")
    combined_df.to_csv(out_path, index=False)
    
    # -------------------------
    # Evaluate dataset
    # -------------------------
    df_analyze = combined_df.loc[:, ~combined_df.columns.str.contains("Reasoning")]
    
    target_col = "observed_disagreement"
    if target_col not in df_analyze.columns:
        raise ValueError(f"{target_col!r} not found in DataFrame")
    
    target_idx = df_analyze.columns.get_loc(target_col)

    # Keep target and everything to its right
    df_analyze = df_analyze.iloc[:, target_idx:]
    
    return df_analyze


##################################
# Evaluation Functions
##################################

def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0, average='macro')),
        "recall": float(recall_score(y_true, y_pred, zero_division=0, average='macro')),
        "f1": float(f1_score(y_true, y_pred, zero_division=0, average='macro')),
    }

def _contingency_2x2(h: np.ndarray, y: np.ndarray) -> np.ndarray:
    n00 = int(((h == 0) & (y == 0)).sum())
    n01 = int(((h == 0) & (y == 1)).sum())
    n10 = int(((h == 1) & (y == 0)).sum())
    n11 = int(((h == 1) & (y == 1)).sum())
    return np.array([[n00, n01], [n10, n11]], dtype=int)

def _odds_ratio_and_ci(
    table_2x2,
    alpha: float = 0.05,
    ci_method: str = "wald",
    add_half_if_zeros: bool = True,
):
    """
    Compute odds ratio and CI from a 2x2 table.

    Args:
        table_2x2: array-like shape (2,2) with counts [[a,b],[c,d]]
        alpha: 1 - confidence level (0.05 -> 95% CI)
        ci_method: "wald" or "exact" (exact may be slower/less stable depending on table)
        add_half_if_zeros: if any cell is 0, add 0.5 to all cells for finite OR/CI

    Returns:
        (or_value, ci_low, ci_high, used_correction_bool)
    """
    table = np.asarray(table_2x2, dtype=float)
    if table.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 table, got shape {table.shape}")

    # Haldane–Anscombe correction (+0.5 to all cells)
    used_correction = False
    if add_half_if_zeros and np.any(table == 0):
        table = table + 0.5
        used_correction = True

    t22 = Table2x2(table)
    or_value = float(t22.oddsratio)
    ci_low, ci_high = t22.oddsratio_confint(alpha=alpha, method=ci_method)
    return or_value, float(ci_low), float(ci_high), used_correction

def _independence_test_2x2(table_2x2, alpha: float = 0.05):
    """
    Returns p-value from Fisher or Chi-square, plus odds ratio + CI (from the table).
    """
    if table_2x2.shape != (2, 2):
        raise ValueError("Expected a 2x2 table.")
    
    row_sums = table_2x2.sum(axis=1)
    col_sums = table_2x2.sum(axis=0)

    if np.any(row_sums == 0) or np.any(col_sums == 0):
        return {
            "test": "degenerate",
            "p_value": np.nan,
            "odds_ratio": np.nan,
             "or_ci_low": np.nan,
            "or_ci_high": np.nan,
            "or_ci_alpha": np.nan,
            "or_used_haldane_anscombe": False
        }

    _, _, _, expected = chi2_contingency(table_2x2, correction=False)
    min_expected = float(np.min(expected))
    use_fisher = bool(min_expected < 5)
    
    if use_fisher:
        ci_method = "exact"
    else:
        ci_method = "wald"
    
    # OR + CI computed consistently for both branches
    odds_ratio, ci_low, ci_high, used_correction = _odds_ratio_and_ci(
        table_2x2,
        alpha=alpha,
        ci_method=ci_method,     
        add_half_if_zeros=True,
    )

    if use_fisher:
        _, p = fisher_exact(table_2x2, alternative="two-sided")
        return {
            "test": "fisher_exact",
            "p_value": float(p),
            "odds_ratio": odds_ratio,
            "or_ci_low": ci_low,
            "or_ci_high": ci_high,
            "or_ci_alpha": float(alpha),
            "or_used_haldane_anscombe": bool(used_correction),
        }
    else:
        chi2, p, dof, expected = chi2_contingency(table_2x2, correction=False)
        return {
            "test": "chi2",
            "p_value": float(p),
            "chi2": float(chi2),
            "dof": int(dof),
            "min_expected": float(np.min(expected)),
            "odds_ratio": odds_ratio,
            "or_ci_low": ci_low,
            "or_ci_high": ci_high,
            "or_ci_alpha": float(alpha),
            "or_used_haldane_anscombe": bool(used_correction),
        }

def add_multiple_testing_corrections(
    df,
    p_col: str = "p_value_raw",
    alpha: float = 0.05,
):
    """
    Adds Bonferroni, Holm–Bonferroni (FWER), and Benjamini–Hochberg (FDR)
    adjusted p-values to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw p-values.
    p_col : str
        Column name for raw p-values.
    alpha : float
        Family-wise or FDR control level (used only for rejection flags).

    Returns
    -------
    pd.DataFrame
        Same dataframe with new adjusted p-value columns.
    """
    pvals = df[p_col].to_numpy()

    # Bonferroni (FWER)
    _, p_bonf, _, _ = multipletests(pvals, alpha=alpha, method="bonferroni")

    # Holm–Bonferroni (FWER, step-down)
    _, p_holm, _, _ = multipletests(pvals, alpha=alpha, method="holm")

    # Benjamini–Hochberg (FDR)
    _, p_fdr_bh, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")

    df = df.copy()
    df["p_value_bonferroni"] = p_bonf
    df["p_value_holm"] = p_holm
    df["p_value_fdr_bh"] = p_fdr_bh
    
    df["is_significant_raw"] = df[p_col] <= alpha
    df["is_significant_any_corrected"] = (
        (p_bonf <= alpha)
        | (p_holm <= alpha)
        | (p_fdr_bh <= alpha)
        )

    return df

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

def _fit_logit_train_test(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float
) -> dict:
    
    # Handle degenerate case
    if X_train.shape[1] == 0:
        y_pred = np.zeros_like(y_test)
        return {
            "ROC–AUC": float("nan"),
            "n_features": 0,
            "y_pred": y_pred
        }

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000
    )

    # Train
    model.fit(X_train.to_numpy(), y_train)

    # Predict on test
    proba = model.predict_proba(X_test.to_numpy())[:, 1]
    y_pred = (proba >= threshold).astype(int)
    roc_val = _safe_auc(y_test, proba)

    return {
        "ROC–AUC": roc_val,
        "n_features": int(X_train.shape[1]),
        "y_pred": y_pred
    }
    
def _bootstrap_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 2000,
    ci_level=0.95,
    random_state: int = 42,
    average: str = "macro",  # keep consistent with your _binary_metrics
) -> dict:
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true and y_pred must have same length. Got {len(y_true)} vs {len(y_pred)}")

    # 1. Point estimate
    f1_hat = float(f1_score(y_true, y_pred, zero_division=0, average=average))
    
    # 2. Bootstrapping
    bootstrapped_scores = []
    n_samples = len(y_true)
    rng = np.random.default_rng(random_state) # Modern way to handle random states

    for _ in range(n_iterations):
        # Generate random indices with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        
        resample_true = y_true[indices]
        resample_pred = y_pred[indices]
        
        score = f1_score(resample_true, resample_pred, zero_division=0, average=average)
        bootstrapped_scores.append(score)
    
    # 3. Calculate Percentiles
    lower_bound = (1 - ci_level) / 2
    upper_bound = 1 - lower_bound
    
    conf_interval = np.percentile(bootstrapped_scores, [lower_bound * 100, upper_bound * 100])
    
    ci_low = float(conf_interval[0])
    ci_high = float(conf_interval[1])

    return {
        "f1": f1_hat,
        "f1_ci_low": ci_low,
        "f1_ci_high": ci_high
    }

############################################################
# Primary evaluation analysis
############################################################
def disagreement_predictiveness_suite(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "observed_disagreement",
    top_ks=(20, 10, 5, -1, -2), # 20, 10, 5, -1, -2
    alpha: float = 0.05,
    outputs_dir: str | os.PathLike = ".",
    prompt_path: str | os.PathLike = "",
    held_out: bool = False,
    eval_model: str = "",
    output_subdir: str = "",
) -> dict:
    
    # ----------------------------
    # 1) Set directories
    # ----------------------------
    base_outputs_dir = os.path.abspath(str(outputs_dir))
    outputs_dir = os.path.abspath(os.path.join(outputs_dir, eval_model, output_subdir)) \
        if output_subdir else os.path.abspath(os.path.join(outputs_dir, eval_model))
    out_dir = Path(outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ----------------------------
    # 2) Preliminary data processing
    # ----------------------------
    if held_out:
        df = df_test.copy()
    else:
        df = df_val.copy()

    # Features = every column other than the target
    candidate_cols = [c for c in df.columns if c != target_col]

    hypothesis_cols = [
        c for c in candidate_cols
        if set(pd.unique(df[c].dropna())).issubset({0, 1})
    ]

    if not hypothesis_cols:
        raise ValueError("No hypothesis columns found after excluding the target (and applying binary_only filter).")

    # ----------------------------
    # 3) Hypothesis predictiveness + significance
    # ----------------------------
    hypo_rows = [] 
    
    # Create target as binary numpy array
    y = _as_binary(df[target_col])
    
    for col in hypothesis_cols:
        # Create hypothesis feature as binary numpy array
        h = _as_binary(df[col])
        # Calculate classification metrics
        metrics = _binary_metrics(y, h)
        # Compute F1 CI via bootstrap
        f1_ci = _bootstrap_f1_ci(
            y_true=y,
            y_pred=h,
            n_iterations=2000,
            ci_level=0.95,
            random_state=42,
            average="macro",
            )
        metrics.update({
            "f1_ci_low": f1_ci["f1_ci_low"],
            "f1_ci_high": f1_ci["f1_ci_high"],
        })
        # Independence test
        table = _contingency_2x2(h, y)
        test = _independence_test_2x2(table)
        # Add results to dictionary
        hypo_rows.append(
                {
                "hypothesis_name": col,
                "hypothesis": _read_hypothesis_text(prompt_path, col),
                **metrics,
                "n00": int(table[0, 0]),
                "n01": int(table[0, 1]),
                "n10": int(table[1, 0]),
                "n11": int(table[1, 1]),
                "test": test.get("test"),
                "p_value_raw": float(test.get("p_value")),
                "odds_ratio": float(test.get("odds_ratio")),
                "or_ci_low": float(test.get("or_ci_low")),
                "or_ci_high": float(test.get("or_ci_high")),
                "or_ci_alpha": float(test.get("or_ci_alpha", 0.05)),
                "or_used_haldane_anscombe": bool(test.get("or_used_haldane_anscombe"))
            }
        )

    K = len(hypothesis_cols)
    hypo_df = pd.DataFrame(hypo_rows)    
    
    # Add multiple testing corrections
    hypo_df = add_multiple_testing_corrections(
        hypo_df,
        p_col="p_value_raw",
        alpha=0.1
        )
    
    # create new variable "significant" based on raw p-value (1 if significant, 0 otherwise)
    hypo_df["significant"] = (hypo_df["p_value_raw"] <= alpha).astype(int)
    hypo_df["significant_corrected"] = (hypo_df["p_value_holm"] <= 0.1).astype(int)
    
    # Sort the values
    hypo_df = (
            hypo_df
            .sort_values(
                by=["significant", "f1", "precision", "recall"],
                ascending=[False, False, False, False],
            )
            .reset_index(drop=True)
        )

    # ranked list of hypothesis names for top-k usage
    ranked_hypotheses = hypo_df["hypothesis_name"].tolist()
    
    # only significant hypotheses
    significant_df = hypo_df[hypo_df["significant"] == 1].reset_index(drop=True)
    significant_hypotheses = significant_df["hypothesis_name"].tolist()
    significant_hypo_w_name = significant_df[["hypothesis_name", "hypothesis"]]
    # only significant after correction
    significant_corrected_df = hypo_df[hypo_df["significant_corrected"] == 1].reset_index(drop=True)    
    significant_hypotheses_corrected = significant_corrected_df["hypothesis_name"].tolist()
    
    if not held_out:
        # Store hypothesis-level results to CSV
        output_csv = os.path.join(outputs_dir, "significant_disagreement_explanations.csv")
        significant_hypo_w_name.to_csv(output_csv, index=False)
        # Store significant hypotheses to JSON
        output_file = os.path.join(outputs_dir, "significant_disagreement_explanations.json")
        with open(output_file, "w", encoding="utf-8-sig") as file:
            json.dump(significant_hypotheses, file, indent=4, ensure_ascii=False)
        # store significant corrected hypotheses to JSON
        corrected_output_path = str(output_file).replace(".json", "_corrected.json")
        corrected_output_path = Path(corrected_output_path)
        corrected_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(corrected_output_path, "w", encoding="utf-8-sig") as file:
            json.dump(significant_hypotheses_corrected, file, indent=4, ensure_ascii=False)
    else:
        # `base_outputs_dir` is the un-nested held-out dir (e.g. <run>/heldout)
        # — `outputs_dir` may have been nested under <eval_model>/<output_subdir>.
        run_dir = os.path.dirname(base_outputs_dir)
        val_dir = os.path.join(run_dir, "validation", eval_model)
        output_file = os.path.join(val_dir, "significant_disagreement_explanations.json")
        # Load significant hypotheses from JSON if held out analysis
        significant_hypotheses = _load_explanations(output_file)
        # load significant corrected hypotheses from JSON
        corrected_output_path = str(output_file).replace(".json", "_corrected.json")
        significant_hypotheses_corrected = _load_explanations(corrected_output_path)

        # Held-out annotations may have been capped via evaluation.top_hypotheses
        # (e.g. only the top 15 of N significant hypotheses were annotated). In
        # that case the held-out CSVs only contain those top-N columns, so we
        # must restrict significant_hypotheses(_corrected) to what's actually
        # available in df_test before any downstream slicing. Order is preserved
        # from the JSON (which is in F1-rank order from the validation pass).
        annotated_hypos = set(df_test.columns)
        full_n = len(significant_hypotheses)
        significant_hypotheses = [h for h in significant_hypotheses if h in annotated_hypos]
        significant_hypotheses_corrected = [
            h for h in significant_hypotheses_corrected if h in annotated_hypos
        ]
        if len(significant_hypotheses) < full_n:
            print(
                f"  [info] held-out: restricted significant_hypotheses to "
                f"{len(significant_hypotheses)}/{full_n} actually-annotated columns "
                f"(top_hypotheses cap from validation eval config)."
            )

    # ----------------------------
    # 4) Overall predictiveness via multivariate logit
    # ----------------------------
    
    # Create train/test splits
    if held_out:
        # Use validation as train, test as test
        df_train = df_val.copy()
        y_train = _as_binary(df_train[target_col])
        df_test = df_test.copy()
        y_test = _as_binary(df_test[target_col])
    else:
        # Use validation as both train and test
        df_train = df_val.copy()
        y_train = _as_binary(df_train[target_col])
        df_test = df_val.copy()
        y_test = _as_binary(df_test[target_col])
    
    # Train logit model at different thresholds
    logit_rows = []
    thresholds = [0.25, 0.5, 0.75]
    
    # Prepare full feature matrix (only for validation set)
    X_all = (
        df_train[hypothesis_cols]
        .apply(pd.to_numeric, errors="raise")
        .fillna(0)
        .astype(int)
    )
    
    for threshold in thresholds:
        # 1) Train logistic regression model using all hypotheses
        logit_all = _fit_logit_train_test(
            X_train=X_all,
            y_train=y_train,
            X_test=X_all,
            y_test=y_train,
            threshold=threshold
        )
        # Get predictions
        yhat = logit_all["y_pred"] 
        # Calculate classification metrics       
        metrics = _binary_metrics(y_train, yhat)
        # Compute F1 CI via bootstrap
        f1_ci = _bootstrap_f1_ci(
            y_true=y_train,
            y_pred=yhat,
            n_iterations=2000,
            ci_level=0.95,
            random_state=42,
            average="macro",
            )
        metrics.update({
            "f1_ci_low": f1_ci["f1_ci_low"],
            "f1_ci_high": f1_ci["f1_ci_high"],
        })
        
        # Independence test
        table = _contingency_2x2(yhat, y_train)
        test = _independence_test_2x2(table)
        # Add results to dictionary
        logit_rows.append(
            {
                "features_set": "all (validation set only)",
                "k_effective": len(hypothesis_cols),
                "hypothesis_names_used": hypothesis_cols,
                "ROC–AUC": logit_all["ROC–AUC"],
                "n_features": logit_all["n_features"],
                "threshold": threshold,
                **metrics,
                "n00": int(table[0, 0]),
                "n01": int(table[0, 1]),
                "n10": int(table[1, 0]),
                "n11": int(table[1, 1]),
                "test": test.get("test"),
                "p_value_raw": float(test.get("p_value")),
                "odds_ratio": float(test.get("odds_ratio")),
                "or_ci_low": float(test.get("or_ci_low")),
                "or_ci_high": float(test.get("or_ci_high")),
                "or_ci_alpha": float(test.get("or_ci_alpha", 0.05)),
                "or_used_haldane_anscombe": bool(test.get("or_used_haldane_anscombe"))
            }
        )

        # 2) Train logistic regression on top-k hypotheses
        seen_kbuckets: set[tuple[str, int]] = set()
        for k in top_ks:
            if k == -1:
                cols_k = list(significant_hypotheses)
                k_eff = len(cols_k)
                features_set = "all_significant"
                bucket = ("all_significant", k_eff)
            elif k == -2:
                cols_k = list(significant_hypotheses_corrected)
                k_eff = len(cols_k)
                features_set = "all_significant_corrected"
                bucket = ("all_significant_corrected", k_eff)
            elif int(k) > 0:
                # Positive k. For held-out, slice the (already-truncated)
                # significant list. For validation, fall back to the full
                # ranked list so unfiltered top-k baselines stay available.
                if held_out:
                    pool = significant_hypotheses
                else:
                    pool = ranked_hypotheses
                k_eff = min(int(k), len(pool))
                if k_eff <= 0:
                    continue
                cols_k = list(pool[:k_eff])
                features_set = f"top_{k_eff}"
                bucket = (features_set, k_eff)
            else:
                continue

            # Dedup: when several requested k's collapse to the same effective
            # k (e.g. k=12 and k=15 both become top_10 because only 10 sig
            # hypotheses exist), only the first appearance is computed. Keeps
            # the CSV from carrying duplicate rows for the same feature set.
            if bucket in seen_kbuckets:
                continue
            seen_kbuckets.add(bucket)

            # If the chosen feature set is empty, skip rather than crash on
            # a zero-column logit fit.
            if not cols_k:
                continue

            X_k = (df_train[cols_k]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .astype(int))

            if held_out:
                X_k_test = (df_test[cols_k]
                    .apply(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                    .astype(int))
            else:
                X_k_test = X_k.copy()
            
            logit_k = _fit_logit_train_test(
                X_train=X_k,
                y_train=y_train,
                X_test=X_k_test,
                y_test=y_test,
                threshold=threshold
            )
            yhat = logit_k["y_pred"]        
            metrics = _binary_metrics(y_test, yhat)
            
            f1_ci = _bootstrap_f1_ci(
                y_true=y_test,
                y_pred=yhat,
                n_iterations=2000,
                ci_level=0.95,
                random_state=42,
                average="macro",
            )

            metrics.update({
                "f1_ci_low": f1_ci["f1_ci_low"],
                "f1_ci_high": f1_ci["f1_ci_high"],
            })
            
            table = _contingency_2x2(yhat, y_test)
            test = _independence_test_2x2(table)
            
            logit_rows.append(
                {
                    "features_set": features_set,
                    "k_effective": k_eff,
                    "hypothesis_names_used": cols_k,
                    "ROC–AUC": logit_k["ROC–AUC"],
                    "n_features": logit_k["n_features"],
                    "threshold": threshold,
                    **metrics,
                    "n00": int(table[0, 0]),
                    "n01": int(table[0, 1]),
                    "n10": int(table[1, 0]),
                    "n11": int(table[1, 1]),
                    "test": test.get("test"),
                    "p_value_raw": float(test.get("p_value")),
                    "odds_ratio": float(test.get("odds_ratio")),
                    "or_ci_low": float(test.get("or_ci_low")),
                    "or_ci_high": float(test.get("or_ci_high")),
                    "or_ci_alpha": float(test.get("or_ci_alpha", 0.05)),
                    "or_used_haldane_anscombe": bool(test.get("or_used_haldane_anscombe"))
                }
            )

    logit_df = pd.DataFrame(logit_rows)
    
    logit_df = add_multiple_testing_corrections(
        logit_df,
        p_col="p_value_raw",
        alpha=0.1
        )
    
    logit_df["significant"] = (logit_df["p_value_raw"] <= alpha).astype(int)
    
    logit_df = (
            logit_df
            .sort_values(
                by=["significant", "f1", "precision", "recall"],
                ascending=[False, False, False, False],
            )
            .reset_index(drop=True)
        )

    # ----------------------------
    # 5) Save CSVs for each printed table
    # ----------------------------
    hypo_path = os.path.join(outputs_dir, "hypo_predictiveness.csv")
    logit_path = os.path.join(outputs_dir, "overall_predictiveness_logit.csv")

    hypo_df.to_csv(hypo_path, index=False)
    logit_df.to_csv(logit_path, index=False)

    return {
        "hypothesis_cols": hypothesis_cols,
        "hypotheses_predictiveness": hypo_df,
        "overall_predictiveness_logit": logit_df,
        "csv_paths": {
            "hypotheses_predictiveness": str(hypo_path),
            "overall_predictiveness_logit": str(logit_path),
        },
    }

############################################################
# Main function
############################################################

def main(
    held_out: bool,
    outputs_dir: str,
    run_name: str,
    eval_model: str,
    prompt_path: str | None = None,
    top_ks: tuple | list | None = None,
) -> None:

    # Add project root so shared modules can be imported
    _root_added = add_project_root()

    # Remove the temporary path to avoid polluting the import namespace
    if _root_added:
        sys.path.remove(_root_added)

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    if prompt_path:
        prompt_path = os.path.abspath(prompt_path)
    else:
        prompt_path = os.path.join(BASE_DIR, "prompts", run_name)
    outputs_dir = os.path.abspath(outputs_dir)
    from common.src.utils import artifacts
    for folder in ([outputs_dir, os.path.join(os.path.dirname(outputs_dir), "validation")]
                   if held_out else [outputs_dir]):
        batches = sorted(Path(folder).glob("eval_*/eval.csv"))
        if not batches:
            raise ValueError("Evaluation requires complete, manifested annotation batches.")
        for batch in batches:
            status = Path(str(batch) + ".status.json")
            manifest = artifacts.manifest_path(batch)
            if not status.is_file() or not manifest.is_file():
                raise ValueError("Evaluation input lacks completion provenance.")
            record = json.loads(status.read_text())
            if record.get("status") != "complete":
                raise ValueError("Evaluation cannot consume incomplete annotation batches.")
            artifacts.reusable(batch, record["identity"])
        for status in Path(folder).glob("eval_*/*.status.json"):
            if json.loads(status.read_text()).get("status") != "complete":
                raise ValueError("An annotation batch remains incomplete.")

    # -------------------------
    # Load eval data
    # -------------------------
    if held_out:
        # validation data
        outputs_dir_parent = os.path.dirname(outputs_dir)
        val_dir = os.path.join(outputs_dir_parent, "validation")
        val_path = os.path.join(val_dir, "validation.csv")
        df_val = prepare_data(val_path, eval_model, val_dir, False)
        # test data
        test_path = os.path.join(outputs_dir, "test.csv")
        df_test = prepare_data(test_path, eval_model, outputs_dir, held_out)
    else:
        val_path = os.path.join(outputs_dir, "validation.csv")
        df_val = prepare_data(val_path, eval_model, outputs_dir, held_out)
        df_test = pd.DataFrame()    
    
    # ----------------------------------------------------------------
    # Sanity check: a single-class observed_disagreement column makes
    # logistic regression impossible. This usually points to an upstream
    # data preprocessing issue (e.g. annotator label columns missing from
    # the processed file, so add_observed_disagreement() defaulted every
    # row to 0). Fail fast with a clear, *non-transient* message so the
    # outer retry loop in run_all_v*.sh doesn't waste attempts on it.
    # ----------------------------------------------------------------
    if "observed_disagreement" in df_val.columns:
        n_classes_val = df_val["observed_disagreement"].dropna().nunique()
        if n_classes_val < 2:
            raise RuntimeError(
                "FATAL (non-transient): validation observed_disagreement has only "
                f"{n_classes_val} class(es) — cannot fit a predictiveness model. "
                f"This typically means the upstream preprocessing "
                f"(preprocessor_disagreement.add_observed_disagreement) saw no "
                f"per-annotator label columns in the processed data file. "
                f"Inspect data/<project>/ "
                f"to confirm annotator columns are present."
            )

    # ----------------------------------------------------------------
    # Predictiveness suite runs:
    #   - Validation: single "max" pass (any iteration positive => positive),
    #     written to <outputs_dir>/<eval_model>/  (no subdir).
    #   - Held-out: "max" pass (default path) PLUS "min" pass
    #     (all iterations must agree positive; binary AND), written to
    #     <outputs_dir>/<eval_model>/min/.
    # If no per-iteration files exist (num_iterations=1), fall back to the
    # already-loaded df_val/df_test (which use the annotator's own aggregation).
    # ----------------------------------------------------------------
    def _has_iter_files(d: str) -> bool:
        return any(glob.glob(os.path.join(sub, "eval_iter*.csv"))
                   for sub in glob.glob(os.path.join(d, "eval_*")) if os.path.isdir(sub))

    runs = []
    if held_out:
        if _has_iter_files(outputs_dir):
            df_val_max = prepare_data(val_path, eval_model, val_dir, False, aggregation="max")
            df_test_max = prepare_data(test_path, eval_model, outputs_dir, held_out, aggregation="max")
            df_test_min = prepare_data(test_path, eval_model, outputs_dir, held_out, aggregation="min")
            runs.append(("", df_val_max, df_test_max))
            runs.append(("min", df_val_max, df_test_min))
        else:
            print("  [skip] No per-iteration files for held-out — using annotator-default aggregation")
            runs.append(("", df_val, df_test))
    else:
        if _has_iter_files(outputs_dir):
            df_val_max = prepare_data(val_path, eval_model, outputs_dir, held_out, aggregation="max")
            runs.append(("", df_val_max, pd.DataFrame()))
        else:
            print("  [skip] No per-iteration files — using annotator-default aggregation")
            runs.append(("", df_val, pd.DataFrame()))

    last_results = None
    # Default top_ks (back-compat for callers that don't pass one).
    effective_top_ks = tuple(top_ks) if top_ks else (20, 10, 5, -1, -2)
    for subdir, df_val_run, df_test_run in runs:
        label = subdir or "max"
        print(f"\n=== Running predictiveness suite [{label}]  top_ks={list(effective_top_ks)} ===")
        last_results = disagreement_predictiveness_suite(
            df_val=df_val_run,
            df_test=df_test_run,
            target_col="observed_disagreement",
            top_ks=effective_top_ks,
            alpha=0.1,
            outputs_dir=outputs_dir,
            prompt_path=prompt_path,
            held_out=held_out,
            eval_model=eval_model,
            output_subdir=subdir,
        )
    results = last_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--held_out",
        action="store_true",
        help="Indicates whether to use held-out set for evaluation (true/false)",
    )
    parser.add_argument(
        "--outputs_dir",
        required=True,
        help="Base outputs directory containing eval files and eval_* folders",
    )
    parser.add_argument(
        "--run_name",
        required=True,
        help="Run name used to read hypothesis texts from prompts folder",
    )
    parser.add_argument(
        "--eval_model",
        required=True,
        help="Evaluation model name (used to find eval_model folder)",
    )
    parser.add_argument(
        "--prompt_path",
        required=False,
        default=None,
        help="Absolute path to versioned prompts dir for this disagreement run.",
    )
    args = parser.parse_args()

    main(args.held_out, args.outputs_dir, args.run_name, args.eval_model, args.prompt_path)