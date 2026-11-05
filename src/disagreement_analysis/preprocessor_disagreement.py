import numpy as np
import os
import pandas as pd
import sys
import krippendorff
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from itertools import combinations

from traitlets import List


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
    """ Preprocess and validate annotation input datasets.
    """
    
    REQUIRED_COLUMNS: list[str] = []
    
    def __init__(self):
        super().__init__()
        # Diagnostic stats from the most recent preprocess() call. Populated as the
        # pipeline runs so that downstream code (e.g. disagreement_explainer) can
        # write a metadata.json without re-deriving these numbers.
        self.stats: dict = {}

    def preprocess(self, 
                   df: pd.DataFrame,
                   example_size: int,
                   more_than_2: bool = False,
                   num_annotations: int = -1,
                   random_state: int = 42,
                   train_path: str = None,
                   validation_path: str = None,
                   test_path: str = None,
                   train_on_test: bool = False) -> list[pd.DataFrame]:
        """
        Full preprocessing pipeline.
        """
        print(f"\n[Preprocessor] Loaded dataset with {len(df)} rows.")
        
        self.REQUIRED_COLUMNS = (["unique_id", "text"])
        self._check_required_columns(df)

        # Optional sampling
        if isinstance(num_annotations, int) and num_annotations > 0:
            if num_annotations > len(df):
                raise ValueError(
                    f"num_annotations={num_annotations} exceeds dataset size ({len(df)})."
                )

            print(f"[Preprocessor] Sampling {num_annotations} rows (random_state={random_state})")
            df = df.sample(n=num_annotations, random_state=random_state).reset_index(drop=True)
        
        df = df.copy()

        # ── Compute corpus-level stats on the (possibly sub-sampled) input ──
        annotator_cols_all = [c for c in df.columns if c not in {"unique_id", "text"}]
        labels_only = df[annotator_cols_all] if annotator_cols_all else pd.DataFrame()
        per_text_counts = labels_only.notna().sum(axis=1) if not labels_only.empty else pd.Series(dtype=int)
        unique_label_vals: set = set()
        for c in annotator_cols_all:
            unique_label_vals.update(df[c].dropna().astype(str).unique().tolist())
        # Disagreement using the same threshold as the train/val/test split (0.9 entropy).
        disagreement_df = self.add_observed_disagreement(
            df, many_annotators_threshold=2, entropy_threshold=0.9
        )
        n_disagreements = int(disagreement_df["observed_disagreement"].sum())
        # Track the *unique_id*s of disagreement rows so we can compute "cases lost".
        self._disagreement_ids = set(
            disagreement_df.loc[disagreement_df["observed_disagreement"] == 1, "unique_id"]
            .astype(str).tolist()
        )

        # ── Raw annotator-pair pruning + pairwise disagreement variance ──
        # For every pair (a, b) of annotators, count co-annotated items (overlap)
        # and the number of pairwise disagreements (both labeled, labels differ).
        # The IAA pre-filter drops pairs with overlap < MIN_PAIR_OVERLAP. We
        # report:
        #   - what fraction of pairs are pruned (37%-style number), and
        #   - what fraction of the pairwise-disagreement signal lives in the
        #     pruned pairs (the "variance lost" number).
        MIN_PAIR_OVERLAP = 50
        n_ann = len(annotator_cols_all)
        raw_total_pairs = n_ann * (n_ann - 1) // 2 if n_ann >= 2 else 0
        raw_pairs_kept = 0
        total_pairwise_disagreements = 0
        pruned_pairwise_disagreements = 0
        if n_ann >= 2 and not labels_only.empty:
            label_arrs = {c: labels_only[c].to_numpy() for c in annotator_cols_all}
            ann_masks = {c: labels_only[c].notna().to_numpy() for c in annotator_cols_all}
            for i in range(n_ann):
                ci = annotator_cols_all[i]
                mi = ann_masks[ci]
                vi = label_arrs[ci]
                for j in range(i + 1, n_ann):
                    cj = annotator_cols_all[j]
                    both = np.logical_and(mi, ann_masks[cj])
                    overlap = int(both.sum())
                    if overlap == 0:
                        continue
                    # pairwise disagreement count = rows where both labeled and labels differ
                    disagree = int((vi[both] != label_arrs[cj][both]).sum())
                    total_pairwise_disagreements += disagree
                    if overlap >= MIN_PAIR_OVERLAP:
                        raw_pairs_kept += 1
                    else:
                        pruned_pairwise_disagreements += disagree
        raw_pairs_pruned = raw_total_pairs - raw_pairs_kept
        pruned_pair_disagreement_pct = (
            100.0 * pruned_pairwise_disagreements / total_pairwise_disagreements
            if total_pairwise_disagreements else 0.0
        )

        self.stats = {
            "input_rows": int(len(df)),
            "num_annotators": int(n_ann),
            "num_label_types": int(len(unique_label_vals)),
            "label_values": sorted(unique_label_vals),
            "avg_annotators_per_text": float(per_text_counts.mean()) if len(per_text_counts) else 0.0,
            "total_disagreements": n_disagreements,
            "disagreement_pct": (100.0 * n_disagreements / len(df)) if len(df) else 0.0,
            # Raw annotator-pair pruning (matches IAA-overlap definition)
            "min_pair_overlap": MIN_PAIR_OVERLAP,
            "raw_annotator_pairs": int(raw_total_pairs),
            "raw_annotator_pairs_kept": int(raw_pairs_kept),
            "raw_annotator_pairs_pruned": int(raw_pairs_pruned),
            "annotator_pairs_pruned_pct": (100.0 * raw_pairs_pruned / raw_total_pairs) if raw_total_pairs else 0.0,
            # Pairwise-disagreement variance carried by pruned pairs.
            "total_pairwise_disagreements": int(total_pairwise_disagreements),
            "pruned_pairwise_disagreements": int(pruned_pairwise_disagreements),
            "pruned_pair_disagreement_pct": pruned_pair_disagreement_pct,
            # Cluster-pair selection (orthogonal: pairs of annotator clusters
            # selected via clustering before chunking). Populated below.
            "candidate_cluster_pairs": 0,
            "kept_cluster_pairs": 0,
            "pruned_cluster_pairs": 0,
            "cluster_pairs_pruned_pct": 0.0,
            # Train-only disagreement retention (post-chunking) — different
            # question (item-level coverage of the chunker), kept for tuning.
            "train_disagreements": 0,
            "disagreement_cases_kept": 0,
            "disagreement_cases_lost": 0,
            "disagreement_cases_lost_pct": 0.0,
            "holdout_disagreements": 0,
            "total_chunk_rows": 0,
            "num_chunks": 0,
        }

        # Stratified train/val/test split
        train_data = self.stratified_train_val_test_split(df, train_path, validation_path, test_path, train_on_test=train_on_test)
        # Capture train-set unique_ids so we can compute train-only disagreement loss.
        train_unique_ids = set(train_data["unique_id"].astype(str).tolist()) if "unique_id" in train_data.columns else set()
        df_list = []
        
        # if more than 2 annotators, cluster annotator labels into k groups
        if more_than_2:
            df_list = self.process_more_than_2_annotators(train_data)
            self.stats["candidate_cluster_pairs"] = int(getattr(self, "_last_candidate_pairs", len(df_list)))
        else:
            df_list.append(train_data)
            self.stats["candidate_cluster_pairs"] = 1
            
        chunks_ls = []
 
        for df in df_list:
            # Validation steps
            self._check_required_columns(df)
            annotator1, annotator2 = self._check_2_annotators(df, more_than_2=more_than_2)
            self._validate_unique_id(df)
            self._validate_text_column(df)
            labels = self.get_unique_labels(df, annotator1, annotator2, more_than_2=more_than_2)
            
            # if labels is an empty list
            if labels == []:
                continue  # skip this pair of annotators
            
            chunks = self.create_multiple_stratified_samples(df,
                                                             example_size, 
                                                             annotator1,
                                                             annotator2,
                                                             labels=labels,
                                                             random_state=random_state)
            
            if len(chunks) >= 100:
                chunks = chunks[:100]
                chunks_ls.append(chunks)
                print(f"[Preprocessor] Warning: Limiting to first 100 chunk sets for manageability.")
            elif len(chunks) < 100 and len(chunks) > 0:
                chunks_ls.append(chunks)
            elif len(chunks) == 0:
                print(f"[Preprocessor] Warning: No chunks created from the data.")
        
        # ── Finalize stats ──────────────────────────────────────────────
        kept_cluster_pairs = sum(1 for chunks in chunks_ls if chunks)
        candidate_cluster_pairs = max(self.stats.get("candidate_cluster_pairs", 0), kept_cluster_pairs)
        pruned_cluster_pairs = max(0, candidate_cluster_pairs - kept_cluster_pairs)
        total_chunk_rows = sum(int(len(c)) for chunks in chunks_ls for c in chunks)
        num_chunks = sum(len(chunks) for chunks in chunks_ls)

        # Disagreement cases retained = unique_ids appearing in ANY kept chunk that
        # were originally flagged as a disagreement row.
        kept_dis_ids: set = set()
        for chunks in chunks_ls:
            for c in chunks:
                if "unique_id" not in c.columns:
                    continue
                kept_dis_ids.update(c["unique_id"].astype(str).tolist())
        kept_dis_ids &= self._disagreement_ids
        kept_dis_count = len(kept_dis_ids)
        # Train-only denominator: disagreements should only be counted as "lost"
        # when they were available to the chunker (i.e., in the train split).
        # Held-out (val + test) disagreements are intentionally reserved for
        # evaluation and are reported separately.
        train_dis_ids = self._disagreement_ids & train_unique_ids if train_unique_ids else self._disagreement_ids
        train_dis_total = len(train_dis_ids)
        holdout_dis_total = len(self._disagreement_ids) - train_dis_total
        lost = max(0, train_dis_total - kept_dis_count)

        self.stats.update({
            "candidate_cluster_pairs": candidate_cluster_pairs,
            "kept_cluster_pairs": kept_cluster_pairs,
            "pruned_cluster_pairs": pruned_cluster_pairs,
            "cluster_pairs_pruned_pct": (100.0 * pruned_cluster_pairs / candidate_cluster_pairs) if candidate_cluster_pairs else 0.0,
            "train_disagreements": train_dis_total,
            "disagreement_cases_kept": kept_dis_count,
            "disagreement_cases_lost": lost,
            "disagreement_cases_lost_pct": (100.0 * lost / train_dis_total) if train_dis_total else 0.0,
            "holdout_disagreements": holdout_dis_total,
            "total_chunk_rows": total_chunk_rows,
            "num_chunks": num_chunks,
        })

        # write combined dataframe to csv    
        combined_df = self.combine_nested_dfs(chunks_ls)
        if train_path is not None:
            combined_df.to_csv(os.path.join(train_path, "chunks_combined.csv"), index=False)
            
        return chunks_ls

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    def _check_required_columns(self, df: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Dataset is missing required column(s): {missing}. "
                f"The dataset must contain columns: {self.REQUIRED_COLUMNS}"
            )
    def _check_2_annotators(self, df: pd.DataFrame, more_than_2: bool = False):
        annotator_cols = [c for c in df.columns if c not in {"unique_id", "text"}]
        if len(annotator_cols) != 2:
            if not more_than_2:
                print("If there are more than 2 annotators, please set more_than_2=True.")
            raise ValueError(
                f"Expected exactly 2 annotator columns, found {len(annotator_cols)}: {annotator_cols}. "
            )
            
        return annotator_cols[0], annotator_cols[1]

    def _validate_unique_id(self, df: pd.DataFrame):
        # Ensure convertible to int
        try:
            df["unique_id"] = df["unique_id"].astype(int)
        except Exception:
            raise ValueError(
                "Column 'unique_id' must contain values convertible to integers."
            )
        
        # Ensure uniqueness
        if df["unique_id"].duplicated().any():
            dupes = df[df["unique_id"].duplicated()]["unique_id"].tolist()
            raise ValueError(
                f"Duplicate unique_id values found: {dupes[:10]}...\n"
                f"Every row must have a unique 'unique_id'."
            )
            
    def _validate_text_column(self, df: pd.DataFrame):
        if df["text"].isna().any():
            idx = df[df["text"].isna()].index[:10].tolist()
            raise ValueError(
                f"'text' column contains null values at rows: {idx}. "
                f"All rows must contain text."
            )

        # Coerce to string
        df["text"] = df["text"].astype(str)

        # Ensure non-empty after strip
        empty_mask = df["text"].str.strip() == ""
        if empty_mask.any():
            idx = df[empty_mask].index[:10].tolist()
            raise ValueError(
                f"'text' column contains empty or whitespace-only strings at rows: {idx}. "
                f"Every row must have non-empty text."
            )

    def get_unique_labels(self, data: pd.DataFrame, annotator1: str, annotator2: str, more_than_2: bool = False) -> list[str]:
        """
        Extracts unique labels from two label columns, verifies they match,
        and returns the sorted list of labels.
        """

        # 1. Get unique values from annotator1 column → convert to string → sort
        annotator1_labels = sorted(data[annotator1].astype(str).unique().tolist())

        # 2. Same for annotator2 column
        annotator2_labels = sorted(data[annotator2].astype(str).unique().tolist())

        # 3. Ensure the two sets match
        if set(annotator1_labels) != set(annotator2_labels) and not more_than_2:
            raise ValueError(
                f"[ERROR] Mismatched labels between columns '{annotator1}' and '{annotator2}'.\n"
                f"  {annotator1}: {annotator1_labels}\n"
                f"  {annotator2}: {annotator2_labels}"
            )
        elif set(annotator1_labels) != set(annotator2_labels) and more_than_2:
            print(f"[Preprocessor] Warning: Mismatched labels between columns '{annotator1}' and '{annotator2}'. Skipping annotators")
            print(f"  {annotator1}: {annotator1_labels}\n"
                f"  {annotator2}: {annotator2_labels}")
            return []
            
        print(f"[Preprocessor] Found labels: {annotator1_labels}")

        # 4. Return the labels (alphabetically sorted)
        return annotator1_labels

    # ------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------
    def stratified_train_val_test_split(self,
                                        dat: pd.DataFrame, 
                                        train_path: str,
                                        validation_path: str,
                                        test_path: str,
                                        train_on_test: bool = False):
        
        dat = self.add_observed_disagreement(dat, many_annotators_threshold=2, entropy_threshold=0.9)
        
        if train_on_test:
            train_data = dat.sample(frac=1, random_state=42).reset_index(drop=True)
            validation_data = dat.sample(frac=1, random_state=42).reset_index(drop=True)
            test_data = dat.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            target_col = "observed_disagreement"
            seed = 42
            
            # 1) 80% training pool, 20% test
            train_pool, test_data = train_test_split(
                dat,
                test_size=0.2,
                random_state=seed,
                stratify=dat[target_col]
                )
            # 2) From training pool, take 25% as validation, stratified to match the train_pool distribution
            train_data, validation_data = train_test_split(
                train_pool,
                test_size=0.25,
                random_state=seed,
                stratify=train_pool[target_col]
                )

        ##### Save to CSVs
        # Create output directories if they don't exist
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(validation_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        # Store train
        train_data = train_data.drop(columns=["observed_disagreement"])
        train_data.to_csv(train_path + "/train.csv", index=False)
        # Store validation and validation_annotate
        validation_data.to_csv(validation_path + "/validation.csv", index=False)
        validation_data = validation_data.rename(columns={"text": "text_to_annotate"})
        validation_data[["unique_id", "text_to_annotate"]].to_csv(validation_path + "/validation_annotate.csv", index=False)
        # Store test and test_annotate
        test_data.to_csv(test_path + "/test.csv", index=False)
        test_data = test_data.rename(columns={"text": "text_to_annotate"})
        test_data[["unique_id", "text_to_annotate"]].to_csv(test_path + "/test_annotate.csv", index=False)
        
        return train_data

    def add_observed_disagreement(
        self,
        df: pd.DataFrame,
        many_annotators_threshold: int = 2,
        entropy_threshold: float = 0.9,
    ) -> pd.DataFrame:
        # Identify label columns
        annotator_cols = [c for c in df.columns if c not in {"unique_id", "text"}]

        def _row_observed_disagreement(row: pd.Series) -> int:
            labels = row[annotator_cols].dropna()
            n = int(labels.shape[0])

            # No/one label => no disagreement
            if n <= 1:
                return 0

            # Few annotators: any disagreement present logic
            if n <= many_annotators_threshold:
                return int(labels.nunique() > 1)

            # Many annotators: entropy-based disagreement
            counts = labels.value_counts()
            probs = counts / n

            # Shannon entropy
            entropy = -np.sum(probs * np.log(probs))

            # Normalize entropy to [0, 1]
            max_entropy = np.log(len(probs))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            return int(normalized_entropy >= entropy_threshold)

        df = df.copy()
        df["observed_disagreement"] = df.apply(
            _row_observed_disagreement, axis=1
        ).astype(int)

        return df
 


    def sample_in_context_examples(self, 
                                   data: pd.DataFrame, 
                                   sample_per_class: int, 
                                   annotator1: str, 
                                   annotator2: str, 
                                   labels: list[str], 
                                   random_state: int):
        samples = []
        for a1_label in labels:
            for a2_label in labels:
                subset = data[(data[annotator1] == a1_label) & (data[annotator2] == a2_label)]
                if len(subset) >= sample_per_class:
                    samples.append(subset.sample(sample_per_class, random_state=random_state))
                else:
                    # Handle case where there aren't enough examples
                    samples.append(subset)
    
        return pd.concat(samples)

    def create_multiple_stratified_samples(self, 
                                           data: pd.DataFrame, 
                                           example_size: int, 
                                           annotator1: str, 
                                           annotator2: str, 
                                           labels: list[str], 
                                           random_state: int):

        num_combinations = len(labels) ** 2
        sample_per_class = example_size // num_combinations
        chunks = []
        
        remaining_data = data.copy()
        
        while len(remaining_data) > 0:
            chunk = self.sample_in_context_examples(remaining_data, 
                                                    sample_per_class,
                                                    annotator1, 
                                                    annotator2, 
                                                    labels, 
                                                    random_state)
                
            remaining_data = remaining_data.drop(chunk.index)
            
            # Add 3 more disagreement examples
            disagree_pool = remaining_data[remaining_data[annotator1] != remaining_data[annotator2]]
            if len(disagree_pool) > 0:
                k_dis = min(3, len(disagree_pool))
                extra_disagree = disagree_pool.sample(k_dis, random_state=random_state)
            else:
                extra_disagree = remaining_data.iloc[0:0]
                
            if len(chunk) < example_size and len(disagree_pool) == 0:
                break  # no more data to sample

            # remove disagree extras before sampling agree extras (prevents overlap)
            remaining_data = remaining_data.drop(extra_disagree.index)

            # Add 3 agreement examples
            agree_pool = remaining_data[remaining_data[annotator1] == remaining_data[annotator2]]
            if len(agree_pool) > 0:
                k_agree = min(3, len(agree_pool))
                extra_agree = agree_pool.sample(k_agree, random_state=random_state)
            else:
                extra_agree = remaining_data.iloc[0:0]
            
            # remove agree extras before sampling agree extras (prevents overlap)
            remaining_data = remaining_data.drop(extra_agree.index)

            # append extras to chunk
            if len(extra_disagree) > 0:
                chunk = pd.concat([chunk, extra_disagree])
            if len(extra_agree) > 0:
                chunk = pd.concat([chunk, extra_agree])
                
            if len(chunk) < example_size:
                break

            chunks.append(chunk)

        return chunks
    
    def normalize_labels(self, s: pd.Series) -> pd.Series:
        # treat whitespace-only as missing too
        s = s.astype("string")
        s = s.str.strip()
        s = s.replace("", pd.NA)
        return s.where(s.isna(), s.str.lower())
    
    def _pairwise_alpha_overlap_only(self, s1: pd.Series, s2: pd.Series) -> float:
        # overlap mask: both present
        mask = s1.notna() & s2.notna()
        n_overlap = int(mask.sum())
        if n_overlap < 50:
            return np.nan  # insufficient data for agreement

        a = s1.loc[mask].astype("string").to_numpy(dtype=str)
        b = s2.loc[mask].astype("string").to_numpy(dtype=str)

        are_equal = np.array_equal(a, b)
        if are_equal:
            return 1.0  # perfect agreement

        data = np.vstack([a, b])
        return krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")


    def process_more_than_2_annotators(self, df):
        """
        Cluster N annotators by inter-annotator agreement (Krippendorff's alpha),
        using all columns except (unique_id, text) as annotator label columns.

        Assumptions:
        - `unique_id` and `text` are non-annotator columns.
        - Remaining columns correspond to annotators' labels for the same items (rows).
        - Alpha is computed pairwise on overlap only.

        - Based on the number of clusters, create all cross-cluster annotator pairs:
        - Example: for 3 clusters (C1, C2, C3):
            (C1, C2), (C1, C3), (C2, C3)

        - Each pair becomes its own DataFrame with columns:
            unique_id, text, <label1>, <label2>
            where <label1> is annotator_a's label, and <label2> is annotator_b's label.
            Rows are restricted to items where BOTH annotators have non-missing labels.

        - Returns: a list of DataFrames, each DataFrame corresponds to a cluster pair and their annotations
        """
        
        # Get annotator columns
        annotator_cols = [c for c in df.columns if c not in ("unique_id", "text")]
        if not annotator_cols:
            raise ValueError("No annotator columns found (expected at least two beyond unique_id/text).")
        n_annotators = len(annotator_cols)
        
        # Need at least 2 annotators
        if n_annotators < 2:
            raise ValueError("At least two annotator columns are required to form cluster pairs.")
        
        # Process annotator labels
        df_clean = df.copy()
        df_clean = df_clean[annotator_cols]
        df_clean[annotator_cols] = df_clean[annotator_cols].apply(self.normalize_labels)
        
        # Compute pairwise Krippendorff's alpha distance matrix       
        alpha_matrix = np.zeros((n_annotators, n_annotators))

        for i in range(n_annotators):
            alpha_matrix[i, i] = 1.0  # perfect agreement with self
            for j in range(i + 1, n_annotators):
                a = self._pairwise_alpha_overlap_only(df_clean[annotator_cols[i]], df_clean[annotator_cols[j]])
                alpha_matrix[i, j] = a
                alpha_matrix[j, i] = a

        # Convert to distance matrix
        distance_matrix = 1 - alpha_matrix
        distance_matrix[np.isnan(distance_matrix)] = 1.0  # treat unknown agreement as "far"
        np.fill_diagonal(distance_matrix, 0.0) # agreement is perfect with self
        
        # Test different numbers of clusters (up to n_annotators)
        silhouette_scores = []
        cluster_range = range(2, n_annotators) 
        
        for n_clusters in cluster_range:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Calculate silhouette score (higher is better)
            score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
            silhouette_scores.append(score)
        
        # Find optimal number of clusters
        optimal_n = list(cluster_range)[np.argmax(silhouette_scores)]
        print(f"\nOptimal number of clusters: {optimal_n}")
        
        if optimal_n > 7:
            print("Warning: Optimal number of clusters is greater than 7. Capping at 7 clusters.")
            optimal_n = 7  # cap at 7 clusters to limit number of pairs
        
        # Fit final model with optimal clusters
        clustering = AgglomerativeClustering(
            n_clusters=optimal_n,
            metric='precomputed',
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        meta = df[["unique_id", "text"]].copy()      
        labels_clean = df[annotator_cols].apply(self.normalize_labels) 
        
        annotated_instances = {  
                               i: set(labels_clean.index[labels_clean[annotator_cols[i]].notna()])  
                               for i in range(n_annotators)  
                               }  
        outputs = []  
        unique_clusters = sorted(set(cluster_labels)) 
        # Total candidate cross-cluster pairs (before any pruning for overlap).
        self._last_candidate_pairs = int(len(list(combinations(unique_clusters, 2))))
        
        def _centroid_member(members):  
            """Member with minimum average intra-cluster distance (centroid proxy)."""
            avg_dists = [] 
            for m in members:  
                dists = [distance_matrix[m, o] for o in members if o != m]
                avg_dists.append(float(np.mean(dists)) if dists else 0.0)
            return members[int(np.argmin(avg_dists))]

        for cluster_a, cluster_b in combinations(unique_clusters, 2): 
            members_a = [i for i, lab in enumerate(cluster_labels) if lab == cluster_a] 
            members_b = [i for i, lab in enumerate(cluster_labels) if lab == cluster_b] 

            if not members_a or not members_b:
                continue 

            centroid_a = _centroid_member(members_a) 
            centroid_b = _centroid_member(members_b) 

            # Find the cross-cluster pair with maximum overlap
            max_overlap = -1 
            best_pair = (None, None)
            for i in members_a:  
                for j in members_b:  
                    overlap = len(annotated_instances[i].intersection(annotated_instances[j])) 
                    if overlap >= max_overlap:
                        if overlap > max_overlap:
                            max_overlap = overlap
                            candidates = [(i, j)]
                        elif overlap == max_overlap:
                            candidates.append((i, j))
                            
            if not candidates:
                continue
            
            if len(candidates) == 1:
                best_pair = candidates[0]
            else:
                # minimize sum of distances to centroid proxies
                def tie_score(pair):
                    i, j = pair
                    return distance_matrix[i, centroid_a] + distance_matrix[j, centroid_b]
                best_pair = min(candidates, key=tie_score)

            i, j = best_pair  
            a_col = annotator_cols[i] 
            b_col = annotator_cols[j]  

            if max_overlap >= 50:
                # Build the representative pair dataframe (overlap rows only)
                mask = labels_clean[a_col].notna() & labels_clean[b_col].notna() 
                pair_df = meta.loc[mask, ["unique_id", "text"]].copy() 
                pair_df[a_col] = labels_clean.loc[mask, a_col].to_numpy() 
                pair_df[b_col] = labels_clean.loc[mask, b_col].to_numpy() 
                
                outputs.append(pair_df)  
            else:
                print(f"Skipping cluster pair ({cluster_a}, {cluster_b}) due to insufficient overlap ({max_overlap} examples).")

        return outputs

    def combine_nested_dfs(self, list_of_lists: list[list[pd.DataFrame]]) -> pd.DataFrame:
        """
        Takes a list of lists of DataFrames.
        Each DataFrame has columns: unique_id, text, <ann1>, <ann2>
        Renames the 3rd and 4th columns to 'A1' and 'A2',
        then concatenates everything row-wise into a single long DataFrame.
        """

        renamed_dfs = []

        for inner_list in list_of_lists:
            for df in inner_list:
                # Defensive copy
                df = df.copy()

                # Rename 3rd and 4th columns by position
                cols = list(df.columns)
                if len(cols) < 4:
                    raise ValueError(f"DataFrame has fewer than 4 columns: {cols}")

                cols[2] = "A1"
                cols[3] = "A2"
                df.columns = cols

                renamed_dfs.append(df)

        if not renamed_dfs:
            return pd.DataFrame()

        # Concatenate row-wise
        combined_df = pd.concat(renamed_dfs, axis=0, ignore_index=True)

        return combined_df