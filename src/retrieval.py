from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np

from .dtw import dtw_distance
from .preprocess import load_transcriptions


# ---------------------------------------------------------------------------
# Load keywords
# ---------------------------------------------------------------------------

def load_keywords(data_root: Path) -> Set[str]:
    """
    Reads keywords.tsv and returns a set of keyword strings,
    e.g. {"y-o-u", "D-e-c-e-m-b-e-r", ...}.

    Assumption: each line contains at least the keyword as the first field.
                Additional columns (if present) are ignored.
    """
    keywords_path = data_root / "keywords.tsv"
    keywords: Set[str] = set()

    with keywords_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            keyword = parts[0]
            keywords.add(keyword)

    return keywords


# ---------------------------------------------------------------------------
# Ranking for a single query
# ---------------------------------------------------------------------------

def rank_train_for_query(
    query_id: str,
    query_feats: np.ndarray,
    train_features: Dict[str, np.ndarray],
    window: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    For a query word (validation), compute the DTW distance to
    all train words and return a ranking.

    query_id: ID of the query word (e.g., "270-05-07"), only useful for logging.
    query_feats: feature matrix (Tq, D) of the query word.
    train_features: dict train_word_id -> feature matrix (Tk, D).
    window: Sakoe-Chiba band (maximum deviation from the diagonal path).
            None = full DTW (no restriction).

    Returns: list of (train_word_id, dist), sorted by dist (ascending).
    """
    distances: List[Tuple[str, float]] = []

    for train_id, train_feats in train_features.items():
        # Quick length rejection - for speedup
        len_q = query_feats.shape[0]
        len_t = train_feats.shape[0]

        ratio = len_q / max(1, len_t)
        if ratio < 0.5 or ratio > 2.0:
            # skip extremely dissimilar lengths (not the same word)
            distances.append((train_id, float("inf")))
            continue

        d = dtw_distance(query_feats, train_feats, window=window)
        distances.append((train_id, d))

    # sort by distance
    distances.sort(key=lambda x: x[1])
    return distances


# ---------------------------------------------------------------------------
# Retrieval loop over all queries
# ---------------------------------------------------------------------------

def run_retrieval_for_all_queries(
    data_root: Path,
    train_features: Dict[str, np.ndarray],
    val_features: Dict[str, np.ndarray],
    use_only_keywords: bool = True,
    window: Optional[int] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Performs keyword spotting (retrieval) for all query words in the validation set.

    data_root: path to the data/ folder.
    train_features: dict train_word_id -> feature matrix (Tk, D).
    val_features: dict val_word_id   -> feature matrix (Tq, D).
    use_only_keywords:
        - True: only consider queries whose transcription appears in keywords.tsv.
        - False: use all val_features as queries.
    window: Sakoe-Chiba band for DTW. None = full DTW.

    Returns:
        dict query_id -> ranking list [(train_id, dist), ...],
        sorted by dist (ascending).
    """
    # load transcriptions: word_id -> "y-o-u"
    transcriptions = load_transcriptions(data_root)

    # load keywords (if desired)
    if use_only_keywords:
        keywords = load_keywords(data_root)
    else:
        keywords = None  # type: Optional[Set[str]]

    retrieval_results: Dict[str, List[Tuple[str, float]]] = {}

    # loop over all validation words with features
    for query_id, query_feats in val_features.items():
        # get transcription of the query word (if available)
        transcription = transcriptions.get(query_id)

        if transcription is None:
            # no transcription -> skip
            continue

        # optional: only use keywords as queries
        if keywords is not None and transcription not in keywords:
            continue

        # compute ranking vs. all train words
        ranking = rank_train_for_query(
            query_id=query_id,
            query_feats=query_feats,
            train_features=train_features,
            window=window,
        )

        retrieval_results[query_id] = ranking

    return retrieval_results
