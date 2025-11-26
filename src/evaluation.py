from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from .preprocess import load_transcriptions


# type alias for clarity
Ranking = List[Tuple[str, float]]  # [(train_id, dist), ...]


# ---------------------------------------------------------------------------
# Precision–Recall for a single query
# ---------------------------------------------------------------------------

def precision_recall_for_query(
    query_id: str,
    ranking: Ranking,
    transcriptions: Dict[str, str],
    train_id_set: Optional[set[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes precision and recall values along the ranking for a single query.

    query_id: ID of the query word (e.g., "270-05-07").
    ranking: List [(train_id, dist), ...], sorted by distance.
    transcriptions: Dict word_id -> transcription (e.g., "y-o-u").
    train_id_set: Set of all valid train IDs. If None, all IDs in the ranking
                  are considered train words.

    Returns:
        recall:    np.ndarray of recall values per rank (Shape (K,))
        precision: np.ndarray of precision values per rank (Shape (K,))
    """
    if train_id_set is None:
        train_id_set = {wid for wid, _ in ranking}

    # label of the query word
    query_label = transcriptions.get(query_id)
    if query_label is None:
        # no label -> empty curve
        return np.array([]), np.array([])

    # relevant train IDs: all train words with the same transcription
    relevant_ids = {
        wid for wid in train_id_set
        if transcriptions.get(wid) == query_label
    }
    num_relevant = len(relevant_ids)

    if num_relevant == 0:
        # no relevant examples in the train set -> empty curve
        return np.array([]), np.array([])

    # go through the ranking and count TP/FP
    tps = 0
    fps = 0

    precision_list: List[float] = []
    recall_list: List[float] = []

    for rank, (train_id, _dist) in enumerate(ranking, start=1):
        if train_id not in train_id_set:
            # safety: ignore unknown IDs
            continue

        if train_id in relevant_ids:
            tps += 1
        else:
            fps += 1

        precision = tps / float(tps + fps)
        recall = tps / float(num_relevant)

        precision_list.append(precision)
        recall_list.append(recall)

        # optional: stop once all relevant items were found
        if tps == num_relevant:
            break

    return np.array(recall_list, dtype=np.float32), np.array(precision_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Average Precision (AP) for a single query
# ---------------------------------------------------------------------------

def average_precision_for_query(
    query_id: str,
    ranking: Ranking,
    transcriptions: Dict[str, str],
    train_id_set: Optional[set[str]] = None,
) -> float:
    """
    Computes the Average Precision (AP) for a single query.

    AP = mean of Precision@k at the ranks k
         where a relevant document is found.

    query_id: ID of the query word.
    ranking:  [(train_id, dist), ...], sorted by distance (ascending).
    transcriptions: word_id -> transcription.
    train_id_set: set of all valid train IDs.

    Returns: AP value (float). 0.0 if no relevant documents.
    """
    if train_id_set is None:
        train_id_set = {wid for wid, _ in ranking}

    query_label = transcriptions.get(query_id)
    if query_label is None:
        return 0.0

    relevant_ids = {
        wid for wid in train_id_set
        if transcriptions.get(wid) == query_label
    }
    num_relevant = len(relevant_ids)

    if num_relevant == 0:
        return 0.0

    tps = 0
    fps = 0
    ap_sum = 0.0

    for rank, (train_id, _dist) in enumerate(ranking, start=1):
        if train_id not in train_id_set:
            continue

        if train_id in relevant_ids:
            tps += 1
            precision_at_k = tps / float(tps + fps)
            ap_sum += precision_at_k

            if tps == num_relevant:
                break
        else:
            fps += 1

    ap = ap_sum / float(num_relevant) if num_relevant > 0 else 0.0
    return ap


# ---------------------------------------------------------------------------
# Global evaluation: mAP and per-query AP
# ---------------------------------------------------------------------------

def evaluate_map(
    data_root: Path,
    retrieval_results: Dict[str, Ranking],
    train_features: Dict[str, object],
) -> Tuple[float, Dict[str, float]]:
    """
    Performs a mAP evaluation (mean Average Precision) across all queries.

    data_root: path to the data/ folder, required for transcription.tsv.
    retrieval_results: Dict query_id -> Ranking [(train_id, dist), ...].
    train_features: Dict train_word_id -> feature matrix. Only needed
                    to obtain the set of train IDs.

    Returns:
        mean_ap:      mAP across all valid queries.
        per_query_ap: Dict query_id -> AP(query_id).
    """
    transcriptions = load_transcriptions(data_root)
    train_id_set = set(train_features.keys())

    per_query_ap: Dict[str, float] = {}

    for query_id, ranking in retrieval_results.items():
        ap = average_precision_for_query(
            query_id=query_id,
            ranking=ranking,
            transcriptions=transcriptions,
            train_id_set=train_id_set,
        )
        # All queries (even with AP = 0.0) are stored.
        per_query_ap[query_id] = ap

    if not per_query_ap:
        return 0.0, per_query_ap

    mean_ap = float(np.mean(list(per_query_ap.values())))
    return mean_ap, per_query_ap
