# test_pipeline.py 

from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from src.preprocess import build_word_image_index, load_transcriptions
from src.features import build_feature_index
from src.retrieval import rank_train_for_query, load_keywords
from src.evaluation import average_precision_for_query

DATA_ROOT = Path("data")


def pick_good_query(
    val_images: Dict[str, np.ndarray],
    train_images: Dict[str, np.ndarray],
    transcriptions: Dict[str, str],
    keywords: set[str],
) -> str:
    """
    Selects a query from the validation set that:
      - has a transcription
      - is a keyword
      - has at least one matching train word with the same transcription.
    """
    train_ids_by_label: Dict[str, list[str]] = {}
    for wid in train_images.keys():
        label = transcriptions.get(wid)
        if label is None:
            continue
        train_ids_by_label.setdefault(label, []).append(wid)

    # iterate over all val IDs and search for a suitable candidate
    for qid in val_images.keys():
        label = transcriptions.get(qid)
        if label is None:
            continue
        if label not in keywords:
            continue
        if label in train_ids_by_label and len(train_ids_by_label[label]) > 0:
            # good candidate found
            return qid

    raise RuntimeError("No suitable query found. Check keywords / splits.")


def show_query_and_top_matches(
    query_id: str,
    train_ranking,
    val_images: Dict[str, np.ndarray],
    train_images: Dict[str, np.ndarray],
    transcriptions: Dict[str, str],
    top_k: int = 5,
) -> None:
    """
    Shows the query image and the top-k matches as a small gallery.
    """
    query_img = val_images[query_id]
    query_label = transcriptions.get(query_id, "UNKNOWN")

    # top-k IDs
    top = train_ranking[:top_k]

    n_cols = top_k + 1
    plt.figure(figsize=(2 * n_cols, 3))

    # Query
    plt.subplot(1, n_cols, 1)
    plt.imshow(query_img, cmap="gray")
    plt.axis("off")
    plt.title(f"Query\n{query_id}\n{query_label}")

    # Matches
    for i, (train_id, dist) in enumerate(top, start=2):
        img = train_images[train_id]
        label = transcriptions.get(train_id, "UNKNOWN")
        plt.subplot(1, n_cols, i)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"{train_id}\n{label}\n{dist:.1f}")

    plt.tight_layout()
    plt.show()


def main():
    print("=== Test Pipeline: Single Query ===")
    print("Data root:", DATA_ROOT)

    # ------------------------------
    # Step 1: Load word images
    # ------------------------------
    print("[1/4] Loading word images (train/validation)...")
    train_images = build_word_image_index(DATA_ROOT, "train")
    val_images = build_word_image_index(DATA_ROOT, "validation")

    print(f"  -> {len(train_images)} train word images")
    print(f"  -> {len(val_images)} validation word images")

    # ------------------------------
    # Step 2: Extract features
    # ------------------------------
    print("[2/4] Extracting features...")
    train_feats = build_feature_index(train_images, window_width=1, step=1)
    val_feats = build_feature_index(val_images, window_width=1, step=1)

    # ------------------------------
    # Load metadata: transcriptions & keywords
    # ------------------------------
    print("[3/4] Loading transcriptions & keywords...")
    transcriptions = load_transcriptions(DATA_ROOT)
    keywords = load_keywords(DATA_ROOT)

    # pick a suitable query
    print("  -> Searching for a meaningful query (keyword, with matching train examples)...")
    query_id = pick_good_query(val_images, train_images, transcriptions, keywords)
    query_label = transcriptions.get(query_id, "UNKNOWN")
    print(f"  -> Selected query: {query_id} (Label: {query_label})")

    # ------------------------------
    # Step 3: DTW ranking for this query
    # ------------------------------
    print("[4/4] Computing ranking (DTW) for this query...")
    query_feats = val_feats[query_id]

    # Optional: set Sakoe-Chiba band, e.g. window=50
    window = None
    train_ranking = rank_train_for_query(
        query_id=query_id,
        query_feats=query_feats,
        train_features=train_feats,
        window=window,
    )

    print(f"  -> Ranking length: {len(train_ranking)}")

    # compute AP for this query
    ap = average_precision_for_query(
        query_id=query_id,
        ranking=train_ranking,
        transcriptions=transcriptions,
        train_id_set=set(train_feats.keys()),
    )
    print(f"\nAverage Precision (AP) for this query: {ap:.4f}\n")

    # Print top-10 results textually
    print("Top-10 matches:")
    for train_id, dist in train_ranking[:10]:
        label = transcriptions.get(train_id, "UNKNOWN")
        is_relevant = (label == query_label)
        mark = "✓" if is_relevant else " "
        print(f" {mark} {train_id:10s}  label={label:20s}  dist={dist:8.2f}")

    # Optional: visualization
    try:
        print("\nShowing query & top-5 matches (window may open in background)...")
        show_query_and_top_matches(
            query_id=query_id,
            train_ranking=train_ranking,
            val_images=val_images,
            train_images=train_images,
            transcriptions=transcriptions,
            top_k=5,
        )
    except Exception as e:
        print(f"Warning: Could not display images ({e}). "
              "Ensure matplotlib is installed and a GUI is available.")


if __name__ == "__main__":
    main()
