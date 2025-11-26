from pathlib import Path

from src.preprocess import build_word_image_index
from src.features import build_feature_index
from src.retrieval import run_retrieval_for_all_queries
from src.evaluation import evaluate_map

DATA_ROOT = Path("data")

print("Step 1: Loading word images...")
train_images = build_word_image_index(DATA_ROOT, "train")
val_images   = build_word_image_index(DATA_ROOT, "validation")

print("Step 2: Extracting features...")
train_feats = build_feature_index(train_images, window_width=3, step=1)
val_feats   = build_feature_index(val_images,   window_width=3, step=1)

print("Step 3: Retrieval (DTW) for all queries...")
retrieval_results = run_retrieval_for_all_queries(
    DATA_ROOT,
    train_features=train_feats,
    val_features=val_feats,
    use_only_keywords=True,  # only keywords as queries
    window=None,             # or e.g. window=50 for Sakoe-Chiba band
)

print(f"Number of queries with ranking: {len(retrieval_results)}")

# Example: show first query
some_query_id = next(iter(retrieval_results.keys()))
print("Example query:", some_query_id)
for train_id, dist in retrieval_results[some_query_id][:5]:
    print("  ", train_id, dist)


print("Step 4: Evaluation (mAP)...")
mean_ap, per_query_ap = evaluate_map(
    data_root=DATA_ROOT,
    retrieval_results=retrieval_results,
    train_features=train_feats,
)

print(f"Mean Average Precision (mAP): {mean_ap:.4f}")

# Optional: print a few examples
for i, (qid, ap) in enumerate(per_query_ap.items()):
    if i >= 5:
        break
    print(f"Query {qid}: AP = {ap:.4f}")
