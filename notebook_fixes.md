# Notebook Fixes 
## 1. Issues Identified

### 1.1 Train/Test Split Inconsistency

The notebook originally used two split strategies:

- `random_split`
- `split_by_ratio`

but mixed them during evaluation:

- model trained on `train_data_ratio`
- evaluated on `test_data`

This produced inconsistent and invalid model evaluation.

### 1.2 Incorrect Prediction/Evaluation DataFrame

Predictions were written to `df_test` and ranking metrics were computed on `df_test`, even though the active training setup used ratio-based split objects.

### 1.3 Mislabelled Content-Based Model

The original section titled as content-based used:

- a book-user interaction pivot matrix
- `NearestNeighbors` similarity on ratings

This is interaction-based item-item similarity, not metadata-based content filtering.

### 1.4 Missing Metadata Content Pipeline

A full metadata-based content recommender (from `books.csv`) was not yet implemented in the earlier version.

---

## 2. Fixes Applied

### 2.1 Unified Official Split

- Marked `split_by_ratio` as the official split for final metrics.
- Marked `random_split` as exploratory only (not final reporting).

### 2.2 Corrected ItemCF Evaluation Flow

- Updated ItemCF training cell:
  - `eval_data=test_data` -> `eval_data=test_data_ratio`
- Updated prediction cell:
  - write predictions to `df_test_ratio["pred"]`
- Updated ranking metrics cell:
  - `precision_recall_at_k(df_test)` -> `precision_recall_at_k(df_test_ratio)`

### 2.3 Relabelled Exploratory Interaction Block

- Updated markdown heading/description above the existing `NearestNeighbors` interaction-pivot block.
- Explicitly labelled it as exploratory interaction-based item-item similarity (not content-based).
- Added note that final content model uses metadata from `books.csv`.

### 2.4 Added True Metadata-Based Content Recommender

Implemented a new scikit-learn content pipeline using only `books.csv` metadata:

- metadata preparation (`books_content`) with missing-value handling
- text fields combined into `content_text`:
  - `title`, `original_title`, `authors`, `language_code`
- TF-IDF features:
  - English stop words
  - `ngram_range=(1,2)`
  - capped vocabulary size
- optional numeric metadata features:
  - `average_rating`, `ratings_count`, `work_ratings_count`, `work_text_reviews_count`, `original_publication_year`
- `log1p` on heavy count columns
- scaling with `StandardScaler`
- sparse feature union via `hstack`
- created:
  - `item_feature_matrix`
  - `book_id_to_idx`
  - `idx_to_book_id`

### 2.5 Added Content Helper Functions

Added notebook-friendly helpers:

- `build_user_profile(...)`
- `score_unseen_books(...)`
- `recommend_top_n_content(...)`

Key constraints enforced:

- user profile built from training data only (`df_train_ratio`)
- seen training items excluded from candidate recommendations

### 2.6 Added Official Content Recommendation Output

- Added top-N content recommendation output for:
  - `user_id = 31933`
  - `n = 10`

### 2.7 Added Content Evaluation Cell

Added explicit content-model evaluation on `df_test_ratio`:

- cosine-score -> approximate 1-5 rating mapping
- predictions dataframe with:
  - `user`, `item`, `label`, `pred`
- computed:
  - `Precision@10`
  - `Recall@10`
  - `RMSE`

### 2.8 Added Final Comparison and Presentation Cells

- Added final comparison table with rows:
  - `ItemCF (Collaborative Filtering)`
  - `Content-TFIDF (Metadata-Based)`
- Metrics columns:
  - `RMSE`, `Precision@10`, `Recall@10`, plus concise notes
- Added final presentation cell:
  - top-5 recommendations for `user_id = 31933` from both models
  - excludes seen training items
  - displays human-readable book metadata
- Added final markdown conclusion section.

---

## 3. Rationale for Changes

- Enforces evaluation integrity by aligning split, training, prediction, and metrics.
- Aligns with lecturer guidance:
  - true content model implemented with scikit-learn and metadata
  - interaction-based block no longer mislabeled as content-based.
- Improves reproducibility and interpretability of final outputs.
- Supports fair comparison between warm-start CF and cold-start-friendly content methods.

---

## 4. Remaining Tasks

- Re-run notebook top-to-bottom to refresh all outputs in final order.
- Verify metric values are generated without stale intermediate objects.
- (If assignment explicitly requires) add a clear full-dataset retraining step after evaluation for final deployment-style recommendations.
- Final proofreading of markdown narrative and section titles.

---

## 5. Final Goal

Produce a clean, reproducible notebook that:

- compares two valid recommender approaches,
- uses one consistent official evaluation split,
- provides top-N recommendations for `user_id = 31933`,
- and ends with a defensible model recommendation for the bookstore.
