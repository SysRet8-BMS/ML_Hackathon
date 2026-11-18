# Colab model and feature artifacts

These artifacts were copied from the completed Colab run. The feature tables are gzip-compressed to stay below GitHub's 100 MB per-file limit. They contain only the 80 permitted training trips and 10 student-testing trips; evaluator-testing data is not included.

## Feature cache

- `features/train_candidates.csv.gz`: 1,369,843 candidate rows for 80 training trips.
- `features/student_test_candidates.csv.gz`: 130,963 candidate rows for 10 student-testing trips.
- The feature columns are `distance_m`, `distance_gap_m`, `heading_cos`, `gps_speed_mps`, and `candidate_rank`, with trip/observation/arc IDs and the candidate `is_match` target.
- `features/candidate_generation_summary.json` records candidate miss counts/rates; `features/cache_manifest.json` records the trip IDs and cache settings.

## MLP

- `mlp/metrics.json`, `mlp/fold_metrics.csv`, and `mlp/loss_history.csv` record metrics and actual epoch loss histories.
- `mlp/mlp_weights.npz` contains the fitted NumPy MLP weights and feature scaling arrays.

## KNN

- `knn/metrics.json` and `knn/fold_metrics.csv` record model metrics.
- `knn/knn_pipeline.joblib.gz` is the gzip-compressed fitted scikit-learn pipeline (scaler plus KNN estimator).

Metrics in these files are experimental results from this run, not guarantees of final performance. Review the candidate-generation and route-alignment assumptions in the project documentation before reporting them.
