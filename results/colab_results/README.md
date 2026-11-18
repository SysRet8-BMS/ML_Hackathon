# Colab experiment outputs

These are the outputs from the completed Colab run. The feature tables contain the 80 training and 10 student-test trips only; evaluator data is not present.

- `features/train_candidates.csv.gz` and `features/student_test_candidates.csv.gz`: compressed candidate datasets with labels and the five model features.
- `features/cache_manifest.json`: trip IDs and feature/cache settings.
- `candidate_generation_summary.json`: observation counts and candidate miss rates.
- `mlp/`: MLP metrics, fold results, actual loss history, and NumPy weight/scaling artifact.
- `knn/`: KNN metrics, fold results, and compressed fitted pipeline.

The uncompressed feature CSVs and KNN joblib remain available locally in this folder but are excluded from Git because they exceed GitHub's per-file limit. The committed gzip files are byte-for-byte compressed copies of those source files.
