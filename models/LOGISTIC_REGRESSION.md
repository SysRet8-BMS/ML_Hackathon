# Logistic Regression experiment - Member 4

`logistic_regression.py` trains a binary candidate classifier using the shared
PathSnap candidate-feature CSVs. It expects one row per `(trip_id,
observation_id, candidate arc)` with the columns `trip_id`, `observation_id`,
`is_match`, plus matching finite numeric feature columns. `arc_id` is optional
metadata retained for the Top-1 prediction file.

The default is L2-regularised logistic regression (`C=1.0`, LBFGS,
`class_weight="balanced"`, maximum 1,000 iterations). Candidate classes are
imbalanced, so balanced weights make the baseline less likely to ignore the
positive candidate; this is an initial setting, not a tuned result.

The script uses 5-fold `GroupKFold` by `trip_id`. The `StandardScaler` sits
inside each fold's pipeline and is therefore fitted only on that fold's training
trips. It then refits on all training trips and evaluates the untouched
student-test CSV. Do not pass evaluator trips or evaluator labels to this
command.

```sh
python models/logistic_regression.py \
  --train train_candidates.csv \
  --student-test student_test_candidates.csv \
  --out results/logistic_regression
```

Outputs:

- `metrics.json`: configuration, per-fold metrics, CV macro-F1 summary, and
  student-test metrics/timing.
- `fold_metrics.csv`: report-ready grouped-CV results.
- `student_top1_predictions.csv`: one highest-probability candidate per GPS
  observation, with coverage and correctness diagnostic fields.
- `logistic_regression.joblib`: final scaler-and-classifier pipeline, feature
  order, and configuration for inference.

`candidate_miss_rate` reports observations for which the shared candidate
generator supplied no positive/reference candidate. `observation_top1_accuracy`
counts those observations as incorrect, while the coverage-conditional Top-1
metric separates classifier decisions from candidate-generation coverage.

Logistic regression has no epoch-wise training history, so do not manufacture
training-loss curves for the report. Use the real fold metrics, student-test
metrics, timings, and optional confusion matrices in `metrics.json` instead.
