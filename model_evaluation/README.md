# Model evaluation

Code and artifacts for comparing models, computing metrics, and producing plots
and confusion matrices.

## Where evaluation currently lives

Each training script computes its own candidate-level and observation-level
metrics and writes them next to its model:

| Model | Script | Results |
|---|---|---|
| MLP | `model_training/mlp.py` | `results/mlp/` |
| KNN | `model_training/knn.py` | `results/knn/` |
| Logistic regression | `model_training/logistic_regression.py` | `results/logistic_regression/` |
| Random forest | `model_training/RF.py` | `results/colab_results/rf_results/` |
| XGBoost | `model_training/train_xgboost.py` | `results/xgboost/` |

Hyperparameter-search comparison plots are produced by
`hyperparameter_optimisation/summarize_knn_tuning.py` into `results/knn_tuning/`.

## Outstanding

- A single consolidated comparison script that reads every model's
  `metrics.json` and emits one table plus confusion matrices.
- Standardise on **macro F1** across all models (the XGBoost log currently
  reports positive-class F1 only).
- Alignment-aware metrics: report a clean-trip vs affected-trip split so the
  route-alignment defect does not silently depress every model's score.
- `build_lr_report.py` builds the per-model logistic-regression DOCX; per-model
  reports should not be submitted in place of the consolidated report.
