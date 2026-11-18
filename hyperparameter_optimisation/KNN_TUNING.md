# Provisional Bayesian KNN experiment

The saved KNN baseline leads the available three-model comparison in CV macro
F1. This experiment does not establish the final team model: XGBoost and Random
Forest results must still join the required five-model comparison.

Run from the repository root:

```sh
.venv/bin/python hyperparameter_optimisation/tune_knn.py --trials 10
```

Optuna TPE maximizes mean candidate macro F1 across five whole-trip folds on
all 80 training trips. The folds use the existing MLP/KNN group-fold helper.
Each trial fits scaling only on its training fold. No row subsampling is used.
Seed 17 controls search sampling. The first trial is the existing baseline.
Four startup trials are followed by adaptive TPE proposals.

Search space: odd neighbor counts 5 through 45 explore local versus smoother
decisions; uniform versus distance weights tests sensitivity to proximity;
Minkowski p=1 or p=2 compares Manhattan and Euclidean distance. Leaf size 40,
KD-tree search, the five shared features, and threshold 0.5 remain fixed.

The student-test CSV is loaded after the winning configuration is selected.
Both baseline and winner are refitted and evaluated on the same machine and
data. The study database and trials.csv retain parameters, status, duration,
objective and fold timing. folds.json records exact validation trip IDs.
comparison.json records results, input hashes, versions, seed and search cost.
The compressed fitted pipeline is knn_tuned.joblib.gz. Use a fresh --out path
for another run; existing study databases are deliberately preserved.

The optimized CV score is a selection score, not an unbiased nested-CV estimate.
Ten trials constitute a limited search and do not guarantee the global optimum.
Route continuity and HMM performance are outside this classifier experiment.

After completion, generate the comparison table, report section and SVG plots:

```sh
.venv/bin/python hyperparameter_optimisation/summarize_knn_tuning.py
```

This run uses scikit-learn log_loss for both baseline and tuned models. The
older MLP/KNN scorer adds 1e-9 inside logarithms, so its log-loss values are not
exactly comparable when KNN outputs zero or one. Use the paired results from
this experiment for tuning conclusions; do not combine old/new timing values
from different machines. Lower log loss measures overall probability quality,
not calibration alone.
