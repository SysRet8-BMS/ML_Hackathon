# Bayesian hyperparameter optimisation of KNN

This completed experiment tunes the current KNN candidate. Final team model selection remains pending the required five-model comparison; the available XGBoost log does not contain student-test results, and Random Forest results are absent.

## Model choice and objective

KNN had the highest saved mean CV Macro F1 among the three comparable completed baselines (0.8086). The objective was mean candidate Macro F1 over five folds of whole training trips. Macro F1 weights match and non-match classes equally. Top-1 accuracy and log loss remain separate route-matching and probability-quality checks.

## Search space and method

Optuna TPE with seed 17 evaluated 10 trials. Trial 0 reproduced the baseline; four startup trials were followed by adaptive TPE proposals. All trials used the full training data and five fixed trip-grouped folds. Each fold fitted its own StandardScaler. No evaluator data was read.

| Parameter | Search | Reason |
|---|---|---|
| Neighbours | Odd integers 5 to 45 | Local decisions versus smoothing |
| Weights | distance or uniform | Influence of nearby candidates |
| Minkowski p | 1 or 2 | Manhattan versus Euclidean geometry |

KD-tree, leaf size 40, the five shared input features, and classification threshold 0.5 were fixed. Input files exactly match the team compressed feature cache. Baseline and tuned evaluations used the same folds, inputs and machine.

## Selected configuration

Baseline: `{"n_neighbors": 15, "weights": "distance", "p": 2}`. Best trial: 4. Selected: `{"n_neighbors": 45, "weights": "uniform", "p": 1}`.

## Measured results

| Measure | Baseline | Tuned | Difference |
|---|---:|---:|---:|
| CV Macro F1 | 0.808625 | 0.816511 | +0.007886 |
| Student candidate accuracy | 0.922062 | 0.923635 | +0.001573 |
| Student Macro F1 | 0.807100 | 0.812620 | +0.005520 |
| Student Top 1 accuracy | 0.596086 | 0.602662 | +0.006576 |
| Student log loss | 0.482268 | 0.281531 | -0.200737 |
| Student conditional Top 1 | 0.803560 | 0.812425 | +0.008865 |
| Candidate miss rate | 0.258194 | 0.258194 | +0.000000 |
| Full training seconds | 0.903747 | 0.825976 | -0.077771 |
| Student inference seconds | 2.126684 | 5.330508 | +3.203824 |

Scores are fractions; a difference of 0.01 is one percentage point. Lower log loss is better. Timing covers one fit/predict measurement per model and excludes CSV loading, candidate generation and HMM decoding.

Both runs use scikit-learn log_loss. The older team KNN scorer adds 1e-9 inside logarithms, which gives different penalties for exact zero/one probabilities. Compare the paired values here rather than mixing old and new log-loss results. Log loss reflects overall probability quality and does not by itself prove calibration.

The CV-winning configuration was fixed before loading student-test data. Both models were refitted on all 80 training trips and evaluated on the same ten disjoint student-test trips.

Search wall time: 476.03 seconds, excluding final refits and student-test evaluation.

## Interpretation and limits

CV Macro F1 changed by +0.789 percentage points; student Macro F1 changed by +0.552 points and Top-1 by +0.658 points. These are different objectives and improvements in one do not establish improvement in all.

The best CV score is a model-selection score, not an unbiased nested-CV estimate. Ten trials are a limited search, not proof of a global optimum. Candidate misses are measured from supplied is_match labels and depend on the shared alignment assumptions. Tuning cannot recover reference arcs absent from candidate sets. Route continuity and HMM ablation require the later decoder experiment.

The winning neighbour count is the upper search boundary (45), so a larger search could change the preferred value. No additional search was chosen based on the student-test results. The measured inference increase is a deployment tradeoff; this result does not establish KNN as the final system model.

## Reproduction and evidence

Run `.venv/bin/python models/tune_knn.py --trials 10 --out results/knn_tuning_new`, then `.venv/bin/python models/summarize_knn_tuning.py --out results/knn_tuning_new`. Existing study directories are protected against overwrite.

`study.sqlite3` retains the full search, `trials.csv` stores trial parameters/status/duration, `folds.json` stores trip assignments, `comparison.json` stores measured results, input SHA256 hashes and versions, and `knn_tuned.joblib.gz` stores the fitted scaler and classifier. SVG charts and baseline_vs_tuned.csv are generated from those saved results.

Use this section in the single consolidated team PDF; it is not a replacement for the required five-model comparison. AI assistance was used to implement and document the experiment; team members should verify and explain it under the event policy.
