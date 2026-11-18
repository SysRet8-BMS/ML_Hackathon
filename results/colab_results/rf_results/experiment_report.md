# PathSnap Random Forest — shared feature experiment

## Inputs
- Training: 1,369,843 candidate rows across 80 trips
- Student test: 130,963 candidate rows across 10 trips
- Features: distance_m, distance_gap_m, heading_cos, gps_speed_mps, candidate_rank

## Cross-validation summary (mean ± sample standard deviation)
- candidate_macro_f1: 0.791838 ± 0.024888
- candidate_roc_auc: 0.943371 ± 0.017784
- candidate_pr_auc: 0.747074 ± 0.062462
- candidate_log_loss: 0.276987 ± 0.023140
- top1_accuracy_covered: 0.787720 ± 0.043954
- top1_accuracy_all_observations: 0.612430 ± 0.122510
- candidate_coverage: 0.774547 ± 0.134720
- candidate_miss_rate: 0.225453 ± 0.134720

## Student-test metrics
- candidate_rows: 130963
- candidate_positive_rate: 0.108527
- candidate_accuracy: 0.885494
- candidate_precision: 0.484966
- candidate_recall: 0.888553
- candidate_f1: 0.627466
- candidate_macro_f1: 0.779908
- candidate_roc_auc: 0.943773
- candidate_pr_auc: 0.630495
- candidate_log_loss: 0.311645
- tn: 103338
- fp: 13412
- fn: 1584
- tp: 12629
- observations: 19160
- covered_observations: 14213
- candidate_coverage: 0.741806
- candidate_miss_rate: 0.258194
- top1_accuracy_covered: 0.818265
- top1_accuracy_all_observations: 0.606994
- final_fit_seconds: 59.763526
- inference_seconds: 0.394911
