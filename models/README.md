# MLP experiment

`mlp.py` implements Member 1's one-hidden-layer MLP as a binary candidate
classifier. It uses NumPy only and records real per-epoch training and
validation binary-cross-entropy histories.

## Input contract

Provide two CSV files, one row per `(trip_id, observation_id, candidate arc)`.
Both files must contain `trip_id`, `observation_id`, `is_match`, and identical
numeric feature columns. `arc_id` is optional metadata. `is_match` must be 1 for
the aligned reference candidate and 0 for other candidates. This module does not
generate labels or determine GPS-to-route alignment; do that in the shared
feature pipeline and document its assumptions first.

`--train` must contain training trips only. The script groups CV folds by
`trip_id`; it fits normalization separately inside each fold and on the full
training CSV for the final artifact. `--student-test` is used only after all
fold models are complete and the final model has been fit. Never pass evaluator
trips here.

## Run

```sh
python models/mlp.py \
  --train results/features/train_candidates.csv \
  --student-test results/features/student_test_candidates.csv \
  --out results/mlp
```

Outputs are JSON metrics, fold-level metrics, actual epoch loss history, and
compressed NumPy weights. All metrics are candidate-level except
`observation_top1_accuracy`, which chooses the highest-scoring candidate for
each observation. Route continuity is not claimed by this module; evaluate it
after mapping predicted candidate IDs through the road graph. Do not publish
experimental conclusions until real data have passed through the shared feature
generator.

The default configuration is a single 32-unit ReLU hidden layer, sigmoid output,
binary cross entropy, mini-batch SGD (learning rate 0.01, batch size 512), L2
weight penalty 1e-4, 40 epochs, and seed 17. These are baseline settings, not
selected or tuned results. The current implementation records held-out fold
validation loss for analysis; it does not early-stop on a fold's validation
data.
