# MLP experiment

`mlp.py` implements Member 1's one-hidden-layer MLP as a binary candidate
classifier. It uses NumPy only and records real per-epoch training and
validation binary-cross-entropy histories.

## Input contract

The recommended workflow reads raw trip directories directly; CSV is optional.
The raw loader parses `.track`, `.nodes`, `.arcs`, and `.route` with NumPy,
vectorizes local metric-coordinate conversion, generates nearest road candidates
with a segment grid, and labels
each GPS point with the nearest supplied route arc while requiring the route
index to move monotonically forward. That is an explicit alignment assumption,
not a given point-to-arc correspondence; inspect it before treating results as
authoritative. Candidate-level rows are held in memory, so no intermediate
feature CSV is required.

For an optional prebuilt CSV workflow, use one row per
`(trip_id, observation_id, candidate arc)` with numeric features and `is_match`.

`--train` must contain training trips only. The script groups CV folds by
`trip_id`; it fits normalization separately inside each fold and on the full
training CSV for the final artifact. `--student-test` is used only after all
fold models are complete and the final model has been fit. Never pass evaluator
trips here.

## Run

```sh
python model_training/mlp.py \
  --train-root datasets/train \
  --student-root datasets/student_test \
  --out results/mlp
```

The raw workflow defaults to the nearest 8 arcs within 100 m (up to an 800 m
fallback radius). Outputs are JSON metrics, fold-level metrics, actual epoch
loss history, and compressed NumPy weights. All metrics are candidate-level except
`observation_top1_accuracy`, which chooses the highest-scoring candidate for
each observation. Route continuity is not claimed by this module; evaluate it
after mapping predicted candidate IDs through the road graph. Do not publish
experimental conclusions until real data have passed through the shared feature
generator.

The default compact feature set is perpendicular distance, distance gap to the
nearest candidate, heading cosine, GPS speed, and candidate rank. Override with
`--features col1,col2,...` after checking the shared feature schema. The default
configuration is a single 32-unit ReLU hidden layer, sigmoid output, binary cross
entropy, mini-batch SGD (learning rate 0.01, batch size 8192), L2 weight penalty
1e-4, 15 epochs, and seed 17. These are baseline settings, not
selected or tuned results. The current implementation records held-out fold
validation loss for analysis; it does not early-stop on a fold's validation
data.

## KNN baseline

Member 1's additional KNN implementation uses the same raw-data candidate
pipeline and evaluation contract:

```sh
python model_training/knn.py \
  --train-root datasets/train \
  --student-root datasets/student_test \
  --out results/knn
```

It uses the same five default features as MLP and applies `StandardScaler` inside
each trip-grouped fold. It uses a distance-weighted KNN (`k=15`, KD-tree) by
default. KNN has no epoch-wise loss
history; report actual fit/inference time and metrics instead. Exact KNN may be
memory- or compute-intensive on large candidate tables, so compare runtime and
consider reducing candidate K or using a documented training-only sample if
needed. Any reduced sample must be disclosed and must not replace the final
full-data evaluation without team approval.
