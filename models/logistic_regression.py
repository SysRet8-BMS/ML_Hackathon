"""Leakage-safe logistic-regression baseline for PathSnap candidate matching.

The input is the shared candidate-feature CSV: one row for every
``(trip_id, observation_id, candidate arc)`` with an ``is_match`` label.  It
does not create candidates or labels, and deliberately never reads evaluator
trips.  Scaling is fitted within each training fold to avoid validation-trip
leakage, then refitted on all training trips for the student-test evaluation
and final saved artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ID_COLUMNS = {"trip_id", "observation_id", "arc_id"}
LABEL_COLUMN = "is_match"


def load_candidates(path: Path):
    """Load and validate one shared candidate-feature CSV."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")

    fields = list(rows[0])
    required = {"trip_id", "observation_id", LABEL_COLUMN}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    features = [field for field in fields if field not in ID_COLUMNS | {LABEL_COLUMN}]
    if not features:
        raise ValueError(f"{path} has no numeric feature columns")

    try:
        x = np.asarray([[float(row[col]) for col in features] for row in rows], dtype=float)
        y = np.asarray([int(row[LABEL_COLUMN]) for row in rows], dtype=int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains non-numeric feature values or invalid labels") from exc
    if not np.isfinite(x).all():
        raise ValueError(f"{path} contains non-finite feature values")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError(f"{path} must use binary {LABEL_COLUMN} values (0/1)")
    return rows, features, x, y


def make_model(c: float, class_weight: str | None, max_iter: int, seed: int) -> Pipeline:
    """Build an interpretable, regularised probability model."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def observation_metrics(y: np.ndarray, probabilities: np.ndarray, rows: list[dict[str, str]]):
    """Score best-candidate decisions without assuming every truth is generated.

    A candidate generator can miss the reference road segment.  Such an
    observation remains an incorrect Top-1 decision, while the coverage-aware
    figure isolates classifier quality where a correct decision is possible.
    """
    observations: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        observations.setdefault((row["trip_id"], row["observation_id"]), []).append(index)

    total_correct = covered_correct = covered = 0
    predictions = []
    for (trip_id, observation_id), indexes in observations.items():
        best = max(indexes, key=lambda i: probabilities[i])
        reference_present = bool(y[indexes].any())
        correct = int(y[best] == 1)
        total_correct += correct
        if reference_present:
            covered += 1
            covered_correct += correct
        predictions.append(
            {
                "trip_id": trip_id,
                "observation_id": observation_id,
                "predicted_arc_id": rows[best].get("arc_id", ""),
                "predicted_probability": float(probabilities[best]),
                "reference_candidate_present": int(reference_present),
                "top1_correct": correct,
            }
        )

    total = len(observations)
    return {
        "observations": total,
        "observations_with_reference_candidate": covered,
        "candidate_miss_rate": float(1 - covered / total) if total else 0.0,
        "observation_top1_accuracy": float(total_correct / total) if total else 0.0,
        "observation_top1_accuracy_when_reference_candidate_present": (
            float(covered_correct / covered) if covered else None
        ),
    }, predictions


def metrics(y: np.ndarray, probabilities: np.ndarray, rows: list[dict[str, str]]):
    """Return the common candidate- and observation-level metric contract."""
    labels = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y, labels, labels=[0, 1]).ravel().tolist()
    grouped, _ = observation_metrics(y, probabilities, rows)
    return {
        "candidate_accuracy": float(accuracy_score(y, labels)),
        "candidate_macro_f1": float(f1_score(y, labels, average="macro", zero_division=0)),
        "candidate_precision": float(precision_score(y, labels, zero_division=0)),
        "candidate_recall": float(recall_score(y, labels, zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "confusion_matrix_tn_fp_fn_tp": matrix,
        **grouped,
    }


def write_csv(path: Path, records: list[dict]):
    if not records:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Candidate CSV from 80 training trips only")
    parser.add_argument("--student-test", type=Path, required=True, help="Candidate CSV from 10 student-test trips")
    parser.add_argument("--out", type=Path, default=Path("results/logistic_regression"))
    parser.add_argument("--folds", type=int, default=5, help="Trip-grouped CV folds")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse L2 regularisation strength")
    parser.add_argument(
        "--class-weight", choices=("balanced", "none"), default="balanced",
        help="Use balanced positive/negative weights, or unweighted logistic loss",
    )
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum LBFGS iterations")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.C <= 0 or args.folds < 2 or args.max_iter < 1:
        parser.error("--C must be positive, --folds at least 2, and --max-iter at least 1")

    args.out.mkdir(parents=True, exist_ok=True)
    train_rows, features, x_train, y_train = load_candidates(args.train)
    student_rows, student_features, x_student, y_student = load_candidates(args.student_test)
    if features != student_features:
        raise ValueError("Train and student-test feature columns must match exactly")
    groups = np.asarray([row["trip_id"] for row in train_rows])
    if len(np.unique(groups)) < args.folds:
        raise ValueError("There must be at least as many distinct training trips as folds")
    class_weight = None if args.class_weight == "none" else args.class_weight

    fold_records = []
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (train_index, valid_index) in enumerate(splitter.split(x_train, y_train, groups), start=1):
        model = make_model(args.C, class_weight, args.max_iter, args.seed + fold)
        start = time.perf_counter()
        model.fit(x_train[train_index], y_train[train_index])
        elapsed = time.perf_counter() - start
        probabilities = model.predict_proba(x_train[valid_index])[:, 1]
        fold_records.append(
            {
                "fold": fold,
                "training_trips": len(np.unique(groups[train_index])),
                "validation_trips": len(np.unique(groups[valid_index])),
                "train_seconds": elapsed,
                **metrics(y_train[valid_index], probabilities, [train_rows[i] for i in valid_index]),
            }
        )

    final_model = make_model(args.C, class_weight, args.max_iter, args.seed)
    start = time.perf_counter()
    final_model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    student_probabilities = final_model.predict_proba(x_student)[:, 1]
    inference_seconds = time.perf_counter() - start
    student_metrics = metrics(y_student, student_probabilities, student_rows)
    _, student_predictions = observation_metrics(y_student, student_probabilities, student_rows)

    results = {
        "model": "l2_regularized_binary_logistic_regression",
        "features": features,
        "config": {
            "C": args.C,
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": class_weight,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "objective": "regularized binary log loss",
            "threshold": 0.5,
        },
        "data_boundary": {
            "training_input": str(args.train),
            "student_test_input": str(args.student_test),
            "evaluator_data_used": False,
        },
        "cv_folds": fold_records,
        "cv_macro_f1_mean": float(np.mean([row["candidate_macro_f1"] for row in fold_records])),
        "cv_macro_f1_std": float(np.std([row["candidate_macro_f1"] for row in fold_records])),
        "student_test": student_metrics,
        "fit_seconds_full_train": fit_seconds,
        "student_inference_seconds": inference_seconds,
    }
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    write_csv(args.out / "fold_metrics.csv", fold_records)
    write_csv(args.out / "student_top1_predictions.csv", student_predictions)
    joblib.dump(
        {"pipeline": final_model, "features": features, "model_name": results["model"], "config": results["config"]},
        args.out / "logistic_regression.joblib",
    )
    print(f"Saved logistic-regression experiment outputs to {args.out}")


if __name__ == "__main__":
    main()
