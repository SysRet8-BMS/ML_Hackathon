
import os
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "distance_m",
    "distance_gap_m",
    "heading_cos",
    "gps_speed_mps",
    "candidate_rank"
]

TARGET = "is_match"

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Training file not found: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "trip_id",
        "observation_id",
        "arc_id",
        TARGET
    ] + FEATURES

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Missing values found in required columns. "
            "Inspect and resolve them before training."
        )

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    df[TARGET] = df[TARGET].astype(int)

    df[FEATURES] = df[FEATURES].astype(float)

    if not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError("Infinite feature values found.")

    return df


# ============================================================
# OBSERVATION LEVEL EVALUATION
# ============================================================

def observation_metrics(df, probabilities):

    temp = df[
        ["trip_id", "observation_id", "is_match", "candidate_rank"]
    ].copy()

    temp["probability"] = probabilities

    # Candidate with highest predicted probability wins.
    # Candidate rank breaks probability ties.
    temp = temp.sort_values(
        [
            "trip_id",
            "observation_id",
            "probability",
            "candidate_rank"
        ],
        ascending=[True, True, False, True]
    )

    top_candidates = temp.drop_duplicates(
        ["trip_id", "observation_id"]
    )

    # A prediction is correct only if the selected candidate
    # is actually labelled positive.
    top1_accuracy = float(
        top_candidates["is_match"].mean()
    )

    # Coverage: observations where at least one positive
    # candidate exists.
    positive_counts = temp.groupby(
        ["trip_id", "observation_id"]
    )["is_match"].sum()

    covered = positive_counts > 0

    coverage = float(covered.mean())

    # Conditional accuracy considers only observations
    # where the correct candidate was present.
    conditional_accuracy = float(
        top_candidates.loc[
            covered.to_numpy(),
            "is_match"
        ].mean()
    ) if covered.any() else 0.0

    return {
        "top1_accuracy_all_observations": top1_accuracy,
        "candidate_coverage": coverage,
        "conditional_top1_accuracy": conditional_accuracy,
        "total_observations": int(len(positive_counts)),
        "observations_with_positive": int(covered.sum()),
        "observations_without_positive": int((~covered).sum())
    }


# ============================================================
# MODEL
# ============================================================

def create_model(scale_pos_weight):

    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
        help="Path to training candidate CSV"
    )

    parser.add_argument(
        "--out",
        default="xgb_results",
        help="Output directory"
    )

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("\nLoading training candidate data...")

    df = load_data(args.train)

    X = df[FEATURES]
    y = df[TARGET]
    groups = df["trip_id"]

    print("\nDATASET SUMMARY")
    print("--------------------------")
    print("Rows:", len(df))
    print("Training trips:", groups.nunique())
    print("Positive candidates:", int(y.sum()))
    print("Negative candidates:", int((y == 0).sum()))

    positive_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    print("Positive class weight:", round(positive_weight, 3))

    observation_groups = df.groupby(
        ["trip_id", "observation_id"]
    )[TARGET].sum()

    print(
        "Observations with no positive:",
        int((observation_groups == 0).sum())
    )

    print(
        "Observations with positive:",
        int((observation_groups > 0).sum())
    )

    if groups.nunique() < N_SPLITS:
        raise ValueError("Not enough trips for 5-fold GroupKFold.")

    # ========================================================
    # GROUP K-FOLD CROSS VALIDATION
    # ========================================================

    cv = GroupKFold(n_splits=N_SPLITS)

    oof_probabilities = np.zeros(len(df))
    oof_predictions = np.zeros(len(df), dtype=int)

    fold_results = []

    print("\nSTARTING GROUP K-FOLD CROSS VALIDATION")

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        print(f"\nFOLD {fold}/{N_SPLITS}")

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        train_trips = set(groups.iloc[train_idx])
        val_trips = set(groups.iloc[val_idx])

        if train_trips.intersection(val_trips):
            raise RuntimeError("Trip leakage detected!")

        fold_weight = (
            (y_train == 0).sum() /
            max((y_train == 1).sum(), 1)
        )

        model = create_model(fold_weight)

        start = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start

        probabilities = model.predict_proba(X_val)[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        oof_probabilities[val_idx] = probabilities
        oof_predictions[val_idx] = predictions

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        accuracy = accuracy_score(
            y_val,
            predictions
        )

        loss = log_loss(
            y_val,
            np.clip(probabilities, 1e-7, 1 - 1e-7),
            labels=[0, 1]
        )

        observation_result = observation_metrics(
            df.iloc[val_idx],
            probabilities
        )

        fold_result = {
            "fold": fold,
            "training_trips": len(train_trips),
            "validation_trips": len(val_trips),
            "candidate_accuracy": accuracy,
            "candidate_precision": precision,
            "candidate_recall": recall,
            "candidate_f1": f1,
            "candidate_log_loss": loss,
            "training_time_seconds": training_time,
            **observation_result
        }

        fold_results.append(fold_result)

        print("Candidate accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1:", round(f1, 4))
        print("Log loss:", round(loss, 4))

        print(
            "Top-1 accuracy (all observations):",
            round(
                observation_result[
                    "top1_accuracy_all_observations"
                ],
                4
            )
        )

        print(
            "Candidate coverage:",
            round(
                observation_result["candidate_coverage"],
                4
            )
        )

        print("Training time:", round(training_time, 2), "seconds")

    # ========================================================
    # OUT-OF-FOLD RESULTS
    # ========================================================

    print("\n================================")
    print("OVERALL OUT-OF-FOLD RESULTS")
    print("================================")

    overall = {
        "candidate_accuracy": float(
            accuracy_score(y, oof_predictions)
        ),
        "candidate_precision": float(
            precision_score(y, oof_predictions, zero_division=0)
        ),
        "candidate_recall": float(
            recall_score(y, oof_predictions, zero_division=0)
        ),
        "candidate_f1": float(
            f1_score(y, oof_predictions, zero_division=0)
        ),
        "candidate_log_loss": float(
            log_loss(
                y,
                np.clip(oof_probabilities, 1e-7, 1 - 1e-7),
                labels=[0, 1]
            )
        )
    }

    overall.update(
        observation_metrics(df, oof_probabilities)
    )

    for key, value in overall.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float)
              else f"{key}: {value}")

    # ========================================================
    # SAVE CV RESULTS
    # ========================================================

    pd.DataFrame(fold_results).to_csv(
        os.path.join(args.out, "cv_fold_metrics.csv"),
        index=False
    )

    pd.DataFrame({
        "actual": y,
        "oof_probability": oof_probabilities,
        "oof_prediction": oof_predictions
    }).to_csv(
        os.path.join(args.out, "oof_predictions.csv"),
        index=False
    )

    # ========================================================
    # FINAL TRAINING ON ALL 80 TRAINING TRIPS
    # ========================================================

    print("\nTraining final XGBoost model on all training trips...")

    final_model = create_model(positive_weight)

    start = time.time()

    final_model.fit(X, y)

    final_training_time = time.time() - start

    model_path = os.path.join(
        args.out,
        "xgboost_road_matching_model.json"
    )

    final_model.save_model(model_path)

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": final_model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        os.path.join(args.out, "feature_importance.csv"),
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "model": "XGBoost",
        "training_trips": int(groups.nunique()),
        "candidate_rows": int(len(df)),
        "features": FEATURES,
        "positive_candidates": int(y.sum()),
        "negative_candidates": int((y == 0).sum()),
        "group_kfold_splits": N_SPLITS,
        "overall_oof_metrics": overall,
        "mean_fold_f1": float(
            np.mean([r["candidate_f1"] for r in fold_results])
        ),
        "std_fold_f1": float(
            np.std([r["candidate_f1"] for r in fold_results], ddof=1)
        ),
        "final_training_time_seconds": final_training_time,
        "model_path": model_path
    }

    with open(
        os.path.join(args.out, "training_summary.json"),
        "w"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\n================================")
    print("XGBOOST TRAINING COMPLETE")
    print("================================")

    print("Saved model:", model_path)
    print("Results directory:", args.out)


if __name__ == "__main__":
    main()
