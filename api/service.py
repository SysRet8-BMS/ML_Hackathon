"""Inference service for the evaluator API.

Prediction uses only the trip track and road network. A reference .route is
opened only by score_after_prediction, after candidate scores and arc choices
have already been produced.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score

from data_processing.candidate_features import SegmentGrid, point_segment, project_many, read_numeric
from data_processing.inference_candidates import INFERENCE_FEATURES, InferenceTrip, build_inference_candidates

ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.getenv("PATHSNAP_DATA_ROOT", str(ROOT / "Datasets"))).expanduser().resolve()


def evaluator_root() -> Path:
    return Path(os.getenv("PATHSNAP_EVALUATOR_DIR", str(data_root() / "evaluator_test"))).expanduser().resolve()


def model_path() -> Path:
    default = ROOT / "colab_results" / "knn" / "knn_pipeline.joblib.gz"
    return Path(os.getenv("PATHSNAP_MODEL_PATH", str(default))).expanduser().resolve()


@lru_cache(maxsize=1)
def load_model():
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact is missing: {path}")
    artifact = joblib.load(path)
    model = artifact.get("model") if isinstance(artifact, dict) else artifact
    features = artifact.get("features", INFERENCE_FEATURES) if isinstance(artifact, dict) else INFERENCE_FEATURES
    if tuple(features) != INFERENCE_FEATURES:
        raise ValueError(
            "The configured model expects a different feature set. Update the inference extractor "
            "to match the final model artifact before deployment."
        )
    return model


def list_evaluator_trips() -> list[str]:
    root = evaluator_root()
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluator trip directory is missing: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit())


def _selected_rows(trip: InferenceTrip):
    if not trip.rows:
        raise ValueError("No road candidates were generated for this trip")
    model = load_model()
    x = np.asarray([[row[name] for name in INFERENCE_FEATURES] for row in trip.rows], dtype=float)
    probas = model.predict_proba(x)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError("Loaded model does not expose a positive candidate class")
    scores = probas[:, int(positive[0])]
    # Stable ordering also makes the selected arc deterministic when scores tie.
    best_by_observation = {}
    for i, (row, score) in enumerate(zip(trip.rows, scores)):
        obs_id = int(row["observation_id"])
        current = best_by_observation.get(obs_id)
        if current is None or score > current["score"]:
            best_by_observation[obs_id] = {**row, "score": float(score), "candidate_index": i}
    return best_by_observation


def _align_route_after_prediction(trip_dir: Path, trip: InferenceTrip) -> list[int]:
    """Read reference labels for scoring only; caller must already have predicted."""
    route_path = trip_dir / f"{trip.trip_id}.route"
    route = read_numeric(route_path).astype(np.int64).ravel()
    valid_route = route[(route >= 0) & (route < len(trip.arcs))]
    if len(valid_route) == 0:
        return [-1] * trip.observations

    lon0 = float(np.mean([p[1] for p in trip.track_latlon]))
    lat0 = float(np.mean([p[0] for p in trip.track_latlon]))
    nodes_xy = project_many(trip.nodes_lonlat, lon0, lat0)
    starts, ends = nodes_xy[trip.arcs[:, 0]], nodes_xy[trip.arcs[:, 1]]
    points = project_many(np.asarray([[p[1], p[0]] for p in trip.track_latlon]), lon0, lat0)
    route_arc_ids = np.unique(valid_route)
    grid = SegmentGrid(starts, ends, indices=route_arc_ids)
    positions: dict[int, list[int]] = {}
    for pos, arc_id in enumerate(valid_route):
        positions.setdefault(int(arc_id), []).append(pos)

    refs: list[int] = []
    cursor = 0
    for point in points:
        best = None
        for radius in (100.0, 800.0):
            for distance, arc_id, _ in grid.nearby(point, radius):
                for route_pos in positions.get(int(arc_id), ()):
                    if route_pos >= cursor and (best is None or distance < best[0]):
                        best = (distance, route_pos, int(arc_id))
            if best is not None:
                break
        if best is None:
            for route_pos in range(cursor, len(valid_route)):
                arc_id = int(valid_route[route_pos])
                distance, _ = point_segment(point, starts[arc_id], ends[arc_id])
                if best is None or distance < best[0]:
                    best = (distance, route_pos, arc_id)
        if best is None:
            refs.append(-1)
        else:
            cursor = best[1]
            refs.append(best[2])
    return refs


def _route_continuity(predicted: list[int], arcs: np.ndarray) -> float | None:
    pairs = [(a, b) for a, b in zip(predicted, predicted[1:]) if a >= 0 and b >= 0]
    if not pairs:
        return None
    connected = sum(1 for a, b in pairs if a == b or int(arcs[a, 1]) == int(arcs[b, 0]))
    return connected / len(pairs)


def _map_network(trip: InferenceTrip, max_segments: int = 5000) -> list[list[list[float]]]:
    track = np.asarray(trip.track_latlon, dtype=float)
    lon_min, lon_max = float(track[:, 1].min()), float(track[:, 1].max())
    lat_min, lat_max = float(track[:, 0].min()), float(track[:, 0].max())
    lon_pad = max((lon_max - lon_min) * .15, .003)
    lat_pad = max((lat_max - lat_min) * .15, .003)
    nodes = trip.nodes_lonlat
    starts, ends = nodes[trip.arcs[:, 0]], nodes[trip.arcs[:, 1]]
    intersects = (
        (np.minimum(starts[:, 0], ends[:, 0]) <= lon_max + lon_pad) &
        (np.maximum(starts[:, 0], ends[:, 0]) >= lon_min - lon_pad) &
        (np.minimum(starts[:, 1], ends[:, 1]) <= lat_max + lat_pad) &
        (np.maximum(starts[:, 1], ends[:, 1]) >= lat_min - lat_pad)
    )
    ids = np.flatnonzero(intersects)
    if len(ids) > max_segments:
        ids = ids[::int(np.ceil(len(ids) / max_segments))][:max_segments]
    return [[[float(nodes[trip.arcs[i, 0], 1]), float(nodes[trip.arcs[i, 0], 0])],
             [float(nodes[trip.arcs[i, 1], 1]), float(nodes[trip.arcs[i, 1], 0])]] for i in ids]


def predict_trip(trip_id: str) -> dict:
    """Predict first; then score against the evaluator reference route server-side."""
    root = evaluator_root()
    trip_dir = (root / trip_id).resolve()
    if trip_dir.parent != root or not trip_dir.is_dir():
        raise FileNotFoundError(f"Evaluator trip does not exist: {trip_id}")

    # This stage loads only .track, .nodes, and .arcs. There is no .route access.
    trip = build_inference_candidates(trip_dir)
    selected = _selected_rows(trip)
    predicted = [int(selected[i]["arc_id"]) if i in selected else -1
                 for i in range(trip.observations)]
    predicted_rows = [selected[i] for i in sorted(selected)]

    # The classifier-only result is completed before any reference data is read.
    decoder_name = "classifier_only"

    # Post-prediction scoring: labels remain local and only aggregate metrics leave the server.
    reference = _align_route_after_prediction(trip_dir, trip)
    scored = [(p, r) for p, r in zip(predicted, reference) if r >= 0]
    accuracy = (sum(p == r for p, r in scored) / len(scored)) if scored else None
    macro_f1 = None
    if scored:
        macro_f1 = float(f1_score([r for _, r in scored], [p for p, _ in scored],
                                  average="macro", zero_division=0))
    candidate_misses = 0
    rows_by_obs: dict[int, set[int]] = {}
    for row in trip.rows:
        rows_by_obs.setdefault(int(row["observation_id"]), set()).add(int(row["arc_id"]))
    valid_reference_count = 0
    for obs_id, ref_arc in enumerate(reference):
        if ref_arc < 0:
            continue
        valid_reference_count += 1
        if ref_arc not in rows_by_obs.get(obs_id, set()):
            candidate_misses += 1
    continuity = _route_continuity(predicted, trip.arcs)

    route_segments = []
    for row in predicted_rows:
        arc_id = int(row["arc_id"])
        a, b = trip.arcs[arc_id]
        route_segments.append([
            [float(trip.nodes_lonlat[a, 1]), float(trip.nodes_lonlat[a, 0])],
            [float(trip.nodes_lonlat[b, 1]), float(trip.nodes_lonlat[b, 0])],
        ])
    display_track = trip.track_latlon
    if len(display_track) > 5000:
        stride = int(np.ceil(len(display_track) / 5000))
        display_track = display_track[::stride]
        if display_track[-1] != trip.track_latlon[-1]:
            display_track.append(trip.track_latlon[-1])

    return {
        "trip_id": trip_id,
        "decoder": decoder_name,
        "gps_points": display_track,
        "road_segments": _map_network(trip),
        "predicted_route_segments": route_segments,
        "metrics": {
            "observation_top1_accuracy": accuracy,
            "observation_macro_f1": macro_f1,
            "candidate_miss_rate": candidate_misses / max(valid_reference_count, 1),
            "route_continuity": continuity,
            "observations": trip.observations,
            "observations_with_candidates": len(selected),
        },
    }
