"""Candidate features for inference; this module never opens a .route file."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from feature_engineering.candidate_features import SegmentGrid, bearing, project_many, read_numeric

INFERENCE_FEATURES = (
    "distance_m",
    "distance_gap_m",
    "heading_cos",
    "gps_speed_mps",
    "candidate_rank",
)


@dataclass
class InferenceTrip:
    trip_id: str
    rows: list[dict]
    track_latlon: list[list[float]]
    nodes_lonlat: np.ndarray
    arcs: np.ndarray
    observations: int
    candidate_misses_without_reference: int


def build_inference_candidates(
    trip_dir: Path,
    *,
    k: int = 8,
    radius_m: float = 100.0,
    max_radius_m: float = 800.0,
) -> InferenceTrip:
    """Generate candidates from track/network inputs without consulting labels.

    Candidate misses returned here only mean no candidate segment was found.
    Determining whether the reference arc was missed requires a separate,
    post-prediction scoring step.
    """
    trip_id = trip_dir.name
    track = read_numeric(trip_dir / f"{trip_id}.track")
    nodes_ll = read_numeric(trip_dir / f"{trip_id}.nodes")
    arcs = read_numeric(trip_dir / f"{trip_id}.arcs").astype(np.int64)
    if track.shape[1] < 3 or nodes_ll.shape[1] < 2 or arcs.shape[1] < 2:
        raise ValueError(f"Malformed input rows for trip {trip_id}")
    if len(track) == 0 or len(nodes_ll) == 0 or len(arcs) == 0:
        raise ValueError(f"Trip {trip_id} has an empty track or road network")
    if not np.isfinite(track[:, :3]).all() or not np.isfinite(nodes_ll[:, :2]).all():
        raise ValueError(f"Trip {trip_id} contains non-finite GPS or node coordinates")
    if np.any((track[:, 0] < -180) | (track[:, 0] > 180) |
              (track[:, 1] < -90) | (track[:, 1] > 90)):
        raise ValueError(f"Trip {trip_id} contains out-of-range GPS coordinates")
    if np.any(arcs < 0) or np.any(arcs[:, :2] >= len(nodes_ll)):
        raise ValueError(f"Trip {trip_id} contains invalid arc endpoint references")

    lon0, lat0 = float(np.mean(track[:, 0])), float(np.mean(track[:, 1]))
    nodes_xy = project_many(nodes_ll[:, :2], lon0, lat0)
    starts, ends = nodes_xy[arcs[:, 0]], nodes_xy[arcs[:, 1]]
    points = project_many(track[:, :2], lon0, lat0)
    grid = SegmentGrid(starts, ends)
    degree = np.bincount(arcs[:, :2].ravel(), minlength=len(nodes_ll))

    headings = np.zeros(len(points), dtype=float)
    speeds = np.zeros(len(points), dtype=float)
    for i in range(1, len(points)):
        headings[i] = bearing(points[i - 1], points[i])
        dt = max(float(track[i, 2] - track[i - 1, 2]), 1e-3)
        speeds[i] = float(np.linalg.norm(points[i] - points[i - 1]) / dt)

    rows: list[dict] = []
    no_candidate = 0
    for obs_id, point in enumerate(points):
        found = []
        search_radius = radius_m
        while search_radius <= max_radius_m and not found:
            found = grid.nearby(point, search_radius)
            search_radius *= 2
        found = found[:k]
        if not found:
            no_candidate += 1
            continue
        best_distance = found[0][0]
        for rank, (distance, arc_id, _fraction) in enumerate(found):
            a, b = arcs[arc_id, :2]
            segment_heading = bearing(starts[arc_id], ends[arc_id])
            heading_delta = 0.0 if obs_id == 0 else np.arctan2(
                np.sin(headings[obs_id] - segment_heading),
                np.cos(headings[obs_id] - segment_heading),
            )
            rows.append({
                "trip_id": trip_id,
                "observation_id": obs_id,
                "arc_id": int(arc_id),
                "distance_m": float(distance),
                "distance_gap_m": float(distance - best_distance),
                "heading_cos": float(np.cos(heading_delta)),
                "gps_speed_mps": float(speeds[obs_id]),
                "candidate_rank": int(rank),
                "from_node": int(a),
                "to_node": int(b),
            })
    track_latlon = [[float(lat), float(lon)] for lon, lat in track[:, :2]]
    return InferenceTrip(
        trip_id=trip_id,
        rows=rows,
        track_latlon=track_latlon,
        nodes_lonlat=nodes_ll[:, :2],
        arcs=arcs[:, :2],
        observations=len(track),
        candidate_misses_without_reference=no_candidate,
    )
