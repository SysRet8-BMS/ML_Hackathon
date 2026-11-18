"""Create an interactive map for a permitted train or student-test trip.

The route layer expands the supplied chronological arc IDs into straight
node-to-node segments. It is a reference visualization, not GPS/route alignment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import folium
from folium.plugins import Fullscreen


ALLOWED_PARTITIONS = {"train", "student_test"}


def read_rows(path):
    with path.open() as f:
        return [line.split() for line in f if line.strip()]


def create_map(data_root: Path, partition: str, trip_id: str, output: Path):
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError("Only train/student_test maps can be generated during development")
    trip = data_root / partition / trip_id
    if not trip.is_dir():
        raise FileNotFoundError(trip)
    track = [(float(r[1]), float(r[0])) for r in read_rows(trip / f"{trip_id}.track")]
    nodes = [(float(r[1]), float(r[0])) for r in read_rows(trip / f"{trip_id}.nodes")]
    arcs = [tuple(map(int, r[:2])) for r in read_rows(trip / f"{trip_id}.arcs")]
    route = [int(r[0]) for r in read_rows(trip / f"{trip_id}.route")]
    if not track:
        raise ValueError("Trip has no valid GPS points")
    m = folium.Map(location=track[len(track) // 2], zoom_start=14, control_scale=True)
    # Road network: draw all links, simplified as straight node-to-node segments.
    road = folium.FeatureGroup(name="Road network", show=True)
    for a, b in arcs:
        if 0 <= a < len(nodes) and 0 <= b < len(nodes):
            folium.PolyLine([nodes[a], nodes[b]], color="#8996a3", weight=1, opacity=.45).add_to(road)
    road.add_to(m)
    gps_group = folium.FeatureGroup(name="GPS observations", show=True)
    folium.PolyLine(track, color="#0878c9", weight=3, opacity=.85, tooltip="Raw GPS track").add_to(gps_group)
    for i, point in enumerate(track):
        folium.CircleMarker(point, radius=2, color="#0878c9", fill=True, fill_opacity=.75,
                            tooltip=f"GPS observation {i}").add_to(gps_group)
    gps_group.add_to(m)
    route_group = folium.FeatureGroup(name="Supplied reference route (arc sequence)", show=True)
    for arc_id in route:
        if 0 <= arc_id < len(arcs):
            a, b = arcs[arc_id]
            if 0 <= a < len(nodes) and 0 <= b < len(nodes):
                folium.PolyLine([nodes[a], nodes[b]], color="#18a558", weight=4, opacity=.9,
                                tooltip=f"Reference arc {arc_id}").add_to(route_group)
    route_group.add_to(m)
    folium.Marker(track[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(track[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen(position="topleft").add_to(m)
    m.fit_bounds(track)
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Datasets"))
    parser.add_argument("--partition", choices=sorted(ALLOWED_PARTITIONS), required=True)
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/eda/sample_map.html"))
    args = parser.parse_args()
    create_map(args.data, args.partition, args.trip_id, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
