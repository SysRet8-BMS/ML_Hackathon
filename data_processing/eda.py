"""Reproducible EDA for permitted PathSnap partitions (train/student_test only).

Uses the Python standard library for parsing and haversine distances; matplotlib
is optional and used only for static trajectory plots.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict, deque
from pathlib import Path


EXTENSIONS = ("track", "nodes", "arcs", "route")


def rows(path: Path) -> list[list[str]]:
    with path.open() as f:
        return [line.split() for line in f if line.strip()]


def haversine(lon1, lat1, lon2, lat2):
    radius = 6_371_008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1, math.sqrt(a)))


def analyze_trip(folder: Path):
    tid = folder.name
    track = rows(folder / f"{tid}.track")
    nodes = rows(folder / f"{tid}.nodes")
    arcs = rows(folder / f"{tid}.arcs")
    route = rows(folder / f"{tid}.route")
    pts, invalid = [], 0
    for row in track:
        try:
            lon, lat, ts = map(float, row[:3])
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError("coordinate out of range")
            pts.append((lon, lat, ts))
        except (ValueError, IndexError):
            invalid += 1
    gaps = [pts[i][2] - pts[i - 1][2] for i in range(1, len(pts))]
    # The graph is undirected for exploratory connectivity only; direction is retained in arcs.
    adj = defaultdict(set)
    bad_arc_refs = 0
    for i, row in enumerate(arcs):
        try:
            a, b = map(int, row[:2])
            if a >= len(nodes) or b >= len(nodes) or min(a, b) < 0:
                bad_arc_refs += 1
                continue
            adj[a].add(b); adj[b].add(a)
        except (ValueError, IndexError):
            bad_arc_refs += 1
    route_ids = [int(row[0]) for row in route if row]
    invalid_route_ids = sum(not 0 <= arc_id < len(arcs) for arc_id in route_ids)
    route_valid = [arc_id for arc_id in route_ids if 0 <= arc_id < len(arcs)]
    route_discontinuities = 0
    for prev_id, next_id in zip(route_valid, route_valid[1:]):
        a, b = arcs[prev_id][:2], arcs[next_id][:2]
        try:
            a0, a1, b0, b1 = map(int, (a[0], a[1], b[0], b[1]))
            if a1 != b0:  # directed path continuity; reverse traversal is not presumed valid
                route_discontinuities += 1
        except (ValueError, IndexError):
            route_discontinuities += 1
    seen, components = set(), []
    for n in range(len(nodes)):
        if n in seen:
            continue
        q, size = deque([n]), 0
        seen.add(n)
        while q:
            u = q.popleft(); size += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        components.append(size)
    out = {
        "partition": folder.parent.name, "trip_id": tid,
        "track_rows": len(track), "valid_gps_rows": len(pts), "invalid_gps_rows": invalid,
        "nodes": len(nodes), "arcs": len(arcs), "route_arcs": len(route_ids),
        "invalid_route_arc_ids": invalid_route_ids,
        "route_directed_discontinuities": route_discontinuities,
        "timestamp_nonincreasing_steps": sum(x <= 0 for x in gaps),
        "timestamp_gap_gt_1s": sum(x > 1 for x in gaps),
        "timestamp_gap_max_s": max(gaps, default=0),
        "timestamp_gap_median_s": sorted(gaps)[len(gaps)//2] if gaps else 0,
        "graph_components_including_isolates": len(components),
        "isolated_nodes": sum(1 for n in range(len(nodes)) if not adj[n]),
        "largest_component_nodes": max(components, default=0),
        "invalid_arc_node_references": bad_arc_refs,
        "gps_lon_min": min((p[0] for p in pts), default=""),
        "gps_lon_max": max((p[0] for p in pts), default=""),
        "gps_lat_min": min((p[1] for p in pts), default=""),
        "gps_lat_max": max((p[1] for p in pts), default=""),
    }
    return out, pts, nodes, arcs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("Datasets"))
    p.add_argument("--out", type=Path, default=Path("results/eda"))
    p.add_argument("--plot-limit", type=int, default=1)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, plot_data = [], []
    for part in ("train", "student_test"):
        folders = sorted((args.data / part).iterdir())
        partition_plots = 0
        for folder in folders:
            if not folder.is_dir():
                continue
            record, pts, nodes, arcs = analyze_trip(folder)
            records.append(record)
            if partition_plots < args.plot_limit:
                plot_data.append((part, folder.name, pts, nodes, arcs))
                partition_plots += 1
    with (args.out / "trip_inventory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader(); w.writerows(records)
    keys = ["track_rows", "valid_gps_rows", "invalid_gps_rows", "nodes", "arcs", "route_arcs",
            "invalid_route_arc_ids", "route_directed_discontinuities",
            "timestamp_nonincreasing_steps", "timestamp_gap_gt_1s", "graph_components_including_isolates",
            "isolated_nodes", "invalid_arc_node_references"]
    with (args.out / "partition_summary.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["partition", "trip_count"] + [f"{k}_sum" for k in keys])
        for part in ("train", "student_test"):
            subset = [r for r in records if r["partition"] == part]
            w.writerow([part, len(subset)] + [sum(float(r[k]) for r in subset) for k in keys])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for part, tid, pts, nodes, arcs in plot_data:
            if not pts:
                continue
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter([n[0] for n in nodes], [n[1] for n in nodes], s=1, c="#b8c3cc", label="road nodes")
            # Bound visual density for very large local networks; the CSV retains
            # complete graph counts and the interactive map script draws all links.
            stride = max(1, len(arcs) // 5000)
            for row in arcs[::stride]:
                try:
                    a, b = map(int, row[:2]); ax.plot([float(nodes[a][0]), float(nodes[b][0])], [float(nodes[a][1]), float(nodes[b][1])], color="#b8c3cc", lw=.3)
                except (ValueError, IndexError):
                    pass
            ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts], ".-", ms=2, lw=.8, c="#1479c9", label="GPS")
            ax.set(title=f"{part} trip {tid} (lon/lat preview)", xlabel="Longitude", ylabel="Latitude")
            ax.legend(); ax.set_aspect("equal", adjustable="datalim"); fig.tight_layout()
            fig.savefig(args.out / f"{part}_{tid}_preview.png", dpi=150); plt.close(fig)
    except ImportError:
        (args.out / "PLOTS_UNAVAILABLE.txt").write_text("Install matplotlib to generate static preview plots.\n")
    print(f"Wrote {len(records)} trip records to {args.out}")


if __name__ == "__main__":
    main()
