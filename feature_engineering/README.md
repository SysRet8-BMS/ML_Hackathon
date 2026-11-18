# Feature engineering

Candidate generation and feature extraction for the shared candidate table.

- `candidate_features.py` — parses `.track`, `.nodes`, `.arcs`, and `.route`
  files, projects coordinates, builds the segment grid, generates nearby
  candidate arcs, and labels them with `is_match`.
- `inference_candidates.py` — generates candidates for a single trip at
  prediction time without reading `.route`.

The shared feature set is `distance_m`, `distance_gap_m`, `heading_cos`,
`gps_speed_mps`, and `candidate_rank`.

> **Known limitation:** the current labelling rule requires the matched route
> arc index to move monotonically forward. This drifts on loop, out-and-back,
> and duplicate-arc trips. See `results/eda/DATA_AUDIT.md` and the alignment
> discussion in the project audit before treating labels as authoritative.

Run from the repository root so `feature_engineering` is importable.
