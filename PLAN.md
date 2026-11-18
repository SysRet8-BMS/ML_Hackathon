# PathSnap Hackathon Plan

## Purpose and evidence

This plan follows `AGENTS.md` and the five supporting PDFs in `Docs/`. The required outputs are a public GitHub repository and one consolidated PDF report. The dataset documentation specifies 80 training trips, 10 student-testing trips for development comparison, and 10 evaluator-testing trips for final evaluation only. No dataset files are currently present in the repository, so the actual file layout, counts, alignment, data quality, and feasible model choices remain to be verified when data is supplied.

## Integrity rules

- Keep evaluator trips and their `.route` labels out of development, feature selection, model selection, tuning, and UI responses. On the inference laptop, labels may be used only after a prediction for final scoring; do not use those scores to revise the system.
- Use trip-grouped validation within the 80 training trips. Fit learned preprocessing only within each training fold; choose the final model and decoder settings using training CV and the 10 student-testing trips as permitted.
- Treat `.route` as supplied reference data, not unquestionable ground truth. Determine and document GPS-to-route alignment from actual files; do not assume row-wise correspondence.
- Generate every reported metric, plot, map, and table from checked-in code and real runs. Do not invent results or loss curves. Preserve configs, seeds, logs, and outputs needed to reproduce claims.
- Keep team implementation independent, retain meaningful human-authored Git history, and ensure all members can explain the pipeline and evidence. Do not use Codex/GPT as commit author.
- Keep credentials and evaluator ground truth out of public-facing artifacts and UI payloads. Review data licensing/privacy and repository size before publishing raw data or large artifacts.

## Technical sequence

### 1. Confirm inputs and establish reproducibility

- Inventory all supplied files and partition manifests; verify counts and naming against the documentation before analysis.
- Add a README, dependency specification, configuration/seeds, and clear module layout. Add data/artifact ignore rules where appropriate; never accidentally publish secrets or restricted data.
- Record environment and command lines for each experiment. Create result files only from actual runs.

### 2. EDA and data alignment

- Inspect training and student-testing partitions only: file availability/sizes, schemas, row counts, coordinate plausibility, invalid/missing values, timestamp ordering/gaps, trip extents, route lengths, graph sizes, isolated nodes, components, and connectivity.
- Establish and document how route arc sequences correspond to GPS observations, including ambiguities and route-label limitations. Validate node/arc indexes and geometry construction.
- Use a justified metric CRS or geodesic calculations for distances; keep longitude/latitude ordering explicit at file, projection, and Folium boundaries.
- Save reproducible inventory/quality tables, trajectory and graph plots, and at least one interactive Folium map. Label which partition each analysis uses.

### 3. Candidate generation and features

- Build a spatially indexed road-segment candidate generator. The current MLP raw-data path implements a grid-based nearest-arc search (initial K=8, radius=100 m with fallback) and reports candidate misses; validate and tune K/radius using training-only grouped validation, then student-testing for development comparison.
- Construct candidate-level labels with the current explicit working rule: nearest supplied route arc under non-decreasing route order. Inspect and document its assumptions/limitations before treating results as authoritative; record candidate misses and alignment uncertainty.
- Engineer spatial, heading/kinematic, and candidate-context features (distances in metres, projection position, arc geometry, bearing agreement, GPS speed/time gaps, candidate rank/count, and useful graph context). Avoid target leakage and fit preprocessing within CV folds.
- Ensure inference-time feature generation uses only GPS/network inputs and does not require reference routes.

### 4. Five-model experiment and phase-one presentation

The four assigned models are MLP and KNN (Member 1), XGBoost (Member 2), Random Forest (Member 3), and Logistic Regression (Member 4). This gives five distinct models; KNN is Member 1's fifth-model contribution. If the team wants an additional sixth model, assign it separately.

For each model, retain the actual configuration, grouped-CV and student-test outputs, confusion matrix, metrics, timing, and model-specific report answers. Member 1's MLP implementation must retain actual epoch-level training/validation loss rather than synthetic curves.

- Keep all candidates in the same trip-grouped CV protocol (GroupKFold, with fold count chosen to suit 80 trips), feature pipeline, and primary selection metric. Compare fold stability as well as means; avoid training accuracy as a selection criterion.
- Record configurations, CV macro F1, student-test observation Top-1 accuracy, macro F1, precision/recall, log loss when valid, candidate miss rate, route continuity, sequence measures, and train/inference times. Retain confusion matrices and actual-result comparison plots.
- Explain each model's suitability, strengths/limits, important hyperparameters/objective, student-test results, and over/underfit evidence in its own report subsection. Mark genuinely inapplicable metrics N/A with explanation.
- Phase-one presentation checkpoint: show verified EDA, candidate/alignment assumptions, baseline feature pipeline, and whatever real baseline-model comparisons have completed. Explicitly distinguish completed evidence from planned work; do not imply all five models are finished if they are not.

### 5. Select and tune one model

- Select a promising model using documented training-CV evidence, probability quality, stability, runtime, and student-test development comparison. State the primary metric and why it fits map matching.
- Run Bayesian HPO (for example Optuna/TPE) with a justified search space and grouped CV on the 80 training trips. Save each trial's parameters, objective, duration, status, and seed.
- Refit with selected parameters and compare tuned/untuned models using the same folds and student-test metrics, including inference time. Do not consult evaluator scores for any change.

### 6. HMM/Viterbi route reconstruction

- Represent each time step's candidate arcs as hidden states; use normalized classifier scores as log emissions with documented smoothing.
- Define directed graph transitions using road connectivity/length and GPS displacement, with explicit behavior for same-arc transitions, unreachable arcs, U-turns, missing candidates, and large timestamp gaps. Use metric distances.
- Tune decoder parameters only on permitted development data. Compare classifier-only and decoded output on student-testing trips using pointwise accuracy, sequence-level measures (for example arc-sequence F1/LCS with definitions), and route continuity. Include both helped and not-helped examples selected reproducibly.

### 7. Maps, API, UI, deployment

- Build reusable Folium output with GPS observations, local road network, reference route where permitted, classifier route, decoded route, clear legend/layer controls, bounds, and reproducible HTML. Do not expose evaluator reference labels in maps/UI.
- Laptop 1: FastAPI loads only inference artifacts and required trip inputs; `/trips` lists evaluator trip IDs, `/predict/{trip_id}` computes predictions through the real pipeline. Keep evaluator `.route` data server-side and use it strictly after prediction for final scoring. Return predictions and approved aggregate metrics, never labels. Log requests/responses without leaking routes or secrets.
- Laptop 2: UI selects a trip, calls the API, and shows returned route/map/metrics. Document host/network setup and CORS/firewall instructions without committing credentials.
- End-to-end test first on student-testing trips. For evaluator trips, verify execution and failure handling without using scores to tune. Capture genuine UI and receiving-terminal API evidence.

### 8. Final report and release

- Produce one consolidated PDF with required sections 2.2–7: EDA/features and partition attribution; five-model table and individual discussions; real comparison plots; Bayesian HPO; HMM and ablation; Folium map evidence; working API/UI screenshots.
- Link every claim to reproducible scripts/configs/results in the public repository. Audit for secrets, evaluator labels, hardcoded predictions, fabricated evidence, and oversized/unlicensed data/artifacts before publication.

## Four-person contribution plan

| Owner | Model ownership | Shared implementation and review |
|---|---|---|
| Member 1 (you) | MLP | Own MLP training/evaluation and real epoch-loss records; coordinate the feature input contract and participate in EDA/alignment review. |
| Member 2 | XGBoost | Own XGBoost training/evaluation; use the shared candidate features, CV splits, metrics, and artifact interface. |
| Member 3 | Random Forest | Own Random Forest training/evaluation; review route continuity and probability output compatibility. |
| Member 4 | Logistic Regression | Own Logistic Regression training/evaluation and shared experiment/report integration. |
| Member 1 (you) | KNN (additional model) | Own distance-weighted KNN training/evaluation under the shared split and metrics contract. |

Cross-cutting tasks (EDA, candidate generation/features, HMM, API/UI, and report integration) should be explicitly coordinated among members; this change assigns model ownership only and does not silently transfer those responsibilities.

All members should review the whole pipeline and practice explaining it for the viva. Use small, reviewable changes and commits authored by the human contributor; agree on commit conventions without using Codex/GPT author identity.

## Checkpoints and gates

1. **Input gate:** dataset partitions/files verified and evaluator data access boundary established.
2. **Phase-one presentation:** EDA and map, documented alignment, candidate/features baseline, and available real baseline comparisons shown with honest status.
3. **Model gate:** the five models (MLP, KNN, XGBoost, Random Forest, Logistic Regression) are trained; common evaluation is logged and the final family is justified.
4. **Tuning/decoder gate:** HPO retained; HMM ablation and edge cases evaluated on student-testing data.
5. **Integration gate:** student-test end-to-end pass; evaluator trip execution check does not feed scores back into development.
6. **Submission gate:** public repository and single PDF are complete, reproducible, scrubbed, and demonstrably consistent with the live system.

## Open decisions to resolve from actual inputs

- Dataset paths, exact partition manifests, licensing/publication constraints, and artifact size limits.
- True coordinate extent and chosen metric CRS.
- GPS-to-route alignment method and uncertainty policy.
- Candidate K/radius, graph representation, CV fold count, model library availability, primary selection metric, and sequence metric definitions.
- Final model and HPO search space, selected only after measured experiments.
