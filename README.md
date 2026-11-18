# PathSnap

PathSnap maps noisy GPS observations to road segments and reconstructs a connected route. The repository contains the candidate feature pipeline, model experiments, a FastAPI prediction service, and a Streamlit evaluator page.

## Development data and model

Keep the 80 training trips and 10 student-testing trips separate from the evaluator partition. Evaluator trips and their `.route` files must stay on the backend laptop. The API predicts from `.track`, `.nodes`, and `.arcs`; it reads `.route` only after the prediction has been produced, for server-side aggregate scoring. The UI receives no reference route or point labels.

The current backend default model is the saved KNN pipeline in `colab_results/knn/knn_pipeline.joblib.gz`. Set `PATHSNAP_MODEL_PATH` to the final tuned model artifact when HPO and model selection are complete. The current API response identifies its decoder as `classifier_only`; connect the HMM teammate's decoder before the final integrated demonstration.

## Laptop 1: API backend

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the local trip folders under `Datasets/evaluator_test/<trip_id>/`, or set `PATHSNAP_DATA_ROOT` and `PATHSNAP_EVALUATOR_DIR` to the local data locations. The API model and dataset folders are ignored by Git where applicable.

Run the server so it is reachable from the second laptop:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Available endpoints:

- `GET /health` reports whether the configured model artifact exists.
- `GET /trips` lists evaluator trip IDs found in the configured evaluator folder.
- `POST /predict/{trip_id}` generates candidates without reading `.route`, predicts candidate arcs, then computes aggregate scores against the server-side reference route.

Useful environment overrides:

```text
PATHSNAP_DATA_ROOT=/absolute/path/to/Datasets
PATHSNAP_EVALUATOR_DIR=/absolute/path/to/Datasets/evaluator_test
PATHSNAP_MODEL_PATH=/absolute/path/to/final_model.joblib
PATHSNAP_CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
```

## Laptop 2: evaluator page

Start Streamlit and point it to Laptop 1's address on the shared network:

```bash
PATHSNAP_API_URL=http://<LAPTOP1_IP>:8000 streamlit run ui/app.py
```

On Windows PowerShell, set the variable before starting Streamlit:

```powershell
$env:PATHSNAP_API_URL = "http://<LAPTOP1_IP>:8000"
streamlit run ui/app.py
```

The page calls the API, displays aggregate metrics, and draws GPS, local road-network, and predicted-route layers in Folium. Configure the API laptop's firewall to allow TCP port 8000 on the local network.

## EDA and report

The editable preprocessing and EDA report section is `report/section_2_2_eda_preprocessing.md`. The current interim PDF comparison is `report/technical_report_draft.pdf`; rebuild it and its actual-result charts with:

```bash
python report/build_report.py
```

The final submission still needs results from at least five trained models, Bayesian hyperparameter optimization, the HMM comparison, and genuine EDA/map and deployment screenshots. Do not describe the current interim report as the completed team report.
