# Deployment

Configuration, dependencies, and instructions required to run PathSnap.

## Dependencies

Install from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Data placement

Keep the 80 training trips and 10 student-test trips separate from the evaluator
partition. Evaluator trips and their `.route` files must stay on the backend
laptop.

```text
Datasets/
  train/            # training trips (never shipped to the evaluator)
  student_test/     # student-test trips
  evaluator_test/   # evaluator trips, backend only
```

Override locations with `PATHSNAP_DATA_ROOT` and `PATHSNAP_EVALUATOR_DIR`.

## Run the API backend (Laptop 1)

```bash
./deployment/run_api.sh
```

or directly:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Endpoints: `GET /health`, `GET /trips`, `POST /predict/{trip_id}`.

## Run the evaluator UI (Laptop 2)

```bash
PATHSNAP_API_URL=http://<LAPTOP1_IP>:8000 ./deployment/run_ui.sh
```

On Windows PowerShell:

```powershell
$env:PATHSNAP_API_URL = "http://<LAPTOP1_IP>:8000"
streamlit run ui/app.py
```

## Environment variables

```text
PATHSNAP_DATA_ROOT=/absolute/path/to/Datasets
PATHSNAP_EVALUATOR_DIR=/absolute/path/to/Datasets/evaluator_test
PATHSNAP_MODEL_PATH=/absolute/path/to/final_model.joblib
PATHSNAP_CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
PATHSNAP_API_URL=http://<LAPTOP1_IP>:8000
```

Allow TCP port 8000 through the backend laptop's firewall so the second laptop
can reach the API.
