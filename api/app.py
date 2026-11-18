"""FastAPI application for the two-laptop PathSnap demo."""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.service import list_evaluator_trips, load_model, model_path, predict_trip

logging.basicConfig(level=os.getenv("PATHSNAP_LOG_LEVEL", "INFO"))
logger = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap Map-Matching API", version="0.1.0")
cors_origins = [s.strip() for s in os.getenv("PATHSNAP_CORS_ORIGINS", "*").split(",") if s.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_log(request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started
    logger.info("%s %s -> %s (%.3fs)", request.method, request.url.path,
                response.status_code, elapsed)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_configured": model_path().is_file()}


@app.get("/trips")
def trips():
    try:
        ids = list_evaluator_trips()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"trips": ids, "count": len(ids)}


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    if not re.fullmatch(r"\d{1,16}", trip_id):
        raise HTTPException(status_code=400, detail="trip_id must contain digits only")
    try:
        # Load before processing so a missing model fails without reading any trip labels.
        load_model()
        result = predict_trip(trip_id)
        logger.info("Completed post-prediction scoring for %s: %s", trip_id, result["metrics"])
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed for trip %s", trip_id)
        raise HTTPException(status_code=500, detail="Prediction failed; inspect API terminal logs") from exc
