"""Streamlit evaluator page; reference labels and route geometry never reach it."""
from __future__ import annotations

import os

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components


def _percent(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


st.set_page_config(page_title="PathSnap Evaluator", page_icon="🗺️", layout="wide")
st.title("PathSnap · Route reconstruction")
st.caption("Choose an evaluator trip to request a live prediction from the backend.")

api_url = st.sidebar.text_input(
    "Backend URL",
    value=os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000"),
).rstrip("/")


@st.cache_data(ttl=10)
def fetch_trips(url: str):
    response = requests.get(f"{url}/trips", timeout=10)
    response.raise_for_status()
    return response.json().get("trips", [])


try:
    trip_ids = fetch_trips(api_url)
except requests.RequestException as exc:
    st.error(f"Could not reach the API at {api_url}: {exc}")
    st.stop()

if not trip_ids:
    st.warning("The API is reachable, but no evaluator trip folders were found.")
    st.stop()

trip_id = st.selectbox("Evaluator trip", trip_ids)
if st.button("Predict route", type="primary"):
    with st.spinner(f"Predicting trip {trip_id}…"):
        try:
            response = requests.post(f"{api_url}/predict/{trip_id}", timeout=300)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")
            st.stop()

    metrics = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observation Top-1", _percent(metrics.get("observation_top1_accuracy")))
    c2.metric("Macro F1", _percent(metrics.get("observation_macro_f1")))
    c3.metric("Route continuity", _percent(metrics.get("route_continuity")))
    c4.metric("Candidate misses", _percent(metrics.get("candidate_miss_rate")))
    st.caption(
        f"Decoder: {result.get('decoder', 'unknown')} · "
        f"{metrics.get('observations_with_candidates', 0):,} of "
        f"{metrics.get('observations', 0):,} observations had candidates"
    )

    gps = result.get("gps_points", [])
    if not gps:
        st.warning("The API returned no GPS points to display.")
    else:
        center = gps[len(gps) // 2]
        map_view = folium.Map(location=center, zoom_start=14, control_scale=True)
        road_group = folium.FeatureGroup(name="Local road network", show=True)
        for segment in result.get("road_segments", []):
            folium.PolyLine(segment, color="#8a96a3", weight=2, opacity=.6).add_to(road_group)
        road_group.add_to(map_view)

        gps_group = folium.FeatureGroup(name="GPS observations", show=True)
        folium.PolyLine(gps, color="#1976d2", weight=3, opacity=.8).add_to(gps_group)
        stride = max(1, len(gps) // 250)
        for i in range(0, len(gps), stride):
            folium.CircleMarker(gps[i], radius=2, color="#1976d2", fill=True,
                                fill_opacity=.75, tooltip=f"GPS sample {i}").add_to(gps_group)
        gps_group.add_to(map_view)

        predicted_group = folium.FeatureGroup(name="Predicted road segments", show=True)
        for segment in result.get("predicted_route_segments", []):
            folium.PolyLine(segment, color="#d62728", weight=4, opacity=.9).add_to(predicted_group)
        predicted_group.add_to(map_view)
        folium.LayerControl(collapsed=False).add_to(map_view)
        map_view.fit_bounds(gps)
        components.html(map_view.get_root().render(), height=660, scrolling=True)
        st.caption("The evaluator reference route is kept on the backend and is not included in this map or API response.")
