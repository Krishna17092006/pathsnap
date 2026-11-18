"""PathSnap evaluator UI (Laptop 2).

Run from the project root:
    streamlit run ui/app.py

Set the API address in the sidebar, or beforehand with the PATHSNAP_API_URL environment
variable (e.g. http://192.168.1.20:8000). The UI talks to the API only over HTTP
(contract: TEAM.md section 2.7). It never reads trip files or ground truth: every route
and score shown here comes from the API response.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.maps import prediction_map  # noqa: E402

# 127.0.0.1 rather than "localhost": on Windows, "localhost" tries IPv6 first and adds ~2 s per request.
DEFAULT_API_URL = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000")
HEALTH_TIMEOUT_S = 3
TRIPS_TIMEOUT_S = 10
PREDICT_TIMEOUT_S = 180

METRIC_INFO = {  # key -> (label, kind)
    "top1_hmm": ("Top-1 accuracy (HMM)", "pct"),
    "top1_classifier": ("Top-1 accuracy (classifier only)", "pct"),
    "route_continuity_hmm": ("Route continuity (HMM)", "pct"),
    "route_continuity_clf": ("Route continuity (classifier only)", "pct"),
    "route_precision": ("Route precision", "pct"),
    "route_recall": ("Route recall", "pct"),
    "route_f1": ("Route F1", "pct"),
    "n_fixes": ("GPS fixes", "int"),
    "n_valid_fixes": ("Scored fixes", "int"),
}


# ----------------------------------------------------------------------------- API client
class ApiError(Exception):
    pass


def api_request(method: str, base: str, path: str, timeout: float) -> dict:
    url = base.rstrip("/") + path
    try:
        r = requests.request(method, url, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Cannot connect to {base}. Check that the API is running on Laptop 1, that the "
                       f"address and port are right, and that the firewall allows the port.") from None
    except requests.exceptions.Timeout:
        raise ApiError(f"{method} {path} timed out after {timeout:.0f} s.") from None
    except requests.exceptions.RequestException as e:
        raise ApiError(f"Request to {url} failed: {e}") from None
    if r.status_code != 200:
        if path == "/health" and r.status_code == 404:
            raise ApiError(f"Something answers at {base}, but it is not the PathSnap API (GET /health returned 404). "
                           f"Check the port: another program, such as Docker, may be using it.")
        try:
            detail = r.json().get("detail", r.text[:200])
        except ValueError:
            detail = r.text[:200]
        raise ApiError(f"{method} {path} returned HTTP {r.status_code}: {detail}")
    try:
        return r.json()
    except ValueError:
        raise ApiError(f"{method} {path} did not return JSON.") from None


@st.cache_data(ttl=10, show_spinner=False)
def check_health(base: str) -> dict:
    return api_request("GET", base, "/health", HEALTH_TIMEOUT_S)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_trips(base: str) -> list[str]:
    return list(api_request("GET", base, "/trips", TRIPS_TIMEOUT_S).get("trips", []))


# ----------------------------------------------------------------------------- formatting
def metric_label(key: str) -> str:
    return METRIC_INFO.get(key, (key.replace("_", " ").capitalize(), ""))[0]


def fmt(key: str, value) -> str:
    if value is None:
        return "—"
    kind = METRIC_INFO.get(key, ("", "num"))[1]
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "int":
        return f"{int(value):,}"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


# ----------------------------------------------------------------------------- page
st.set_page_config(page_title="PathSnap", page_icon=":material/route:", layout="wide")
st.session_state.setdefault("results", {})
st.session_state.setdefault("history", [])
results: dict = st.session_state["results"]
history: list = st.session_state["history"]

with st.sidebar:
    st.subheader("API server (Laptop 1)")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL,
                            help="Address of the laptop running the PathSnap API, e.g. http://192.168.1.20:8000")
    connected, health = False, {}
    try:
        health = check_health(api_url)
        connected = True
    except ApiError as e:
        st.error(str(e), icon=":material/error:")
    if connected:
        st.success("Connected" + (" to the mock server" if health.get("mock") else ""), icon=":material/check_circle:")
    if st.button("Refresh", icon=":material/refresh:", width="stretch"):
        check_health.clear()
        fetch_trips.clear()
        st.rerun()
    st.divider()
    st.caption("Reference routes stay on the API server. This page only receives predicted routes and "
               "aggregate scores.")

st.title("PathSnap")
st.caption("Map matching and route reconstruction. Pick an evaluation trip: the API server runs the full "
           "pipeline (candidate roads → classifier → HMM/Viterbi) and scores its own prediction.")

if not connected:
    st.info("Connect to the API server to load the evaluation trips.", icon=":material/link_off:")
    st.stop()
try:
    trips = fetch_trips(api_url)
except ApiError as e:
    st.error(str(e), icon=":material/error:")
    st.stop()
if not trips:
    st.warning("The API reports no trips. Put the evaluation trips in its trips folder, then press Refresh.")
    st.stop()

st.subheader(f"Evaluation trips ({len(trips)})")
# Labels must not change between reruns (e.g. adding a mark to finished trips): Streamlit would treat it
# as a new widget and reset the selection. Finished runs are listed in "Runs this session" instead.
trip = st.pills("Trip", trips, selection_mode="single", default=trips[0], label_visibility="collapsed", key="trip")

c_run, c_note = st.columns([1, 3], vertical_alignment="center")
run = c_run.button("Run map matching", type="primary", icon=":material/play_arrow:", disabled=trip is None,
                   width="stretch")
if trip is None:
    c_note.caption("Select a trip.")
elif trip in results:
    c_note.caption(f"Last run at {results[trip]['at']}. Press Run to request it again.")

if run and trip is not None:
    with st.spinner(f"Running the pipeline for trip {trip} on the API server…"):
        t0 = time.perf_counter()
        try:
            resp = api_request("POST", api_url, f"/predict/{trip}", PREDICT_TIMEOUT_S)
        except ApiError as e:
            st.error(str(e), icon=":material/error:")
            st.stop()
        roundtrip_ms = (time.perf_counter() - t0) * 1000
        m = prediction_map(resp.get("gps") or [], resp.get("predicted_route") or [], title=f"Trip {trip}")
    metrics = resp.get("metrics") or {}
    results[trip] = {"resp": resp, "roundtrip_ms": roundtrip_ms, "at": time.strftime("%H:%M:%S"),
                     "map_html": m.get_root().render()}
    history.append({
        "Time": results[trip]["at"], "Trip": trip,
        "Top-1 (HMM)": fmt("top1_hmm", metrics.get("top1_hmm")),
        "Top-1 (classifier)": fmt("top1_classifier", metrics.get("top1_classifier")),
        "Continuity (HMM)": fmt("route_continuity_hmm", metrics.get("route_continuity_hmm")),
        "GPS fixes": resp.get("n_fixes"),
        "Server ms": (resp.get("timing_ms") or {}).get("total"),
        "Round trip ms": round(roundtrip_ms),
        "Mock": bool(resp.get("mock")),
    })
    st.rerun()

rec = results.get(trip) if trip is not None else None
if rec is None:
    if trip is not None:
        st.info(f"Press **Run map matching** to process trip {trip}.", icon=":material/touch_app:")
else:
    resp = rec["resp"]
    metrics = resp.get("metrics") or {}
    timing = resp.get("timing_ms") or {}
    if resp.get("mock"):
        st.warning("Mock API response for UI development: no map matching was run, the orange line is the raw "
                   "GPS path and no scores were computed.", icon=":material/science:")

    hmm, clf = metrics.get("top1_hmm"), metrics.get("top1_classifier")
    delta = f"{(hmm - clf) * 100:+.1f} pts vs classifier" if hmm is not None and clf is not None else None
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(metric_label("top1_hmm"), fmt("top1_hmm", hmm), delta=delta, border=True)
    k2.metric(metric_label("top1_classifier"), fmt("top1_classifier", clf), border=True)
    k3.metric(metric_label("route_continuity_hmm"), fmt("route_continuity_hmm", metrics.get("route_continuity_hmm")),
              border=True)
    k4.metric("Response time", f"{rec['roundtrip_ms'] / 1000:.2f} s", border=True,
              help=f"Round trip from this page. Server pipeline: {timing.get('total', '—')} ms")

    left, right = st.columns([2.3, 1], gap="large")
    with left:
        components.html(rec["map_html"], height=600)
    with right:
        st.markdown("**Scores**")
        if metrics:
            st.dataframe(pd.DataFrame([{"Metric": metric_label(k), "Value": fmt(k, v)} for k, v in metrics.items()]),
                         hide_index=True, width="stretch")
        else:
            st.caption("The API returned no scores.")
        st.markdown("**Server pipeline timing**")
        if timing:
            st.dataframe(pd.DataFrame([{"Stage": k.replace("_", " ").capitalize(), "ms": v} for k, v in timing.items()]),
                         hide_index=True, width="stretch")
        st.markdown("**Response**")
        st.markdown(f"- GPS fixes: {resp.get('n_fixes', 0):,}\n"
                    f"- Predicted arcs: {len(resp.get('predicted_arcs') or []):,}\n"
                    f"- Received at {rec['at']}, round trip {rec['roundtrip_ms']:.0f} ms")

if history:
    st.subheader("Runs this session")
    st.dataframe(pd.DataFrame(history[::-1]), hide_index=True, width="stretch")
