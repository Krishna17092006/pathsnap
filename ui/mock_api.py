"""Development stand-in for the PathSnap API, used to build and test the UI before the
real backend (api/main.py) exists.

It serves the same endpoints and response shape as the API contract (TEAM.md
section 2.7), but performs NO map matching: the "predicted route" it returns is
just the raw GPS path, and every metric is null. Every response carries
"mock": true, so the UI labels it clearly. It is never used for evaluation.

Run from the project root:
    python -m ui.mock_api                                  # port 8765, student_test trips
    python -m ui.mock_api --port 8766 --trips-dir data/student_test

Port 8765 is used by default because 8000 is often taken by other local services (e.g. Docker).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.data_io import DATA_DIR, read_raw

TRIP_ID = re.compile(r"^\d{8}$")
METRIC_KEYS = ["top1_classifier", "top1_hmm", "route_continuity_clf", "route_continuity_hmm",
               "route_precision", "route_recall"]


def make_handler(trips_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _trips(self) -> list[str]:
            return sorted(p.name for p in trips_dir.iterdir() if p.is_dir() and TRIP_ID.match(p.name))

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok", "mock": True})
            elif self.path == "/trips":
                self._send(200, {"trips": self._trips(), "mock": True})
            else:
                self._send(404, {"detail": "not found"})

        def do_POST(self):  # noqa: N802
            m = re.fullmatch(r"/predict/(\w+)", self.path)
            if not m or not TRIP_ID.match(m.group(1)) or m.group(1) not in self._trips():
                self._send(404, {"detail": "unknown trip"})
                return
            t0 = time.perf_counter()
            tid = m.group(1)
            track = read_raw(trips_dir / tid / f"{tid}.track", "track").dropna()
            gps = [[round(la, 6), round(lo, 6)] for lo, la in zip(track["lon"], track["lat"])]
            total_ms = round((time.perf_counter() - t0) * 1000, 1)
            self._send(200, {
                "trip_id": tid,
                "n_fixes": len(gps),
                "gps": gps,
                "predicted_route": gps[::3],
                "predicted_arcs": [],
                "metrics": {k: None for k in METRIC_KEYS},
                "timing_ms": {"total": total_ms},
                "mock": True,
            })

        def log_message(self, fmt, *args):
            print(f"[mock-api {time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Mock PathSnap API for UI development (no map matching)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--trips-dir", type=Path, default=DATA_DIR / "student_test")
    args = ap.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.trips_dir.resolve()))
    print(f"Mock PathSnap API on http://{args.host}:{args.port}  (trips from {args.trips_dir})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
