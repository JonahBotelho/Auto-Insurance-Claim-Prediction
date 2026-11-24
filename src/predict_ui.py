"""Minimalistic local browser UI for prediction-only inference with model 1200001 / 1200002."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import sys
import threading
import time
import webbrowser

import pandas as pd

from src.data_loading import load_test_data_from_buffer
from src.feature_schema import TARGET_COLUMN
from src.model_io import load_model
from src.predict import generate_submission

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTOR_MODEL_PATH = REPO_ROOT / "models" / "em_timeline_sibling_rich_pricing_ensemble.joblib"
PAGE_PATH = Path(__file__).resolve().parent / "predict_ui_page.html"
MODEL_LABEL = "1200001 / 1200002"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local browser claim predictor (predict-only).")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_PREDICTOR_MODEL_PATH,
                        help="Path to the pre-trained joblib model (default: 1200001 / 1200002).")
    parser.add_argument("--host", default="127.0.0.1", help="Local bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8001, help="Local bind port (default: 8001).")
    parser.add_argument("--no-browser", action="store_true", help="Start server without opening a browser tab.")
    return parser.parse_args(argv)


DEFAULT_OUTPUT_FILENAME = "predictedclaimamount.csv"


def _safe_output_filename(raw_name: str) -> str:
    stem = Path(raw_name or DEFAULT_OUTPUT_FILENAME).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or Path(DEFAULT_OUTPUT_FILENAME).stem
    return f"{cleaned}.csv"


def _score_csv_bytes(model, payload: bytes):
    """Parse uploaded CSV bytes and return predicted CSV bytes plus summary statistics."""
    if not payload:
        raise ValueError("Uploaded file is empty.")
    try:
        test_data = load_test_data_from_buffer(io.BytesIO(payload))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not parse CSV file: {error}") from error

    started = time.perf_counter()
    submission = generate_submission(model, test_data)
    elapsed = time.perf_counter() - started
    total_rows = len(submission)
    nonzero_claims = int((submission[TARGET_COLUMN] > 0).sum())
    nonzero_percent = 100.0 * nonzero_claims / total_rows if total_rows else 0.0
    csv_bytes = submission.to_csv(index=False).encode("utf-8")
    return csv_bytes, {
        "total_rows": total_rows,
        "nonzero_claims": nonzero_claims,
        "nonzero_percent": nonzero_percent,
        "elapsed_seconds": elapsed,
    }


def _make_handler(model, model_path: Path, page_bytes: bytes):
    predict_lock = threading.Lock()

    class PredictorRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format_string, *args):
            return

        def _send_bytes(self, status, content_type, body, extra_headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for header, value in (extra_headers or {}).items():
                self.send_header(header, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status, payload):
            self._send_bytes(status, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", page_bytes)
                return
            if self.path == "/api/health":
                self._send_json(HTTPStatus.OK, {
                    "ready": True,
                    "model_label": MODEL_LABEL,
                    "model_file": model_path.name,
                })
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})

        def do_POST(self):
            if self.path != "/api/predict":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length header."})
                return
            if content_length <= 0:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "No CSV data received."})
                return
            if content_length > MAX_UPLOAD_BYTES:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "CSV exceeds the 100 MB upload limit."})
                return

            payload = self.rfile.read(content_length)
            output_name = _safe_output_filename(self.headers.get("X-Filename", DEFAULT_OUTPUT_FILENAME))
            try:
                with predict_lock:
                    csv_bytes, stats = _score_csv_bytes(model, payload)
            except ValueError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except Exception as error:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Prediction failed: {error}"})
                return

            headers = {
                "Content-Disposition": f'attachment; filename="{output_name}"',
                "X-Total-Rows": str(stats["total_rows"]),
                "X-Nonzero-Claims": str(stats["nonzero_claims"]),
                "X-Nonzero-Percent": f"{stats['nonzero_percent']:.4f}",
                "X-Elapsed-Seconds": f"{stats['elapsed_seconds']:.2f}",
            }
            self._send_bytes(HTTPStatus.OK, "text/csv; charset=utf-8", csv_bytes, headers)

    return PredictorRequestHandler


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.model_path.is_file():
        print(f"Model file not found: {args.model_path}", file=sys.stderr)
        print("Train the 1200001 / 1200002 model first with:", file=sys.stderr)
        print("  python -m src.train --config configs/claim_history/em_timeline_sibling_rich_pricing_ensemble.json "
              f"--full-fit-only --model-path {args.model_path}", file=sys.stderr)
        return 1

    print(f"Loading model {MODEL_LABEL} from {args.model_path}...", flush=True)
    model = load_model(args.model_path)
    page_bytes = PAGE_PATH.read_bytes()
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(model, args.model_path, page_bytes))
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Local predictor ready at {url} (Press Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local predictor.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
