#!/usr/bin/env python3
"""Tiny frontend for live next-block gas direction predictions.

Run:
    python3 frontend_app.py

Then open:
    http://localhost:8000

The app fetches recent Etherscan block rows, rebuilds the same cleaned and
engineered features, loads trained model artifacts from model_outputs/, and
predicts whether the next block will have higher gas_used than the current
latest block.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from clean_blocks import clean_blocks
from feature_engineering import add_engineered_features
from scrape_blocks import build_session, fetch_page, page_url, parse_blocks_from_html


ARTIFACT_PATH = Path("model_outputs/model_artifacts.pkl")
PAGE_SIZE = 100
LIVE_PAGES = 3
HOST = "127.0.0.1"
PORT = 8000


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ethereum Next Block Prediction</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #152033;
      --muted: #5d6b82;
      --line: #d9e1ec;
      --blue: #2563eb;
      --green: #15803d;
      --red: #b91c1c;
      --amber: #b45309;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
    }
    main {
      max-width: 1040px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.15;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }
    .toolbar {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin: 24px 0;
    }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--blue);
      color: white;
      padding: 11px 16px;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary {
      background: #334155;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    .wide { grid-column: 1 / -1; }
    h2 {
      margin: 0 0 14px;
      font-size: 18px;
    }
    .kv {
      display: grid;
      grid-template-columns: 190px 1fr;
      gap: 8px 12px;
      font-size: 14px;
    }
    .kv div:nth-child(odd) {
      color: var(--muted);
    }
    .prediction {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .model-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .model-name {
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .direction {
      margin-top: 8px;
      font-size: 24px;
      font-weight: 800;
    }
    .higher { color: var(--green); }
    .lower { color: var(--red); }
    .confidence {
      margin-top: 4px;
      color: var(--muted);
    }
    .status {
      min-height: 24px;
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 14px;
    }
    .error {
      color: var(--red);
      font-weight: 700;
    }
    .warn {
      color: var(--amber);
      font-weight: 700;
    }
    @media (max-width: 760px) {
      .grid, .prediction { grid-template-columns: 1fr; }
      .kv { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <h1>Ethereum Next Block Gas Prediction</h1>
    <p>Fetches the latest Etherscan block rows, rebuilds the engineered features, and predicts whether the next block will have higher gas used than the current block.</p>

    <div class="toolbar">
      <button id="getBtn">Get Current Block</button>
      <button id="checkBtn" class="secondary" disabled>Check Next Block</button>
    </div>
    <div id="status" class="status"></div>

    <div class="grid">
      <section class="panel">
        <h2>Current Block</h2>
        <div id="blockDetails" class="kv">
          <div>Block</div><div>-</div>
          <div>Timestamp</div><div>-</div>
          <div>Transactions</div><div>-</div>
          <div>Gas Used</div><div>-</div>
          <div>Gas Limit</div><div>-</div>
          <div>Base Fee</div><div>-</div>
          <div>Congestion Ratio</div><div>-</div>
          <div>Rows Used</div><div>-</div>
        </div>
      </section>

      <section class="panel">
        <h2>Verification</h2>
        <div id="verification" class="kv">
          <div>Next block</div><div>-</div>
          <div>Actual direction</div><div>-</div>
          <div>Result</div><div>-</div>
        </div>
        <p class="status">After a prediction, the app waits about 12 seconds and checks whether the next block was actually higher or lower.</p>
      </section>

      <section class="panel wide">
        <h2>Model Predictions</h2>
        <div id="predictions" class="prediction">
          <div class="model-card">
            <div class="model-name">Logistic Regression</div>
            <div class="direction">-</div>
            <div class="confidence">Confidence: -</div>
          </div>
          <div class="model-card">
            <div class="model-name">Random Forest</div>
            <div class="direction">-</div>
            <div class="confidence">Confidence: -</div>
          </div>
        </div>
      </section>
    </div>
  </main>

  <script>
    let lastPrediction = null;
    let checkTimer = null;

    const statusEl = document.getElementById("status");
    const getBtn = document.getElementById("getBtn");
    const checkBtn = document.getElementById("checkBtn");

    function fmt(value) {
      if (value === null || value === undefined || value === "") return "-";
      if (typeof value === "number") return value.toLocaleString(undefined, { maximumFractionDigits: 6 });
      return value;
    }

    function setStatus(message, kind = "") {
      statusEl.className = kind ? `status ${kind}` : "status";
      statusEl.textContent = message;
    }

    function renderBlock(block, rowsUsed) {
      document.getElementById("blockDetails").innerHTML = `
        <div>Block</div><div>${fmt(block.block_number)}</div>
        <div>Timestamp</div><div>${fmt(block.block_datetime_utc)}</div>
        <div>Transactions</div><div>${fmt(block.txn_count)}</div>
        <div>Gas Used</div><div>${fmt(block.gas_used)}</div>
        <div>Gas Limit</div><div>${fmt(block.gas_limit)}</div>
        <div>Base Fee</div><div>${fmt(block.base_fee_gwei)} Gwei</div>
        <div>Congestion Ratio</div><div>${fmt(block.congestion_ratio)}</div>
        <div>Rows Used</div><div>${fmt(rowsUsed)}</div>
      `;
    }

    function renderPredictions(predictions) {
      const cards = predictions.map((prediction) => {
        const cls = prediction.direction === "higher" ? "higher" : "lower";
        return `
          <div class="model-card">
            <div class="model-name">${prediction.model}</div>
            <div class="direction ${cls}">${prediction.direction.toUpperCase()}</div>
            <div class="confidence">Confidence: ${(prediction.confidence * 100).toFixed(1)}%</div>
            <div class="confidence">P(higher): ${(prediction.probability_higher * 100).toFixed(1)}%</div>
          </div>
        `;
      }).join("");
      document.getElementById("predictions").innerHTML = cards;
    }

    function renderVerification(data) {
      const result = data.match === null ? "-" : (data.match ? "Matched prediction" : "Did not match prediction");
      document.getElementById("verification").innerHTML = `
        <div>Next block</div><div>${fmt(data.next_block_number)}</div>
        <div>Actual direction</div><div>${fmt(data.actual_direction)}</div>
        <div>Result</div><div>${result}</div>
      `;
    }

    async function postJson(url, payload = {}) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    async function getCurrentBlock() {
      if (checkTimer) clearInterval(checkTimer);
      getBtn.disabled = true;
      checkBtn.disabled = true;
      setStatus("Fetching recent Etherscan rows and running both models...");
      try {
        const data = await postJson("/api/current-block");
        lastPrediction = data;
        renderBlock(data.current_block, data.rows_used);
        renderPredictions(data.predictions);
        checkBtn.disabled = false;
        setStatus("Prediction ready. Waiting 12 seconds before checking the next block...");
        let seconds = 12;
        checkTimer = setInterval(() => {
          seconds -= 1;
          if (seconds > 0) {
            setStatus(`Prediction ready. Checking next block in ${seconds} seconds...`);
          } else {
            clearInterval(checkTimer);
            checkNextBlock();
          }
        }, 1000);
      } catch (error) {
        setStatus(error.message, "error");
      } finally {
        getBtn.disabled = false;
      }
    }

    async function checkNextBlock() {
      if (!lastPrediction) return;
      checkBtn.disabled = true;
      setStatus("Fetching latest block page to verify the next block...");
      try {
        const primary = lastPrediction.predictions[0];
        const data = await postJson("/api/check-next", {
          block_number: lastPrediction.current_block.block_number,
          gas_used: lastPrediction.current_block.gas_used,
          predicted_direction: primary.direction,
        });
        renderVerification(data);
        if (data.match === null) {
          setStatus(data.message || "Next block not available yet.", "warn");
          checkBtn.disabled = false;
        } else {
          setStatus("Verification complete.");
        }
      } catch (error) {
        setStatus(error.message, "error");
        checkBtn.disabled = false;
      }
    }

    getBtn.addEventListener("click", getCurrentBlock);
    checkBtn.addEventListener("click", checkNextBlock);
  </script>
</body>
</html>
"""


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "n/a", "na", "-"}:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def row_value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None:
        return ""
    return value


def public_block_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_number": int(to_float(row_value(row, "block_number")) or 0),
        "block_datetime_utc": row_value(row, "block_datetime_utc"),
        "txn_count": int(to_float(row_value(row, "txn_count")) or 0),
        "gas_used": int(to_float(row_value(row, "gas_used")) or 0),
        "gas_limit": int(to_float(row_value(row, "gas_limit")) or 0),
        "base_fee_gwei": to_float(row_value(row, "base_fee_gwei")),
        "congestion_ratio": to_float(row_value(row, "congestion_ratio")),
    }


def fetch_live_rows(pages: int = LIVE_PAGES) -> list[dict[str, Any]]:
    session = build_session()
    raw_rows: list[dict[str, str]] = []
    for page in range(1, pages + 1):
        url = page_url(page, PAGE_SIZE)
        scraped_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        html = fetch_page(session, url)
        raw_rows.extend(parse_blocks_from_html(html, url, page, scraped_at_utc))

    cleaned_rows, _summary = clean_blocks(raw_rows)
    engineered_rows = add_engineered_features(cleaned_rows)
    if not engineered_rows:
        raise RuntimeError("No live rows could be cleaned and engineered from Etherscan.")
    return engineered_rows


def load_artifacts() -> dict[str, Any]:
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(
            "Missing model_outputs/model_artifacts.pkl. Run `python3 model_training.py` after installing requirements."
        )
    try:
        with ARTIFACT_PATH.open("rb") as handle:
            artifact = pickle.load(handle)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Cannot load model artifact because Python package `{exc.name}` is missing. "
            "Run `python3 -m pip install -r requirements.txt`."
        ) from exc

    feature_cols = artifact.get("feature_cols", [])
    if any(str(col).startswith("target_") for col in feature_cols):
        raise RuntimeError(
            "Model artifact includes target columns as features. Rerun `python3 model_training.py` to create a clean artifact."
        )
    return artifact


def feature_vector(row: dict[str, Any], feature_cols: list[str]) -> list[float]:
    values: list[float] = []
    missing: list[str] = []
    for column in feature_cols:
        value = to_float(row.get(column))
        if value is None:
            missing.append(column)
        else:
            values.append(value)
    if missing:
        raise RuntimeError(f"Current block is missing model features: {', '.join(missing[:6])}")
    return values


def predict_current_block(row: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Cannot run prediction because `numpy` is missing. Run `python3 -m pip install -r requirements.txt`.") from exc

    artifact = load_artifacts()
    feature_cols = artifact["feature_cols"]
    vector = feature_vector(row, feature_cols)
    x_array = np.array([vector], dtype=np.float64)
    x_scaled = artifact["scaler"].transform(x_array)

    output: list[dict[str, Any]] = []
    for key, label in [
        ("logistic_regression", "Logistic Regression"),
        ("random_forest", "Random Forest"),
    ]:
        model = artifact[key]
        probability_higher = float(model.predict_proba(x_scaled)[0][1])
        direction = "higher" if probability_higher >= 0.5 else "lower"
        confidence = probability_higher if direction == "higher" else 1.0 - probability_higher
        output.append(
            {
                "model": label,
                "direction": direction,
                "confidence": confidence,
                "probability_higher": probability_higher,
            }
        )
    return output


def current_block_response() -> dict[str, Any]:
    rows = fetch_live_rows()
    current = rows[-1]
    return {
        "rows_used": len(rows),
        "current_block": public_block_payload(current),
        "predictions": predict_current_block(current),
    }


def check_next_response(payload: dict[str, Any]) -> dict[str, Any]:
    current_block_number = int(payload["block_number"])
    current_gas_used = float(payload["gas_used"])
    predicted_direction = str(payload.get("predicted_direction", "")).lower()

    rows = fetch_live_rows(pages=1)
    next_row = next(
        (row for row in rows if int(to_float(row.get("block_number")) or 0) == current_block_number + 1),
        None,
    )

    if next_row is None:
        latest_block = int(to_float(rows[-1].get("block_number")) or 0)
        return {
            "next_block_number": current_block_number + 1,
            "actual_direction": "",
            "match": None,
            "message": f"Next block is not visible yet. Latest visible block is {latest_block}. Try again in a few seconds.",
        }

    next_gas_used = float(to_float(next_row.get("gas_used")) or 0.0)
    actual_direction = "higher" if next_gas_used > current_gas_used else "lower"
    return {
        "next_block_number": current_block_number + 1,
        "next_gas_used": int(next_gas_used),
        "actual_direction": actual_direction,
        "match": actual_direction == predicted_direction if predicted_direction else None,
        "message": "",
    }


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in {"/", "/index.html"}:
            self.send_error(404)
            return
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        request_body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(request_body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            json_response(self, 400, {"error": "Invalid JSON request body."})
            return

        try:
            if self.path == "/api/current-block":
                json_response(self, 200, current_block_response())
            elif self.path == "/api/check-next":
                json_response(self, 200, check_next_response(payload))
            else:
                json_response(self, 404, {"error": "Unknown endpoint."})
        except Exception as exc:  # noqa: BLE001 - frontend should receive a readable demo error.
            json_response(self, 500, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Frontend running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping frontend.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
