#!/usr/bin/env python3
"""
Persistent Whisper STT server — keeps model warm in memory.
Accepts WAV files via HTTP POST, returns transcription.

Run from mlx-env:
  source ~/mlx-env/.venv/bin/activate
  python3 ~/projects/reachy-hermes/stt_server.py

Endpoint:
  POST http://localhost:8678/transcribe
  Body: WAV file (multipart/form-data or raw bytes)
  Response: {"text": "transcribed text"}
"""

import io
import json
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

from lightning_whisper_mlx import LightningWhisperMLX

MODEL_SIZE = "distil-large-v3"
PORT = 8678

logging.basicConfig(level=logging.INFO, format="%(asctime)s [STT] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("stt-server")

# Load model once at startup
log.info(f"Loading Whisper {MODEL_SIZE}...")
t0 = time.time()
whisper = LightningWhisperMLX(model=MODEL_SIZE, batch_size=12)
log.info(f"Whisper ready in {time.time()-t0:.1f}s")


class STTHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/transcribe":
            self.send_error(404)
            return

        # Read the WAV data from request body
        content_length = int(self.headers.get("Content-Length", 0))
        wav_data = self.rfile.read(content_length)

        if not wav_data:
            self._respond({"text": "", "error": "no data"})
            return

        # Write to temp file (whisper needs a file path)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_data)
            tmp_path = f.name

        try:
            t0 = time.time()
            result = whisper.transcribe(tmp_path, language="en")
            text = result.get("text", "").strip()

            # Filter non-English hallucinations
            if text and not all(ord(c) < 128 for c in text):
                log.info(f"Filtered hallucination: {text[:40]}")
                text = ""

            elapsed = time.time() - t0
            log.info(f"{elapsed:.2f}s | \"{text[:60]}\"" if text else f"{elapsed:.2f}s | (empty)")
            self._respond({"text": text, "elapsed": round(elapsed, 3)})
        except Exception as e:
            log.error(f"Transcription error: {e}")
            self._respond({"text": "", "error": str(e)})
        finally:
            os.unlink(tmp_path)

    def _respond(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress default HTTP logs


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), STTHandler)
    log.info(f"STT server listening on http://localhost:{PORT}/transcribe")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
