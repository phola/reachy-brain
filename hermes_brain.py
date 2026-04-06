#!/usr/bin/env python3
"""
Hermes Brain for Reachy Mini — v0.2

Full local conversation loop via robot's mic and speaker:
  Robot mic → WebRTC → STT (Whisper) → Gemma 4 → TTS (Kokoro) → WebRTC → Robot speaker
  + expressive head movements and antenna animations

Requirements:
  - Reachy Mini running and awake (firmware v1.6.0+)
  - mlx_lm.server with Gemma 4 on port 1234
  - ~/mlx-env with mlx_audio (STT + TTS)

Usage:
  cd ~/projects/reachy-hermes
  .venv/bin/python3 hermes_brain.py
"""

import os
import sys
import time
import json
import wave
import asyncio
import logging
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import urllib.request

# ── Fix GStreamer on macOS with Homebrew Python ─────────────
os.environ["GST_PLUGIN_SCANNER"] = ""
os.environ.setdefault("GST_PLUGIN_SYSTEM_PATH", "")

# ── Config ──────────────────────────────────────────────────

REACHY_HOST = "192.168.68.120"
LLM_URL = "http://localhost:1234/v1/chat/completions"
LLM_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"

# mlx-env paths for STT/TTS
MLX_VENV = Path.home() / "mlx-env" / ".venv" / "bin"
MLX_PYTHON = str(MLX_VENV / "python3")

# STT config (lightning-whisper-mlx — runs in-process via mlx-env)
STT_MODEL_SIZE = "small"  # small/medium/large-v3-turbo

# TTS config (Kokoro ONNX — fast, natural voice)
KOKORO_MODEL = str(Path.home() / "mlx-env" / "kokoro-v1.0.onnx")
KOKORO_VOICES = str(Path.home() / "mlx-env" / "voices-v1.0.bin")
TTS_VOICE = "im_nicola"

# Audio config
SPEECH_MIN_DURATION = 1.0      # minimum seconds of speech to process
SILENCE_THRESHOLD = 0.015      # RMS threshold for speech detection
SILENCE_AFTER_SPEECH = 2.0     # seconds of silence after speech to trigger processing
DOA_POLL_INTERVAL = 0.15       # how often to check DOA

# Conversation
SYSTEM_PROMPT = """You are Reachy Mini, a small expressive tabletop robot in the family kitchen.
You are friendly, curious, and helpful. Keep responses SHORT — 1-3 sentences max.
You're having a spoken conversation, not writing an essay.
When someone asks you a question, ANSWER it directly. Don't deflect or ask them to tell you instead.
Be natural, warm, and a bit playful."""

MAX_HISTORY = 10  # conversation turns to keep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet down noisy loggers
logging.getLogger("reachy_mini").setLevel(logging.WARNING)
logging.getLogger("reachy_mini.media.audio_control_utils").setLevel(logging.CRITICAL)
log = logging.getLogger("hermes-brain")


# ── HTTP helpers (for robot expressions) ────────────────────

def reachy_post(path: str, data: dict = None) -> dict:
    url = f"http://{REACHY_HOST}:8000{path}"
    body = json.dumps(data).encode() if data else b""
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log.debug(f"Reachy API error: {e}")
        return {}


def reachy_get(path: str) -> dict:
    url = f"http://{REACHY_HOST}:8000{path}"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def is_speech_detected() -> bool:
    """Check robot's hardware VAD via DOA endpoint."""
    doa = reachy_get("/api/state/doa")
    return doa.get("speech_detected", False)


def reachy_goto(pitch=0, yaw=0, roll=0, antennas=None, duration=0.8):
    data = {
        "head_pose": {"x": 0, "y": 0, "z": 0, "roll": roll, "pitch": pitch, "yaw": yaw},
        "duration": duration,
        "interpolation": "minjerk",
    }
    if antennas:
        data["antennas"] = antennas
    return reachy_post("/api/move/goto", data)


# ── Robot expressions (non-blocking) ───────────────────────

def express(emotion: str):
    """Trigger a robot expression in a background thread."""
    def _do():
        try:
            if emotion == "listening":
                reachy_goto(pitch=-0.08, roll=0.05, duration=0.5)
            elif emotion == "thinking":
                reachy_goto(pitch=-0.05, roll=-0.15, duration=0.6)
            elif emotion == "speaking":
                reachy_goto(pitch=-0.05, roll=0, duration=0.4)
            elif emotion == "happy":
                reachy_goto(pitch=-0.1, antennas=[-0.3, 0.3], duration=0.4)
                time.sleep(0.5)
                reachy_goto(antennas=[0.3, -0.3], duration=0.3)
                time.sleep(0.4)
                reachy_goto(antennas=[0, 0], duration=0.3)
            elif emotion == "idle":
                reachy_goto(duration=0.8)
        except Exception:
            pass
    threading.Thread(target=_do, daemon=True).start()


# ── LLM (local Gemma 4) ────────────────────────────────────

class Conversation:
    def __init__(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})

        if len(self.messages) > MAX_HISTORY * 2 + 1:
            self.messages = self.messages[:1] + self.messages[-(MAX_HISTORY * 2):]

        payload = {
            "model": LLM_MODEL,
            "messages": self.messages,
            "max_tokens": 150,
            "temperature": 0.7,
        }

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            LLM_URL, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                reply = data["choices"][0]["message"]["content"]
                self.messages.append({"role": "assistant", "content": reply})
                return reply
        except Exception as e:
            log.error(f"LLM error: {e}")
            return "Sorry, I had a little hiccup. Could you say that again?"


# ── STT (lightning-whisper-mlx) ──────────────────────────────

_whisper = None

def get_whisper():
    """Lazy-load whisper model (runs in mlx-env subprocess)."""
    global _whisper
    if _whisper is None:
        log.info(f"Loading Whisper ({STT_MODEL_SIZE})...")
        # Import from mlx-env — run as subprocess to use correct venv
        pass
    return _whisper


STT_SERVER_URL = "http://localhost:8678/transcribe"

def transcribe(audio_path: str) -> str:
    """Transcribe audio — uses STT server if running, else subprocess fallback."""
    # Try the persistent STT server first (model already warm = fast)
    try:
        with open(audio_path, "rb") as f:
            wav_data = f.read()
        req = urllib.request.Request(
            STT_SERVER_URL, data=wav_data, method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("text", "")
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        pass  # server not running, fall back to subprocess

    # Fallback: subprocess (cold start, ~1s slower)
    try:
        result = subprocess.run(
            [
                MLX_PYTHON, "-c",
                f"""
from lightning_whisper_mlx import LightningWhisperMLX
w = LightningWhisperMLX(model='{STT_MODEL_SIZE}', batch_size=12)
r = w.transcribe('{audio_path}', language='en')
print(r['text'].strip())
"""
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            log.error(f"STT error: {result.stderr[:300]}")
            return ""
        text = result.stdout.strip()
        if text and all(ord(c) < 128 for c in text):
            return text
        return ""
    except subprocess.TimeoutExpired:
        log.error("STT timed out")
        return ""


# ── TTS (Kokoro ONNX) ───────────────────────────────────────

_kokoro = None

def get_kokoro():
    """Lazy-load Kokoro TTS model."""
    global _kokoro
    if _kokoro is None:
        log.info("Loading Kokoro TTS...")
        # Run in subprocess since kokoro-onnx uses the mlx-env
        pass
    return _kokoro


def synthesize(text: str) -> Optional[str]:
    """Synthesize speech using Kokoro ONNX (subprocess), return path to wav."""
    wav_path = tempfile.mktemp(suffix=".wav")

    try:
        result = subprocess.run(
            [
                MLX_PYTHON, "-c",
                f"""
import kokoro_onnx, soundfile as sf
model = kokoro_onnx.Kokoro('{KOKORO_MODEL}', '{KOKORO_VOICES}')
samples, sr = model.create('''{text.replace("'", "\\'")}''', voice='{TTS_VOICE}', speed=1.0)
sf.write('{wav_path}', samples, sr)
"""
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            log.error(f"TTS error: {result.stderr[:300]}")
            return None
        return wav_path
    except subprocess.TimeoutExpired:
        log.error("TTS timed out")
        return None


# ── Audio utilities ─────────────────────────────────────────

def rms(audio: np.ndarray) -> float:
    """Calculate RMS of audio array."""
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def save_wav(audio_data: np.ndarray, sample_rate: int, path: str):
    """Save float32 audio as 16-bit mono WAV."""
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]  # take first channel
    audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def load_wav_as_float32(path: str) -> tuple:
    """Load WAV file, return (float32 mono array, sample_rate)."""
    with wave.open(path, "r") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    if sw == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    elif sw == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483647.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")

    if n_ch > 1:
        samples = samples.reshape(-1, n_ch)[:, 0]

    return samples, sr


# ── Play audio through robot speaker ───────────────────────

def speak(media, text: str, output_sr: int):
    """Synthesize text and play through robot's speaker via WebRTC."""
    log.info(f"  TTS: synthesizing...")
    t0 = time.time()
    wav_path = synthesize(text)
    if wav_path is None:
        log.warning("TTS failed, skipping speech")
        return
    log.info(f"  TTS: done in {time.time()-t0:.1f}s")

    try:
        audio, sr = load_wav_as_float32(wav_path)

        # Resample to match robot output
        if sr != output_sr:
            from scipy.signal import resample
            new_len = int(len(audio) * output_sr / sr)
            audio = resample(audio, new_len).astype(np.float32)

        # Push to robot speaker in chunks
        chunk_duration = 0.05  # 50ms chunks
        chunk_size = int(output_sr * chunk_duration)

        log.info(f"  Playing {len(audio)/output_sr:.1f}s of audio...")
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            media.push_audio_sample(chunk)
            time.sleep(chunk_duration * 0.9)

        # Small pause after speaking
        time.sleep(0.3)
    except Exception as e:
        log.error(f"Playback error: {e}")
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


# ── Preflight checks ───────────────────────────────────────

def check_reachy() -> bool:
    try:
        req = urllib.request.Request(f"http://{REACHY_HOST}:8000/api/daemon/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = json.loads(resp.read())
        if status.get("state") != "running":
            log.error(f"Reachy daemon not running (state: {status.get('state')})")
            return False
        log.info(f"Reachy Mini: v{status.get('version')}, {status.get('wlan_ip')}")
        return True
    except Exception as e:
        log.error(f"Cannot reach Reachy at {REACHY_HOST}: {e}")
        return False


def check_llm() -> bool:
    try:
        req = urllib.request.Request("http://localhost:1234/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            models = json.loads(resp.read())
        log.info(f"LLM: {models['data'][0]['id']}")
        return True
    except Exception as e:
        log.error(f"LLM not running on :1234 — start with:")
        log.error("  source ~/mlx-env/.venv/bin/activate && mlx_lm.server --model mlx-community/gemma-4-26b-a4b-it-4bit --port 1234")
        return False


def connect_robot():
    """Connect to robot via SDK with WebRTC media."""
    from reachy_mini import ReachyMini
    robot = ReachyMini(host=REACHY_HOST, connection_mode='network')
    return robot


# ── Main ────────────────────────────────────────────────────

def main():
    log.info("=" * 50)
    log.info("  Hermes Brain for Reachy Mini v0.2")
    log.info("=" * 50)

    if not check_reachy() or not check_llm():
        sys.exit(1)

    # Enable motors and wake up
    log.info("Enabling motors...")
    reachy_post("/api/motors/set_mode/enabled")
    reachy_post("/api/move/play/wake_up")
    time.sleep(2)

    log.info("Connecting via SDK (WebRTC)...")
    try:
        robot = connect_robot()
        media = robot.media
        if media is None:
            raise RuntimeError("No media manager")
        log.info(f"Media backend: {media.backend}")
    except Exception as e:
        log.error(f"SDK connection failed: {e}")
        log.info("Falling back to text mode")
        text_mode(Conversation())
        return

    # Start audio
    log.info("Starting audio pipelines...")
    media.start_recording()
    media.start_playing()
    time.sleep(2)  # let WebRTC settle

    input_sr = media.get_input_audio_samplerate()
    output_sr = media.get_output_audio_samplerate()
    log.info(f"Audio: mic={input_sr}Hz, speaker={output_sr}Hz")

    # Quick mic test
    sample = media.get_audio_sample()
    if sample is not None:
        log.info(f"Mic working: shape={sample.shape}, rms={rms(sample):.4f}")
    else:
        log.warning("No mic data yet — will keep trying")

    # Camera test
    frame = media.get_frame()
    if frame is not None:
        log.info(f"Camera working: {frame.shape}")
    else:
        log.warning("No camera frame yet")

    conv = Conversation()

    # Greet
    express("happy")
    greet = "Hi there! I'm Reachy. What would you like to talk about?"
    log.info(f"Reachy: {greet}")
    speak(media, greet, output_sr)
    # Flush mic echo from greeting
    time.sleep(1.0)
    while media.get_audio_sample() is not None:
        pass

    # Main conversation loop
    # Strategy: DOA triggers start, RMS tracks ongoing speech, silence ends it
    log.info("")
    log.info("Listening (hybrid DOA+RMS)... Ctrl+C to stop")
    log.info("-" * 40)

    audio_buffer = []
    speech_active = False
    silence_start = None
    robot_speaking = False
    last_doa_check = 0

    try:
        while True:
            # Skip while robot is speaking
            if robot_speaking:
                time.sleep(0.05)
                continue

            # Read audio
            sample = media.get_audio_sample()
            if sample is None:
                time.sleep(0.01)
                continue

            mono = sample[:, 0] if sample.ndim > 1 else sample
            level = rms(mono)

            if not speech_active:
                # Check DOA periodically to start listening
                now = time.time()
                if now - last_doa_check > DOA_POLL_INTERVAL:
                    last_doa_check = now
                    if is_speech_detected() and level > SILENCE_THRESHOLD:
                        speech_active = True
                        audio_buffer = [mono.copy()]
                        silence_start = None
                        log.info(f"🎤 Speech detected (DOA + RMS={level:.3f})")
                        express("listening")
            elif speech_active:
                # Already listening — use RMS to track speech
                audio_buffer.append(mono.copy())

                if level > SILENCE_THRESHOLD * 0.5 and is_speech_detected():
                    # Still hearing actual speech (both RMS and DOA confirm)
                    silence_start = None
                elif level > SILENCE_THRESHOLD and silence_start is not None:
                    # RMS is up but DOA says no speech — might be noise
                    # Let the silence timer keep running
                    pass
                else:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_AFTER_SPEECH:
                        # Utterance done
                        speech_active = False
                        silence_start = None

                        total_samples = sum(len(s) for s in audio_buffer)
                        duration = total_samples / input_sr
                        if duration < SPEECH_MIN_DURATION:
                            log.debug(f"Too short ({duration:.1f}s), skipping")
                            audio_buffer = []
                            continue

                        log.info(f"  {duration:.1f}s of speech captured")
                        express("thinking")

                        audio_data = np.concatenate(audio_buffer)
                        audio_buffer = []

                        wav_path = tempfile.mktemp(suffix=".wav")
                        save_wav(audio_data, input_sr, wav_path)

                        # STT
                        log.info("  Transcribing...")
                        t0 = time.time()
                        text = transcribe(wav_path)
                        try:
                            os.unlink(wav_path)
                        except OSError:
                            pass

                        if not text or len(text.strip()) < 2:
                            log.info("  (empty transcription)")
                            express("idle")
                            continue

                        log.info(f"  STT ({time.time()-t0:.1f}s): \"{text}\"")

                        # LLM
                        log.info("  Thinking...")
                        t0 = time.time()
                        reply = conv.chat(text)
                        log.info(f"  LLM ({time.time()-t0:.1f}s): \"{reply}\"")

                        # Speak
                        express("speaking")
                        robot_speaking = True
                        speak(media, reply, output_sr)
                        time.sleep(1.0)
                        while media.get_audio_sample() is not None:
                            pass
                        robot_speaking = False

                        express("happy")
                        time.sleep(0.5)
                        express("idle")

    except KeyboardInterrupt:
        log.info("\nShutting down...")

    finally:
        express("idle")
        try:
            media.stop_recording()
            media.stop_playing()
        except Exception:
            pass


# ── Text mode fallback ──────────────────────────────────────

def text_mode(conv: Conversation):
    """Fallback text-only mode when audio isn't available."""
    log.info("Text mode — type messages, robot responds with movements")
    express("happy")

    try:
        while True:
            text = input("\nYou: ").strip()
            if not text:
                continue
            if text.lower() in ("quit", "exit", "bye"):
                break

            express("thinking")
            t0 = time.time()
            reply = conv.chat(text)
            log.info(f"Reachy ({time.time()-t0:.1f}s): {reply}")
            express("speaking")
            time.sleep(1)
            express("happy")
            time.sleep(0.5)
            express("idle")

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        express("idle")


if __name__ == "__main__":
    main()
