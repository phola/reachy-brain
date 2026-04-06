# Reachy Brain 🧠🤖

**A fully local AI brain for the [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) robot.**

No cloud APIs. No subscriptions. Everything runs on your Mac.

> ⚠️ **Alpha** — This is an early experiment. It works, it's fun, but expect rough edges. Contributions and ideas welcome.

## What It Does

Turns your Reachy Mini into a conversational companion powered entirely by local AI:

```
Robot mic → Whisper STT → Gemma 4 LLM → Kokoro TTS → Robot speaker
                    + expressive head movements
```

- **Listens** through the robot's microphone (WebRTC)
- **Understands** speech using Whisper (distil-large-v3, ~0.5s)
- **Thinks** using Gemma 4 26B locally via MLX (~0.8s)
- **Speaks** through the robot's speaker using Kokoro TTS (~1.7s)
- **Moves** its head and antennas expressively during conversation
- **Filters noise** using the robot's hardware voice activity detection (DOA)

Total turnaround: ~3 seconds from you finishing a sentence to the robot responding.

## Why Local?

The default Reachy Mini conversation app routes everything through OpenAI's cloud API. This project replaces that with models running on your own hardware:

| Component | Default App | Reachy Brain |
|-----------|------------|--------------|
| STT | OpenAI Realtime API | Whisper (MLX, on-device) |
| LLM | GPT-4 / Gemma via OpenRouter | Gemma 4 26B (MLX, on-device) |
| TTS | OpenAI Realtime API | Kokoro 82M (ONNX, on-device) |
| VAD | Server-side | Robot hardware DOA + RMS |
| Latency | Network dependent | ~3s total |
| Cost | Per-token API billing | Free after hardware |
| Privacy | Audio sent to cloud | Nothing leaves your network |

## Requirements

- **Reachy Mini** with firmware v1.6.0+ (WiFi version)
- **Apple Silicon Mac** with 16GB+ RAM (tested on M3 Max 36GB)
- Python 3.13+, [uv](https://github.com/astral-sh/uv)

## Quick Start

### 1. Install dependencies

```bash
# Clone
git clone https://github.com/phola/reachy-brain.git
cd reachy-brain

# Create venv and install Reachy SDK
uv venv --python 3.13
uv pip install reachy-mini openai scipy numpy

# Set up MLX environment (STT + TTS)
cd ~
uv venv mlx-env --python 3.13
source mlx-env/.venv/bin/activate
pip install mlx-lm lightning-whisper-mlx kokoro-onnx soundfile

# Download Kokoro TTS model (~340MB)
curl -L -o mlx-env/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o mlx-env/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### 2. Start the services

```bash
# Terminal 1 — LLM server
source ~/mlx-env/.venv/bin/activate
mlx_lm.server --model mlx-community/gemma-4-26b-a4b-it-4bit --port 1234

# Terminal 2 — STT server (keeps Whisper warm for fast transcription)
source ~/mlx-env/.venv/bin/activate
python3 ~/projects/reachy-brain/stt_server.py

# Terminal 3 — The brain
cd ~/projects/reachy-brain
.venv/bin/python3 hermes_brain.py
```

### 3. Talk to your robot

That's it. The robot will greet you and start listening.

## Files

| File | What it does |
|------|-------------|
| `hermes_brain.py` | Main conversation loop — connects everything |
| `stt_server.py` | Persistent Whisper server (warm model = fast STT) |
| `reachy.py` | Standalone HTTP control bridge for motor control |
| `PLAN.md` | Development roadmap and architecture notes |

## Configuration

Edit the top of `hermes_brain.py`:

```python
REACHY_HOST = "192.168.68.120"    # Your robot's IP
LLM_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
TTS_VOICE = "im_nicola"           # See voice options below
STT_MODEL_SIZE = "small"          # or distil-large-v3 (more accurate)
```

### Voice Options

Kokoro includes 50+ voices across accents:

| Prefix | Accent | Examples |
|--------|--------|----------|
| `af_` | American Female | `af_heart`, `af_bella`, `af_nova`, `af_sky` |
| `am_` | American Male | `am_adam`, `am_puck`, `am_echo` |
| `bf_` | British Female | `bf_emma`, `bf_isabella` |
| `bm_` | British Male | `bm_george`, `bm_fable`, `bm_lewis` |
| `ef_` | European Female | `ef_dora` |
| `em_` | European Male | `em_alex` |
| `ff_` | French Female | `ff_siwis` |
| `if_` | Italian Female | `if_sara` |
| `im_` | Italian Male | `im_nicola` |

## Known Issues

- **GStreamer on macOS**: Set `GST_PLUGIN_SCANNER=""` before running (handled automatically in the script)
- **Whisper hallucinations**: Forced to English language + ASCII filter to prevent gibberish on noise
- **Echo**: The robot can hear itself speak — a mic flush + cooldown after each response mitigates this
- **Kitchen noise**: Hardware DOA helps but isn't perfect — the hybrid DOA+RMS approach works best

## Roadmap

- [x] Full local conversation loop
- [x] Hardware VAD (DOA + RMS hybrid)
- [x] Persistent STT server (warm Whisper)
- [x] Natural TTS with voice selection (Kokoro ONNX)
- [x] Expressive head movements
- [ ] Camera vision — describe what the robot sees, feed as context
- [ ] Intent classification — ignore background chatter
- [ ] Agent escalation — hand off complex questions to a full AI agent with tool access
- [ ] Wake word detection ("Hey Reachy")
- [ ] Idle behaviours — scan room, track people, react to sounds
- [ ] Streaming TTS — start speaking before full generation

## Architecture

```
  Reachy Mini (kitchen)              Mac (M3 Max)
  ┌──────────────────┐               ┌──────────────────────────┐
  │ Microphone ──────►──── WebRTC ──►│ Whisper STT (0.5s)       │
  │                  │               │     ↓                    │
  │                  │               │ Gemma 4 LLM (0.8s)       │
  │                  │               │     ↓                    │
  │ Speaker ◄────────◄──── WebRTC ──◄│ Kokoro TTS (1.7s)        │
  │                  │               │     +                    │
  │ Head motors ◄────◄──── HTTP ────◄│ Expressions              │
  │ Camera ──────────►──── WebRTC ──►│ (planned: vision)        │
  └──────────────────┘               └──────────────────────────┘
```

## Standalone Motor Control

`reachy.py` can be used independently to control the robot:

```python
from reachy import ReachyMini
r = ReachyMini()
r.look("left")
r.nod()
r.emote("happy")
r.scan()
```

Or from the command line:

```bash
python3 reachy.py emote curious
python3 reachy.py look up
python3 reachy.py nod
```

## Credits

Built with:
- [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) by Pollen Robotics
- [MLX](https://github.com/ml-explore/mlx) by Apple
- [Gemma 4](https://ai.google.dev/gemma) by Google
- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad
- [Whisper](https://github.com/openai/whisper) by OpenAI
- [lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx) by Mustafa Aljadery
- [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) by thewh1teagle

## License

MIT
