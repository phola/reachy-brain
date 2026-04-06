# Hermes Brain for Reachy Mini — Project Plan

## Architecture

```
  Reachy Mini (kitchen)              Mac (M3 Max, 36GB)
  ┌──────────────────┐               ┌──────────────────────────┐
  │ Microphone ──────►──── WebRTC ──►│ STT (Whisper, local)     │
  │                  │               │     ↓                    │
  │                  │               │ Gemma 4 (local LLM)      │
  │                  │               │   or Hermes (escalation)  │
  │                  │               │     ↓                    │
  │ Speaker ◄────────◄──── WebRTC ──◄│ TTS (Kokoro ONNX)       │
  │                  │               │     +                    │
  │ Motors ◄─────────◄──── HTTP ────◄│ Expressions/movements   │
  │ Camera ──────────►──── WebRTC ──►│ Vision context (planned) │
  └──────────────────┘               └──────────────────────────┘
```

## Completed

- [x] HTTP control bridge (reachy.py) — movements, emotions, gestures
- [x] SDK connection via WebRTC (firmware v1.6.0)
- [x] Full audio pipeline — robot mic → Mac → robot speaker
- [x] Gemma 4 26B MoE on mlx_lm.server (port 1234, ~69 tok/s)
- [x] STT — Whisper small via lightning-whisper-mlx (~1.6s)
- [x] TTS — Kokoro ONNX 82M (~1.7s, natural voice)
- [x] Hybrid VAD — robot hardware DOA + RMS (filters dishwasher noise)
- [x] Head expressions during conversation (listening/thinking/speaking/happy)
- [x] Auto motor enable on startup
- [x] Echo suppression (mic flush after speaking)
- [x] English-only STT + hallucination filter
- [x] GStreamer macOS fix (GST_PLUGIN_SCANNER="")
- [x] Hermes skill saved (reachy-mini-control)

## In Progress

- [ ] **Optimise STT latency** — keep Whisper warm in memory instead of
      subprocess-per-call. Could use a persistent server or in-process loading.
      Target: shave ~1s off current ~1.6s.

## Next Up

- [ ] **Camera context** — periodic frame grabs from robot camera, describe
      scene via Gemma 4 vision, feed as system prompt context.
      ("person at counter, kettle on, two people in room")

- [ ] **Intent classification** — use Gemma 4 to classify each utterance:
      "directed at robot" vs "background conversation". Only respond to
      directed speech. Possibly use a wake word ("Hey Reachy").

## Later

- [ ] **Hermes API server** — enable the built-in OpenAI-compatible API on
      gateway port 8642. Gives full Hermes agent (tools, memory, web, files).

- [ ] **Escalation path** — Gemma 4 handles casual chat locally. When it
      detects a request needing tools/memory/web, escalate to Hermes.
      Trigger: "Hey Hermes" wake word, or intent classification.
      Robot says "Let me think about that..." while waiting.

- [ ] **Streaming TTS** — start speaking before full generation completes.
      Kokoro generates fast enough this may not be needed.

- [ ] **Better TTS** — try Orpheus 3B via llama.cpp (best quality, ~5-8s
      currently, may improve with WavTokenizer decoder).

- [ ] **Wake word** — "Hey Reachy" to activate from idle, avoid processing
      background conversations entirely.

- [ ] **Idle behaviours** — robot scans room, reacts to sounds, tracks
      people with head when not in conversation.

- [ ] **Persistent memory** — Hermes remembers past conversations with
      family members across sessions.

## Files

```
~/projects/reachy-hermes/
├── hermes_brain.py      # Main conversation app (v0.2)
├── reachy.py            # HTTP control bridge (standalone)
├── conv-app/            # Cloned official conversation app (reference)
├── PLAN.md              # This file
└── .venv/               # Python 3.13, reachy-mini SDK v1.6.0
```

## Running

```bash
# Terminal 1: Start Gemma 4
source ~/mlx-env/.venv/bin/activate
mlx_lm.server --model mlx-community/gemma-4-26b-a4b-it-4bit --port 1234

# Terminal 2: Run the brain
cd ~/projects/reachy-hermes
.venv/bin/python3 hermes_brain.py

# Or background with logging:
PYTHONUNBUFFERED=1 .venv/bin/python3 hermes_brain.py > /tmp/reachy_brain.log 2>&1 &
tail -f /tmp/reachy_brain.log
```

## Dependencies

- Reachy Mini firmware v1.6.0+ (matching SDK)
- mlx_lm.server with Gemma 4 on port 1234
- lightning-whisper-mlx (in ~/mlx-env/.venv)
- kokoro-onnx + model files (in ~/mlx-env/)
- GStreamer fix: GST_PLUGIN_SCANNER="" before import

## Key Learnings

- SDK v1.3.1 (Zenoh) can't connect remotely; v1.6.0 (WebSocket) matches firmware
- GST_PLUGIN_SCANNER="" fixes GStreamer Python framework path on Homebrew
- Whisper hallucinates foreign text on noise — force language='en' + ASCII filter
- Robot DOA `speech_detected` alone is too flickery — hybrid DOA+RMS works best
- Kokoro via f5-tts-mlx has dependency hell; kokoro-onnx is clean and fast
- Must flush mic buffer after robot speaks to avoid echo loops
- Hermes has built-in OpenAI-compatible API server (port 8642) for escalation
