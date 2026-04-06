# Built a fully local AI brain for a robot — Whisper + Gemma 4 + Kokoro on Apple Silicon

I have a Reachy Mini (small tabletop robot by Pollen Robotics) and wanted to see if I could replace its cloud-dependent conversation app with something fully local on my M3 Max.

**The pipeline:**

```
Robot mic → Whisper STT (0.5s) → Gemma 4 26B (0.8s) → Kokoro TTS (1.7s) → Robot speaker
```

~3 second turnaround. No cloud APIs, no subscriptions. The robot sits in the kitchen and my family talks to it.

**Stack:**
- **STT:** Whisper distil-large-v3 via lightning-whisper-mlx. Runs as a persistent server so the model stays warm — first call ~0.4s, subsequent ~0.1-0.5s
- **LLM:** Gemma 4 26B MoE 4-bit via mlx_lm.server. Only 4B params active per token so it's fast
- **TTS:** Kokoro 82M via kokoro-onnx. 50+ voice options, generates faster than real-time. Tried Kokoro via mlx-audio but dependency hell (spacy/phonemizer chain) — the ONNX version just works
- **Audio:** WebRTC streaming through the robot's own mic and speaker using the reachy-mini SDK
- **VAD:** The robot has hardware voice activity detection (direction of arrival mic array) which I combine with RMS threshold. Filters out kitchen noise pretty well

**Lessons learned:**
- Keeping Whisper warm in a server (vs subprocess per call) cut STT from 1.6s to 0.5s — 3x speedup
- Whisper hallucinates foreign text on noise — forcing `language='en'` + filtering non-ASCII fixes it
- The robot hears itself talk — need to flush the mic buffer and add a cooldown after each response
- Hardware VAD alone is flickery, pure RMS triggers on dishwashers. Hybrid (DOA AND RMS) works best

**What's next:**
- Camera vision context (the robot has a camera, feed scene descriptions to the LLM)
- Intent classification to ignore background chatter
- Agent escalation for complex questions (web search, memory, tools)

Repo (MIT): https://github.com/phola/reachy-brain

Anyone else running local AI on physical robots? Curious what setups people have.
