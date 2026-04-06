# Fully local conversation app alternative (no cloud API)

I built an open-source alternative to the default conversation app that runs entirely locally — no OpenAI API key needed.

**How it works:**
- STT: Whisper distil-large-v3 via MLX (~0.5s)
- LLM: Gemma 4 26B via mlx_lm.server (~0.8s)
- TTS: Kokoro 82M via ONNX (~1.7s)
- VAD: Robot's hardware DOA + RMS hybrid
- Audio: WebRTC through the robot's own mic and speaker

Total latency is about 3 seconds end-to-end. Everything runs on an M3 Max MacBook.

It uses the reachy-mini SDK (v1.6.0) for WebRTC audio streaming and the HTTP API for motor control — the robot moves its head and antennas expressively during conversation.

Still early/alpha but works well enough for my kids to have conversations with it in the kitchen.

**Repo:** https://github.com/phola/reachy-brain

Would love feedback from other Reachy Mini owners. Especially interested in:
- Whether the WebRTC audio setup works on other machines
- Ideas for better voice activity detection
- What other local models people have tried

Happy to answer questions about the setup.
