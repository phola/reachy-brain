"""
Reachy Mini control bridge for Hermes.

Wraps the Reachy Mini HTTP API into a simple Python interface.
Use from Hermes execute_code or as a standalone module.

Usage:
    from reachy import ReachyMini
    r = ReachyMini()          # auto-discovers or uses default IP
    r.look_at("left")         # natural language directions
    r.nod()                   # gestures
    r.shake_head()
    r.emote("curious")        # compound expressions
    r.say("Hello!")            # TTS via system speaker
    r.status()                # full state dump
"""

import json
import time
import urllib.request
import urllib.error
import math
from typing import Optional

DEFAULT_HOST = "192.168.68.120"
DEFAULT_PORT = 8000


class ReachyMini:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.base = f"http://{host}:{port}"
        self._check_connection()

    def _check_connection(self):
        try:
            self._get("/api/daemon/status")
        except Exception as e:
            raise ConnectionError(f"Cannot reach Reachy Mini at {self.base}: {e}")

    def _get(self, path: str) -> dict:
        url = f"{self.base}{path}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, data: dict = None) -> dict:
        url = f"{self.base}{path}"
        body = json.dumps(data).encode() if data else b""
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"} if data else {}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    # ── Status ──────────────────────────────────────────────

    def status(self) -> dict:
        """Get daemon status (state, backend ready, version, etc.)."""
        return self._get("/api/daemon/status")

    def state(self) -> dict:
        """Get full robot state (head pose, body yaw, antennas)."""
        return self._get("/api/state/full")

    def head_pose(self) -> dict:
        """Get current head pose (x, y, z, roll, pitch, yaw)."""
        return self._get("/api/state/present_head_pose")

    def doa(self) -> dict:
        """Direction of arrival from microphone + speech detection."""
        return self._get("/api/state/doa")

    def is_speaking(self) -> bool:
        """Check if someone is currently speaking (via DOA)."""
        return self.doa().get("speech_detected", False)

    def motor_status(self) -> dict:
        return self._get("/api/motors/status")

    # ── Daemon control ──────────────────────────────────────

    def wake_up(self):
        """Start daemon and wake up the robot."""
        return self._post("/api/daemon/start?wake_up=true")

    def sleep(self):
        """Stop daemon and put robot to sleep."""
        return self._post("/api/daemon/stop?goto_sleep=true")

    def play_wake_up(self):
        """Play the wake-up animation."""
        return self._post("/api/move/play/wake_up")

    def play_sleep(self):
        """Play the go-to-sleep animation."""
        return self._post("/api/move/play/goto_sleep")

    # ── Movement primitives ─────────────────────────────────

    def goto(
        self,
        x: float = 0, y: float = 0, z: float = 0,
        roll: float = 0, pitch: float = 0, yaw: float = 0,
        antennas: tuple = None,
        body_yaw: float = None,
        duration: float = 1.0,
        interpolation: str = "minjerk",
        wait: bool = True,
    ) -> str:
        """
        Move to an absolute pose.

        Head pose values are in radians (roughly -0.5 to 0.5 for most axes).
        Antennas: tuple of (left, right) in radians.
        Body yaw: body rotation in radians.

        Returns move UUID.
        """
        data = {
            "head_pose": {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw},
            "duration": duration,
            "interpolation": interpolation,
        }
        if antennas is not None:
            data["antennas"] = list(antennas)
        if body_yaw is not None:
            data["body_yaw"] = body_yaw
        result = self._post("/api/move/goto", data)
        if wait:
            time.sleep(duration + 0.2)
        return result.get("uuid", "")

    def center(self, duration: float = 1.0):
        """Return head to center position."""
        return self.goto(duration=duration)

    def stop_move(self):
        """Stop any currently running move."""
        return self._post("/api/move/stop", {})

    # ── Natural direction commands ──────────────────────────

    DIRECTIONS = {
        "left":       {"yaw": 0.4},
        "right":      {"yaw": -0.4},
        "up":         {"pitch": -0.3},
        "down":       {"pitch": 0.25},
        "up-left":    {"pitch": -0.25, "yaw": 0.3},
        "up-right":   {"pitch": -0.25, "yaw": -0.3},
        "down-left":  {"pitch": 0.2, "yaw": 0.3},
        "down-right": {"pitch": 0.2, "yaw": -0.3},
        "center":     {},
        "tilt-left":  {"roll": 0.3},
        "tilt-right": {"roll": -0.3},
    }

    def look(self, direction: str, duration: float = 1.0):
        """
        Look in a named direction.
        Options: left, right, up, down, up-left, up-right,
                 down-left, down-right, center, tilt-left, tilt-right
        """
        d = direction.lower().replace(" ", "-")
        if d not in self.DIRECTIONS:
            raise ValueError(f"Unknown direction '{direction}'. Options: {list(self.DIRECTIONS.keys())}")
        return self.goto(**self.DIRECTIONS[d], duration=duration)

    # ── Gestures ────────────────────────────────────────────

    def nod(self, times: int = 3, speed: float = 0.3):
        """Nod yes."""
        for _ in range(times):
            self.goto(pitch=-0.15, duration=speed, wait=True)
            self.goto(pitch=0.05, duration=speed, wait=True)
        self.center(duration=speed)

    def shake_head(self, times: int = 3, speed: float = 0.3):
        """Shake head no."""
        for _ in range(times):
            self.goto(yaw=-0.2, duration=speed, wait=True)
            self.goto(yaw=0.2, duration=speed, wait=True)
        self.center(duration=speed)

    def tilt_curious(self, duration: float = 0.8):
        """Curious head tilt to the right."""
        self.goto(roll=-0.25, pitch=-0.1, duration=duration)

    def perk_up(self, duration: float = 0.5):
        """Quick upward look — attention/surprise."""
        self.goto(pitch=-0.25, duration=duration * 0.4, wait=True)
        time.sleep(0.3)
        self.center(duration=duration)

    def droop(self, duration: float = 1.0):
        """Sad/tired drooping head."""
        self.goto(pitch=0.25, roll=0.05, duration=duration)

    def wiggle_antennas(self, times: int = 3, speed: float = 0.2):
        """Wiggle antennas excitedly."""
        for _ in range(times):
            self.goto(antennas=(-0.5, 0.5), duration=speed, wait=True)
            self.goto(antennas=(0.5, -0.5), duration=speed, wait=True)
        self.goto(antennas=(0, 0), duration=speed)

    def scan(self, duration: float = 1.5):
        """Scan the room — slow look left to right."""
        self.goto(yaw=0.5, duration=duration, wait=True)
        self.goto(yaw=-0.5, duration=duration * 2, wait=True)
        self.center(duration=duration)

    # ── Compound expressions ────────────────────────────────

    def emote(self, emotion: str, intensity: float = 1.0):
        """
        Play a compound expression.
        Options: happy, sad, curious, surprised, angry, sleepy,
                 excited, confused, attentive, bored
        """
        s = 0.3 + (1 - intensity) * 0.5  # speed scales with intensity
        i = intensity

        emotions = {
            "happy": lambda: (
                self.goto(pitch=-0.1 * i, duration=s, wait=True),
                self.wiggle_antennas(times=2, speed=s * 0.6),
                self.nod(times=2, speed=s),
                self.center(duration=s),
            ),
            "sad": lambda: (
                self.goto(pitch=0.2 * i, roll=0.05 * i, duration=s * 2, wait=True),
                self.goto(antennas=(-0.3 * i, -0.3 * i), duration=s),
                time.sleep(1),
                self.center(duration=s * 2),
                self.goto(antennas=(0, 0), duration=s),
            ),
            "curious": lambda: (
                self.tilt_curious(duration=s),
                time.sleep(0.5),
                self.goto(pitch=-0.15 * i, roll=-0.2 * i, duration=s, wait=True),
                time.sleep(0.5),
                self.center(duration=s),
            ),
            "surprised": lambda: (
                self.perk_up(duration=s),
                self.wiggle_antennas(times=2, speed=s * 0.4),
                self.center(duration=s),
            ),
            "angry": lambda: (
                self.goto(pitch=0.1 * i, duration=s * 0.5, wait=True),
                self.shake_head(times=2, speed=s * 0.5),
                self.center(duration=s),
            ),
            "sleepy": lambda: (
                self.goto(pitch=0.2 * i, roll=0.1 * i, duration=s * 3, wait=True),
                time.sleep(0.5),
                self.goto(pitch=0.15 * i, roll=-0.08 * i, duration=s * 2, wait=True),
                self.center(duration=s * 2),
            ),
            "excited": lambda: (
                self.wiggle_antennas(times=3, speed=s * 0.3),
                self.nod(times=3, speed=s * 0.5),
                self.goto(pitch=-0.15 * i, duration=s * 0.5, wait=True),
                self.center(duration=s),
            ),
            "confused": lambda: (
                self.tilt_curious(duration=s),
                time.sleep(0.3),
                self.goto(roll=0.2 * i, duration=s, wait=True),
                time.sleep(0.3),
                self.tilt_curious(duration=s),
                self.center(duration=s),
            ),
            "attentive": lambda: (
                self.goto(pitch=-0.1 * i, duration=s, wait=True),
                self.goto(antennas=(0.3 * i, 0.3 * i), duration=s),
            ),
            "bored": lambda: (
                self.scan(duration=s * 2),
                self.droop(duration=s * 2),
                time.sleep(1),
                self.center(duration=s * 2),
            ),
        }

        e = emotion.lower()
        if e not in emotions:
            raise ValueError(f"Unknown emotion '{emotion}'. Options: {list(emotions.keys())}")
        emotions[e]()

    # ── Audio ───────────────────────────────────────────────

    def volume(self) -> int:
        """Get current volume level."""
        return self._get("/api/volume/current").get("volume", 0)

    def set_volume(self, level: int):
        """Set volume (0-100)."""
        return self._post("/api/volume/set", {"volume": level})

    def test_sound(self):
        """Play a test sound on the robot speaker."""
        return self._post("/api/volume/test-sound")

    # ── Apps ────────────────────────────────────────────────

    def list_apps(self) -> list:
        """List available apps from HuggingFace."""
        return self._get("/api/apps/list-available")

    def start_app(self, name: str):
        """Start an installed app by name."""
        return self._post(f"/api/apps/start-app/{name}")

    def stop_app(self):
        """Stop the currently running app."""
        return self._post("/api/apps/stop-current-app")

    def app_status(self) -> dict:
        """Get current app status."""
        return self._get("/api/apps/current-app-status")

    # ── Turn toward sound ───────────────────────────────────

    def turn_to_sound(self, duration: float = 1.0):
        """Turn head toward detected sound direction (using DOA)."""
        d = self.doa()
        angle = d.get("angle", 0)
        # DOA angle is in radians, map to yaw
        # Clamp to safe range
        yaw = max(-0.5, min(0.5, angle - math.pi / 2))
        return self.goto(yaw=yaw, duration=duration)

    # ── Convenience ─────────────────────────────────────────

    def __repr__(self):
        try:
            s = self.status()
            return f"ReachyMini(state={s['state']}, v{s['version']}, ip={s['wlan_ip']})"
        except:
            return f"ReachyMini(base={self.base})"


# ── Quick CLI test ──────────────────────────────────────────

if __name__ == "__main__":
    import sys

    r = ReachyMini()
    print(repr(r))

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "status":
            print(json.dumps(r.status(), indent=2))
        elif cmd == "state":
            print(json.dumps(r.state(), indent=2))
        elif cmd == "look" and len(sys.argv) > 2:
            r.look(sys.argv[2])
        elif cmd == "nod":
            r.nod()
        elif cmd == "shake":
            r.shake_head()
        elif cmd == "emote" and len(sys.argv) > 2:
            r.emote(sys.argv[2])
        elif cmd == "scan":
            r.scan()
        elif cmd == "sleep":
            r.play_sleep()
        elif cmd == "wake":
            r.play_wake_up()
        elif cmd == "center":
            r.center()
        elif cmd == "wiggle":
            r.wiggle_antennas()
        elif cmd == "sound":
            r.turn_to_sound()
        elif cmd == "test-sound":
            r.test_sound()
        else:
            print(f"Unknown command: {cmd}")
            print("Commands: status state look <dir> nod shake emote <emotion> scan sleep wake center wiggle sound test-sound")
    else:
        print("Commands: status state look <dir> nod shake emote <emotion> scan sleep wake center wiggle sound test-sound")
