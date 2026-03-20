#!/usr/bin/env python3
"""
murmel – push-to-talk dictation daemon for Linux
Transcribes with NVIDIA Parakeet ASR models via NeMo.

Usage:
  murmel start [--model MODEL] [--inject MODE]
  murmel toggle        # send SIGUSR1 to daemon (bind this in Hyprland)
  murmel stop
  murmel config [--set KEY=VALUE ...]
  murmel models        # list available models
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

CONFIG_DIR  = Path.home() / ".config" / "murmel"
CONFIG_FILE = CONFIG_DIR / "config.json"
PID_FILE    = Path("/tmp/murmel.pid")
LOG_FILE    = CONFIG_DIR / "murmel.log"

# NVIDIA Parakeet model registry
MODELS: dict[str, str] = {
    "parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",  # multilingual (25 langs)
    "parakeet-tdt-0.6b":    "nvidia/parakeet-tdt-0.6b-v2",
    "parakeet-tdt-1.1b":    "nvidia/parakeet-tdt-1.1b",
    "parakeet-ctc-0.6b":    "nvidia/parakeet-ctc-0.6b",
    "parakeet-rnnt-0.6b":   "nvidia/parakeet-rnnt-0.6b",
}

DEFAULT_CONFIG: dict = {
    "model":        "parakeet-tdt-0.6b-v3",   # default model key (multilingual)
    "sample_rate":  16000,
    "channels":     1,
    "inject_mode":  "auto",    # auto | wtype | xdotool | clipboard
    "notify":       False,
    "log_level":    "INFO",
    "preload":      True,      # load model on daemon start
    "min_duration": 0.4,       # seconds — discard shorter clips
    "trailing_silence": 0.3,   # seconds of silence to append to recording
}

# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO"):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", "%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(fmt)
    root.addHandler(fh)

log = logging.getLogger("murmel")

# ── Config ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            user = json.load(f)
        return {**DEFAULT_CONFIG, **user}
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG.copy()

def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

# ── Notifications ─────────────────────────────────────────────────────────────

def notify(title: str, body: str = "", urgency: str = "normal",
           icon: str = "audio-input-microphone") -> None:
    import subprocess
    try:
        subprocess.run(
            ["notify-send", "-u", urgency, "-i", icon, "--", title, body],
            capture_output=True, timeout=3,
        )
    except Exception:
        pass  # notifications are best-effort

# ── Text Injection ─────────────────────────────────────────────────────────────

def inject_text(text: str, mode: str = "auto") -> None:
    text = text.strip()
    if not text:
        return

    if mode == "auto":
        if os.environ.get("WAYLAND_DISPLAY"):
            mode = "wtype"
        elif os.environ.get("DISPLAY"):
            mode = "xdotool"
        else:
            mode = "clipboard"

    log.debug(f"Injecting via {mode!r}: {text!r}")

    dispatch = {
        "wtype":    _inject_wtype,
        "xdotool":  _inject_xdotool,
        "clipboard": _inject_clipboard,
    }
    fn = dispatch.get(mode, _inject_clipboard)
    fn(text)


def _inject_wtype(text: str) -> None:
    import subprocess
    try:
        subprocess.run(["wtype", "--", text], check=True, timeout=15)
    except FileNotFoundError:
        log.warning("wtype not found – falling back to clipboard")
        _inject_clipboard(text)
    except subprocess.CalledProcessError as e:
        log.warning(f"wtype failed ({e}) – falling back to clipboard")
        _inject_clipboard(text)


def _inject_xdotool(text: str) -> None:
    import subprocess
    try:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text],
            check=True, timeout=15,
        )
    except FileNotFoundError:
        log.warning("xdotool not found – falling back to clipboard")
        _inject_clipboard(text)
    except subprocess.CalledProcessError as e:
        log.warning(f"xdotool failed ({e}) – falling back to clipboard")
        _inject_clipboard(text)


def _inject_clipboard(text: str) -> None:
    """Copy to clipboard, then simulate Ctrl+V paste."""
    import subprocess

    copied = False
    for copy_cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"],
                     ["xsel", "--clipboard", "--input"]):
        try:
            subprocess.run(copy_cmd, input=text.encode(), check=True, timeout=5)
            copied = True
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    if not copied:
        log.error("Could not copy to clipboard – no wl-copy/xclip/xsel found")
        return

    time.sleep(0.05)

    for paste_cmd in (["wtype", "-M", "ctrl", "v", "-m", "ctrl"],
                      ["xdotool", "key", "--clearmodifiers", "ctrl+v"]):
        try:
            subprocess.run(paste_cmd, check=True, timeout=5)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    log.warning("Could not auto-paste; text is in clipboard – paste manually")

# ── Audio Recording ───────────────────────────────────────────────────────────

class Recorder:
    """Thread-safe audio recorder backed by sounddevice."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels    = channels
        self._frames: list[np.ndarray] = []
        self._active  = False
        self._stream  = None
        self._lock    = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._active

    def start(self) -> None:
        import sounddevice as sd
        with self._lock:
            self._frames = []
            self._active = True
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=self._callback,
                blocksize=1024,
            )
            self._stream.start()
        log.info("Recording started")

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug(f"Audio status: {status}")
        if self._active:
            self._frames.append(indata.copy())

    def stop(self, trailing_silence: float = 0.3) -> np.ndarray | None:
        with self._lock:
            self._active = False
            if self._stream:
                # let the last chunk flush
                time.sleep(trailing_silence)
                self._stream.stop()
                self._stream.close()
                self._stream = None
        if not self._frames:
            return None
        audio = np.concatenate(self._frames, axis=0).flatten()
        duration = len(audio) / self.sample_rate
        log.info(f"Recording stopped — {duration:.2f}s captured")
        return audio

# ── ASR ───────────────────────────────────────────────────────────────────────

class ParakeetASR:
    """Wrapper around NeMo Parakeet ASR models."""

    def __init__(self, model_key: str):
        hf_id = MODELS.get(model_key)
        if hf_id is None:
            # Allow passing a raw HuggingFace ID directly
            hf_id = model_key
        self.model_key = model_key
        self.hf_id     = hf_id
        self._model    = None
        self._lock     = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            log.info(f"Loading model {self.hf_id} …")
            try:
                import nemo.collections.asr as nemo_asr  # noqa: F401
            except ImportError:
                _die(
                    "NeMo ASR not installed.\n"
                    "Install with:  pip install 'nemo_toolkit[asr]'\n"
                    "GPU (recommended): pip install 'nemo_toolkit[asr]' torch torchvision torchaudio"
                )
            self._model = nemo_asr.models.ASRModel.from_pretrained(self.hf_id)
            self._model.eval()
            log.info("Model loaded and ready")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.load()
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            tmp = fh.name
        try:
            sf.write(tmp, audio, sample_rate)
            results = self._model.transcribe([tmp])
            # NeMo returns list[str] or list[Hypothesis] depending on model type
            r = results[0]
            return r if isinstance(r, str) else r.text
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

# ── Daemon ────────────────────────────────────────────────────────────────────

class DictationDaemon:

    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.recorder = Recorder(cfg["sample_rate"], cfg["channels"])
        self.asr      = ParakeetASR(cfg["model"])
        self._lock    = threading.Lock()

    # ── Signal handlers ──────────────────────────────────────────────────────

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGUSR1, self.handle_toggle)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT,  self.handle_shutdown)

    def handle_toggle(self, signum, frame) -> None:
        """SIGUSR1 – toggle recording on/off."""
        with self._lock:
            if self.recorder.is_recording:
                self._stop_and_transcribe()
            else:
                self._start_recording()

    def handle_shutdown(self, signum, frame) -> None:
        log.info("Shutting down…")
        if self.recorder.is_recording:
            self.recorder.stop()
        PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    # ── Recording flow ────────────────────────────────────────────────────────

    def _start_recording(self) -> None:
        if self.cfg["notify"]:
            notify("🎙 murmel", "Recording…")
        self.recorder.start()

    def _stop_and_transcribe(self) -> None:
        audio = self.recorder.stop(self.cfg["trailing_silence"])

        if audio is None:
            notify("murmel", "No audio captured", icon="dialog-warning")
            return

        min_samples = int(self.cfg["min_duration"] * self.cfg["sample_rate"])
        if len(audio) < min_samples:
            notify("murmel", "Clip too short – discarded", icon="dialog-warning")
            return

        # Transcription is slow; run in a background thread
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _transcribe(self, audio: np.ndarray) -> None:
        if self.cfg["notify"]:
            notify("murmel", "Transcribing…", icon="system-run")
        try:
            text = self.asr.transcribe(audio, self.cfg["sample_rate"])
            log.info(f"→ {text!r}")
            if text.strip():
                inject_text(text, self.cfg["inject_mode"])
                if self.cfg["notify"]:
                    preview = text[:70] + ("…" if len(text) > 70 else "")
                    notify("✅ murmel", preview)
            else:
                notify("murmel", "Nothing recognised", icon="dialog-information")
        except Exception as exc:
            log.exception("Transcription failed")
            notify("murmel error", str(exc), urgency="critical", icon="dialog-error")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        PID_FILE.write_text(str(os.getpid()))
        log.info(f"murmel daemon started  PID={os.getpid()}")
        log.info(f"Model : {self.cfg['model']} ({MODELS.get(self.cfg['model'], 'custom')})")
        log.info(f"Inject: {self.cfg['inject_mode']}")
        log.info("Send SIGUSR1 to toggle recording  |  SIGTERM to quit")

        if self.cfg.get("preload"):
            self.asr.load()

        self._install_signal_handlers()
        log.info("Signal handlers registered — ready")

        try:
            while True:
                signal.pause()
        finally:
            PID_FILE.unlink(missing_ok=True)

# ── CLI helpers ───────────────────────────────────────────────────────────────

def _die(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(1)


def _running_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)   # check if process exists
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return None

# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_start(args) -> None:
    if _running_pid():
        _die("murmel is already running. Use 'murmel stop' first.")

    cfg = load_config()
    if args.model:
        cfg["model"] = args.model
    if args.inject:
        cfg["inject_mode"] = args.inject
    if args.no_preload:
        cfg["preload"] = False

    setup_logging(cfg.get("log_level", "INFO"))
    DictationDaemon(cfg).run()


def cmd_toggle(_) -> None:
    pid = _running_pid()
    if not pid:
        _die("murmel is not running. Start it with: murmel start")
    os.kill(pid, signal.SIGUSR1)
    log.debug(f"SIGUSR1 → {pid}")


def cmd_stop(_) -> None:
    pid = _running_pid()
    if not pid:
        print("murmel is not running")
        return
    os.kill(pid, signal.SIGTERM)
    print(f"Stopped murmel (PID {pid})")


def cmd_status(_) -> None:
    pid = _running_pid()
    if pid:
        cfg = load_config()
        print(f"Running  PID={pid}  model={cfg['model']}  inject={cfg['inject_mode']}")
    else:
        print("Not running")


def cmd_config(args) -> None:
    cfg = load_config()
    if args.set:
        for kv in args.set:
            k, _, v = kv.partition("=")
            # Best-effort type coercion
            for cast in (int, float,
                         lambda x: {"true": True, "false": False}[x.lower()], str):
                try:
                    cfg[k] = cast(v)
                    break
                except Exception:
                    pass
        save_config(cfg)
        print(f"Config saved → {CONFIG_FILE}")
    print(json.dumps(cfg, indent=2))


def cmd_models(_) -> None:
    print("Available Parakeet models:\n")
    cfg = load_config()
    for key, hf_id in MODELS.items():
        mark = " ← current" if key == cfg["model"] else ""
        print(f"  {key:<26}  {hf_id}{mark}")
    print(f"\nSet with:  murmel config --set model=<key>")

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="murmel",
        description="Push-to-talk dictation daemon for Linux using Parakeet ASR",
    )
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")

    # start
    sp = sub.add_parser("start", help="Start the daemon")
    sp.add_argument(
        "--model", choices=list(MODELS),
        help="ASR model (default from config)",
    )
    sp.add_argument(
        "--inject", choices=["auto", "wtype", "xdotool", "clipboard"],
        help="Text injection method (default: auto)",
    )
    sp.add_argument(
        "--no-preload", action="store_true",
        help="Don't preload model on start (load on first use instead)",
    )
    sp.set_defaults(func=cmd_start)

    sub.add_parser("toggle", help="Toggle recording on/off").set_defaults(func=cmd_toggle)
    sub.add_parser("stop",   help="Stop the daemon").set_defaults(func=cmd_stop)
    sub.add_parser("status", help="Show daemon status").set_defaults(func=cmd_status)
    sub.add_parser("models", help="List available models").set_defaults(func=cmd_models)

    cp = sub.add_parser("config", help="View or edit config")
    cp.add_argument("--set", nargs="+", metavar="KEY=VALUE",
                    help="Set config values, e.g. murmel config --set model=parakeet-tdt-1.1b notify=false")
    cp.set_defaults(func=cmd_config)

    args = p.parse_args()
    if not hasattr(args, "func"):
        p.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
