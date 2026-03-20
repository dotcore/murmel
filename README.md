# murmel

Push-to-talk dictation for Linux/Wayland. Runs locally on your GPU.

Bind a hotkey, hold to talk, release to transcribe. murmel captures audio, transcribes it with [NVIDIA Parakeet](https://huggingface.co/collections/nvidia/parakeet-702d58e33ff18e42e5e2e29d) (via NeMo), and types the result at your cursor. No cloud services, no API keys.

### How it works

1. A daemon listens for `SIGUSR1`. Bind any key to `murmel toggle`.
2. Hold the key to record, release to stop.
3. Audio is transcribed locally (GPU or CPU).
4. Text is injected at the cursor via `wtype` (Wayland), `xdotool` (X11), or clipboard fallback.

## Quick install (Arch Linux)

```bash
git clone <this-repo>
cd murmel
chmod +x install.sh
./install.sh
```

The installer handles pacman packages, sets up a Python 3.13 venv with `uv`, installs NeMo, and configures the systemd service.


## Manual install

```bash
# System deps
sudo pacman -S --needed wtype libnotify portaudio

# Create venv with Python 3.13 (NeMo doesn't fully support 3.14 yet)
uv venv --python 3.13 .venv

# Install Python deps (pulls in PyTorch, ~2 GB first time)
uv pip install sounddevice soundfile numpy 'nemo_toolkit[asr]'

# Install wrapper script
mkdir -p ~/.local/bin
cat > ~/.local/bin/murmel <<EOF
#!/bin/sh
exec "$(pwd)/.venv/bin/python3" "$(pwd)/murmel.py" "\$@"
EOF
chmod +x ~/.local/bin/murmel
```


## Hyprland keybind

Add to your Hyprland config (e.g. `~/.config/hypr/bindings.conf`):

```ini
# Push-to-talk: hold Super+D to record, release to transcribe
bindd = SUPER, D, Start dictation, exec, ~/.local/bin/murmel toggle
bindrd = SUPER, D, Stop dictation, exec, ~/.local/bin/murmel toggle

# Or simple toggle with F9 (press to start, press again to stop)
bind = , F9, exec, ~/.local/bin/murmel toggle
```

> **Note:** Use the full path (`~/.local/bin/murmel`) since Hyprland's exec environment may not include `~/.local/bin` in `PATH`.

Start the daemon (auto-starts on login if the service is enabled):

```bash
systemctl --user start murmel   # or: murmel start
```


## Available models

| Key | HuggingFace ID | Size | Notes |
|-----|---------------|------|-------|
| `parakeet-tdt-0.6b-v3` *(default)* | nvidia/parakeet-tdt-0.6b-v3 | 0.6B | Multilingual (25 langs), auto language detection |
| `parakeet-tdt-0.6b` | nvidia/parakeet-tdt-0.6b-v2 | 0.6B | English only, fast |
| `parakeet-tdt-1.1b` | nvidia/parakeet-tdt-1.1b | 1.1B | English, best accuracy |
| `parakeet-ctc-0.6b` | nvidia/parakeet-ctc-0.6b | 0.6B | CTC variant |
| `parakeet-rnnt-0.6b` | nvidia/parakeet-rnnt-0.6b | 0.6B | RNN-T variant |

List models and switch:

```bash
murmel models
murmel config --set model=parakeet-tdt-1.1b
```

You can also pass any HuggingFace model ID directly:

```bash
murmel start --model nvidia/parakeet-tdt-1.1b
```


## CLI reference

```
murmel start [--model KEY] [--inject MODE] [--no-preload]
murmel toggle        # send SIGUSR1 to the running daemon
murmel stop          # send SIGTERM
murmel status        # show PID and current config
murmel models        # list models
murmel config        # view config
murmel config --set KEY=VALUE ...
```


## Config (`~/.config/murmel/config.json`)

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `parakeet-tdt-0.6b-v3` | Model key or HF ID |
| `inject_mode` | `auto` | `auto` / `wtype` / `xdotool` / `clipboard` |
| `sample_rate` | `16000` | Audio sample rate (Hz) |
| `notify` | `true` | Desktop notifications |
| `preload` | `true` | Load model on daemon start |
| `min_duration` | `0.4` | Discard clips shorter than N seconds |
| `trailing_silence` | `0.3` | Extra silence to capture after toggle-off |
| `log_level` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

```bash
# Examples
murmel config --set model=parakeet-tdt-1.1b
murmel config --set notify=false
murmel config --set inject_mode=clipboard
murmel config --set log_level=DEBUG
```


## Text injection methods

| Method | When to use |
|--------|-------------|
| `wtype` (default on Wayland) | Hyprland or any Wayland compositor. Requires `wtype` package. |
| `xdotool` | XWayland apps or pure X11. May miss focus. |
| `clipboard` | Fallback. Copies text and simulates Ctrl+V. |

`auto` checks `$WAYLAND_DISPLAY` and picks `wtype`, falling back to `xdotool` on X11.


## Waybar module (optional)

Show recording state in Waybar with a `custom/murmel` module:

**`~/.config/waybar/config`**:
```json
"custom/murmel": {
    "exec": "[ -f /tmp/murmel.pid ] && echo '🎙' || echo ''",
    "interval": 1,
    "on-click": "murmel toggle",
    "tooltip": true
}
```


## Logs

```bash
journalctl --user -u murmel -f     # systemd logs
tail -f ~/.config/murmel/murmel.log
```


## GPU acceleration

NeMo uses PyTorch and will use a CUDA GPU if one is available. The 0.6B model runs in roughly 0.5-2s on CPU on modern hardware.


## Troubleshooting

**No audio captured**: check your default input device:
```bash
.venv/bin/python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

**wtype fails in XWayland apps**: switch inject mode to `xdotool` or `clipboard`:
```bash
murmel config --set inject_mode=clipboard
```

**Model download fails**: NeMo caches models in `~/.cache/huggingface/hub/`. If a download is interrupted, delete the partial files and restart.

## License

[MIT](LICENSE)
