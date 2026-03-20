#!/usr/bin/env bash
# install.sh – Set up murmel on Arch Linux / Hyprland
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
INSTALL_DIR="$HOME/.local/bin"
SERVICE_DIR="$HOME/.config/systemd/user"

echo "==> Installing murmel"

# ── System deps (Arch) ─────────────────────────────────────────────────────
if command -v pacman &>/dev/null; then
    echo "==> Installing system packages via pacman"
    sudo pacman -S --needed --noconfirm \
        wtype \
        libnotify \
        portaudio
fi

# ── uv ───────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# ── Virtual environment ──────────────────────────────────────────────────
echo "==> Setting up Python 3.13 virtual environment at $VENV_DIR"
uv venv --python 3.13 "$VENV_DIR"

echo "==> Installing Python dependencies (this is large ~2GB, may take a while)"
uv pip install --python "$VENV_DIR/bin/python3" \
    sounddevice \
    soundfile \
    numpy \
    'nemo_toolkit[asr]'

# ── Install murmel wrapper ─────────────────────────────────────────────
mkdir -p "$INSTALL_DIR"
cat > "$INSTALL_DIR/murmel" <<EOF
#!/bin/sh
exec "$VENV_DIR/bin/python3" "$SCRIPT_DIR/murmel.py" "\$@"
EOF
chmod +x "$INSTALL_DIR/murmel"
echo "==> Installed: $INSTALL_DIR/murmel"

# ── Systemd user service ──────────────────────────────────────────────────
mkdir -p "$SERVICE_DIR"
cp "$SCRIPT_DIR/murmel.service" "$SERVICE_DIR/murmel.service"
systemctl --user daemon-reload
systemctl --user enable murmel.service
echo "==> systemd service installed and enabled"

# ── Hyprland keybind hint ─────────────────────────────────────────────────
cat <<'BANNER'

════════════════════════════════════════════════════════════
  murmel installed!
════════════════════════════════════════════════════════════

Add to ~/.config/hypr/bindings.conf:

    # Push-to-talk (hold Super+D to record, release to transcribe)
    bindd = SUPER, D, Start dictation, exec, ~/.local/bin/murmel toggle
    bindrd = SUPER, D, Stop dictation, exec, ~/.local/bin/murmel toggle

Start the daemon now:
    systemctl --user start murmel

Or manually:
    murmel start

List / change models:
    murmel models
    murmel config --set model=parakeet-tdt-1.1b

Default model: parakeet-tdt-0.6b-v3 (nvidia/parakeet-tdt-0.6b-v3, multilingual)
Config file:   ~/.config/murmel/config.json
Log file:      ~/.config/murmel/murmel.log

════════════════════════════════════════════════════════════
BANNER
