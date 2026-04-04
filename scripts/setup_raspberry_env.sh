#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/Bioattend-raspberry}"
cd "$PROJECT_DIR"

echo "[1/7] Installation des paquets systeme..."
sudo apt update
sudo apt install -y python3-venv python3-picamera2 python3-libcamera libcap-dev

echo "[2/7] Suppression ancien venv..."
rm -rf venv

echo "[3/7] Creation nouveau venv avec paquets systeme visibles..."
python3 -m venv --system-site-packages venv
source venv/bin/activate

echo "[4/7] Upgrade pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

echo "[5/7] Installation dependances GPIO/Python..."
pip install --no-cache-dir rpi-lgpio lgpio

echo "[6/7] Verification imports critiques..."
python - <<'PY'
import importlib

checks = [
    ("RPi.GPIO", "GPIO"),
    ("lgpio", "lgpio"),
    ("picamera2", "Picamera2"),
]

for mod, label in checks:
    importlib.import_module(mod)
    print(f"OK: {label}")
PY

echo "[7/7] Termine. Pour lancer:"
echo "source venv/bin/activate && python3 src/main.py"
