#!/usr/bin/env bash
set -euo pipefail

# One-command environment bootstrap for a fresh machine.
# - Creates venv if missing
# - Installs system packages (only when needed)
# - Installs Python dependencies only when requirements changed

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
MARKER_FILE="$VENV_DIR/.requirements.sha256"

FORCE_REINSTALL=false
MODE="auto" # auto | prod | dev

for arg in "$@"; do
  case "$arg" in
    --force) FORCE_REINSTALL=true ;;
    --prod) MODE="prod" ;;
    --dev) MODE="dev" ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: scripts/bootstrap_env.sh [--force] [--prod|--dev]"
      exit 1
      ;;
  esac
done

is_raspberry_pi() {
  if [[ "$MODE" == "prod" ]]; then
    return 0
  fi
  if [[ "$MODE" == "dev" ]]; then
    return 1
  fi

  if [[ -f /proc/device-tree/model ]] && grep -qi "raspberry" /proc/device-tree/model; then
    return 0
  fi
  return 1
}

install_system_packages() {
  if ! command -v apt >/dev/null 2>&1; then
    echo "[INFO] apt not found, skipping system package installation."
    return
  fi

  echo "[1/4] Installing required system packages..."
  sudo apt update

  if is_raspberry_pi; then
    sudo apt install -y python3-venv python3-picamera2 python3-libcamera libcap-dev
  else
    sudo apt install -y python3-venv
  fi
}

choose_requirements_file() {
  if is_raspberry_pi; then
    echo "$PROJECT_DIR/requirements.txt"
  else
    echo "$PROJECT_DIR/requirements-dev.txt"
  fi
}

compute_hash() {
  local req_file="$1"
  # Include python version + requirements file content in hash.
  {
    python3 --version
    cat "$req_file"
  } | sha256sum | awk '{print $1}'
}

create_or_reuse_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[2/4] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
  else
    echo "[2/4] Reusing existing virtual environment."
  fi
}

install_python_dependencies_if_needed() {
  local req_file="$1"
  local current_hash
  current_hash="$(compute_hash "$req_file")"

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  if [[ "$FORCE_REINSTALL" == "true" ]]; then
    echo "[3/4] Forced reinstall requested."
    pip install --upgrade pip setuptools wheel
    pip install --no-cache-dir -r "$req_file"
    echo "$current_hash" > "$MARKER_FILE"
    return
  fi

  if [[ -f "$MARKER_FILE" ]] && [[ "$(cat "$MARKER_FILE")" == "$current_hash" ]]; then
    echo "[3/4] Dependencies unchanged. Skipping pip install."
    return
  fi

  echo "[3/4] Installing/updating Python dependencies from $(basename "$req_file")..."
  pip install --upgrade pip setuptools wheel
  pip install --no-cache-dir -r "$req_file"
  echo "$current_hash" > "$MARKER_FILE"
}

print_next_step() {
  echo "[4/4] Environment ready."
  echo "Activate and run:"
  echo "source venv/bin/activate && python3 src/main.py"
}

cd "$PROJECT_DIR"
REQ_FILE="$(choose_requirements_file)"

echo "[INFO] Project: $PROJECT_DIR"
echo "[INFO] Mode   : $(is_raspberry_pi && echo 'Raspberry/production' || echo 'Development')"
echo "[INFO] Using  : $(basename "$REQ_FILE")"

install_system_packages
create_or_reuse_venv
install_python_dependencies_if_needed "$REQ_FILE"
print_next_step
