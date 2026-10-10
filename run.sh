#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "  brGradio2CheXagent"
echo "  Stanford AIMI CheXagent-2-3b + Meta NLLB-200"
echo "=========================================================="

source "$SCRIPT_DIR/.venv/bin/activate"
python "$SCRIPT_DIR/app.py"
