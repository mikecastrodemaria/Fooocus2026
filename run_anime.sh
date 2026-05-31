#!/usr/bin/env bash
# Fooocus2026 - preset anime (Mac / Linux).
cd "$(dirname "$0")" || exit 1

if [ -n "$FOOOCUS_PYTHON" ]; then PYTHON="$FOOOCUS_PYTHON"
elif [ -x "./.venv/bin/python" ]; then PYTHON="./.venv/bin/python"
elif [ -x "./venv/bin/python" ]; then PYTHON="./venv/bin/python"
elif [ -x "./python_embeded/bin/python" ]; then PYTHON="./python_embeded/bin/python"
elif command -v python3.10 >/dev/null 2>&1; then PYTHON="python3.10"
elif command -v python3 >/dev/null 2>&1; then PYTHON="python3"
else PYTHON="python"
fi

while true; do
    "$PYTHON" -s entry_with_update.py --preset anime "$@"
    rc=$?
    if [ "$rc" -eq 42 ]; then
        echo
        echo "[Restart UI] Code de sortie 42 detecte - relance de Fooocus..."
        echo
        continue
    fi
    break
done
