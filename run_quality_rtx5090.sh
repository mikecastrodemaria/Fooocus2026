#!/usr/bin/env bash
# Fooocus2026 - RTX 5090 Quality Max (Linux + NVIDIA).
# Sur Mac il n'y a pas de 5090 ni de CUDA : utilise run.sh a la place.
cd "$(dirname "$0")" || exit 1

echo "============================================"
echo " Fooocus2026 - Optimise RTX 5090 (32GB VRAM)"
echo " Mode: Qualite Maximale"
echo "============================================"
echo

if [ -n "$FOOOCUS_PYTHON" ]; then PYTHON="$FOOOCUS_PYTHON"
elif [ -x "./.venv/bin/python" ]; then PYTHON="./.venv/bin/python"
elif [ -x "./venv/bin/python" ]; then PYTHON="./venv/bin/python"
elif command -v python3.10 >/dev/null 2>&1; then PYTHON="python3.10"
elif command -v python3 >/dev/null 2>&1; then PYTHON="python3"
else PYTHON="python"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ATTENTION] nvidia-smi absent : pas de GPU NVIDIA detecte."
    echo "            Ces flags CUDA ne s'appliquent qu'a une carte NVIDIA."
    echo "            Sur Mac/AMD, prefere ./run.sh"
    echo
fi

# Optimisations CUDA RTX 5090
export NVIDIA_TF32_OVERRIDE=1
export CUDA_CACHE_MAXSIZE=4294967296
export CUDA_AUTO_BOOST=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

while true; do
    "$PYTHON" -s entry_with_update.py \
        --always-gpu \
        --disable-offload-from-vram \
        --unet-in-bf16 \
        --vae-in-bf16 \
        --async-cuda-allocation \
        --attention-pytorch \
        --preview-option taesd "$@"
    rc=$?
    if [ "$rc" -eq 42 ]; then
        echo
        echo "[Restart UI] Code de sortie 42 detecte - relance de Fooocus..."
        echo
        continue
    fi
    break
done
