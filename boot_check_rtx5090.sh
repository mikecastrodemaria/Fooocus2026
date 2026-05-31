#!/usr/bin/env bash
# Fooocus2026 - Boot Check RTX 5090 (Linux + NVIDIA).
# Diagnostic GPU + tuning CUDA + lancement qualite. Sur Mac, utilise run.sh.
cd "$(dirname "$0")" || exit 1

echo "===================================================="
echo "   FOOOCUS2026 - RTX 5090 Boot Diagnostic"
echo "===================================================="
echo

# --- Detection interpreteur Python ---
if [ -n "$FOOOCUS_PYTHON" ]; then PYTHON="$FOOOCUS_PYTHON"
elif [ -x "./.venv/bin/python" ]; then PYTHON="./.venv/bin/python"
elif [ -x "./venv/bin/python" ]; then PYTHON="./venv/bin/python"
elif command -v python3.10 >/dev/null 2>&1; then PYTHON="python3.10"
elif command -v python3 >/dev/null 2>&1; then PYTHON="python3"
else PYTHON="python"
fi

# --- 1. GPU Detection ---
echo "[1/5] Detection GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.draw \
        --format=csv,noheader,nounits | while IFS=, read -r name drv vtot vfree temp power; do
        echo "   GPU        :$name"
        echo "   Driver     :$drv"
        echo "   VRAM Total :$vtot MB"
        echo "   VRAM Libre :$vfree MB"
        echo "   Temperature:$temp C"
        echo "   Power      :$power W"
    done
    # Garde VRAM (lecture directe pour le test)
    VFREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    if [ -n "$VFREE" ] && [ "$VFREE" -lt 8000 ] 2>/dev/null; then
        echo "   [ATTENTION] Seulement ${VFREE} MB de VRAM libre."
        printf "   Continuer quand meme ? [o/N] "
        read -r ans
        case "$ans" in o|O|oui|y|Y) ;; *) exit 0;; esac
    fi
    echo "   [OK]"
else
    echo "   [ATTENTION] nvidia-smi absent : pas de GPU NVIDIA."
    echo "   Sur Mac/AMD, ce script n'a pas de sens. Utilise ./run.sh"
    echo
fi
echo

# --- 2. Python ---
echo "[2/5] Verification interpreteur Python..."
echo "   PYTHON = $PYTHON"
if ! "$PYTHON" --version >/dev/null 2>&1; then
    echo "   [ERREUR] Python introuvable. Definis FOOOCUS_PYTHON ou installe python3.10."
    exit 1
fi
"$PYTHON" --version
echo "   [OK]"
echo

# --- 3. Dependances (best-effort) ---
echo "[3/5] Verification dependances..."
if ! "$PYTHON" -c "import packaging" >/dev/null 2>&1; then
    echo "   [FIX] packaging manquant, installation..."
    "$PYTHON" -m pip install packaging --quiet
fi
echo "   [OK]"
echo

# --- 4. PyTorch / CUDA ---
echo "[4/5] Verification PyTorch et CUDA..."
"$PYTHON" -c "import torch; print('   PyTorch '+str(torch.__version__)); print('   CUDA dispo: '+str(torch.cuda.is_available())); print('   CUDA: '+str(torch.version.cuda)); print('   bf16: '+str(torch.cuda.is_bf16_supported()))" 2>/dev/null \
    || echo "   [ATTENTION] PyTorch/CUDA non detecte. Le 1er lancement installera les deps."
echo

# --- 5. Modeles ---
echo "[5/5] Verification des modeles..."
CKPT_DIR="$HOME/sdlibs/models/Stable-diffusion"
[ -d "/mnt/d/Github/sdlibs/models/Stable-diffusion" ] && CKPT_DIR="/mnt/d/Github/sdlibs/models/Stable-diffusion"
if [ -d "$CKPT_DIR" ]; then
    n=$(find "$CKPT_DIR" -maxdepth 1 -name "*.safetensors" | wc -l)
    echo "   Checkpoints trouves: $n"
else
    echo "   [INFO] Dossier checkpoints non trouve, telechargement au 1er lancement."
fi
echo "   [OK]"
echo

# --- Tuning CUDA ---
echo "----------------------------------------------------"
echo " Application des optimisations RTX 5090..."
echo "----------------------------------------------------"
export NVIDIA_TF32_OVERRIDE=1
export CUDA_CACHE_MAXSIZE=4294967296
export CUDA_AUTO_BOOST=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "   Variables CUDA configurees. [OK]"
echo
echo "===================================================="
echo "   Lancement en mode Qualite Max..."
echo "===================================================="
sleep 2

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
