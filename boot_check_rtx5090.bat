@echo off
setlocal enabledelayedexpansion
title Fooocus2026 - Boot Check RTX 5090
color 0A
cd /d "%~dp0"

echo ====================================================
echo    FOOOCUS2026 - RTX 5090 Boot Diagnostic
echo    Pre-launch check and optimization
echo ====================================================
echo.

REM =====================================================
REM  Detection de l'interpreteur Python
REM =====================================================
set "PYTHON="
if defined FOOOCUS_PYTHON set "PYTHON=%FOOOCUS_PYTHON%"
if not defined PYTHON if exist ".\python_embeded\python.exe" set "PYTHON=.\python_embeded\python.exe"
if not defined PYTHON if exist ".\python_embeded\Scripts\python.exe" set "PYTHON=.\python_embeded\Scripts\python.exe"
if not defined PYTHON if exist ".\venv\Scripts\python.exe" set "PYTHON=.\venv\Scripts\python.exe"
if not defined PYTHON if exist ".\.venv\Scripts\python.exe" set "PYTHON=.\.venv\Scripts\python.exe"
if not defined PYTHON where py >nul 2>&1 && set "PYTHON=py -3.10"
if not defined PYTHON set "PYTHON=python"

REM =====================================================
REM  1. GPU Detection
REM =====================================================
echo [1/7] Detection GPU...
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.draw --format=csv,noheader,nounits > "%TEMP%\fooocus_gpu_check.txt" 2>nul
if errorlevel 1 (
    echo    [ERREUR] nvidia-smi introuvable. Verifie les drivers NVIDIA.
    pause
    exit /b 1
)
for /f "tokens=1,2,3,4,5,6 delims=," %%a in (%TEMP%\fooocus_gpu_check.txt) do (
    set "GPU_NAME=%%a"
    set "DRIVER_VER=%%b"
    set "VRAM_TOTAL=%%c"
    set "VRAM_FREE=%%d"
    set "GPU_TEMP=%%e"
    set "GPU_POWER=%%f"
)
echo    GPU        : !GPU_NAME!
echo    Driver     : !DRIVER_VER!
echo    VRAM Total : !VRAM_TOTAL! MB
echo    VRAM Libre : !VRAM_FREE! MB
echo    Temperature: !GPU_TEMP! C
echo    Power Draw : !GPU_POWER! W
echo    [OK]
echo.

REM =====================================================
REM  2. VRAM Check
REM =====================================================
echo [2/7] Verification VRAM disponible...
for /f "tokens=* delims= " %%x in ("!VRAM_FREE!") do set "VRAM_FREE_TRIMMED=%%x"
set /a VRAM_CHECK=!VRAM_FREE_TRIMMED! 2>nul
if !VRAM_CHECK! LSS 8000 (
    echo    [ATTENTION] Seulement !VRAM_FREE! MB de VRAM libre.
    choice /C ON /M "   Continuer quand meme ? O=Oui, N=Non"
    if errorlevel 2 exit /b 0
) else (
    echo    !VRAM_FREE! MB disponibles - suffisant.
    echo    [OK]
)
echo.

REM =====================================================
REM  3. Temperature Check
REM =====================================================
echo [3/7] Verification temperature GPU...
for /f "tokens=* delims= " %%x in ("!GPU_TEMP!") do set "TEMP_TRIMMED=%%x"
set /a TEMP_CHECK=!TEMP_TRIMMED! 2>nul
if !TEMP_CHECK! GEQ 80 (
    echo    [ATTENTION] GPU a !GPU_TEMP! C - deja chaude, throttling possible.
) else if !TEMP_CHECK! GEQ 65 (
    echo    !GPU_TEMP! C - tiede, acceptable.
    echo    [OK]
) else (
    echo    !GPU_TEMP! C - frais, parfait.
    echo    [OK]
)
echo.

REM =====================================================
REM  4. Python Environment Check
REM =====================================================
echo [4/7] Verification interpreteur Python...
echo    PYTHON = %PYTHON%
%PYTHON% --version 2>nul
if errorlevel 1 (
    echo    [ERREUR] Interpreteur Python introuvable.
    echo    Definis FOOOCUS_PYTHON, ou installe Python 3.10, ou place un python_embeded.
    pause
    exit /b 1
)
echo    [OK]
echo.

REM =====================================================
REM  5. Dependances Python (best-effort)
REM =====================================================
echo [5/7] Verification dependances...
%PYTHON% -c "import packaging" 2>nul
if errorlevel 1 (
    echo    [FIX] Module 'packaging' manquant, installation...
    %PYTHON% -m pip install packaging --quiet
    echo    [OK] packaging installe.
) else (
    echo    packaging: present
)
echo    [OK]
echo.

REM =====================================================
REM  6. PyTorch / CUDA Check
REM =====================================================
echo [6/7] Verification PyTorch et CUDA...
%PYTHON% -c "import torch; print('    PyTorch '+str(torch.__version__)); print('    CUDA dispo: '+str(torch.cuda.is_available())); print('    CUDA: '+str(torch.version.cuda)); print('    bf16: '+str(torch.cuda.is_bf16_supported()))" 2>nul
if errorlevel 1 (
    echo    [ATTENTION] PyTorch/CUDA non detecte. Le premier lancement installera les deps.
) else (
    echo    [OK]
)
echo.

REM =====================================================
REM  7. Model Files Check
REM =====================================================
echo [7/7] Verification des modeles...
if exist "D:\Github\sdlibs\models\Stable-diffusion" (
    set "CKPT_COUNT=0"
    for %%f in ("D:\Github\sdlibs\models\Stable-diffusion\*.safetensors") do set /a CKPT_COUNT+=1
    echo    Checkpoints trouves: !CKPT_COUNT!
) else (
    echo    [INFO] Dossier checkpoints non accessible, telechargement au 1er lancement.
)
if exist "D:\Github\sdlibs\models\Lora" (
    set "LORA_COUNT=0"
    for %%f in ("D:\Github\sdlibs\models\Lora\*.safetensors") do set /a LORA_COUNT+=1
    echo    LoRAs trouves: !LORA_COUNT!
)
echo    [OK]
echo.

REM =====================================================
REM  Optimisations CUDA RTX 5090
REM =====================================================
echo ----------------------------------------------------
echo  Application des optimisations RTX 5090...
echo ----------------------------------------------------
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
echo    Variables CUDA configurees.
echo    [OK]
echo.

echo ====================================================
echo    Checks passes. Lancement en mode Qualite Max...
echo ====================================================
echo    Precision : bf16 UNet + VAE
echo    VRAM      : always-gpu, sans offload
echo    Attention : PyTorch native ^| Preview: TAESD
echo.
timeout /t 3 /nobreak >nul

:fooocus_start
%PYTHON% -s entry_with_update.py ^
  --always-gpu ^
  --disable-offload-from-vram ^
  --unet-in-bf16 ^
  --vae-in-bf16 ^
  --async-cuda-allocation ^
  --attention-pytorch ^
  --preview-option taesd

if %ERRORLEVEL% EQU 42 (
    echo.
    echo [Restart UI] Code de sortie 42 detecte - relance de Fooocus...
    echo.
    goto fooocus_start
)
echo.
echo ----------------------------------------------------
echo  Fooocus s'est arrete. En cas de crash: run.bat (mode standard).
echo ----------------------------------------------------
pause
