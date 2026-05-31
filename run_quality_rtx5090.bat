@echo off
title Fooocus2026 - RTX 5090 Quality Max
cd /d "%~dp0"
echo ============================================
echo  Fooocus2026 - Optimise RTX 5090 (32GB VRAM)
echo  Mode: Qualite Maximale
echo ============================================
echo.
REM bf16 UNet + VAE (meilleure precision que fp16 sur Blackwell)
REM tout en VRAM, allocation CUDA asynchrone, attention PyTorch native, preview TAESD.
REM Note: pas de --preset quality_max dans Fooocus2026, la qualite vient des flags.

set "PYTHON="
if defined FOOOCUS_PYTHON set "PYTHON=%FOOOCUS_PYTHON%"
if not defined PYTHON if exist ".\python_embeded\python.exe" set "PYTHON=.\python_embeded\python.exe"
if not defined PYTHON if exist ".\python_embeded\Scripts\python.exe" set "PYTHON=.\python_embeded\Scripts\python.exe"
if not defined PYTHON if exist ".\venv\Scripts\python.exe" set "PYTHON=.\venv\Scripts\python.exe"
if not defined PYTHON if exist ".\.venv\Scripts\python.exe" set "PYTHON=.\.venv\Scripts\python.exe"
if not defined PYTHON where py >nul 2>&1 && set "PYTHON=py -3.10"
if not defined PYTHON set "PYTHON=python"

REM Optimisations CUDA RTX 5090
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID

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
echo  Fooocus s'est arrete. En cas de crash, relance run.bat (mode standard).
echo ----------------------------------------------------
pause
