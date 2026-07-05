@echo off
title Fooocus2026 - XYZ batch de nuit
REM ============================================================
REM  Exemple custom-15.3 : trois grilles X/Y/Z enchainees en CLI.
REM  Adaptez prompts/axes, lancez avant d'aller dormir.
REM  Chaque planche finit dans Fooocus\outputs\xyz_grids\
REM ============================================================
cd /d "%~dp0Fooocus"
set PY=..\python_embeded\Scripts\python.exe
set FLAGS=--always-gpu --disable-offload-from-vram --unet-in-bf16 --vae-in-bf16

echo [1/3] Calibrage CFG x Steps...
%PY% -s xyz_cli.py --prompt "portrait of a woman, cinematic lighting, masterpiece" ^
    --x "CFG:3,5,7" --y "Steps:20,30,40" --seed 12345 %FLAGS%

echo [2/3] Duel de checkpoints...
%PY% -s xyz_cli.py --prompt "portrait of a woman, cinematic lighting, masterpiece" ^
    --x "Checkpoint:juggernaut,realvis" --y "CFG:4,7" --seed 12345 %FLAGS%

echo [3/3] Variantes de prompt (S/R)...
%PY% -s xyz_cli.py --prompt "portrait of a woman, cinematic lighting, masterpiece" ^
    --x "Prompt S/R:cinematic lighting,golden hour,neon glow,candlelight" --seed 12345 %FLAGS%

echo.
echo Termine. Planches dans Fooocus\outputs\xyz_grids\
pause
