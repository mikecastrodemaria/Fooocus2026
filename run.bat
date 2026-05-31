@echo off
REM Fooocus2026 - lancement standard.
cd /d "%~dp0"

REM --- Detection de l'interpreteur Python ---
REM Surcharge possible: set FOOOCUS_PYTHON=C:\chemin\python.exe avant de lancer.
set "PYTHON="
if defined FOOOCUS_PYTHON set "PYTHON=%FOOOCUS_PYTHON%"
if not defined PYTHON if exist ".\python_embeded\python.exe" set "PYTHON=.\python_embeded\python.exe"
if not defined PYTHON if exist ".\python_embeded\Scripts\python.exe" set "PYTHON=.\python_embeded\Scripts\python.exe"
if not defined PYTHON if exist ".\venv\Scripts\python.exe" set "PYTHON=.\venv\Scripts\python.exe"
if not defined PYTHON if exist ".\.venv\Scripts\python.exe" set "PYTHON=.\.venv\Scripts\python.exe"
if not defined PYTHON where py >nul 2>&1 && set "PYTHON=py -3.10"
if not defined PYTHON set "PYTHON=python"

:fooocus_start
%PYTHON% -s entry_with_update.py
if %ERRORLEVEL% EQU 42 (
    echo.
    echo [Restart UI] Code de sortie 42 detecte - relance de Fooocus...
    echo.
    goto fooocus_start
)
pause
