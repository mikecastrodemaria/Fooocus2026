@echo off
REM custom-26 : protocole CLI de la famille crispz (voir fooocus_protocol.py et
REM comics2crispz/docs/CLI_PROTOCOL.md). Toujours UNE ligne JSON sur stdout.
REM Codes : 0 ok, 1 erreur d'execution, 2 spec invalide, 3 non supporte, 4 pas de route.
REM
REM   czp.bat caps
REM   czp.bat gen --spec spec.json        (--spec - : spec sur stdin)
REM   czp.bat upscale --spec spec.json
REM   czp.bat inpaint --spec spec.json
REM   options : --remote URL, --local, --timeout S
setlocal
set "CZP_CALLER_CWD=%CD%"
cd /d "%~dp0"

set "PYTHON="
if defined FOOOCUS_PYTHON set "PYTHON=%FOOOCUS_PYTHON%"
if not defined PYTHON if exist ".\python_embeded\python.exe" set "PYTHON=.\python_embeded\python.exe"
if not defined PYTHON if exist "..\python_embeded\python.exe" set "PYTHON=..\python_embeded\python.exe"
if not defined PYTHON if exist ".\venv\Scripts\python.exe" set "PYTHON=.\venv\Scripts\python.exe"
if not defined PYTHON if exist ".\.venv\Scripts\python.exe" set "PYTHON=.\.venv\Scripts\python.exe"
if not defined PYTHON where py >nul 2>&1 && set "PYTHON=py -3.10"
if not defined PYTHON set "PYTHON=python"

%PYTHON% -s fooocus_protocol.py %*
exit /b %ERRORLEVEL%
