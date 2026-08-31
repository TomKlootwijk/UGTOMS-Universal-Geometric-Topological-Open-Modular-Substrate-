@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "APP_ROOT=%~dp0"
set "GUI_SCRIPT=%APP_ROOT%scripts\local_coder_gui.py"
set "GUI_STARTED="

if not exist "%GUI_SCRIPT%" (
    echo UGTOMS Local Coder could not find:
    echo   %GUI_SCRIPT%
    echo Keep this launcher at the repository root and try again.
    pause
    exit /b 1
)

pushd "%APP_ROOT%" >nul 2>nul
if errorlevel 1 (
    echo UGTOMS Local Coder could not open its repository directory:
    echo   %APP_ROOT%
    pause
    exit /b 1
)

for /f "delims=" %%P in ('"%SystemRoot%\System32\where.exe" python.exe 2^>nul') do (
    call :TRY_PYTHON "%%~fP"
    if defined GUI_STARTED goto :LAUNCHED
)

for /f "delims=" %%P in ('"%SystemRoot%\System32\where.exe" py.exe 2^>nul') do (
    call :TRY_PY_LAUNCHER "%%~fP"
    if defined GUI_STARTED goto :LAUNCHED
)

popd
echo UGTOMS Local Coder needs Python 3 with tkinter.
echo Install Python 3, including the optional Tcl/Tk component, then double-click this file again.
pause
exit /b 1

:LAUNCHED
popd
exit /b 0

:TRY_PYTHON
"%~1" -c "import sys, tkinter; raise SystemExit(sys.version_info[:2] ^< (3, 10))" >nul 2>nul
if errorlevel 1 exit /b 0
start "" /B /D "%APP_ROOT%" "%~1" "%GUI_SCRIPT%"
if not errorlevel 1 set "GUI_STARTED=1"
exit /b 0

:TRY_PY_LAUNCHER
"%~1" -3 -c "import sys, tkinter; raise SystemExit(sys.version_info[:2] ^< (3, 10))" >nul 2>nul
if errorlevel 1 exit /b 0
start "" /B /D "%APP_ROOT%" "%~1" -3 "%GUI_SCRIPT%"
if not errorlevel 1 set "GUI_STARTED=1"
exit /b 0
