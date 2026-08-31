@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "APP_ROOT=%~dp0"
set "GUI_BOOTSTRAP=%APP_ROOT%scripts\start_local_coder_gui.ps1"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

if not exist "%GUI_BOOTSTRAP%" (
    echo UGTOMS Local Coder could not find:
    echo   %GUI_BOOTSTRAP%
    echo Keep this launcher at the repository root and try again.
    pause
    exit /b 1
)

if not exist "%POWERSHELL_EXE%" (
    echo UGTOMS Local Coder needs machine-owned Windows PowerShell at:
    echo   %POWERSHELL_EXE%
    pause
    exit /b 1
)

"%POWERSHELL_EXE%" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%GUI_BOOTSTRAP%" -ProjectRoot "%APP_ROOT%"
if errorlevel 1 (
    echo.
    echo UGTOMS Local Coder could not start. Review the error above, install Python 3.10+
    echo with tkinter if needed, then double-click this file again.
    pause
    exit /b 1
)

exit /b 0
