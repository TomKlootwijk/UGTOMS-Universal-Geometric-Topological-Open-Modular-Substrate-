[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SetupScript = Join-Path $PSScriptRoot "setup_local_coder.ps1"
$Launcher = Join-Path $PSScriptRoot "local_coder.py"
& $SetupScript

$Python = Get-Command python.exe -ErrorAction SilentlyContinue
if ($null -ne $Python) {
    & $Python.Source $Launcher @Arguments
    exit $LASTEXITCODE
}

$PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($null -eq $PyLauncher) {
    throw "Python 3 was not found on PATH."
}

& $PyLauncher.Source -3 $Launcher @Arguments
exit $LASTEXITCODE
