[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$InstallRoot = Join-Path $ProjectRoot ".local-coder"
$NpmCache = Join-Path $InstallRoot "npm-cache"
$OpenCodeExe = Join-Path $InstallRoot "node_modules\opencode-ai\bin\opencode.exe"
$PinnedVersion = "1.18.25"

function Test-PinnedOpenCode {
    if (-not (Test-Path -LiteralPath $OpenCodeExe -PathType Leaf)) {
        return $false
    }

    $VersionOutput = @(& $OpenCodeExe --version 2>$null)
    $VersionExitCode = $LASTEXITCODE
    if ($VersionExitCode -ne 0 -or $VersionOutput.Count -eq 0) {
        return $false
    }
    return $VersionOutput[0].Trim() -eq $PinnedVersion
}

if (Test-PinnedOpenCode) {
    Write-Host "OpenCode $PinnedVersion is already installed locally."
    return
}

$Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($null -eq $Npm) {
    throw "npm.cmd was not found. Install Node.js with npm, then rerun this script."
}

New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
Write-Host "Installing OpenCode $PinnedVersion under $InstallRoot ..."
& $Npm.Source install --prefix $InstallRoot --cache $NpmCache --save-exact --no-audit --no-fund "opencode-ai@$PinnedVersion"
if ($LASTEXITCODE -ne 0) {
    throw "The project-local OpenCode installation failed with exit code $LASTEXITCODE."
}

if (-not (Test-PinnedOpenCode)) {
    throw "The local OpenCode executable is missing or is not version $PinnedVersion."
}

Write-Host "OpenCode $PinnedVersion is ready at $OpenCodeExe"
