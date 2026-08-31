[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$InstallRoot = Join-Path $ProjectRoot ".local-coder"
$NpmCache = Join-Path $InstallRoot "npm-cache"
$OpenCodeExe = Join-Path $InstallRoot "node_modules\opencode-ai\bin\opencode.exe"
$PinnedVersion = "1.18.25"
$PinnedSha256 = "ef06e41a35795066e95acde276a42fbbf85d7a683c2787f6a19ed20bcde9b6ff"

function Assert-SafeProjectPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root
    )

    $RootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $PathFull = [System.IO.Path]::GetFullPath($Path)
    $Prefix = $RootFull + [System.IO.Path]::DirectorySeparatorChar
    if (-not $PathFull.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Local-coder path escapes the project root: $PathFull"
    }

    $Relative = $PathFull.Substring($Prefix.Length)
    $Current = $RootFull
    foreach ($Component in ($Relative -split '[\\/]')) {
        if ([string]::IsNullOrEmpty($Component)) {
            continue
        }
        $Current = Join-Path $Current $Component
        if (Test-Path -LiteralPath $Current) {
            $Item = Get-Item -LiteralPath $Current -Force
            if (($Item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Local-coder path traverses a symlink, junction, or reparse point: $Current"
            }
        }
    }
}

function Test-PinnedOpenCode {
    if (-not (Test-Path -LiteralPath $OpenCodeExe -PathType Leaf)) {
        return $false
    }

    $ActualSha256 = (Get-FileHash -LiteralPath $OpenCodeExe -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualSha256 -ne $PinnedSha256) {
        return $false
    }

    $VersionOutput = @(& $OpenCodeExe --version 2>$null)
    $VersionExitCode = $LASTEXITCODE
    if ($VersionExitCode -ne 0 -or $VersionOutput.Count -eq 0) {
        return $false
    }
    return $VersionOutput[0].Trim() -eq $PinnedVersion
}

Assert-SafeProjectPath -Path $InstallRoot -Root $ProjectRoot
Assert-SafeProjectPath -Path $NpmCache -Root $ProjectRoot
Assert-SafeProjectPath -Path $OpenCodeExe -Root $ProjectRoot

if (Test-PinnedOpenCode) {
    Write-Host "OpenCode $PinnedVersion is already installed locally."
    return
}

$Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($null -eq $Npm) {
    throw "npm.cmd was not found. Install Node.js with npm, then rerun this script."
}

New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
Assert-SafeProjectPath -Path $InstallRoot -Root $ProjectRoot
Write-Host "Installing OpenCode $PinnedVersion under $InstallRoot ..."
& $Npm.Source install --prefix $InstallRoot --cache $NpmCache --save-exact --no-audit --no-fund "opencode-ai@$PinnedVersion"
if ($LASTEXITCODE -ne 0) {
    throw "The project-local OpenCode installation failed with exit code $LASTEXITCODE."
}

Assert-SafeProjectPath -Path $InstallRoot -Root $ProjectRoot
Assert-SafeProjectPath -Path $OpenCodeExe -Root $ProjectRoot
if (-not (Test-PinnedOpenCode)) {
    if (-not (Test-Path -LiteralPath $OpenCodeExe -PathType Leaf)) {
        throw "The local OpenCode executable is missing after package installation."
    }
    $InstalledSha256 = (Get-FileHash -LiteralPath $OpenCodeExe -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($InstalledSha256 -ne $PinnedSha256) {
        throw "The installed OpenCode executable digest differs from the pinned package output. Expected $PinnedSha256, got $InstalledSha256."
    }
    throw "The local OpenCode executable is not version $PinnedVersion."
}

Assert-SafeProjectPath -Path $OpenCodeExe -Root $ProjectRoot

Write-Host "OpenCode $PinnedVersion ($PinnedSha256) is ready at $OpenCodeExe"
