[CmdletBinding()]
param(
    [switch]$FunctionsOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$InstallRoot = Join-Path $ProjectRoot ".local-coder"
$NpmCache = Join-Path $InstallRoot "npm-cache"
$OpenCodeExe = Join-Path $InstallRoot "node_modules\opencode-ai\bin\opencode.exe"
$BundledArchive = Join-Path $ProjectRoot "vendor\opencode\opencode-windows-x64-1.18.25.zip"
$PinnedVersion = "1.18.25"
$PinnedArchiveBytes = 62030007
$PinnedArchiveSha256 = "35fe618642f733aa1db8e26a78a1c9ee7cfce47e94cdbd36a37312c9d55e2a45"
$PinnedExecutableBytes = 179651624
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

function Get-Sha256File {
    param(
        [Parameter(Mandatory = $true)][string]$Path
    )

    $Stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::Read
    )
    $Sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $Bytes = $Sha256.ComputeHash($Stream)
        return [System.BitConverter]::ToString($Bytes).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $Sha256.Dispose()
        $Stream.Dispose()
    }
}

function Test-PinnedOpenCode {
    param(
        [string]$ExecutablePath = $OpenCodeExe
    )

    Assert-SafeProjectPath -Path $ExecutablePath -Root $ProjectRoot
    if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) {
        return $false
    }
    if ((Get-Item -LiteralPath $ExecutablePath -Force).Length -ne $PinnedExecutableBytes) {
        return $false
    }

    $ActualSha256 = Get-Sha256File -Path $ExecutablePath
    if ($ActualSha256 -ne $PinnedSha256) {
        return $false
    }

    $VersionOutput = @(& $ExecutablePath --version 2>$null)
    $VersionExitCode = $LASTEXITCODE
    if ($VersionExitCode -ne 0 -or $VersionOutput.Count -eq 0) {
        return $false
    }
    if ($VersionOutput[0].Trim() -ne $PinnedVersion) {
        return $false
    }

    $PostRunSha256 = Get-Sha256File -Path $ExecutablePath
    return $PostRunSha256 -eq $PinnedSha256
}

function Get-VerifiedPlatformOpenCode {
    $Candidates = @(
        (Join-Path $InstallRoot "node_modules\opencode-windows-x64\bin\opencode.exe"),
        (Join-Path $InstallRoot "node_modules\opencode-windows-x64-baseline\bin\opencode.exe")
    )
    $VerifiedCandidates = @()
    $Observed = @()

    foreach ($Candidate in $Candidates) {
        Assert-SafeProjectPath -Path $Candidate -Root $ProjectRoot
        if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
            continue
        }
        $Item = Get-Item -LiteralPath $Candidate -Force
        $Digest = Get-Sha256File -Path $Candidate
        $Observed += "${Candidate}:bytes=$($Item.Length),sha256=${Digest}"
        if ($Item.Length -eq $PinnedExecutableBytes -and $Digest -eq $PinnedSha256) {
            $VerifiedCandidates += $Candidate
        }
    }

    if ($VerifiedCandidates.Count -ne 1) {
        $Detail = if ($Observed.Count -eq 0) { "no Windows x64 package binary was installed" } else { $Observed -join "; " }
        throw "Exactly one installed platform package must match the pinned OpenCode executable before first execution; $Detail"
    }
    return $VerifiedCandidates[0]
}

function Install-VerifiedExecutableFile {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath
    )

    Assert-SafeProjectPath -Path $SourcePath -Root $ProjectRoot
    $OpenCodeExeParent = Split-Path -Parent $OpenCodeExe
    New-Item -ItemType Directory -Path $OpenCodeExeParent -Force | Out-Null
    Assert-SafeProjectPath -Path $OpenCodeExeParent -Root $ProjectRoot
    Assert-SafeProjectPath -Path $OpenCodeExe -Root $ProjectRoot
    $TemporaryExe = Join-Path $OpenCodeExeParent (".opencode." + [Guid]::NewGuid().ToString("N") + ".tmp")
    Assert-SafeProjectPath -Path $TemporaryExe -Root $ProjectRoot
    try {
        [System.IO.File]::Copy($SourcePath, $TemporaryExe, $false)
        $TemporaryItem = Get-Item -LiteralPath $TemporaryExe -Force
        $TemporarySha256 = Get-Sha256File -Path $TemporaryExe
        if ($TemporaryItem.Length -ne $PinnedExecutableBytes -or $TemporarySha256 -ne $PinnedSha256) {
            throw "The verified OpenCode executable changed while being copied. Expected $PinnedExecutableBytes bytes / $PinnedSha256, got $($TemporaryItem.Length) bytes / $TemporarySha256."
        }
        Move-Item -LiteralPath $TemporaryExe -Destination $OpenCodeExe -Force
    }
    finally {
        if (Test-Path -LiteralPath $TemporaryExe) {
            Remove-Item -LiteralPath $TemporaryExe -Force
        }
    }
}

function Install-OpenCodeFromBundledArchive {
    Assert-SafeProjectPath -Path $BundledArchive -Root $ProjectRoot
    $ArchiveItem = Get-Item -LiteralPath $BundledArchive -Force
    if ($ArchiveItem.Length -ne $PinnedArchiveBytes) {
        throw "Bundled OpenCode archive size mismatch. Expected $PinnedArchiveBytes bytes, got $($ArchiveItem.Length)."
    }
    $ArchiveSha256 = Get-Sha256File -Path $BundledArchive
    if ($ArchiveSha256 -ne $PinnedArchiveSha256) {
        throw "Bundled OpenCode archive digest mismatch. Expected $PinnedArchiveSha256, got $ArchiveSha256."
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $OpenCodeExeParent = Split-Path -Parent $OpenCodeExe
    New-Item -ItemType Directory -Path $OpenCodeExeParent -Force | Out-Null
    Assert-SafeProjectPath -Path $OpenCodeExeParent -Root $ProjectRoot
    $ExtractedExe = Join-Path $OpenCodeExeParent (".opencode.archive." + [Guid]::NewGuid().ToString("N") + ".tmp")
    Assert-SafeProjectPath -Path $ExtractedExe -Root $ProjectRoot
    $Archive = $null
    try {
        $Archive = [System.IO.Compression.ZipFile]::OpenRead($BundledArchive)
        if ($Archive.Entries.Count -ne 1) {
            throw "Bundled OpenCode archive must contain exactly one entry."
        }
        $Entry = $Archive.Entries[0]
        if ($Entry.FullName -cne "opencode.exe" -or $Entry.Name -cne "opencode.exe") {
            throw "Bundled OpenCode archive entry must be exactly opencode.exe."
        }
        if ($Entry.Length -ne $PinnedExecutableBytes) {
            throw "Bundled OpenCode executable size mismatch. Expected $PinnedExecutableBytes bytes, got $($Entry.Length)."
        }
        $InputStream = $Entry.Open()
        $OutputStream = [System.IO.File]::Open(
            $ExtractedExe,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        try {
            $InputStream.CopyTo($OutputStream)
        }
        finally {
            $OutputStream.Dispose()
            $InputStream.Dispose()
        }
    }
    catch {
        if (Test-Path -LiteralPath $ExtractedExe) {
            Remove-Item -LiteralPath $ExtractedExe -Force
        }
        throw
    }
    finally {
        if ($null -ne $Archive) {
            $Archive.Dispose()
        }
    }

    try {
        $PostReadArchiveSha256 = Get-Sha256File -Path $BundledArchive
        if ($PostReadArchiveSha256 -ne $PinnedArchiveSha256) {
            throw "Bundled OpenCode archive changed during extraction."
        }
        Install-VerifiedExecutableFile -SourcePath $ExtractedExe
    }
    finally {
        if (Test-Path -LiteralPath $ExtractedExe) {
            Remove-Item -LiteralPath $ExtractedExe -Force
        }
    }
}

function Install-OpenCodeFromNpm {
    $Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($null -eq $Npm) {
        throw "npm.cmd was not found and the verified bundled archive is absent. Install Node.js with npm, then rerun this script."
    }

    New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
    Assert-SafeProjectPath -Path $InstallRoot -Root $ProjectRoot
    Write-Host "Bundled OpenCode archive is absent; installing OpenCode $PinnedVersion under $InstallRoot with lifecycle scripts disabled ..."
    & $Npm.Source install --prefix $InstallRoot --cache $NpmCache --save-exact --ignore-scripts --no-audit --no-fund "opencode-ai@$PinnedVersion"
    if ($LASTEXITCODE -ne 0) {
        throw "The project-local OpenCode installation failed with exit code $LASTEXITCODE."
    }
    $VerifiedPlatformExe = Get-VerifiedPlatformOpenCode
    Install-VerifiedExecutableFile -SourcePath $VerifiedPlatformExe
}

function Invoke-LocalCoderPreflight {
    Assert-SafeProjectPath -Path $InstallRoot -Root $ProjectRoot
    Assert-SafeProjectPath -Path $NpmCache -Root $ProjectRoot
    Assert-SafeProjectPath -Path $OpenCodeExe -Root $ProjectRoot
    if (-not (Test-PinnedOpenCode -ExecutablePath $OpenCodeExe)) {
        throw "The project-local OpenCode executable failed its pinned size, digest, or version preflight."
    }
    $Item = Get-Item -LiteralPath $OpenCodeExe -Force
    return [pscustomobject]@{
        Path = $Item.FullName
        Version = $PinnedVersion
        Bytes = $Item.Length
        Sha256 = $PinnedSha256
    }
}

function Install-PinnedOpenCode {
    Assert-SafeProjectPath -Path $InstallRoot -Root $ProjectRoot
    Assert-SafeProjectPath -Path $NpmCache -Root $ProjectRoot
    Assert-SafeProjectPath -Path $OpenCodeExe -Root $ProjectRoot

    if (Test-PinnedOpenCode -ExecutablePath $OpenCodeExe) {
        Write-Host "OpenCode $PinnedVersion is already installed locally."
        $Result = Invoke-LocalCoderPreflight
        return $Result
    }

    New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
    Assert-SafeProjectPath -Path $InstallRoot -Root $ProjectRoot
    if (Test-Path -LiteralPath $BundledArchive -PathType Leaf) {
        Write-Host "Installing OpenCode $PinnedVersion from the verified bundled archive ..."
        Install-OpenCodeFromBundledArchive
    }
    else {
        Install-OpenCodeFromNpm
    }

    $Result = Invoke-LocalCoderPreflight
    Write-Host "OpenCode $PinnedVersion ($PinnedSha256) is ready at $OpenCodeExe"
    return $Result
}

if (-not $FunctionsOnly) {
    Install-PinnedOpenCode | Out-Null
}
