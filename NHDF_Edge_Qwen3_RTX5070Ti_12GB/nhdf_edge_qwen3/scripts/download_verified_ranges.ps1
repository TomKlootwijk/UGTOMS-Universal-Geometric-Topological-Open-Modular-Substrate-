param(
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [Parameter(Mandatory = $true)]
    [long]$ExpectedBytes,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ExpectedSha256,

    [Parameter(Mandatory = $true)]
    [string]$PartsDirectory,

    [ValidateRange(1, 128)]
    [int]$Segments = 16,

    [ValidateRange(1, 32)]
    [int]$Parallel = 8,

    [switch]$DeletePartsAfterMerge
)

$ErrorActionPreference = 'Stop'
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
$resolvedParts = [System.IO.Path]::GetFullPath($PartsDirectory)
$outputParent = [System.IO.Path]::GetDirectoryName($resolvedOutput)
New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
New-Item -ItemType Directory -Force -Path $resolvedParts | Out-Null

$chunk = [long][math]::Ceiling($ExpectedBytes / $Segments)
0..($Segments - 1) | ForEach-Object -Parallel {
    $index = $_
    $start = [long]$index * $using:chunk
    $end = [math]::Min($using:ExpectedBytes - 1, $start + $using:chunk - 1)
    $needed = $end - $start + 1
    $part = Join-Path $using:resolvedParts ('part-{0:D3}.bin' -f $index)

    if ((Test-Path -LiteralPath $part) -and (Get-Item -LiteralPath $part).Length -eq $needed) {
        "segment $index already complete"
        return
    }

    curl.exe -L --silent --show-error --fail --retry 5 --retry-delay 2 `
        --range "$start-$end" -o $part $using:Url
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for segment $index with exit code $LASTEXITCODE"
    }
    $actual = (Get-Item -LiteralPath $part).Length
    if ($actual -ne $needed) {
        throw "segment $index length $actual != $needed"
    }
    "segment $index complete ($actual bytes)"
} -ThrottleLimit $Parallel

$partFiles = Get-ChildItem -LiteralPath $resolvedParts -Filter 'part-*.bin' | Sort-Object Name
if ($partFiles.Count -ne $Segments) {
    throw "expected $Segments segments, found $($partFiles.Count)"
}

$outStream = [System.IO.File]::Open(
    $resolvedOutput,
    [System.IO.FileMode]::Create,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
)
try {
    foreach ($partFile in $partFiles) {
        $inStream = [System.IO.File]::OpenRead($partFile.FullName)
        try {
            $inStream.CopyTo($outStream, 8MB)
        }
        finally {
            $inStream.Dispose()
        }
        if ($DeletePartsAfterMerge) {
            $outStream.Flush($true)
            Remove-Item -LiteralPath $partFile.FullName -Force
        }
    }
}
finally {
    $outStream.Dispose()
}

$item = Get-Item -LiteralPath $resolvedOutput
if ($item.Length -ne $ExpectedBytes) {
    throw "merged length $($item.Length) != $ExpectedBytes"
}
$actualSha = (Get-FileHash -LiteralPath $resolvedOutput -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha -ne $ExpectedSha256.ToLowerInvariant()) {
    throw "SHA-256 mismatch: $actualSha"
}

[pscustomobject]@{
    Path = $item.FullName
    Bytes = $item.Length
    Sha256 = $actualSha
    Verified = $true
    PartsDirectory = $resolvedParts
}
