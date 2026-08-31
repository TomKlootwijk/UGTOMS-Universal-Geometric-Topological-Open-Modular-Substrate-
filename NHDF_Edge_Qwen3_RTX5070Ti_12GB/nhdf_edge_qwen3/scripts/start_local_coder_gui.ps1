[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ErrorActionPreference = "Stop"

function Clear-PythonBootstrapEnvironment {
    $ProcessEnvironment = [System.Environment]::GetEnvironmentVariables(
        [System.EnvironmentVariableTarget]::Process
    )
    foreach ($NameObject in @($ProcessEnvironment.Keys)) {
        $Name = [string]$NameObject
        $UpperName = $Name.ToUpperInvariant()
        $IsPythonBootstrapVariable = (
            $UpperName.StartsWith("PYTHON") -or
            $UpperName.StartsWith("PY_") -or
            $UpperName.StartsWith("PYLAUNCHER") -or
            $UpperName.StartsWith("CONDA_") -or
            $UpperName -eq "VIRTUAL_ENV" -or
            $UpperName -eq "VIRTUAL_ENV_PROMPT" -or
            $UpperName -eq "__PYVENV_LAUNCHER__"
        )
        if ($IsPythonBootstrapVariable) {
            Microsoft.PowerShell.Management\Remove-Item `
                -LiteralPath ("Env:" + $Name) `
                -Force `
                -ErrorAction Stop
        }
    }
}

function Resolve-UnaliasedApplication {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$BlockedRoot,
        [Parameter(Mandatory = $true)][string[]]$AllowedRoots
    )

    if (-not [System.IO.Path]::IsPathRooted($Candidate)) {
        throw "Interpreter path is not absolute: $Candidate"
    }
    $Full = [System.IO.Path]::GetFullPath($Candidate)
    if (-not [string]::Equals(
        $Candidate,
        $Full,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Interpreter path is not canonical: $Candidate"
    }
    $Resolved = (Resolve-Path -LiteralPath $Full -ErrorAction Stop).Path
    if (-not [string]::Equals($Full, $Resolved, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Interpreter resolves through an alias: $Full"
    }
    $Blocked = [System.IO.Path]::GetFullPath($BlockedRoot).TrimEnd('\', '/')
    $BlockedPrefix = $Blocked + [System.IO.Path]::DirectorySeparatorChar
    if ($Full.StartsWith($BlockedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing a Python interpreter from inside the repository: $Full"
    }

    $InsideAllowedRoot = $false
    foreach ($AllowedRoot in $AllowedRoots) {
        if ([string]::IsNullOrWhiteSpace($AllowedRoot)) {
            continue
        }
        $Allowed = [System.IO.Path]::GetFullPath($AllowedRoot).TrimEnd('\', '/')
        $AllowedPrefix = $Allowed + [System.IO.Path]::DirectorySeparatorChar
        if ($Full.StartsWith(
            $AllowedPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            $InsideAllowedRoot = $true
            break
        }
    }
    if (-not $InsideAllowedRoot) {
        throw "Interpreter is outside the approved Windows installation roots: $Full"
    }

    $Root = [System.IO.Path]::GetPathRoot($Full)
    $Relative = $Full.Substring($Root.Length)
    $Current = $Root
    foreach ($Component in ($Relative -split '[\\/]')) {
        if ([string]::IsNullOrWhiteSpace($Component)) {
            continue
        }
        $Current = Join-Path $Current $Component
        $Item = Get-Item -LiteralPath $Current -Force -ErrorAction Stop
        if (($Item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Interpreter path traverses a symlink, junction, or reparse point: $Current"
        }
    }
    $Leaf = Get-Item -LiteralPath $Full -Force
    if ($Leaf.PSIsContainer -or $Leaf.Extension -ne ".exe") {
        throw "Interpreter candidate is not an executable file: $Full"
    }
    return $Full
}

function Get-RegisteredPythonCandidates {
    param(
        [Parameter(Mandatory = $true)][string[]]$MachineRoots,
        [Parameter(Mandatory = $true)][string[]]$UserRoots
    )

    $Records = [System.Collections.Generic.List[object]]::new()
    $Hives = @(
        [pscustomobject]@{
            Hive = [Microsoft.Win32.RegistryHive]::LocalMachine
            Label = "HKLM"
            AllowedRoots = $MachineRoots
        },
        [pscustomobject]@{
            Hive = [Microsoft.Win32.RegistryHive]::CurrentUser
            Label = "HKCU"
            AllowedRoots = $UserRoots
        }
    )
    $Views = @(
        [Microsoft.Win32.RegistryView]::Registry64,
        [Microsoft.Win32.RegistryView]::Registry32
    )

    foreach ($HiveRecord in $Hives) {
        foreach ($View in $Views) {
            $BaseKey = $null
            $CoreKey = $null
            try {
                $BaseKey = [Microsoft.Win32.RegistryKey]::OpenBaseKey(
                    $HiveRecord.Hive,
                    $View
                )
                $CoreKey = $BaseKey.OpenSubKey("Software\Python\PythonCore", $false)
                if ($null -eq $CoreKey) {
                    continue
                }
                $Versions = @(
                    $CoreKey.GetSubKeyNames() |
                        Where-Object { $_ -match '^3\.\d+$' } |
                        Sort-Object { [version]$_ } -Descending
                )
                foreach ($Version in $Versions) {
                    $InstallKey = $null
                    try {
                        $InstallKey = $CoreKey.OpenSubKey("$Version\InstallPath", $false)
                        if ($null -eq $InstallKey) {
                            continue
                        }
                        $Executable = $null
                        if ($InstallKey.GetValueNames() -contains "ExecutablePath") {
                            if ($InstallKey.GetValueKind("ExecutablePath") -eq `
                                [Microsoft.Win32.RegistryValueKind]::String) {
                                $Executable = [string]$InstallKey.GetValue(
                                    "ExecutablePath",
                                    $null,
                                    [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames
                                )
                            }
                        }
                        if ([string]::IsNullOrWhiteSpace($Executable)) {
                            if (($InstallKey.GetValueNames() -contains "") -and `
                                $InstallKey.GetValueKind("") -eq `
                                [Microsoft.Win32.RegistryValueKind]::String) {
                                $InstallRoot = [string]$InstallKey.GetValue(
                                    "",
                                    $null,
                                    [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames
                                )
                                if (-not [string]::IsNullOrWhiteSpace($InstallRoot)) {
                                    $Executable = Join-Path $InstallRoot "python.exe"
                                }
                            }
                        }
                        if (-not [string]::IsNullOrWhiteSpace($Executable)) {
                            [void]$Records.Add([pscustomobject]@{
                                Name = "$($HiveRecord.Label) $View Python $Version"
                                Executable = $Executable
                                Prefix = @()
                                AllowedRoots = $HiveRecord.AllowedRoots
                            })
                        }
                    }
                    finally {
                        if ($null -ne $InstallKey) {
                            $InstallKey.Close()
                        }
                    }
                }
            }
            finally {
                if ($null -ne $CoreKey) {
                    $CoreKey.Close()
                }
                if ($null -ne $BaseKey) {
                    $BaseKey.Close()
                }
            }
        }
    }
    return @($Records)
}

function Get-FixedPythonLauncherCandidates {
    param(
        [Parameter(Mandatory = $true)][string]$WindowsRoot,
        [Parameter(Mandatory = $true)][string[]]$MachineRoots,
        [Parameter(Mandatory = $true)][string]$LocalApplicationData
    )

    $Records = [System.Collections.Generic.List[object]]::new()
    $Paths = [System.Collections.Generic.List[string]]::new()
    [void]$Paths.Add((Join-Path $WindowsRoot "py.exe"))
    [void]$Paths.Add((Join-Path $WindowsRoot "System32\py.exe"))
    foreach ($MachineRoot in $MachineRoots) {
        if (-not [string]::IsNullOrWhiteSpace($MachineRoot)) {
            [void]$Paths.Add((Join-Path $MachineRoot "Python Launcher\py.exe"))
        }
    }
    [void]$Paths.Add(
        (Join-Path $LocalApplicationData "Programs\Python\Launcher\py.exe")
    )
    foreach ($Path in $Paths) {
        if (-not [string]::IsNullOrWhiteSpace($Path)) {
            [void]$Records.Add([pscustomobject]@{
                Name = "fixed py.exe launcher"
                Executable = $Path
                Prefix = @("-3")
                AllowedRoots = @($MachineRoots + $LocalApplicationData)
            })
        }
    }
    return @($Records)
}

function Test-GuiPython {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$PrefixArguments = @()
    )

    $Arguments = @($PrefixArguments) + @(
        "-I",
        "-E",
        "-c",
        "import os, sys, tkinter; blocked = ('PYTHON', 'PY_', 'PYLAUNCHER', 'CONDA_'); exact = {'VIRTUAL_ENV', 'VIRTUAL_ENV_PROMPT', '__PYVENV_LAUNCHER__'}; raise SystemExit(sys.version_info[:2] < (3, 10) or not sys.flags.isolated or not sys.flags.ignore_environment or any(name.upper().startswith(blocked) or name.upper() in exact for name in os.environ))"
    )
    & $Executable @Arguments *> $null
    return $LASTEXITCODE -eq 0
}

$Root = (Resolve-Path -LiteralPath $ProjectRoot -ErrorAction Stop).Path
$GuiScript = Join-Path $Root "scripts\local_coder_gui.py"
if (-not (Test-Path -LiteralPath $GuiScript -PathType Leaf)) {
    throw "UGTOMS Local Coder GUI script is missing: $GuiScript"
}

$WindowsRoot = [System.Environment]::GetFolderPath(
    [System.Environment+SpecialFolder]::Windows
)
$ProgramFiles = [System.Environment]::GetFolderPath(
    [System.Environment+SpecialFolder]::ProgramFiles
)
$ProgramFilesX86 = [System.Environment]::GetFolderPath(
    [System.Environment+SpecialFolder]::ProgramFilesX86
)
$LocalApplicationData = [System.Environment]::GetFolderPath(
    [System.Environment+SpecialFolder]::LocalApplicationData
)
if (
    [string]::IsNullOrWhiteSpace($WindowsRoot) -or
    [string]::IsNullOrWhiteSpace($ProgramFiles) -or
    [string]::IsNullOrWhiteSpace($LocalApplicationData)
) {
    throw "Windows did not return its trusted Python installation roots."
}
$MachineRoots = @($WindowsRoot, $ProgramFiles, $ProgramFilesX86)
$UserRoots = @($LocalApplicationData)

Clear-PythonBootstrapEnvironment
$Failures = [System.Collections.Generic.List[string]]::new()
$Candidates = @(
    @(Get-RegisteredPythonCandidates `
        -MachineRoots $MachineRoots `
        -UserRoots $UserRoots) +
    @(Get-FixedPythonLauncherCandidates `
        -WindowsRoot $WindowsRoot `
        -MachineRoots $MachineRoots `
        -LocalApplicationData $LocalApplicationData)
)
$Seen = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)

foreach ($Kind in $Candidates) {
    try {
        if (-not (Test-Path -LiteralPath $Kind.Executable -PathType Leaf)) {
            continue
        }
        $Executable = Resolve-UnaliasedApplication `
            -Candidate $Kind.Executable `
            -BlockedRoot $Root `
            -AllowedRoots $Kind.AllowedRoots
        if (-not $Seen.Add($Executable)) {
            continue
        }
        if (-not (Test-GuiPython -Executable $Executable -PrefixArguments $Kind.Prefix)) {
            $Failures.Add("$Executable does not provide isolated Python 3.10+ with tkinter.")
            continue
        }
        $Arguments = @($Kind.Prefix) + @("-I", "-E", ('"{0}"' -f $GuiScript))
        Start-Process `
            -FilePath $Executable `
            -ArgumentList $Arguments `
            -WorkingDirectory $Root `
            -WindowStyle Hidden `
            -ErrorAction Stop | Out-Null
        exit 0
    }
    catch {
        $Failures.Add("$($Kind.Name): $($_.Exception.Message)")
    }
}

$Detail = if ($Failures.Count -gt 0) {
    "`nChecked candidates:`n - " + ($Failures -join "`n - ")
}
else {
    ""
}
throw "UGTOMS Local Coder needs Python 3.10 or newer with tkinter.$Detail"
