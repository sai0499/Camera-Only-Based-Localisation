$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot
if (-not [Environment]::Is64BitOperatingSystem -or -not [Environment]::Is64BitProcess) {
    throw 'Use 64-bit Windows and 64-bit PowerShell.'
}

$setupRoot = Join-Path $projectRoot '.setup'
$uvRoot = Join-Path $setupRoot 'uv'
$uvExe = Join-Path $uvRoot 'uv.exe'
$uvVersion = '0.11.6'
$uvZipHash = '99aa60edd017a256dbf378f372d1cff3292dbc6696e0ea01716d9158d773ab77'
New-Item -ItemType Directory -Force -Path $uvRoot | Out-Null
if (-not (Test-Path -LiteralPath $uvExe -PathType Leaf)) {
    Write-Host 'Downloading the pinned uv bootstrap...'
    $uvZip = Join-Path $setupRoot 'uv.zip'
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $uvZip
    if ((Get-FileHash -LiteralPath $uvZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $uvZipHash) {
        throw 'uv download checksum mismatch. Nothing was executed.'
    }
    Expand-Archive -LiteralPath $uvZip -DestinationPath $uvRoot -Force
}
$actualUv = & $uvExe --version
if ($LASTEXITCODE -ne 0 -or $actualUv -notmatch '^uv 0\.11\.6(?:\s|$)') {
    throw 'The local uv bootstrap has the wrong version.'
}

# Keep managed Python and caches local; never use the MSYS/system interpreter.
$env:UV_CACHE_DIR = Join-Path $setupRoot 'cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $setupRoot 'python'
$env:UV_PYTHON_BIN_DIR = Join-Path $setupRoot 'bin'
$env:PATH = $env:UV_PYTHON_BIN_DIR + [IO.Path]::PathSeparator + $env:PATH
$env:UV_PYTHON_PREFERENCE = 'only-managed'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $projectRoot '.venv'
$env:UV_HTTP_TIMEOUT = '600'
$env:PYTHONDONTWRITEBYTECODE = '1'
Write-Host 'Installing the shared Python 3.12.13 runtime...'
& $uvExe python install 3.12.13 --no-progress
if ($LASTEXITCODE -ne 0) { throw 'Managed Python installation failed.' }
Write-Host 'Synchronizing the frozen development environment (first download is several GB)...'
& $uvExe sync --locked --no-progress
if ($LASTEXITCODE -ne 0) { throw 'Dependency synchronization failed.' }
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
& $uvExe pip check --python $pythonExe
if ($LASTEXITCODE -ne 0) { throw 'Installed dependency validation failed.' }
& $pythonExe -B scripts/prepare_weights.py
if ($LASTEXITCODE -ne 0) { throw 'Pretrained-weight preparation failed.' }
& $pythonExe -B scripts/check_environment.py
if ($LASTEXITCODE -ne 0) { throw 'Environment smoke check failed.' }
Write-Host ''
Write-Host 'Setup complete. Open START_HERE.md and use .venv\Scripts\python.exe.'
