param([Parameter(ValueFromRemainingArguments=$true)][string[]]$AppArguments)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$uvVersion='0.13.0'
$archiveHash='088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c'
$binaryHash='2bfbe53d3cc95799e98fa08323b04844fc01d4f8c87dd5c00089e43f6554f162'

function File-Sha256([string]$path) {
    $stream=[IO.File]::OpenRead($path)
    $algorithm=[Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
    finally { $algorithm.Dispose(); $stream.Dispose() }
}

function Local-Directory([string]$relative) {
    $target=[IO.Path]::GetFullPath((Join-Path $projectRoot $relative))
    if (-not $target.StartsWith($projectRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {
        throw 'Runtime directory escapes the project.'
    }
    $current=$projectRoot
    foreach ($part in $relative.Split([char[]]@('\','/'),[StringSplitOptions]::RemoveEmptyEntries)) {
        $current=Join-Path $current $part
        if (Test-Path -LiteralPath $current) {
            $item=Get-Item -LiteralPath $current -Force
            if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw ('Runtime directory must be a local directory: '+$current)
            }
        } else { New-Item -ItemType Directory -Path $current | Out-Null }
    }
    return $target
}

try {
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw 'The automatic Windows runtime supports x86-64.' }
    $toolDir=Local-Directory ('.local\uv-tools\'+$uvVersion)
    $pythonDir=Local-Directory '.local\uv-python'
    $cacheDir=Local-Directory '.local\uv-cache'
    $uvPath=Join-Path $toolDir 'uv.exe'
    $valid=(Test-Path -LiteralPath $uvPath -PathType Leaf)
    if ($valid) { $valid=(File-Sha256 $uvPath) -eq $binaryHash }
    if (-not $valid) {
        Write-Host 'DM100: Downloading the verified Python bootstrap tool (first setup).'
        [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12
        $archive=Join-Path $toolDir 'uv-download.zip'
        Invoke-WebRequest -UseBasicParsing -Uri ('https://github.com/astral-sh/uv/releases/download/'+$uvVersion+'/uv-x86_64-pc-windows-msvc.zip') -OutFile $archive -TimeoutSec 180
        if ((File-Sha256 $archive) -ne $archiveHash) { throw 'uv download checksum mismatch; it was not executed.' }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $zip=[IO.Compression.ZipFile]::OpenRead($archive)
        try {
            $entry=$zip.GetEntry('uv.exe')
            if ($null -eq $entry) { throw 'uv.exe is absent from the verified archive.' }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$uvPath,$true)
        } finally { $zip.Dispose() }
        if ((File-Sha256 $uvPath) -ne $binaryHash) { throw 'uv executable checksum mismatch.' }
    }
    $env:UV_PYTHON_INSTALL_DIR=$pythonDir
    $env:UV_CACHE_DIR=$cacheDir
    foreach ($variable in @('PYTHONHOME','PYTHONPATH','UV_PYTHON','UV_CONFIG_FILE','UV_PROJECT','UV_WORKING_DIR','UV_NO_MANAGED_PYTHON','UV_PYTHON_DOWNLOADS','UV_OFFLINE','UV_PYTHON_INSTALL_MIRROR','UV_PYTHON_DOWNLOADS_JSON_URL')) {
        Remove-Item -LiteralPath ('Env:\'+$variable) -ErrorAction SilentlyContinue
    }
    & $uvPath --no-config python install --no-bin --no-registry 3.12.15
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare Python. Check internet access and retry.' }
    $runtime=& $uvPath --no-config python find --managed-python 3.12.15
    if ($LASTEXITCODE -ne 0) { throw 'The managed Python executable could not be located.' }
    $pythonPath=$runtime.Trim()
    & $pythonPath -I -X utf8 (Join-Path $projectRoot 'scripts\launch_dm100.py') @AppArguments
    exit $LASTEXITCODE
} catch {
    Write-Host ('DM100: '+$_.Exception.Message)
    exit 1
}
