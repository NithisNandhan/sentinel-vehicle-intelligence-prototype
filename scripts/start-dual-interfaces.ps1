param([switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
$dockerExecutable = if ($dockerCommand) { $dockerCommand.Source } else {
    Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
}
if (!(Test-Path -LiteralPath $dockerExecutable)) { throw 'Docker Desktop was not found.' }
Push-Location $projectRoot
try {
    $buildArgs = @('--build')
    if ($SkipBuild) { $buildArgs = @() }
    & $dockerExecutable compose up -d @buildArgs api dashboard
    if ($LASTEXITCODE -ne 0) { throw 'The central services could not be started.' }
    $running = & $dockerExecutable ps --filter name=sentinel-live-detection-viewer --format '{{.Names}}'
    if ($running -contains 'sentinel-live-detection-viewer') {
        Write-Host 'An existing engineering service is running. Stop it explicitly before using updated code.'
    } else {
        & (Join-Path $PSScriptRoot 'start-live-detection-viewer.ps1') -Background -DeliverAccepted -LifetimeMinutes 480
    }
    Write-Host 'Operator dashboard: http://127.0.0.1:5173/'
    Write-Host 'Engineering console: http://127.0.0.1:7780/'
    Write-Host 'Unlock each using the local operator key. Start one shared camera analysis explicitly.'
} finally { Pop-Location }
