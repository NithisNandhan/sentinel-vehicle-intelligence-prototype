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
    Write-Host 'Legacy launcher: use start-recorded-analysis.ps1 for the current recorded-footage product.'
    Write-Host 'Sentinel dashboard: http://127.0.0.1:5173/'
    Write-Host 'The worker remains a background service. Diagnostics now open in the dashboard Inspector.'
} finally { Pop-Location }
