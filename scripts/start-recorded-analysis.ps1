param(
    # indian_adapted needs the detector-training volume and is not offered here.
    [ValidateSet('legacy', 'general', 'plate_adapted')]
    [string]$DetectorProfile = 'general',
    # Process as fast as this machine allows instead of at most real time.
    # Frames are never dropped in either mode.
    [switch]$Unpaced,
    [ValidateRange(1, 64)]
    [int]$CpuThreads = 4,
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
$dockerExecutable = if ($dockerCommand) { $dockerCommand.Source } else {
    Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
}
if (!(Test-Path -LiteralPath $dockerExecutable)) { throw 'Docker Desktop was not found.' }
if (!(Test-Path -LiteralPath (Join-Path $projectRoot '.env'))) { throw 'Run scripts/init-env.ps1 first.' }

# The models come from the handoff release; they are never downloaded here.
& $dockerExecutable volume inspect sentinel-prototype_model_cache *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'The model volume is missing. Restore the model-cache bundle first: scripts/restore-teammate-handoff.ps1.'
}
# The delivery queue volume is shared with the live launcher and never removed by compose.
& $dockerExecutable volume create sentinel_shared_recognition *> $null
if ($LASTEXITCODE -ne 0) { throw 'The delivery queue volume could not be created.' }
$running = & $dockerExecutable ps --filter name=sentinel-live-detection-viewer --format '{{.Names}}'
if ($running -contains 'sentinel-live-detection-viewer') {
    throw 'The live engineering service is using port 7780. Stop it before starting recorded analysis.'
}
$recordings = Join-Path $projectRoot 'data\recordings'
New-Item -ItemType Directory -Path $recordings -Force | Out-Null
$videos = @(Get-ChildItem -LiteralPath $recordings -File | Where-Object { $_.Name -notlike '*.receipt.json' })
$receipts = @(Get-ChildItem -LiteralPath $recordings -File -Filter '*.receipt.json')
Write-Host "Recordings folder: $recordings ($($videos.Count) file(s), $($receipts.Count) receipt(s))"
if ($receipts.Count -eq 0) {
    Write-Host 'No receipts yet. A video is listed only after tools/register_recording.py has written its receipt.'
}

Push-Location $projectRoot
try {
    $env:RECORDED_DETECTOR_PROFILE = $DetectorProfile
    $env:SENTINEL_CPU_THREADS = "$CpuThreads"
    $env:SENTINEL_RECORDED_UNPACED = if ($Unpaced) { 'true' } else { 'false' }
    $buildArgs = @('--build')
    if ($SkipBuild) { $buildArgs = @() }
    # The API is rebuilt with the service: an older API would reject recorded events.
    & $dockerExecutable compose --profile recorded up -d @buildArgs api dashboard recorded-analysis
    if ($LASTEXITCODE -ne 0) { throw 'Recorded analysis could not be started. Check Docker and the restored models.' }
    Write-Host 'Operator dashboard:  http://127.0.0.1:5173/'
    Write-Host 'Engineering console: http://127.0.0.1:7780/'
    Write-Host 'Unlock each with the local operator key, choose a recording, and start analysis explicitly.'
    Write-Host "Profile: $DetectorProfile on CPU. Footage is labelled RECORDED. Model acceptance is not verified accuracy."
    Write-Host 'Stop with: docker compose --profile recorded stop recorded-analysis'
} finally { Pop-Location }
