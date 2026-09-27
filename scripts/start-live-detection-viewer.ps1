param(
    [ValidateSet('legacy', 'general', 'plate_adapted', 'indian_adapted')]
    [string]$DetectorProfile = 'general',
    [switch]$DeliverAccepted,
    [switch]$Background,
    [ValidateRange(1, 480)]
    [int]$LifetimeMinutes = 30
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
$dockerExecutable = if ($dockerCommand) { $dockerCommand.Source } else {
    Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
}
if (!(Test-Path -LiteralPath $dockerExecutable)) { throw 'Docker Desktop was not found.' }
if (!(Test-Path -LiteralPath (Join-Path $projectRoot '.env'))) { throw 'The local .env file is required.' }
Write-Host 'Open http://127.0.0.1:7780/ and unlock with your local operator key. No footage is recorded.'
Write-Host "The shared session pauses when idle and the service shuts down after $LifetimeMinutes minutes."
Write-Host "Experimental tracked profile: $DetectorProfile. Model acceptance is not verified plate accuracy."
$backgroundArgs = @()
if ($Background) { $backgroundArgs = @('-d') }
$deliveryArgs = @()
if ($DeliverAccepted) { $deliveryArgs = @('--deliver') }
& $dockerExecutable run --rm --name sentinel-live-detection-viewer --gpus all `
    @backgroundArgs `
    --network sentinel-prototype_default --env-file (Join-Path $projectRoot '.env') `
    -e MODEL_CACHE_DIR=/models -e MEDIA_RELAY_BASE_URL=rtsp://mediamtx:8554 `
    -e CENTRAL_API_URL=http://api:8000 -e EDGE_DB_PATH=/data/shared-recognition.sqlite3 `
    -e YOLO_CONFIG_DIR=/tmp/sentinel-yolo `
    -p 127.0.0.1:7780:7780 `
    --mount "type=bind,source=$projectRoot/edge,target=/workspace/edge,readonly" `
    --mount "type=bind,source=$projectRoot/tools,target=/workspace/tools,readonly" `
    --mount type=volume,source=sentinel-prototype_model_cache,target=/models,readonly `
    --mount type=volume,source=sentinel-prototype_alpr_cache,target=/root/.cache,readonly `
    --mount type=volume,source=sentinel_detector_training,target=/training,readonly `
    --mount type=volume,source=sentinel_shared_recognition,target=/data `
    -w /workspace/edge sentinel-prototype-edge-worker:latest `
    python /workspace/tools/run_ocr_research.py -m sentinel_edge.live_detection_viewer `
    --cache /models/experiments/awiros `
    --checkpoint /models/experiments/awiros/adaptations/indian-corpus-encoder.safetensors `
    --container-bind --detector-profile $DetectorProfile --lifetime ($LifetimeMinutes * 60) --configured-relays @deliveryArgs
if ($LASTEXITCODE -ne 0) { throw 'The viewer stopped with an error. Check Docker and the cached models.' }
