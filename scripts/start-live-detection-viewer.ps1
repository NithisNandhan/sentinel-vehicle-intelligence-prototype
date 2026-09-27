$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
$dockerExecutable = if ($dockerCommand) { $dockerCommand.Source } else {
    Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
}
if (!(Test-Path -LiteralPath $dockerExecutable)) { throw 'Docker Desktop was not found.' }
if (!(Test-Path -LiteralPath (Join-Path $projectRoot '.env'))) { throw 'The local .env file is required.' }
Write-Host 'Open http://127.0.0.1:7780/ and choose CAM01 or CAM02. No footage is recorded.'
Write-Host 'The viewer pauses when idle and shuts down after 30 minutes. Ctrl+C stops it sooner.'
& $dockerExecutable run --rm --name sentinel-live-detection-viewer --gpus all `
    --network sentinel-prototype_default --env-file (Join-Path $projectRoot '.env') `
    -e MODEL_CACHE_DIR=/models -e MEDIA_RELAY_BASE_URL=rtsp://mediamtx:8554 `
    -e CAMERA_IDS=cam01,cam02 -e YOLO_CONFIG_DIR=/tmp/sentinel-yolo `
    -p 127.0.0.1:7780:7780 `
    --mount "type=bind,source=$projectRoot,target=/workspace,readonly" `
    --mount type=volume,source=sentinel-prototype_model_cache,target=/models,readonly `
    --mount type=volume,source=sentinel-prototype_alpr_cache,target=/root/.cache,readonly `
    -w /workspace/edge sentinel-prototype-edge-worker:latest `
    python /workspace/tools/run_ocr_research.py -m sentinel_edge.live_detection_viewer `
    --cache /models/experiments/awiros `
    --checkpoint /models/experiments/awiros/adaptations/indian-corpus-encoder.safetensors `
    --container-bind
if ($LASTEXITCODE -ne 0) { throw 'The viewer stopped with an error. Check Docker and the cached models.' }
