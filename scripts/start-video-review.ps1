param(
    [string]$RunDirectory = 'data/benchmarks/full-video-review-verified-20261110',
    [int]$Port = 7795
)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
Push-Location $taskRoot
try {
    if (!(Test-Path -LiteralPath (Join-Path $RunDirectory 'manifest.json'))) {
        throw 'The complete video run is not ready yet; check its progress.json first.'
    }
    python tools/serve_video_review.py --run $RunDirectory --video data/recordings/footage.mp4 --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'The local review server could not start.' }
} finally { Pop-Location }
