param(
    [string]$RunDirectory = 'data/benchmarks/vehicle-tracking-960-20261110',
    [int]$Port = 7798
)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
Push-Location $taskRoot
try {
    if (!(Test-Path -LiteralPath (Join-Path $RunDirectory 'report.json'))) {
        throw 'The tracking benchmark is not ready. Complete capture, replay and thumbnails first.'
    }
    python tools/serve_tracking_review.py --run $RunDirectory --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'The local tracking review server could not start.' }
} finally { Pop-Location }
