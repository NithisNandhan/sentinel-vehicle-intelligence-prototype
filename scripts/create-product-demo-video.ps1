$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$deliverables = Join-Path $projectRoot 'deliverables'
New-Item -ItemType Directory -Path $deliverables -Force | Out-Null

$workDirectory = Join-Path $env:TEMP ('sentinel-demo-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $workDirectory | Out-Null
$scenes = @(
    @{ id = 'opening'; title = 'From camera feeds to operator action'; voiceover = 'Sentinel Vehicle Intelligence is a prototype that connects regional camera analysis with a central operator workflow.' },
    @{ id = 'challenge'; title = 'One review workflow for selected feeds'; voiceover = 'The goal is to turn selected live feeds into reviewable vehicle events, while keeping uncertain recognition separate from confirmed facts.' },
    @{ id = 'architecture'; title = 'Regional analysis. Central command.'; voiceover = 'An edge worker reads selected streams, tracks vehicles, and submits privacy-limited events to a central API and command dashboard.' },
    @{ id = 'recognition'; title = 'Recognition gathers evidence over time'; voiceover = 'For each tracked vehicle, the system gathers plate crops from multiple frames. OCR alternatives are fused over time; blur is not magically restored.' },
    @{ id = 'operator'; title = 'The operator command centre'; voiceover = 'Operators can see camera health, search authorised cameras, open preview-only feeds, review alerts, reconstruct routes, and prepare reports.' },
    @{ id = 'synthetic-alert'; title = 'Alert handling, clearly labelled'; voiceover = 'The highlighted alert is synthetic demo data. It shows watchlist matching, acknowledgement, audit history, and resolution—not a real plate recognition.' },
    @{ id = 'engineering'; title = 'A separate view for model debugging'; voiceover = 'A separate engineering console reveals detection boxes, original pixel crops, OCR suggestions, candidate status, and processing timings for debugging.' },
    @{ id = 'privacy'; title = 'Designed for resilience and privacy'; voiceover = 'Images and full plate text stay out of persistent records. The local outbox retries metadata after a central outage and deduplicates events.' },
    @{ id = 'evidence'; title = 'What the current evidence says'; voiceover = 'Live tests show vehicle detections and occasional plate crops, but we have not verified a correct live plate read against ground truth.' },
    @{ id = 'roadmap'; title = 'The next proof points'; voiceover = 'Next steps are labelled Indian-plate evaluation, better camera and image-quality coverage, and measured accuracy before any production claim.' }
)
$scenes | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $workDirectory 'scenes.json') -Encoding utf8

$runtimeRoot = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node'
$nodeExecutable = Join-Path $runtimeRoot 'bin\node.exe'
$nodeModules = Join-Path $runtimeRoot 'node_modules'
if (!(Test-Path -LiteralPath $nodeExecutable)) {
    $nodeCommand = Get-Command node -ErrorAction SilentlyContinue
    if (!$nodeCommand) { throw 'Node.js is required to record this video.' }
    $nodeExecutable = $nodeCommand.Source
}
if (Test-Path -LiteralPath $nodeModules) { $env:NODE_PATH = $nodeModules }
$renderer = Join-Path $PSScriptRoot 'render-product-demo-video.cjs'
& $nodeExecutable $renderer $workDirectory $deliverables
if ($LASTEXITCODE -ne 0) { throw 'The video renderer failed.' }
Write-Host ''
Write-Host 'Video:    ' (Join-Path $deliverables 'Sentinel_Vehicle_Intelligence_Demo.webm')
Write-Host 'Captions: ' (Join-Path $deliverables 'Sentinel_Vehicle_Intelligence_Demo.vtt')
Write-Host 'Temporary build files: ' $workDirectory
