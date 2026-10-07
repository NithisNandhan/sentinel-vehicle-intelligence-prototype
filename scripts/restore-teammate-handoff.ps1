param(
    [Parameter(Mandatory=$true)][string]$AssetsPath,
    [switch]$SkipTraining,
    [switch]$SkipBenchmarks,
    [switch]$SkipModels,
    [switch]$SkipCentral
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$assets = (Resolve-Path -LiteralPath $AssetsPath).Path
$tools = Join-Path $projectRoot 'tools'
$restored = Join-Path $projectRoot 'data\handoff-restored'
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.env'))) {
    throw 'Run scripts/init-env.ps1 first. Do not copy the original laptop credentials.'
}
function Run-Docker {
    param([string[]]$DockerArgs)
    & docker @DockerArgs
    if ($LASTEXITCODE -ne 0) { throw 'Docker handoff step failed. Existing data has not been deliberately overwritten.' }
}
if (-not $SkipCentral) {
    $apiContainer = & docker compose ps -q api
    if ($apiContainer) { throw 'Restore central events before starting the API, on a fresh installation only.' }
}
Run-Docker -DockerArgs @('compose','--profile','offline','build','api','dashboard','demo-simulator')
$helper = 'sentinel-prototype-api:latest'
function Restore-Archive {
    param([string]$Name, [string]$Volume)
    $manifest = Join-Path $assets "$Name.manifest.json"
    if (-not (Test-Path -LiteralPath $manifest)) { throw "Missing manifest: $Name" }
    if ($Volume) {
        Run-Docker -DockerArgs @('volume','create',$Volume)
        $destination = "type=volume,source=$Volume,target=/restore"
    } else {
        $folder = Join-Path $restored $Name
        New-Item -ItemType Directory -Path $folder -Force | Out-Null
        $destination = "type=bind,source=$folder,target=/restore"
    }
    Run-Docker -DockerArgs @('run','--rm','--entrypoint','python',
        '--mount',"type=bind,source=$tools,target=/handoff-tools,readonly",
        '--mount',"type=bind,source=$assets,target=/assets,readonly",
        '--mount',$destination,$helper,'/handoff-tools/handoff_archives.py','restore',
        '--manifest',"/assets/$Name.manifest.json",'--destination','/restore')
}
if (-not $SkipModels) {
    Restore-Archive -Name 'model-cache' -Volume 'sentinel-prototype_model_cache'
    Restore-Archive -Name 'alpr-cache' -Volume 'sentinel-prototype_alpr_cache'
}
if (-not $SkipTraining) {
    Restore-Archive -Name 'detector-training' -Volume 'sentinel_detector_training'
}
if (-not $SkipBenchmarks) { Restore-Archive -Name 'benchmark-data' }
Restore-Archive -Name 'saved-events'
Restore-Archive -Name 'presentation-assets'
if (-not $SkipCentral) {
    $events = Join-Path $restored 'saved-events'
    Run-Docker -DockerArgs @('compose','up','-d','database')
    Run-Docker -DockerArgs @('compose','run','--rm','--entrypoint','sh',
        '-v',"${tools}:/handoff-tools:ro",
        '-v',"${events}:/snapshot:ro",'api','-c',
        'alembic upgrade head && python /handoff-tools/handoff_events.py restore-central --snapshot /snapshot/central-events.json')
    $queues = @(
        @('sentinel-prototype_edge_data','production-edge-outbox.sqlite3','outbox.sqlite3'),
        @('sentinel-prototype_demo_edge_data','offline-demo-outbox.sqlite3','outbox.sqlite3'),
        @('sentinel_shared_recognition','shared-recognition-shared-recognition.sqlite3','shared-recognition.sqlite3')
    )
    foreach ($queue in $queues) {
        Run-Docker -DockerArgs @('volume','create',$queue[0])
        Run-Docker -DockerArgs @('run','--rm','--entrypoint','python',
            '--mount',"type=bind,source=$tools,target=/handoff-tools,readonly",
            '--mount',"type=bind,source=$events,target=/snapshot,readonly",
            '--mount',"type=volume,source=$($queue[0]),target=/restore",$helper,
            '/handoff-tools/handoff_events.py','restore-outbox',
            '--snapshot',"/snapshot/$($queue[1])",'--destination',"/restore/$($queue[2])")
    }
    Run-Docker -DockerArgs @('compose','up','-d','api','dashboard','mediamtx')
}
Write-Host "Handoff restored. Supplemental files: $restored"
Write-Host 'Operator dashboard: http://127.0.0.1:5173/ (unlock with your new local operator key)'
Write-Host 'Recorded-video dashboard integration is the next task; no Sentinel credentials were restored.'
