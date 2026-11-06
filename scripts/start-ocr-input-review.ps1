param(
    [string]$ReviewRoot = 'data/benchmarks/recorded-footage-20261104/manual-review',
    [string]$OutputPath = 'data/benchmarks/reports/ocr-input-review-20261106',
    [ValidateRange(1024, 65535)]
    [int]$Port = 7777,
    [switch]$SkipGenerate
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$reviewPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $ReviewRoot))
$outputDirectory = [IO.Path]::GetFullPath((Join-Path $projectRoot $OutputPath))
if (!(Test-Path -LiteralPath (Join-Path $reviewPath 'manifest.json'))) {
    throw 'The locally labelled crop set is missing. Restore its manifest, labels and images first.'
}
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
if (!$SkipGenerate) {
    $dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
    $dockerExecutable = if ($dockerCommand) { $dockerCommand.Source } else {
        Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
    }
    if (!(Test-Path -LiteralPath $dockerExecutable)) { throw 'Docker Desktop was not found.' }
    $arguments = @('run', '--rm', '--gpus', 'all', '--network', 'none', '--workdir', '/workspace',
        '--mount', 'type=volume,source=sentinel-prototype_model_cache,target=/models,readonly',
        '--mount', "type=bind,source=$projectRoot/edge,target=/workspace/edge,readonly",
        '--mount', "type=bind,source=$projectRoot/tools,target=/workspace/tools,readonly",
        '--mount', "type=bind,source=$reviewPath,target=/review,readonly",
        '--mount', "type=bind,source=$outputDirectory,target=/output",
        '--entrypoint', 'python', 'sentinel-prototype-edge-worker:latest',
        '/workspace/tools/run_ocr_research.py', '-m', 'sentinel_edge.ocr_input_review',
        '--review-root', '/review', '--output', '/output', '--device', 'gpu')
    & $dockerExecutable @arguments
    if ($LASTEXITCODE -ne 0) { throw 'The OCR input review could not be generated.' }
}
if (!(Test-Path -LiteralPath (Join-Path $outputDirectory 'index.html'))) {
    throw 'The generated review is missing. Run again without -SkipGenerate.'
}
$url = "http://127.0.0.1:$Port/"
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    try {
        $served = Invoke-WebRequest -Uri ($url + 'review.json') -TimeoutSec 3 -UseBasicParsing
        $expected = Get-Content -LiteralPath (Join-Path $outputDirectory 'review.json') -Raw
        if ($served.Content -ne $expected) { throw 'Different content' }
    } catch { throw "Port $Port is occupied by a different service. Choose another -Port." }
} else {
    $python = (Get-Command python -ErrorAction Stop).Source
    $server = Start-Process -FilePath $python -WindowStyle Hidden -PassThru -WorkingDirectory $outputDirectory `
        -ArgumentList @('-m', 'http.server', "$Port", '--bind', '127.0.0.1') `
        -RedirectStandardOutput (Join-Path $outputDirectory 'server.stdout.log') `
        -RedirectStandardError (Join-Path $outputDirectory 'server.stderr.log')
    $server.Id | Set-Content -LiteralPath (Join-Path $outputDirectory 'server.pid')
    $ready = $false
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $url -TimeoutSec 1 -UseBasicParsing
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        } catch { Start-Sleep -Milliseconds 200 }
    }
    if (!$ready) { throw 'The local review server did not start. Check server.stderr.log.' }
}
Write-Host "OCR crop review: $url"
Write-Host 'This local page contains your labelled recording crops and full OCR diagnostics.'
