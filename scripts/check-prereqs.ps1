$ErrorActionPreference = 'Continue'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $dockerBin = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin'
    $dockerExe = Join-Path $dockerBin 'docker.exe'
    $credentialHelper = Join-Path $dockerBin 'docker-credential-desktop.exe'
    if ((Test-Path -LiteralPath $dockerExe) -and (Test-Path -LiteralPath $credentialHelper)) {
        $env:PATH = "$dockerBin;$env:PATH"
        Write-Output 'Docker Desktop: per-user command folder added for this PowerShell process only'
    }
}

function Show-CommandStatus {
    param([string]$Name)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        Write-Output "${Name}: missing"
    } else {
        Write-Output "${Name}: $($command.Source)"
    }
}

Show-CommandStatus 'docker'
Show-CommandStatus 'node'
Show-CommandStatus 'npm'
Show-CommandStatus 'ffmpeg'

try {
    $wslOutput = & wsl --status 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0) {
        Write-Output "WSL: available"
    } else {
        $cleanWslOutput = ($wslOutput -replace "`0", '' -replace '[\r\n]+', ' ').Trim()
        Write-Output "WSL: not ready ($cleanWslOutput)"
    }
} catch {
    Write-Output 'WSL: unavailable'
}

try {
    $gpu = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
    Write-Output "NVIDIA GPU: $gpu"
} catch {
    Write-Output 'NVIDIA GPU: unavailable'
}

try {
    $freeGb = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
    Write-Output "C: free: $freeGb GB"
} catch {
    Write-Output 'C: free: unknown'
}

try {
    $memory = (Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory
    Write-Output "RAM: $([math]::Round($memory / 1GB, 1)) GB"
} catch {
    Write-Output 'RAM: could not be read; check Task Manager > Performance > Memory'
}

if (Get-Command docker -ErrorAction SilentlyContinue) {
    try {
        docker info --format 'Docker engine: {{.ServerVersion}}' 2>$null
        if ($LASTEXITCODE -ne 0) { Write-Output 'Docker engine: not running' }
    } catch { Write-Output 'Docker engine: not running' }
}

