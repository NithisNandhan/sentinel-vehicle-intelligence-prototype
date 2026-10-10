param(
    [switch]$Force
)

$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $workspace '.env.example'
$target = Join-Path $workspace '.env'
if ((Test-Path -LiteralPath $target) -and -not $Force) {
    Write-Output '.env already exists; leaving it unchanged.'
    exit 0
}

function New-SecretHex {
    param([int]$ByteCount = 32)
    $bytes = New-Object byte[] $ByteCount
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    return -join ($bytes | ForEach-Object { $_.ToString('x2') })
}

$settings = Get-Content -LiteralPath $source -Raw
$settings = $settings.Replace('GENERATE_DB_PASSWORD', (New-SecretHex 24))
$settings = $settings.Replace('GENERATE_EDGE_API_KEY', (New-SecretHex 32))
$settings = $settings.Replace('GENERATE_OPERATOR_API_KEY', (New-SecretHex 32))
$settings = $settings.Replace('GENERATE_HMAC_KEY', (New-SecretHex 32))
# Docker Compose must not see a byte-order mark before the first key. Windows
# PowerShell 5.1 has no utf8NoBOM encoding name, so write the bytes directly.
[System.IO.File]::WriteAllText($target, $settings, (New-Object System.Text.UTF8Encoding($false)))
Write-Output 'Created .env with random local keys. Add Sentinel credentials there only when using online mode.'

