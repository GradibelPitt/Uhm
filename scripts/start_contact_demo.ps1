param(
    [string]$Video = 'data\raw\pov\prague_demo_4k.mp4',
    [string]$Timeline = 'runs\pov_demo.json',
    [string]$Config = 'configs\pov_motion.yaml',
    [int]$Port = 8001,
    [switch]$Refresh
)
$ErrorActionPreference = 'Stop'
$uhmRepository = Split-Path -Parent $PSScriptRoot
$uhmPython = Join-Path $uhmRepository '.venv\Scripts\python.exe'
$uhmConfig = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $uhmRepository $Config }
$uhmVideo = if ([IO.Path]::IsPathRooted($Video)) { $Video } else { Join-Path $uhmRepository $Video }
$uhmTimeline = if ([IO.Path]::IsPathRooted($Timeline)) { $Timeline } else { Join-Path $uhmRepository $Timeline }
if (-not (Test-Path -LiteralPath $uhmPython)) { throw 'Install the project dependencies first: uv sync' }
if (-not (Test-Path -LiteralPath $uhmVideo)) { throw "Video not found: $uhmVideo" }
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Port must be between 1 and 65535.' }
$uhmReuse = $false
if (-not $Refresh -and (Test-Path -LiteralPath $uhmTimeline)) {
    $uhmReport = Get-Content -LiteralPath $uhmTimeline -Raw | ConvertFrom-Json
    $uhmReuse = ($uhmReport.video -eq (Resolve-Path -LiteralPath $uhmVideo).Path) -and
        ($uhmReport.video_sha256 -eq (Get-FileHash -LiteralPath $uhmVideo -Algorithm SHA256).Hash.ToLowerInvariant()) -and
        ($uhmReport.config_sha256 -eq (Get-FileHash -LiteralPath $uhmConfig -Algorithm SHA256).Hash.ToLowerInvariant())
    if ($uhmReport.task -eq 'egocentric_motion' -and -not $uhmReport.analysis_complete) { $uhmReuse = $false }
}
Push-Location -LiteralPath $uhmRepository
try {
    if (-not $uhmReuse) {
        & (Join-Path $PSScriptRoot 'start_local_vision.ps1')
        Write-Output 'Analyzing video with the local model. This preparation is offline, not live inference.'
        & $uhmPython -m uhm infer-local-contacts --video $uhmVideo --output $uhmTimeline --config $uhmConfig
        if ($LASTEXITCODE -ne 0) { throw 'Local contact analysis failed.' }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $uhmRepository 'frontend\node_modules'))) {
        & npm.cmd --prefix frontend ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    }
    & npm.cmd --prefix frontend run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    Write-Output "Haptic Studio: http://127.0.0.1:$Port"
    & $uhmPython -m uhm serve-actions $uhmVideo $uhmTimeline --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Contact demo server stopped with an error.' }
}
finally { Pop-Location }
