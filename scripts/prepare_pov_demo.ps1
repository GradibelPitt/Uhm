$ErrorActionPreference = 'Stop'
$uhmRepository = Split-Path -Parent $PSScriptRoot
$uhmPython = Join-Path $uhmRepository '.venv\Scripts\python.exe'
$uhmSource = Join-Path $uhmRepository 'data\raw\pov\prague_4k.webm'
$uhmDemo = Join-Path $uhmRepository 'data\raw\pov\prague_demo_4k.mp4'
$uhmProxy = Join-Path $uhmRepository 'data\raw\pov\prague_demo_4k.analysis.mp4'
if (-not (Test-Path -LiteralPath $uhmSource)) { throw "Missing source video: $uhmSource" }
$uhmFfmpeg = & $uhmPython -X utf8 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'
& $uhmFfmpeg -hide_banner -loglevel warning -nostdin -y -ss 2 -i $uhmSource -t 118 -map 0:v:0 -an -c:v libx264 -preset fast -crf 19 -pix_fmt yuv420p -r 30000/1001 -movflags +faststart $uhmDemo
if ($LASTEXITCODE -ne 0) { throw 'Demo trim failed.' }
& $uhmFfmpeg -hide_banner -loglevel warning -nostdin -y -i $uhmDemo -vf scale=640:360 -an -c:v libx264 -preset veryfast -crf 20 $uhmProxy
if ($LASTEXITCODE -ne 0) { throw 'Analysis proxy generation failed.' }
@{
    source_sha256 = (Get-FileHash -LiteralPath $uhmDemo -Algorithm SHA256).Hash.ToLowerInvariant()
    source_start_s = 2
    source_end_s = 120
} | ConvertTo-Json | Set-Content -LiteralPath ([IO.Path]::ChangeExtension($uhmProxy, '.json')) -Encoding utf8NoBOM
Write-Output "Prepared 4K demo from original 00:02 to 02:00: $uhmDemo"
