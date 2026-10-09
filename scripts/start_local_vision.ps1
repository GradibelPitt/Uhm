$ErrorActionPreference = 'Stop'
$uhmRepository = Split-Path -Parent $PSScriptRoot
$uhmRuntime = Join-Path $env:LOCALAPPDATA 'UHM\runtimes\ollama-v0.40.0\ollama.exe'
if (-not (Test-Path -LiteralPath $uhmRuntime)) { throw 'Official Ollama runtime is not installed at the configured location.' }
try {
    $uhmExisting = Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/version' -TimeoutSec 2
    if ($uhmExisting.version) { Write-Output 'Local vision runtime is already running.'; return }
}
catch { }
$uhmPreviousHost = $env:OLLAMA_HOST
$uhmPreviousModels = $env:OLLAMA_MODELS
$uhmPreviousParallel = $env:OLLAMA_NUM_PARALLEL
try {
    $env:OLLAMA_HOST = '127.0.0.1:11435'
    $env:OLLAMA_MODELS = Join-Path $env:LOCALAPPDATA 'UHM\models'
    $env:OLLAMA_NUM_PARALLEL = '1'
    New-Item -ItemType Directory -Path $env:OLLAMA_MODELS -Force | Out-Null
    $uhmProcess = Start-Process -FilePath $uhmRuntime -ArgumentList 'serve' -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $uhmRepository 'runs\ollama.stdout.log') -RedirectStandardError (Join-Path $uhmRepository 'runs\ollama.stderr.log')
    [pscustomobject]@{process_id=$uhmProcess.Id;endpoint='http://127.0.0.1:11435';models=$env:OLLAMA_MODELS} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $uhmRepository 'runs\local_vision_runtime.json') -Encoding utf8
    Write-Output "Local vision runtime started (PID $($uhmProcess.Id))."
}
finally {
    $env:OLLAMA_HOST = $uhmPreviousHost
    $env:OLLAMA_MODELS = $uhmPreviousModels
    $env:OLLAMA_NUM_PARALLEL = $uhmPreviousParallel
}
