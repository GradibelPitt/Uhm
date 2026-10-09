param(
    [Parameter(Mandatory = $true)][string]$Video,
    [Parameter(Mandatory = $true)][string]$Output,
    [double]$MaxSeconds = 4
)

$ErrorActionPreference = 'Stop'
$uhmRepository = Split-Path -Parent $PSScriptRoot
$uhmKeyPath = Join-Path $uhmRepository 'runs\openai-key.clixml'
$uhmSavedSecret = Import-Clixml -LiteralPath $uhmKeyPath
if ($uhmSavedSecret -isnot [System.Security.SecureString]) {
    throw 'The saved API key is not a Windows encrypted SecureString.'
}
$uhmPreviousKey = $env:OPENAI_API_KEY
Push-Location -LiteralPath $uhmRepository
try {
    $env:OPENAI_API_KEY = ([pscredential]::new('OpenAI', $uhmSavedSecret)).GetNetworkCredential().Password
    & '.\.venv\Scripts\python.exe' -m uhm infer-actions --video $Video --output $Output --max-seconds $MaxSeconds
    if ($LASTEXITCODE -ne 0) {
        throw "Action inference failed with exit code $LASTEXITCODE."
    }
}
finally {
    $env:OPENAI_API_KEY = $uhmPreviousKey
    Pop-Location
}
