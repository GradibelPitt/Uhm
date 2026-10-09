$ErrorActionPreference = 'Stop'
$uhmRepository = Split-Path -Parent $PSScriptRoot
$uhmKeyDirectory = Join-Path $uhmRepository 'runs'
New-Item -ItemType Directory -Path $uhmKeyDirectory -Force | Out-Null
$uhmSavedSecret = Read-Host 'Paste the new OpenAI API key (input is hidden)' -AsSecureString
if ($uhmSavedSecret.Length -eq 0) {
    throw 'No API key entered.'
}
$uhmSavedSecret | Export-Clixml -LiteralPath (Join-Path $uhmKeyDirectory 'openai-key.clixml')
Write-Output 'API key saved with Windows user encryption in the ignored runs folder.'
