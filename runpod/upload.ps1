# Upload Windows -> pod RunPod (volume /workspace).
# Remplir .env a la racine (RUNPOD_HOST, RUNPOD_PORT, RUNPOD_SSH_KEY),
# puis :  .\runpod\upload.ps1
#
# runpodctl (gros fichiers) :
#   runpodctl config --apiKey $env:RUNPOD_API_KEY
#   runpodctl send data\crop130_rgb25d
param(
    [string]$HostName = $env:RUNPOD_HOST,
    [string]$Port = $(if ($env:RUNPOD_PORT) { $env:RUNPOD_PORT } else { "22" }),
    [string]$IdentityFile = $env:RUNPOD_SSH_KEY,
    [string]$Remote = "/workspace"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$dotenv = Join-Path $Root ".env"
if (Test-Path $dotenv) {
    Get-Content $dotenv | ForEach-Object {
        if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
        $eq = $_.IndexOf("=")
        if ($eq -lt 1) { return }
        $k = $_.Substring(0, $eq).Trim()
        $v = $_.Substring($eq + 1).Trim().Trim('"').Trim("'")
        if ($k) { Set-Item -Path "Env:$k" -Value $v }
    }
}
if (-not $HostName) { $HostName = $env:RUNPOD_HOST }
if (-not $Port) { $Port = $env:RUNPOD_PORT }
if (-not $Port) { $Port = "22" }
if (-not $IdentityFile) { $IdentityFile = $env:RUNPOD_SSH_KEY }
if (-not $HostName) { throw "definir RUNPOD_HOST dans .env" }

$Root = Split-Path -Parent $PSScriptRoot
$items = @(
    @{ Local = "data\crop130_rgb25d"; Remote = "data/crop130_rgb25d" },
    @{ Local = "data\labels\labels_llm_gpt56sol.csv"; Remote = "data/labels/" },
    @{ Local = "data\csv\train.csv"; Remote = "data/csv/" },
    @{ Local = "data\csv\train_series.csv"; Remote = "data/csv/" },
    @{ Local = "weights\dinov3-vitb16-lvd1689m\model.safetensors"; Remote = "weights/dinov3-vitb16-lvd1689m/" },
    @{ Local = "weights\dinov3-vitb16-lvd1689m\config.json"; Remote = "weights/dinov3-vitb16-lvd1689m/" }
)

$sshArgs = @("-p", $Port)
$scpArgs = @("-P", $Port)
if ($IdentityFile) {
    if (-not (Test-Path $IdentityFile)) { throw "RUNPOD_SSH_KEY introuvable : $IdentityFile" }
    $sshArgs += @("-i", $IdentityFile)
    $scpArgs += @("-i", $IdentityFile)
}

ssh @sshArgs "root@$HostName" "mkdir -p $Remote/data/labels $Remote/data/csv $Remote/data/crop130_rgb25d $Remote/weights/dinov3-vitb16-lvd1689m $Remote/checkpoints"

foreach ($it in $items) {
    $src = Join-Path $Root $it.Local
    if (-not (Test-Path $src)) { throw "manquant : $src" }
    Write-Host "scp $($it.Local) -> $($it.Remote)"
    scp @scpArgs -r $src "root@${HostName}:$Remote/$($it.Remote)"
}

Write-Host "ensuite sur le pod : git clone / rsync du repo, puis bash runpod/setup.sh"
