# Qwen3.8-Flash-Next on the Windows build of Gufo. start.ps1 (same folder)
# is the interactive launcher for every model; this script is the scriptable one.
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next.ps1                # serve: UD-Q4_K_XL, thinking, adaptive MTP
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next.ps1 -Think off     # instruct mode (thinking off)
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next.ps1 -Draft mtp3    # MTP capped at 3 drafts
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next.ps1 -Draft off     # autoregressive baseline
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next.ps1 -Mode bench    # pp/tg at depths 0..128K
#   ... -DraftVocab -Survival -Lookup   the opt-in MTP options (docs\models\qwen3.8-flash-next)
#
# Files come from the Hugging Face cache (see docs\WINDOWS.md for the
# download); -Snapshot points at another copy of unsloth/Qwen3.8-Flash-Next-GGUF.
# Ready when the log shows event=load_completed and GET /health returns 200.
param(
  [ValidateSet("serve", "bench")] [string]$Mode = "serve",
  [ValidateSet("on", "off")] [string]$Think = "on",
  [ValidateSet("mtp", "mtp3", "off")] [string]$Draft = "mtp",
  [switch]$Greedy,
  [switch]$DraftVocab,
  [switch]$Survival,
  [switch]$Lookup,
  [int]$Context = 262144,
  [int]$Port = 8080,
  [string]$Snapshot = "",
  [string]$MtpModel = ""
)
$ErrorActionPreference = "Stop"
$bin = Join-Path $PSScriptRoot "..\..\build\release"
if (-not (Test-Path "$bin\gufo.exe")) {
  throw "gufo.exe not found in $bin; build first: powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1"
}

$modelFile = "UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
$mtpFile = "MTP\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
# Each Hugging Face revision is its own snapshot folder, holding only the files
# downloaded at that revision: pick the newest one that has each file.
$snapshots = @()
if (-not $Snapshot) {
  $cache = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE }
           elseif ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" }
           else { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }
  $snapshots = @(Get-ChildItem (Join-Path $cache "models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots") `
    -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending |
    Select-Object -ExpandProperty FullName)
  $Snapshot = $snapshots | Where-Object { Test-Path (Join-Path $_ $modelFile) } | Select-Object -First 1
  if (-not $Snapshot) {
    throw "$modelFile not found in unsloth/Qwen3.8-Flash-Next-GGUF under $cache; see docs\WINDOWS.md or pass -Snapshot"
  }
}
$model = Join-Path $Snapshot $modelFile
if (-not (Test-Path $model)) { throw "model not found: $model" }
if (-not $MtpModel) {
  $MtpModel = @($Snapshot) + $snapshots | ForEach-Object { Join-Path $_ $mtpFile } |
    Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $MtpModel) { $MtpModel = Join-Path $Snapshot $mtpFile }
}
if ($Draft -ne "off" -and -not (Test-Path $MtpModel)) { throw "MTP model not found: $MtpModel (or use -Draft off)" }
# gufo finds the vision sidecar only beside the model or one folder up, i.e.
# in the same snapshot; one downloaded at another revision needs --mmproj.
$vision = @(@($Snapshot) + $snapshots | ForEach-Object { Join-Path $_ "mmproj-BF16.gguf" } |
  Where-Object { Test-Path $_ }) | Select-Object -First 1

# Never greedy by default. Thinking uses the sampler embedded in the GGUF
# (general.sampling.*: temp 1.0, top-p 0.95, top-k 20); instruct uses Qwen's
# non-thinking convention (0.7 / 0.8 / 20). Clients may override per request.
$sampling = if ($Think -eq "on") {
  @("--temperature", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0")
} else {
  @("--temperature", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0")
}
# -Greedy: temperature 0, the method of gufo's published tables, for
# like-for-like comparisons only.
if ($Greedy) { $sampling = @("--temperature", "0") }
$speculative = switch ($Draft) {
  "mtp" { @("--speculative", "mtp", "--mtp-model", $MtpModel) }
  "mtp3" { @("--speculative", "mtp", "--mtp-model", $MtpModel, "--draft-tokens", "3") }
  "off" { @() }
}
if ($Draft -ne "off" -and $Mode -eq "serve") {
  if ($DraftVocab) { $speculative += @("--mtp-draft-vocab", "latin") }
  if ($Survival) { $speculative += @("--mtp-policy", "survival") }
  if ($Lookup) { $speculative += "--prompt-lookup" }
}

if ($Mode -eq "serve") {
  # 32 MiB requests: base64 images next to a long conversation.
  $arguments = @("serve", "--host", "127.0.0.1", "--port", "$Port", "--sessions", "1",
    "--max-request-bytes", "33554432",
    "llm", "--model", $model, "--served-model-name", "flash-next",
    "--context", "$Context", "--max-output-bytes", "8388608",
    "--think", $Think) + $sampling + $speculative
  if ($vision) { $arguments += @("--mmproj", $vision) }
} else {
  # Same shape as upstream's single-user table: pp2048/tg128 per depth.
  # (bench has no thinking switch; -Think only picks the sampler here.)
  $arguments = @("bench", "--model", $model, "--n-prompt", "2048", "--n-gen", "128",
    "--n-depth", "0,4096,16384,32768,65536,131072", "--repetitions", "2") +
    $sampling + $speculative
}

# GUFO_* variables (GUFO_PLATFORM_TUNING, diagnostics) reach the server; show
# them, since one left in a shell changes what runs.
Get-ChildItem Env: | Where-Object { $_.Name -like "GUFO_*" } | ForEach-Object {
  Write-Host "using $($_.Name)=$($_.Value) from this shell" -ForegroundColor Yellow
}
Write-Host "gufo $($arguments -join ' ')"
& "$bin\gufo.exe" @arguments
