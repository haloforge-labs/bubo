# Qwen3.8-Flash-Next on the Windows build of Gufo.
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next.ps1                # serve: UD-Q4_K_XL, thinking, adaptive MTP
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next.ps1 -Think off     # instruct mode (thinking off)
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next.ps1 -Draft mtp3    # MTP capped at 3 drafts
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next.ps1 -Draft off     # autoregressive baseline
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next.ps1 -Mode bench    # pp/tg at depths 0..128K
#
# Files come from the Hugging Face cache (see docs\WINDOWS.md for the
# download); -Snapshot points at another copy of unsloth/Qwen3.8-Flash-Next-GGUF.
# Ready when the log shows event=load_completed and GET /health returns 200.
param(
  [ValidateSet("serve", "bench")] [string]$Mode = "serve",
  [ValidateSet("on", "off")] [string]$Think = "on",
  [ValidateSet("mtp", "mtp3", "off")] [string]$Draft = "mtp",
  [switch]$Greedy,
  [int]$Context = 262144,
  [int]$Port = 8080,
  [string]$Snapshot = "",
  [string]$MtpModel = ""
)
$ErrorActionPreference = "Stop"
$bin = Resolve-Path "$PSScriptRoot\..\..\build\release"

if (-not $Snapshot) {
  $cache = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE }
           elseif ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" }
           else { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }
  $snapshots = Join-Path $cache "models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots"
  $Snapshot = Get-ChildItem $snapshots -Directory -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
  if (-not $Snapshot) { throw "unsloth/Qwen3.8-Flash-Next-GGUF not found under $cache; see docs\WINDOWS.md or pass -Snapshot" }
}
$model = "$Snapshot\UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
if (-not $MtpModel) { $MtpModel = "$Snapshot\MTP\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" }
if (-not (Test-Path $model)) { throw "model not found: $model" }
if ($Draft -ne "off" -and -not (Test-Path $MtpModel)) { throw "MTP model not found: $MtpModel (or use -Draft off)" }

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

if ($Mode -eq "serve") {
  $arguments = @("serve", "--host", "127.0.0.1", "--port", "$Port", "--sessions", "1",
    "llm", "--model", $model, "--served-model-name", "flash-next",
    "--context", "$Context", "--max-tokens", "32768",
    "--think", $Think) + $sampling + $speculative
} else {
  # Same shape as upstream's single-user table: pp2048/tg128 per depth.
  # (bench has no thinking switch; -Think only picks the sampler here.)
  $arguments = @("bench", "--model", $model, "--n-prompt", "2048", "--n-gen", "128",
    "--n-depth", "0,4096,16384,32768,65536,131072", "--repetitions", "2") +
    $sampling + $speculative
}

# GUFO_* variables left in a shell switch on diagnostics or change decoding;
# the server gets none of them.
Get-ChildItem Env: | Where-Object { $_.Name -like "GUFO_*" } | ForEach-Object {
  Write-Host "ignoring $($_.Name)=$($_.Value) from this shell" -ForegroundColor Yellow
  Remove-Item "Env:$($_.Name)"
}
Write-Host "gufo $($arguments -join ' ')"
& "$bin\gufo.exe" @arguments
