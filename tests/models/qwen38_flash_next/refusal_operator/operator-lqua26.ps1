# Adapted from decode-twin-v2-scan/operator-head.ps1. OPERATOR ONLY.
# Five loads, 50 prompts per arm, 160 output tokens: about 45-55 minutes.
param(
  [Parameter(Mandatory)][string]$Runtime,
  [Parameter(Mandatory)][string]$Direction,
  [Parameter(Mandatory)][string]$Ista,
  [Parameter(Mandatory)][string]$Out,
  # Production Huihui package directory: holds mtp\ and draft_head\.
  [Parameter(Mandatory)][string]$PackageDir,
  # Shard 1 of the production Huihui model (hard links of the production shards).
  [Parameter(Mandatory)][string]$HuihuiModel,
  # Exclusive GPU-harness lock file, created and removed by this script.
  [Parameter(Mandatory)][string]$GpuLock,
  [Parameter(Mandatory)][string]$TempDir,
  # AdvBench harmful_behaviors.csv (public: llm-attacks data/advbench).
  [Parameter(Mandatory)][string]$AdvBenchCsv,
  # Optional wrapper script that takes the shared build/speed lock and runs a command.
  [string]$BuildLockScript = '',
  [switch]$Reasoning,
  [switch]$Speed,
  [switch]$SkipStream,
  [switch]$InsideBuildLock
)
$ErrorActionPreference = 'Stop'
$port = 8098
$lock = $GpuLock
# Explicit paths avoid manifest size restrictions. Existing hard links contain
# the production shards; the production MTP and draft head are shared by all arms.
$modelPath = $HuihuiModel
$mtpPath = "$PackageDir\mtp\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
$draftHeadPath = "$PackageDir\draft_head\draft-out-q4k.gguf"

function Assert-NoOtherServer {
  $others = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(bubo|gufo)' -and $_.ProcessId -ne $PID })
  if ($others.Count) {
    $others | ForEach-Object { "running: PID $($_.ProcessId) $($_.CommandLine)" }
    throw 'Another bubo/gufo process is running. Not stopping it. Start this script when the GPU is free.'
  }
}
function Assert-ProjectionCounters([string[]]$Lines, [string]$Arm, [double]$FittedStrength = 1.955837064160251) {
  $counters = @($Lines | Select-String '\[refusal\] request enabled=(\d+) projections=(\d+) request_projections=(\d+) decode=(\d+) verify=(\d+) prefill=(\d+) batch=(\d+) graph_captures=(\d+) added_launches=(\d+) mode=(sites|stream) strength=([-+0-9.eE]+)')
  if ($counters.Count -lt 50) { throw "${Arm}: missing per-request projection counters; aborting" }
  $m = $counters[-1].Matches[0]
  $enabled = [long]$m.Groups[1].Value; $total = [long]$m.Groups[2].Value
  $decode = [long]$m.Groups[4].Value; $verify = [long]$m.Groups[5].Value
  $prefill = [long]$m.Groups[6].Value; $batch = [long]$m.Groups[7].Value
  $captures = [long]$m.Groups[8].Value
  $maxLaunches = ($counters | ForEach-Object { [long]$_.Matches[0].Groups[9].Value } | Measure-Object -Maximum).Maximum
  $mode = $m.Groups[10].Value
  $strength = [double]::Parse($m.Groups[11].Value,[Globalization.CultureInfo]::InvariantCulture)
  if ($Arm -in 'ista-sites-fit','ista-sites-1.0','ista-stream-1.0') {
    if ($enabled -ne 1 -or $total -le 0 -or $prefill -le 0 -or $verify -le 0 -or ($decode + $verify + $batch) -le 0 -or $captures -le 0) {
      throw "${Arm}: no-op or missing prompt/decode/capture projection evidence; aborting"
    }
    $expectedMode = if ($Arm -eq 'ista-stream-1.0') { 'stream' } else { 'sites' }
    $expectedStrength = if ($Arm -eq 'ista-sites-fit') { $FittedStrength } else { 1.0 }
    if ($mode -ne $expectedMode -or [Math]::Abs($strength - $expectedStrength) -gt 0.00001 -or
        ($mode -eq 'stream' -and $maxLaunches -le 0) -or ($mode -eq 'sites' -and $maxLaunches -ne 0)) {
      throw "${Arm}: wrong mode, strength or launch evidence; aborting"
    }
  } elseif ($enabled -ne 0 -or $total -ne 0 -or $decode -ne 0 -or $verify -ne 0 -or $prefill -ne 0 -or $batch -ne 0 -or $captures -ne 0) {
    throw "${Arm}: projections ran in an off arm; aborting"
  }
  "${Arm}: $($counters[-1].Line.Trim())"
}

Assert-NoOtherServer
if (!$InsideBuildLock -and $BuildLockScript) {
  $pwsh = Join-Path (Get-AppxPackage -Name Microsoft.PowerShell).InstallLocation 'pwsh.exe'
  $forward = @('-NoProfile','-File',$PSCommandPath,'-Runtime',$Runtime,'-Direction',$Direction,'-Ista',$Ista,'-Out',$Out,
    '-PackageDir',$PackageDir,'-HuihuiModel',$HuihuiModel,'-GpuLock',$GpuLock,'-TempDir',$TempDir,'-AdvBenchCsv',$AdvBenchCsv,'-InsideBuildLock')
  if ($Reasoning) { $forward += '-Reasoning' }; if ($Speed) { $forward += '-Speed' }; if ($SkipStream) { $forward += '-SkipStream' }
  $env:BUILD_LOCK_OWNER = 'operator-lqua26'
  & $BuildLockScript $pwsh @forward
  exit $LASTEXITCODE
}
foreach ($f in $modelPath,$mtpPath,$draftHeadPath,$Ista,$Direction,$AdvBenchCsv,"$Runtime\bubo.exe") {
  if (!(Test-Path -LiteralPath $f)) { throw "missing $f" }
}
$fittedText = & python -B -c 'from gguf import GGUFReader; import sys; f=GGUFReader(sys.argv[1]).get_field("refusal.fitted_strength"); print(f.contents() if f else "1.955837064160251")' $Direction
if ($LASTEXITCODE) { throw 'Cannot read fitted strength from the direction file' }
$fittedStrength = [double]::Parse(($fittedText | Select-Object -Last 1),[Globalization.CultureInfo]::InvariantCulture)
New-Item -ItemType Directory -Force -Path $Out,$TempDir | Out-Null
if (Get-ChildItem -LiteralPath $Out -Filter '*.jsonl') { throw 'Output directory already contains a run; use a new directory.' }
for ($w=0; $w -lt 540 -and (Test-Path $lock); $w++) { Start-Sleep 5 }
if (Test-Path $lock) { throw "gpu lock still held: $(Get-Content $lock -Raw)" }
# Exclusive creation closes the wait/create race. Never remove another owner's lock.
$owned = [IO.File]::Open($lock,'CreateNew','Write','Read')
$ownerLine = [Text.Encoding]::UTF8.GetBytes("operator-lqua26 $PID $((Get-Date).ToString('o'))")
$owned.Write($ownerLine,0,$ownerLine.Length); $owned.Flush()
try { Assert-NoOtherServer } catch { $owned.Dispose(); Remove-Item -LiteralPath $lock -ErrorAction SilentlyContinue; throw }

$knobNames = @('GUFO_REFUSAL_DIRECTION','GUFO_DEBUG_HOOKS','BUBO_RUNTIME_DIR','BUBO_RUNTIME_KIND','TEMP','TMP') +
  @(Get-ChildItem Env: | Where-Object { $_.Name -like 'GUFO_*' -or $_.Name -like 'BUBO_*' } | ForEach-Object { $_.Name })
$knobNames = @($knobNames | Select-Object -Unique)
$saved = @{}; foreach ($n in $knobNames) { $saved[$n] = [Environment]::GetEnvironmentVariable($n,'Process') }
try {
  $arms = @('huihui','ista-off','ista-sites-fit','ista-sites-1.0')
  if (!$SkipStream) { $arms += 'ista-stream-1.0' }
  foreach ($arm in $arms) {
    Start-Sleep 3
    Assert-NoOtherServer
    foreach ($n in $knobNames) { if ($n -notin 'TEMP','TMP') { Remove-Item "Env:$n" -ErrorAction SilentlyContinue } }
    $env:TEMP = $TempDir; $env:TMP = $env:TEMP
    $env:BUBO_RUNTIME_DIR = $Runtime; $env:BUBO_RUNTIME_KIND = 'explicit'
    $env:GUFO_DEBUG_HOOKS = '1'
    $model = if ($arm -eq 'huihui') { $modelPath } else { $Ista }
    $log = "$Out\serve-$arm.err"
    $serverArgs = @('serve','llm','--model',('"'+$model+'"'),'--mtp-model',('"'+$mtpPath+'"'),
      '--draft-head-model',('"'+$draftHeadPath+'"'),'--speculative','mtp','--host','127.0.0.1','--port',"$port",
      '--served-model-name','flash-next','--context','8192','--npu-prefill','off','--cache-ram-bytes','8589934592',
      '--no-cache-disk','--sessions','1','--no-vision')
    if ($arm -in 'ista-sites-fit','ista-sites-1.0','ista-stream-1.0') {
      $mode = if ($arm -eq 'ista-stream-1.0') { 'stream' } else { 'sites' }
      $serverArgs += @('--refusal-direction','on','--refusal-direction-file',('"'+$Direction+'"'),
        '--refusal-direction-mode',$mode)
      if ($arm -ne 'ista-sites-fit') { $serverArgs += @('--refusal-direction-strength','1.0') }
      if ($mode -eq 'stream') { $serverArgs += @('--refusal-direction-layer-start','4','--refusal-direction-layer-end','44') }
    }
    $p = Start-Process "$Runtime\bubo.exe" -ArgumentList $serverArgs -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput "$Out\serve-$arm.out" -RedirectStandardError $log
    try {
      $deadline = (Get-Date).AddSeconds(900); $ready = $false
      while (!$ready) {
        $p.Refresh(); if ($p.HasExited) { throw "Server exited: $log" }
        if ((Get-Date) -gt $deadline) { throw "Server not ready after 900 s: $log" }
        try { $ready = (Invoke-WebRequest "http://127.0.0.1:$port/ready" -TimeoutSec 3).StatusCode -eq 200 } catch { Start-Sleep 1 }
      }
      $arguments = @('-B',"$PSScriptRoot\refusal-check-lqua26.py",'--port',"$port",'--arm',$arm,'--out',$Out,'--data',$AdvBenchCsv)
      if ($Reasoning) { $arguments += '--reasoning' }; if ($Speed) { $arguments += '--speed' }
      & python @arguments
      if ($LASTEXITCODE) { throw "${arm}: collection failed; aborting" }
    } finally {
      $p.Refresh()
      if (!$p.HasExited) {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)").CommandLine
        if ($cmd -and $cmd -match "--port $port" -and $cmd -match [regex]::Escape("$Runtime")) { Stop-Process -Id $p.Id; $p.WaitForExit() }
        else { Write-Warning "Did not stop PID $($p.Id): command line does not match this script's server." }
      }
    }
    if (!$p.HasExited) { throw "${arm}: child still running; cannot read flushed counters" }
    Assert-ProjectionCounters (Get-Content -LiteralPath $log) $arm $fittedStrength
  }
  $compareArgs = @('-B',"$PSScriptRoot\refusal-check-lqua26.py",'--compare','--out',$Out)
  if ($SkipStream) { $compareArgs += '--skip-stream' }
  & python @compareArgs
  if ($LASTEXITCODE) { throw 'Comparison failed' }
} finally {
  foreach ($n in $knobNames) { [Environment]::SetEnvironmentVariable($n,$saved[$n],'Process') }
  $owned.Dispose()
  Remove-Item -LiteralPath $lock -ErrorAction SilentlyContinue
}
