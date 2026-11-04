# Operator-only Stage B checks. CPU jobs may invoke only -DryRun.
[CmdletBinding()]
param(
 [Parameter(Mandatory)][string]$Runtime,
 [Parameter(Mandatory)][string]$PreviousRuntime,
 [switch]$OperatorSlot, [switch]$DryRun,
 [Parameter(Mandatory)][string]$Model,
 [Parameter(Mandatory)][string]$Mtp,
 [Parameter(Mandatory)][string]$Twin,
 [Parameter(Mandatory)][string]$Manifest,
 [Parameter(Mandatory)][string]$Prompt32K,
 [Parameter(Mandatory)][string]$OutRoot,
 [Parameter(Mandatory)][string]$TempDir,
 [string[]]$HoldFile=@(),
 [ValidateRange(1,5)][int]$StopAfterStep=5,
 [ValidateRange(1,120)][int]$ProbeTimeoutMinutes=30
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not $DryRun -and -not $OperatorSlot){throw 'Live execution requires -OperatorSlot'}
if(-not $Runtime){throw 'Pass -Runtime with the newly staged kcut candidate'}
# The exclusive-slot wrapper admits this script before it starts. One
# monotonic allowance covers preflight, every child, and all five steps.
$groupAllowanceSeconds=28*60
$groupWatch=[Diagnostics.Stopwatch]::StartNew()
$expectedModelName='Qwen3.8-Flash-Next-Hybrid-GSQ-RCO-Q4KXL-00001-of-00004.gguf'
$modelLeaf=[IO.Path]::GetFileName([IO.Path]::GetFullPath($Model))
$runtimeFull=[IO.Path]::GetFullPath($Runtime).TrimEnd('\','/')
$previousFull=[IO.Path]::GetFullPath($PreviousRuntime).TrimEnd('\','/')
if($modelLeaf -ine $expectedModelName){throw 'Only the pinned ISTA hybrid is allowed'}
if([IO.Path]::GetFileName($previousFull) -ine 'kcut-99a643ba'){throw 'Previous runtime must be kcut-99a643ba'}
if([IO.Path]::GetFileName($runtimeFull) -notlike 'kcut-*' -or $runtimeFull -ieq $previousFull){throw 'A separate staged kcut candidate is required'}
$manifestSha='cbb55a66c0a43866a6b203252af389b39b5b91b2eaf90f889c6c9b4cc72daab9'
$out=Join-Path $OutRoot "stageb-$((date).ToString('yyyyMMdd-HHmmss-fff'))"
$tempRoot=$TempDir
$stopFile=Join-Path $out 'STOP'
$python=if($DryRun){'python'}else{(Get-Command python -CommandType Application|Select-Object -First 1).Source}
$probe=Join-Path $Runtime 'qwen38_flash_next_gpu_probe.exe'
$previousProbe=Join-Path $PreviousRuntime 'qwen38_flash_next_gpu_probe.exe'
$quality=Join-Path $Runtime 'qwen38_flash_next_batch_quality_probe.exe'
$oracle=Join-Path $Runtime 'qwen38_flash_next_stageb_ops_test.exe'
$resetNames=@('GUFO_PREFILL_EXPERTS','GUFO_PREFILL_EXPERTS_MODE','GUFO_PREFILL_EXPERTS_AUDIT','GUFO_PREFILL_EXPERTS_MASS','GUFO_PREFILL_EXPERTS_FOLD',
 'GUFO_PREFILL_TELEMETRY','GUFO_PREFILL_TELEMETRY_LAYERS','GUFO_PREFILL_STAGEB','GUFO_PREFILL_STAGEB_LAYERS','GUFO_PREFILL_STAGEB_STRIDE','GUFO_PREFILL_STAGEB_ROWS',
 'GUFO_PREFILL_DIAG','GUFO_MOE_BENCH','GUFO_PREFILL_PHASE_PROBE','GUFO_PREFILL_PHASE_PROBE_DEBUG','GUFO_PREFILL_FORWARD_TOKENS',
 'GUFO_DENSE_TWIN_MODEL','GUFO_DENSE_TWIN_BATCH','GUFO_DENSE_TWIN_MAX_N','GUFO_DENSE_TWIN_FORCE',
 'GUFO_IQ4_WIDE_VARIANT','GUFO_IQ4_WMMA_WAVES','GUFO_QUALITY_V0_CONTROL','GUFO_ROUTED_LANEROW','GUFO_ROUTED_WMMA_IQ','GUFO_IQ_ROUTE_MIN_ROWS','GUFO_EXPERT_V2K',
 'BUBO_RUNTIME_DIR','PATH','TEMP','TMP')
$inherited=@(Get-ChildItem Env:|Where-Object{$_.Name -like 'GUFO_PREFILL_EXPERTS*' -or $_.Name -like 'GUFO_PREFILL_TELEMETRY*' -or $_.Name -like 'GUFO_PREFILL_STAGEB*'}|Select-Object -ExpandProperty Name)
$resetNames=@(@($resetNames)+$inherited|Select-Object -Unique)
$saved=@{}; foreach($key in $resetNames){$saved[$key]=[Environment]::GetEnvironmentVariable($key,'Process')}
$baseProbe=(17.461+32768.0/1247.9)/60
$estimates=@(
 [pscustomobject]@{Step=1;Name='previous unset vs candidate unset, two cold 32K prefills';Base=2*$baseProbe;Budget=4.0},
 [pscustomobject]@{Step=2;Name='k10 Stage B on vs saved candidate unset';Base=$baseProbe;Budget=4.0},
 [pscustomobject]@{Step=3;Name='CPU identity/sample gate and tiny GPU synthetic oracle';Base=0.05;Budget=0.5},
 [pscustomobject]@{Step=4;Name='5504-row NLL manifest, 337834 prompt tokens, then completeness gate';Base=(17.461+337834.0/1247.9)/60;Budget=19.5},
 [pscustomobject]@{Step=5;Name='derive 48-layer fold files on CPU';Base=0;Budget=0}
)
Write-Output 'INFERRED minutes: 17.461 s load; 32768/1247.9 = 26.259 s cold prefill; manifest 337834/1247.9 = 4.512 min prefill. Sources: timing-sources.txt.'
Write-Output 'Base excludes diagnostic FP64/atomic cost, continuation NLL scoring, and file writes. Step margins total 28 minutes; one hard group allowance of 28 minutes caps every child. Instrumented captures do not qualify throughput.'
Write-Output 'k=10 is an output-consistency gate, not an independent validation of which tensor or token was read.'
foreach($e in $estimates){Write-Output ('Step {0}: base {1:N2} min; budget {2:N2} min: {3}' -f $e.Step,$e.Base,$e.Budget,$e.Name)}
Write-Output "Artifacts: $out; TEMP=TMP=$tempRoot"
function Check-Hold {
 if($groupWatch.Elapsed.TotalSeconds -ge $groupAllowanceSeconds){throw "Stage B group timeout after $groupAllowanceSeconds seconds"}
 foreach($p in (@($HoldFile)+@($stopFile))){if(Test-Path -LiteralPath $p){throw "Stop requested: $p"}}
}
function Reset-ForRun([string]$RunRuntime,[string]$StageB='') {
 if($DryRun){
  Write-Output "RESET: unset GUFO_PREFILL_EXPERTS*, GUFO_PREFILL_TELEMETRY*, GUFO_PREFILL_STAGEB*, diagnostic/route/twin overrides; runtime=$RunRuntime"
  if($StageB){Write-Output "SET: GUFO_PREFILL_STAGEB=$StageB; default all 48 layers, STRIDE=1, ROWS=64; expert knob unset (k10)"}
  return
 }
 foreach($key in $resetNames){if($key -notin @('PATH','BUBO_RUNTIME_DIR','TEMP','TMP')){[Environment]::SetEnvironmentVariable($key,$null,'Process')}}
 $env:TEMP=$tempRoot; $env:TMP=$tempRoot; $env:BUBO_RUNTIME_DIR=$RunRuntime; $env:PATH="$RunRuntime;$($saved['PATH'])"
 if($StageB){$env:GUFO_PREFILL_STAGEB=$StageB}
}
function Quote-Display([string]$Value) { "'" + $Value.Replace("'", "''") + "'" }
function Quote-Native([string]$Value) {
    if ($Value.Contains('"') -or $Value.Contains("`n") -or $Value.Contains("`r")) { throw 'Unsupported quote/newline in argument' }
    '"' + [regex]::Replace($Value, '(\\+)$', '$1$1') + '"'
}
function Invoke-Captured([string]$Exe, [string[]]$Arguments, [string]$Name, [double]$TimeoutSeconds) {
    $stdout = Join-Path $out "$Name.out"
    $stderr = Join-Path $out "$Name.err"
    $display = '& ' + (Quote-Display $Exe) + ' ' + (($Arguments | ForEach-Object { Quote-Display $_ }) -join ' ') +
        ' 1> ' + (Quote-Display $stdout) + ' 2> ' + (Quote-Display $stderr)
    Write-Output $display
    if ($DryRun) { return }
    $started = date
    $environment = [ordered]@{}
    foreach ($key in (@($resetNames) + @('TEMP','TMP'))) { $environment[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $process = $null
    $exitCode = $null
    $allowedSeconds = $null
    try {
        Check-Hold
        $allowedSeconds = [Math]::Min($TimeoutSeconds, $groupAllowanceSeconds - $groupWatch.Elapsed.TotalSeconds)
        $launch = @{FilePath=$Exe; WorkingDirectory=(Split-Path $Exe -Parent); WindowStyle='Hidden'; PassThru=$true; RedirectStandardOutput=$stdout; RedirectStandardError=$stderr}
        # Start-Process rejects an empty -ArgumentList; the operator exes have no args.
        if ($Arguments.Count -gt 0) { $launch.ArgumentList = ($Arguments | ForEach-Object { Quote-Native $_ }) -join ' ' }
        $process = Start-Process @launch
        # Retain the native handle before waiting. Windows PowerShell otherwise
        # can return a null ExitCode for a short Start-Process child.
        $null = $process.Handle
        while (-not $process.WaitForExit(250)) {
            Check-Hold
            if ($watch.Elapsed.TotalSeconds -ge $allowedSeconds) { throw "Timeout after $allowedSeconds seconds: $Name" }
        }
        $process.WaitForExit()
        Check-Hold
        if ($watch.Elapsed.TotalSeconds -ge $allowedSeconds) { throw "Timeout after $allowedSeconds seconds: $Name" }
        $exitCode = $process.ExitCode
        if ($exitCode -ne 0) { throw "Command failed: $Name rc=$exitCode (see $stderr)" }
        foreach ($path in @($stdout,$stderr)) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing log: $path" }
        }
    } finally {
        if ($null -ne $process -and -not $process.HasExited) {
            $process.Kill()
            $process.WaitForExit()
        }
        $watch.Stop()
        $record = [ordered]@{name=$Name; started=$started.ToString('o'); ended=(date).ToString('o'); seconds=$watch.Elapsed.TotalSeconds; timeout_seconds=$allowedSeconds; group_seconds=$groupWatch.Elapsed.TotalSeconds; exit_code=$exitCode; command=$display; environment=$environment}
        $record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out "$Name.process.json") -Encoding UTF8
        if ($null -ne $process) { $process.Dispose() }
    }
}
function Check-Python([string]$Name, [string[]]$Arguments) {
    Invoke-Captured $python $Arguments $Name 180
    if (-not $DryRun) {
        $text = Get-Content -LiteralPath (Join-Path $out "$Name.out") -Raw
        if ([string]::IsNullOrWhiteSpace($text)) { throw "Missing/short checker result: $Name" }
        Write-Output $text.TrimEnd()
    }
}



function Run-Probe([string]$Exe,[string]$Name,[bool]$StageB=$false) {
 $dir=if($StageB){Join-Path $out "$Name-stageb"}else{''}
 Reset-ForRun (Split-Path $Exe -Parent) $dir
 $dump=Join-Path $out "$Name.f32"
 Invoke-Captured $Exe @('--model',$Model,'--prompt-file',$Prompt32K,'--batch','4096','--context','65536','--dump',$dump) $Name ($ProbeTimeoutMinutes*60)
 if($DryRun){Write-Output 'REQUIRE prompt tokens=32768; dump bytes=63569920; finite full-vocabulary logits; byte equality (greedy agreement is insufficient)'}
 else{
  $text=Get-Content -LiteralPath (Join-Path $out "$Name.out") -Raw
  if([regex]::Matches($text,'(?m)^prompt tokens \(32768\)\r?$').Count -ne 1){throw "Wrong/short prompt: $Name"}
  if(-not(Test-Path -LiteralPath $dump -PathType Leaf) -or (Get-Item -LiteralPath $dump).Length -ne 63569920){throw "Missing/short logit dump: $Name"}
  if(Select-String -LiteralPath (Join-Path $out "$Name.err") -SimpleMatch 'prefill_experts=' -Quiet){throw "Unexpected active expert policy: $Name"}
 }
 Check-Python "$Name-finite" @((Join-Path $PSScriptRoot 'check-logits.py'),$dump)
}
function Compare-Logits([string]$Left,[string]$Right) {
 Check-Python "$Left-vs-$Right" @((Join-Path $PSScriptRoot 'check-logits.py'),(Join-Path $out "$Left.f32"),'--compare',(Join-Path $out "$Right.f32"))
}
function Run-Nll {
 $dir=Join-Path $out 'nll-k10-stageb'
 Reset-ForRun $Runtime $dir
 if($DryRun){Write-Output "SET: twin=$Twin batch=1 variant=4 waves=4 forward=4096; expert knob unset; telemetry unset"}
 else{
  $env:GUFO_DENSE_TWIN_MODEL=$Twin; $env:GUFO_DENSE_TWIN_BATCH='1'
  $env:GUFO_IQ4_WIDE_VARIANT='4'; $env:GUFO_IQ4_WMMA_WAVES='4'; $env:GUFO_PREFILL_FORWARD_TOKENS='4096'
 }
 Invoke-Captured $quality @('--model',$Model,'--mtp-model',$Mtp,'--manifest',$Manifest,'--manifest-sha256',$manifestSha,'--c','4','--arm','wmma','--score-prefill','0','--out',(Join-Path $out 'nll-k10'),'--dump-logits','0','--per-token-nll',(Join-Path $out 'nll-k10.jsonl')) 'nll-k10' ($ProbeTimeoutMinutes*60)
 Check-Python 'nll-manifest-completion' @((Join-Path $PSScriptRoot 'manifest_gate.py'),'--manifest',$Manifest,'--manifest-sha256',$manifestSha,'--nll',(Join-Path $out 'nll-k10.jsonl'))
 Check-Python 'calibration-completion' @((Join-Path $PSScriptRoot 'check_stageb.py'),'--directory',$dir,'--manifest',$Manifest,'--manifest-sha256',$manifestSha,'--map-out',(Join-Path $out 'stageb-nll-map.jsonl'))
}
$transcriptStarted=$false
try {
 if(-not $DryRun){
  Check-Hold
  foreach($p in @($Model,$Mtp,$Twin,$Manifest,$Prompt32K,$probe,$previousProbe,$quality,$oracle)){
   if(-not(Test-Path -LiteralPath $p -PathType Leaf) -or (Get-Item -LiteralPath $p).Length -eq 0){throw "Missing/empty input: $p"}
  }
  if(Test-Path -LiteralPath $out){throw 'Output directory must be fresh'}
  New-Item -ItemType Directory -Path $out|Out-Null
  $env:TEMP=$tempRoot; $env:TMP=$tempRoot
  Start-Transcript -LiteralPath (Join-Path $out 'qualification.log')|Out-Null; $transcriptStarted=$true
  $estimates|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $out 'estimates.json') -Encoding UTF8
 }
 Check-Python 'manifest-preflight' @((Join-Path $PSScriptRoot 'manifest_gate.py'),'--manifest',$Manifest,'--manifest-sha256',$manifestSha,'--manifest-only')
 for($step=1;$step -le $StopAfterStep;$step++){
  $started=date; $status='FAIL'; $failure=$null
  Write-Output "STEP $step START $($started.ToString('o'))"
  try {
   if(-not $DryRun){Check-Hold}
   switch($step){
    1 { Run-Probe $previousProbe 'previous-unset'; Run-Probe $probe 'candidate-unset'; Compare-Logits 'previous-unset' 'candidate-unset' }
    2 { Run-Probe $probe 'k10-on' $true; Compare-Logits 'candidate-unset' 'k10-on' }
    3 {
     Check-Python 'cold32k-stageb-selfcheck' @((Join-Path $PSScriptRoot 'check_stageb.py'),'--directory',(Join-Path $out 'k10-on-stageb'),'--tokens','32768','--allow-final-trim')
     Check-Python 'k10-identity' @((Join-Path $PSScriptRoot 'derive_fold.py'),(Join-Path $out 'k10-on-stageb\stageb-totals.jsonl'),'--check-only')
     Reset-ForRun $Runtime
     Invoke-Captured $oracle @((Join-Path $PSScriptRoot 'stageb-synthetic.json')) 'synthetic-oracle' 120
    }
    4 { Run-Nll }
    5 { Check-Python 'derive-fold' @((Join-Path $PSScriptRoot 'derive_fold.py'),(Join-Path $out 'nll-k10-stageb\stageb-totals.jsonl'),'--out',(Join-Path $out 'fold')) }
   }
   if(-not $DryRun){Check-Hold}
   $status=if($DryRun){'DRY-RUN'}else{'PASS'}
  } catch { $failure=$_.Exception.Message; throw }
  finally {
   $ended=date
   if(-not $DryRun){[ordered]@{step=$step;status=$status;started=$started.ToString('o');ended=$ended.ToString('o');seconds=($ended-$started).TotalSeconds;error=$failure}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $out "step-$step.json") -Encoding UTF8}
   Write-Output "STEP $step $status $($ended.ToString('o'))"
  }
 }
 $result=if($DryRun){'DRY-RUN ONLY: no executable launched and no output directory created.'}elseif($StopAfterStep -lt 5){'PARTIAL: requested step boundary reached; full Stage B group incomplete.'}else{"PASS: exact-logit controls, synthetic oracle, k10 output-consistency gate, complete 5504 NLL rows, calibration and folds passed. Artifacts: $out"}
 Write-Output $result
 if(-not $DryRun){$result|Set-Content -LiteralPath (Join-Path $out 'RESULT.txt') -Encoding UTF8}
} finally {
 if(-not $DryRun){
  foreach($key in $saved.Keys){[Environment]::SetEnvironmentVariable($key,$saved[$key],'Process')}
  if($transcriptStarted){Stop-Transcript|Out-Null}
 }
}
