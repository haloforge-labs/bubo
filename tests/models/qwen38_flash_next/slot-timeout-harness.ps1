# CPU-only harness: import actual operator functions, never its main/GPU path.
param([ValidateSet('stall','prelaunch','cumulative')][string]$Mode='stall',
      [string]$OutputRoot=$PSScriptRoot)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$source=Join-Path $PSScriptRoot 'gpu-checks-stageb.ps1'
$tokens=$null; $errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($source,[ref]$tokens,[ref]$errors)
if($errors.Count){throw $errors[0]}
foreach($node in $ast.FindAll({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst]},$false)){
 if($node.Name -in @('Quote-Display','Quote-Native','Check-Hold','Invoke-Captured')){Invoke-Expression $node.Extent.Text}
}
$out=Join-Path $OutputRoot "slot-test-$Mode-$((date).ToString('yyyyMMdd-HHmmss-fff'))"
New-Item -ItemType Directory -Path $out|Out-Null
$DryRun=$false; $stopFile=Join-Path $out 'STOP'; $resetNames=@()
$groupAllowanceSeconds=0.6
$groupWatch=[Diagnostics.Stopwatch]::StartNew()
$fake=(Get-Command powershell -CommandType Application|Select-Object -First 1).Source
$status='FAIL'; $failure=$null
try {
 if($Mode -eq 'prelaunch'){Start-Sleep -Milliseconds 700}
 if($Mode -eq 'cumulative'){
  $groupAllowanceSeconds=1.5
  Invoke-Captured $fake @('-NoProfile','-Command','Write-Output $PID; Start-Sleep -Milliseconds 300') 'first' 10
 }
 Invoke-Captured $fake @('-NoProfile','-Command','Write-Output $PID; Start-Sleep -Seconds 3') 'fake-child' 10
 $status='PASS'
} catch {$failure=$_.Exception.Message}
finally {
 [ordered]@{status=$status;error=$failure;seconds=$groupWatch.Elapsed.TotalSeconds;mode=$Mode;output=$out}|
  ConvertTo-Json|Set-Content -LiteralPath (Join-Path $out 'step-test.json') -Encoding UTF8
}
Get-Content -LiteralPath (Join-Path $out 'step-test.json') -Raw
