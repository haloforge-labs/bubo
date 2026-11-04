# Pure-function tests. Does not evaluate operator-lqua26.ps1 at top level.
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile("$PSScriptRoot\operator-lqua26.ps1",[ref]$tokens,[ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
$function = $ast.Find({ param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Assert-ProjectionCounters' },$true)
Invoke-Expression $function.Extent.Text
function Line([string]$Values, [int]$Launches = 0, [string]$Mode = 'sites', [string]$Strength = '1') {
  return "[refusal] request $Values added_launches=$Launches mode=$Mode strength=$Strength"
}
$off = Line 'enabled=0 projections=0 request_projections=0 decode=0 verify=0 prefill=0 batch=0 graph_captures=0'
$unit = Line 'enabled=1 projections=960 request_projections=96 decode=96 verify=96 prefill=768 batch=0 graph_captures=2'
$fit = Line 'enabled=1 projections=960 request_projections=96 decode=96 verify=96 prefill=768 batch=0 graph_captures=2' 0 'sites' '1.95583706'
$stream = Line 'enabled=1 projections=410 request_projections=41 decode=41 verify=41 prefill=328 batch=0 graph_captures=2' 41 'stream' '1'
Assert-ProjectionCounters (@($off)*50) 'huihui'
Assert-ProjectionCounters (@($off)*50) 'ista-off'
Assert-ProjectionCounters (@($unit)*50) 'ista-sites-1.0'
Assert-ProjectionCounters (@($fit)*50) 'ista-sites-fit'
Assert-ProjectionCounters (@($stream)*50) 'ista-stream-1.0'
$bad = @(
  @{Lines=@($off)*49; Arm='ista-off'},
  @{Lines=@($off)*50; Arm='ista-sites-1.0'},
  @{Lines=@($unit)*50; Arm='ista-off'},
  @{Lines=@(($unit -replace 'prefill=768','prefill=0'))*50; Arm='ista-sites-1.0'},
  @{Lines=@(($unit -replace 'graph_captures=2','graph_captures=0'))*50; Arm='ista-sites-1.0'},
  @{Lines=@(($unit -replace 'verify=96','verify=0'))*50; Arm='ista-sites-1.0'},
  @{Lines=@($unit)*50; Arm='ista-sites-fit'},
  @{Lines=@($unit)*50; Arm='ista-stream-1.0'},
  @{Lines=@(($stream -replace 'added_launches=41','added_launches=0'))*50; Arm='ista-stream-1.0'}
)
foreach ($case in $bad) {
  $failed=$false
  try { Assert-ProjectionCounters $case.Lines $case.Arm } catch { $failed=$true }
  if (!$failed) { throw 'Wrong/missing/no-op counter accepted' }
}
& python -B "$PSScriptRoot\operator_policy_test.py"
if ($LASTEXITCODE) { throw 'Python policy tests failed' }
'PASS operator counter and collection policy; no process, model, HTTP or GPU use'
