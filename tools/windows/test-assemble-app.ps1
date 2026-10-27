# Package-independent contract proof: inspect parameters, execute the actual
# builder invocation with an argument recorder. No packaged executable runs.
$ErrorActionPreference = 'Stop'
$path = Join-Path $PSScriptRoot 'assemble-app.ps1'
$command = Get-Command $path
$ast = $command.ScriptBlock.Ast
$checks = 0
foreach ($name in @('RuntimeChanged','RuntimeDownloadBytes','ModelsChanged','ModelsNeedsPrepare','Notes')) {
  $parameter = $command.Parameters[$name]
  if (-not $parameter -or -not @($parameter.Attributes | Where-Object { $_ -is [Management.Automation.ParameterAttribute] -and $_.Mandatory }).Count) { throw "$name must be mandatory" }
  $node = $ast.ParamBlock.Parameters | Where-Object { $_.Name.VariablePath.UserPath -eq $name }
  if ($null -ne $node.DefaultValue) { throw "$name must have no default" }
  $checks += 2
}
$call = @($ast.FindAll({ param($node) $node -is [Management.Automation.Language.CommandAst] -and $node.Extent.Text -match 'run ./cmd/bubo-release -Zip' }, $true))
if ($call.Count -ne 1) { throw 'Expected one actual release-builder invocation' }
function Capture-Go { $script:forwarded = @($args); $global:LASTEXITCODE = 0 }
$Go = 'Capture-Go'
$zip = 'C:\package with spaces\bubo.zip'
$outPath = 'C:\output with spaces'
$Version = '0.1.1'; $Tag = 'v0.1.1'; $Commit = 'source'; $MinUpdatableFrom = '0.1.0'
$Notes = 'C:\notes with spaces\summary.txt'; $released = '2026-10-26T20:00:00Z'
foreach ($changed in @($false,$true)) {
  $RuntimeChanged = $changed; $ModelsChanged = $changed
  $RuntimeDownloadBytes = if ($changed) { [long]12345678901 } else { [long]0 }
  $ModelsNeedsPrepare = if ($changed) { 'model-one,model-two' } else { '' }
  & ([scriptblock]::Create($call[0].Extent.Text))
  foreach ($value in @("-RuntimeChanged=$($changed.ToString().ToLowerInvariant())", "-RuntimeDownloadBytes=$RuntimeDownloadBytes", "-ModelsChanged=$($changed.ToString().ToLowerInvariant())", "-ModelsNeedsPrepare=$ModelsNeedsPrepare")) {
    if ($value -cnotin $script:forwarded) { throw "Builder lost explicit decision: $value; args=$script:forwarded" }
    $checks++
  }
  $index = [array]::IndexOf($script:forwarded, '-Notes')
  if ($index -lt 0 -or $script:forwarded[$index+1] -cne $Notes) { throw 'Builder lost summary file path' }
  $checks++
}
Write-Output "PASS: assembly interface $checks checks; false/zero/empty and changed/list/64-bit bytes preserved; no package required"
