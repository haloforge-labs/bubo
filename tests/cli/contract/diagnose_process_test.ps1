param([Parameter(Mandatory)][string]$Test, [Parameter(Mandatory)][string]$Build)
$ErrorActionPreference = 'Stop'
$path = Join-Path $Build ('diagnose-' + [Guid]::NewGuid().ToString('N') + '.jsonl')
& $Test $path
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$passed = 0
$failed = 0
foreach ($line in [IO.File]::ReadAllLines($path)) {
    if (Test-Json -Json $line -SchemaFile (Join-Path $PSScriptRoot 'event.schema.json')) { $passed++ } else { $failed++ }
}
foreach ($arguments in @(@('--bad','--json'), @('--json','--warmup','x'), @('--repetitions=-1','--json'), @('--json','--section'), @('--json','--jsonl'))) {
    $lines = @(& $Test @arguments 2>$null)
    $code = $LASTEXITCODE
    if ($code -eq 2 -and $lines.Count -eq 1 -and (Test-Json -Json $lines[0] -SchemaFile (Join-Path $PSScriptRoot 'event.schema.json'))) { $passed++ }
    else { $failed++; Write-Output "FAIL: diagnose args $arguments exit $code lines $($lines.Count)" }
}
$null = & $Test --bad --section all 2>$null
if ($LASTEXITCODE -eq 2) { $passed++ } else { $failed++ }
Write-Output "Diagnose emitted schema and malformed args: $passed passed, $failed failed"
if ($failed) { exit 1 }
