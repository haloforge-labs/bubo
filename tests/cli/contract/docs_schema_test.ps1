param([string]$Root = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$repo = Split-Path (Split-Path (Split-Path $Root))
$document = [IO.File]::ReadAllText((Join-Path $repo 'docs/CLI-CONTRACT.md'))
$passed = 0
$failed = 0
foreach ($block in [regex]::Matches($document, '(?s)```jsonl\r?\n(.*?)\r?\n```')) {
    foreach ($line in $block.Groups[1].Value.Split("`n")) {
        if (Test-Json -Json $line -SchemaFile (Join-Path $Root 'event.schema.json') -ErrorAction SilentlyContinue) { $passed++ }
        else { $failed++; Write-Output 'FAIL: documentation event example' }
    }
}
foreach ($block in [regex]::Matches($document, '(?s)```json\r?\n(.*?)\r?\n```')) {
    foreach ($line in $block.Groups[1].Value.Split("`n")) {
        if (Test-Json -Json $line -SchemaFile (Join-Path $Root 'input.schema.json') -ErrorAction SilentlyContinue) { $passed++ }
        else { $failed++; Write-Output 'FAIL: documentation input example' }
    }
}
Write-Output "Documentation emitted schema examples: $passed passed, $failed failed"
if ($failed -or $passed -eq 0) { exit 1 }
