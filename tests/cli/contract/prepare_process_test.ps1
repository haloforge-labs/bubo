param([Parameter(Mandatory)][string]$Test, [Parameter(Mandatory)][string]$Build)
$ErrorActionPreference = 'Stop'
$root = Join-Path $Build ('machine-prepare-' + [Guid]::NewGuid().ToString('N'))
& $Test $root
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$passed = 0
$failed = 0
$bytes = [IO.File]::ReadAllBytes((Join-Path $root 'events.jsonl'))
if ($bytes.Length -lt 1 -or $bytes[-1] -ne 10 -or
    ($bytes.Length -ge 3 -and $bytes[0] -eq 239 -and $bytes[1] -eq 187 -and $bytes[2] -eq 191) -or
    $bytes.Contains([byte]13)) { throw 'Wire transcript must use LF, without CR or BOM' }
# Throw on invalid UTF-8, including a code point split across JSON records.
$text = [Text.UTF8Encoding]::new($false, $true).GetString($bytes)
foreach ($line in $text.Split("`n", [StringSplitOptions]::RemoveEmptyEntries)) {
    if (Test-Json -Json $line -SchemaFile (Join-Path $PSScriptRoot 'event.schema.json') -ErrorAction SilentlyContinue) { $passed++ }
    else { $failed++; Write-Output "FAIL: $line" }
    $event = $line | ConvertFrom-Json
    if ($event.type -eq 'plan') {
        foreach ($volume in $event.payload.volumes) {
            if ($volume.bytes_to_write -ne ($volume.required_bytes - $volume.safety_margin_bytes) -or
                $volume.free_after_download_bytes -ne [Math]::Max([long]0, [long]($volume.free_bytes - $volume.bytes_to_write))) {
                throw 'Prepare remaining space includes the safety margin or underflows'
            }
            $passed++
        }
    }
}
Write-Output "Prepare emitted schema: $passed passed, $failed failed"
if ($failed) { exit 1 }
