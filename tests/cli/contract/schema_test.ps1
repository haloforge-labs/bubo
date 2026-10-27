param([string]$Root = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$passed = 0
$failed = 0
foreach ($pair in @(@('valid','invalid','event.schema.json'), @('input-valid','input-invalid','input.schema.json'))) {
    foreach ($kind in $pair[0..1]) {
        foreach ($file in Get-ChildItem -LiteralPath (Join-Path $Root $kind) -Filter '*.json') {
            $valid = Test-Json -Json ([IO.File]::ReadAllText($file.FullName)) -SchemaFile (Join-Path $Root $pair[2]) -ErrorAction SilentlyContinue
            if ($valid -eq ($kind -eq $pair[0])) { $passed++ }
            else { $failed++; Write-Output "FAIL: $($file.Name)" }
        }
    }
}
# The additive recommendation detail is typed at both producer locations.
# Omission remains valid for older version-1 records.
$record = [IO.File]::ReadAllText((Join-Path $Root 'valid/model-estimate-flash-next.json'))
foreach ($location in @('estimate', 'row')) {
    foreach ($case in @(
        @{json='[]'; valid=$true},
        @{json='["gpu_headroom"]'; valid=$true},
        @{json='["host_floor"]'; valid=$true},
        @{json='["gpu_headroom","host_floor","unconfirmed"]'; valid=$true},
        @{json='["unconfirmed"]'; valid=$true},
        @{json='["unknown"]'; valid=$false},
        @{json='["host_floor","host_floor"]'; valid=$false},
        @{json='[1]'; valid=$false},
        @{json='"host_floor"'; valid=$false},
        @{json='null'; valid=$false},
        @{json=$null; valid=$true}
    )) {
        $event = $record | ConvertFrom-Json -AsHashtable
        $target = $event.payload.memory_estimate
        if ($location -eq 'row') { $target = $target.recommended_context.by_sessions[0] }
        if ($null -eq $case.json) { $target.Remove('recommendation_failures') | Out-Null }
        else { $target.recommendation_failures = ConvertFrom-Json -InputObject $case.json -NoEnumerate }
        $valid = Test-Json -Json ($event | ConvertTo-Json -Depth 100 -Compress) -SchemaFile (Join-Path $Root 'event.schema.json') -ErrorAction SilentlyContinue
        if ($valid -eq $case.valid) { $passed++ }
        else { $failed++; Write-Output "FAIL: recommendation failures $location $($case.json)" }
    }
}
Write-Output "Contract fixtures: $passed passed, $failed failed"
if ($failed) { exit 1 }
