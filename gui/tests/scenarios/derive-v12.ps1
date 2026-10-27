# Historical running observations cannot be re-recorded without a GPU/server.
# Make explicitly derived fixtures: replace only the current CLI profile rows.
param(
    [Parameter(Mandatory)][string]$PreparedModelPath,
    [Parameter(Mandatory)][string]$PreparedPackageID
)
$ErrorActionPreference = 'Stop'
$dest = Join-Path $PSScriptRoot 'recordings\v12'
$current = Get-Content (Join-Path $dest 'status-main.json') -Raw | ConvertFrom-Json
$registry = (Get-Content (Join-Path $dest 'switches.json') -Raw | ConvertFrom-Json).payload.switches
$byName = @{}
foreach ($row in $registry) { $byName[$row.name] = $row }
$catalogue = (Get-Content (Join-Path $dest 'models-list-empty.json') -Raw | ConvertFrom-Json).payload.catalogue
$modelNames = @{}
foreach ($row in $catalogue) { $modelNames[$row.id] = $row.display_name }
$package = [pscustomobject]@{ package_id=$PreparedPackageID }
$modelPath = $PreparedModelPath
$oldModelPath = 'C:\bubo\run\gui-build\k1cpu\contract-serve-control\fixture-model.gguf'
function Sync-Identity($Value, [string[]]$Segments, $Changes) {
    if ($Value -is [array]) {
        for ($i=0; $i -lt $Value.Count; $i++) { Sync-Identity $Value[$i] ($Segments + [string]$i) $Changes }
    } elseif ($Value -is [pscustomobject]) {
        foreach ($field in $Value.PSObject.Properties) {
            if ($field.Value -is [string]) {
                $before = $field.Value
                $after = $before.Replace($oldModelPath, $modelPath).Replace('fixture-model.gguf', [IO.Path]::GetFileName($modelPath)).Replace('bubo-test-only-fake', $package.package_id)
                if ($after -cne $before) {
                    $field.Value = $after
                    $Changes.Add([ordered]@{ path=@($Segments + $field.Name); before=$before; after=$after })
                }
            } else { Sync-Identity $field.Value ($Segments + $field.Name) $Changes }
        }
    }
}
function Sync-ModelNames($Value, [string]$Path, [Collections.Generic.List[string]]$Replaced) {
    if ($Value -is [array]) {
        for ($i = 0; $i -lt $Value.Count; $i++) { Sync-ModelNames $Value[$i] "$Path[$i]" $Replaced }
    } elseif ($Value -is [pscustomobject]) {
        if ($Value.PSObject.Properties['id'] -and $Value.PSObject.Properties['display_name'] -and $modelNames[$Value.id]) {
            if ($Value.display_name -cne $modelNames[$Value.id]) {
                $Value.display_name = $modelNames[$Value.id]
                $Replaced.Add("$Path.display_name")
            }
        }
        foreach ($field in $Value.PSObject.Properties) { Sync-ModelNames $field.Value "$Path.$($field.Name)" $Replaced }
    }
}
function Sync-Constraints($Value, [string]$Path, [Collections.Generic.List[string]]$Replaced) {
    if ($Value -is [array]) {
        for ($i = 0; $i -lt $Value.Count; $i++) { Sync-Constraints $Value[$i] "$Path[$i]" $Replaced }
    } elseif ($Value -is [pscustomobject]) {
        if ($Value.PSObject.Properties['name'] -and $Value.PSObject.Properties['allowed']) {
            $ownRow = $byName[$Value.name]
            if ($ownRow -and ($Value.allowed | ConvertTo-Json -Depth 30 -Compress) -cne ($ownRow.allowed | ConvertTo-Json -Depth 30 -Compress)) {
                $Value.allowed = $ownRow.allowed
                $Replaced.Add("$Path.allowed")
            }
        }
        foreach ($setting in $Value.PSObject.Properties) {
            $row = $byName[$setting.Name]
            if ($row -and $setting.Value -is [pscustomobject] -and $setting.Value.PSObject.Properties['allowed']) {
                if (($setting.Value.allowed | ConvertTo-Json -Depth 30 -Compress) -cne ($row.allowed | ConvertTo-Json -Depth 30 -Compress)) {
                    $setting.Value.allowed = $row.allowed
                    $Replaced.Add("$Path.$($setting.Name).allowed")
                }
            }
            Sync-Constraints $setting.Value "$Path.$($setting.Name)" $Replaced
        }
    }
}
$sources = [ordered]@{
    'status-w-idle.json'='w/status-ready-flash-next.json'
    'status-gate-idle.json'='gate3/status-0001-230211.374.json'
    'status-gate-busy.json'='gate3/status-0007-230224.488.json'
    'status-lrel8-busy.json'='lrel8/status-0027-002858.110.json'
    'status-engine-missing.json'='b4/status.json'
    'status-page-facts-false.json'='b4/page-facts-false.json'
    'status-page-facts-true.json'='b4/page-facts-true.json'
}
$derived = @()
foreach ($entry in $sources.GetEnumerator()) {
    $source = Join-Path $PSScriptRoot "recordings\$($entry.Value)"
    $record = Get-Content $source -Raw | ConvertFrom-Json
    $record.product = $current.product
    $record.payload.versions = $current.payload.versions
    $record.payload.capabilities.profiles = $current.payload.capabilities.profiles
    $replaced = [Collections.Generic.List[string]]::new()
    $replaced.Add('payload.capabilities.profiles')
    $replaced.Add('product')
    $replaced.Add('payload.versions')
    Sync-Constraints $record '' $replaced
    Sync-ModelNames $record '' $replaced
    $identityChanges = [Collections.Generic.List[object]]::new()
    Sync-Identity $record @() $identityChanges
    $path = Join-Path $dest $entry.Key
    [IO.File]::WriteAllText($path, ($record | ConvertTo-Json -Depth 100 -Compress) + "`n", [Text.UTF8Encoding]::new($false))
    $derived += [ordered]@{ file=$entry.Key; _source='fixture'; source=$entry.Value; source_sha256=(Get-FileHash $source).Hash.ToLowerInvariant(); replaced=$replaced; identity_replacements=@($identityChanges.ToArray()); identity_from=$current.payload.prepared_default; copied_from='v12/status-main.json'; copied_sha256=(Get-FileHash (Join-Path $dest 'status-main.json')).Hash.ToLowerInvariant(); registry_from='v12/switches.json'; registry_sha256=(Get-FileHash (Join-Path $dest 'switches.json')).Hash.ToLowerInvariant(); fixture_sha256=(Get-FileHash $path).Hash.ToLowerInvariant() }
}
[IO.File]::WriteAllText((Join-Path $dest 'derived-statuses.metadata.json'), ($derived | ConvertTo-Json -Depth 10).Replace("`r`n", "`n") + "`n")
$transferSource = Join-Path $PSScriptRoot '..\..\wails\frontend\tests\recorded-observations.json'
$transfer = (Get-Content $transferSource -Raw | ConvertFrom-Json).transfer
$progress = [ordered]@{
    schema_version=1; type='progress'; operation_id=$current.operation_id; seq=1
    time=$current.time; product=$current.product
    payload=[ordered]@{ component='model'; phase='download'; file_id=$transfer.file_id; bytes_done=$transfer.bytes_done; bytes_total=$transfer.bytes_total; resume_offset=0; verification_method='sha256' }
}
[IO.File]::WriteAllText((Join-Path $dest 'progress-model.json'), ($progress | ConvertTo-Json -Depth 10 -Compress) + "`n")
[ordered]@{ _source='fixture'; note=$transfer.note; observed_source=$transfer.source; observed_sha256=$transfer.sha256; copied_from='v12/status-main.json'; fixture_sha256=(Get-FileHash (Join-Path $dest 'progress-model.json')).Hash.ToLowerInvariant() } | ConvertTo-Json | Set-Content (Join-Path $dest 'progress-model.metadata.json') -Encoding utf8
Write-Host "PASS: $($derived.Count) historical status fixtures; captured product/profile/constraint/model-name metadata replaced"
