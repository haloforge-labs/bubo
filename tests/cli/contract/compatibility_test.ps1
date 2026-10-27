param([string]$Root=$PSScriptRoot)
$ErrorActionPreference='Stop'
$passed=0
$failed=0
$schema=Join-Path $Root 'event.schema.json'
foreach($file in Get-ChildItem -LiteralPath (Join-Path $Root 'v1-legacy') -Filter '*.json') {
  $record=[IO.File]::ReadAllText($file.FullName)
  if(Test-Json -Json $record -SchemaFile $schema -ErrorAction SilentlyContinue){$passed++}else{$failed++; "FAIL old fixture $($file.Name)"}
  $value=$record | ConvertFrom-Json -AsHashtable
  $value['future_envelope'] = @{any='ignored'}
  $value['payload']['future_payload'] = @{any='ignored'}
  if(Test-Json -Json ($value | ConvertTo-Json -Depth 100 -Compress) -SchemaFile $schema -ErrorAction SilentlyContinue){$passed++}else{$failed++; "FAIL future keys $($file.Name)"}
  $value.Remove('operation_id') | Out-Null
  if(-not (Test-Json -Json ($value | ConvertTo-Json -Depth 100 -Compress) -SchemaFile $schema -ErrorAction SilentlyContinue)){$passed++}else{$failed++; "FAIL missing required $($file.Name)"}
}
$index=([IO.File]::ReadAllText((Join-Path $Root 'fixture-index.json')) | ConvertFrom-Json -AsHashtable)['fixtures']
$actual=@('valid','invalid','input-valid','input-invalid','v1-legacy') | ForEach-Object {
  $folder=$_
  Get-ChildItem -LiteralPath (Join-Path $Root $folder) -Filter '*.json' | ForEach-Object { "$folder/$($_.Name)" }
}
if(@(Compare-Object @($index.Keys | Sort-Object) @($actual | Sort-Object)).Count -eq 0){$passed++}else{$failed++; 'FAIL fixture directory index differs'}
"Old fixtures, additive fields, required fields and index: $passed passed, $failed failed"
if($failed){exit 1}
