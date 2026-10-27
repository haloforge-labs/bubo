param([Parameter(Mandatory)][string]$Bootstrap,[Parameter(Mandatory)][string]$MemoryTest)
$ErrorActionPreference='Stop'
$passed=0
$schema=Join-Path $PSScriptRoot 'event.schema.json'
$env:BUBO_MANIFEST='C:\no-model-fixture\missing.json'
$env:BUBO_API_KEY='synthetic-auth-key-not-for-output'
$env:HOST='synthetic-host-not-for-output'
$env:PORT='49152'
$lines=@(& $Bootstrap serve --list-switches --json)
if($LASTEXITCODE -ne 0 -or $lines.Count -ne 1){throw 'Bootstrap registry failed'}
$record=$lines[0]
if(-not(Test-Json -Json $record -SchemaFile $schema)){throw 'Registry event schema failed'}
$passed++
$event=$record | ConvertFrom-Json -AsHashtable
$expected=[IO.File]::ReadAllText((Join-Path $PSScriptRoot 'switches.golden.json')) | ConvertFrom-Json -AsHashtable
if(($event.payload | ConvertTo-Json -Depth 100 -Compress) -ne ($expected | ConvertTo-Json -Depth 100 -Compress)){throw 'Registry emitted golden differs'}
$passed++
if($record.Contains($env:BUBO_API_KEY) -or $record.Contains($env:HOST)){throw 'Registry leaked environment'}
$passed++
$bias = @($event.payload.switches | Where-Object name -eq '--think-marker-bias')
if($bias.Count -ne 1 -or $bias[0].allowed.type -ne 'number' -or $bias[0].allowed.format -ne 'finite' -or
   $bias[0].allowed.suggested.minimum -ne -2 -or $bias[0].allowed.suggested.maximum -ne 0 -or
   $bias[0].allowed.ContainsKey('minimum') -or $bias[0].allowed.ContainsKey('maximum') -or
   $bias[0].allowed.ContainsKey('step')) { throw 'Reasoning recommendations must preserve the finite-number reader' }
$passed++
foreach($row in $event.payload.switches | Where-Object secret) {
    if($row.echo -or $row.gui_control) { throw 'Secret row is echoed or GUI-writable' }
}
$passed++
$future=$event
$future['future_envelope']=@{anything='ignored'}
$future.payload.switches[0]['future_switch']=@{anything='ignored'}
if(-not(Test-Json -Json ($future | ConvertTo-Json -Depth 100 -Compress) -SchemaFile $schema)){throw 'Unknown registry keys rejected'}
$passed++
$lines=@(& $MemoryTest)
if($LASTEXITCODE -ne 0){throw 'Memory test failed'}
$jsonLines=@($lines | Where-Object { $_.StartsWith('{') })
if($jsonLines.Count -ne 2){throw 'Expected emitted Pi and memory JSON records'}
$snippet=$jsonLines[0]
$estimate=$jsonLines[1]
if(-not(Test-Json -Json $snippet -SchemaFile (Join-Path $PSScriptRoot 'pi-snippet.schema.json'))){throw 'Pi schema failed'}
$passed++
if($snippet.Contains($env:BUBO_API_KEY) -or -not $snippet.Contains('BUBO_API_KEY')){throw 'Pi authentication key leaked or missing env reference'}
$passed++
$fixture=[IO.File]::ReadAllText((Join-Path $PSScriptRoot 'valid/status.json')) | ConvertFrom-Json -AsHashtable
$fixture.payload['memory_estimate']=$estimate | ConvertFrom-Json -AsHashtable
$fixture.payload['memory_estimate_reason']=$null
$fixture.payload['pi_snippet']=$snippet | ConvertFrom-Json -AsHashtable
if(-not(Test-Json -Json ($fixture | ConvertTo-Json -Depth 100 -Compress) -SchemaFile $schema)){throw 'Memory event schema failed'}
$passed++
$fixture.payload.memory_estimate.weights.Remove('evidence') | Out-Null
if(Test-Json -Json ($fixture | ConvertTo-Json -Depth 100 -Compress) -SchemaFile $schema -ErrorAction SilentlyContinue){throw 'Memory without evidence accepted'}
$passed++
"Bootstrap listing, emitted memory, Pi schema and authentication: $passed passed, 0 failed"
