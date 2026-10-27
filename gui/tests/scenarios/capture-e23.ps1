# Raw-byte capture method reused from S1 capture-cpu.ps1/capture-registry.ps1.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PreparedDefault,
    [Parameter(Mandatory)][string]$GuardDir,
    [Parameter(Mandatory)][string]$RuntimeDir,
    [Parameter(Mandatory)][string]$Exe,
    [Parameter(Mandatory)][string]$Scratch,
    [string[]]$Only = @(),
    [Parameter(Mandatory)][string]$PlanRoot
)
$ErrorActionPreference = 'Stop'
$dest = Join-Path $PSScriptRoot 'recordings\e23'
$prior = Get-Content (Join-Path $dest 'metadata.json') -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Path $Scratch -Force | Out-Null
$profile = Join-Path $Scratch 'local'
$empty = Join-Path $Scratch 'local-empty'
New-Item -ItemType Directory -Path (Join-Path $profile 'Bubo'), $empty -Force | Out-Null
Copy-Item -LiteralPath $PreparedDefault -Destination (Join-Path $profile 'Bubo\prepared-default.json')
function Hash([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
if ((Hash $Exe) -ne '35ff7246e534653d3f5de1bc91763145e6e1f468e4d841121570ff5a3eb8efb6') { throw 'Wrong c4 executable' }
if ((Hash (Join-Path (Split-Path $Exe) 'bubo-engine.dll')) -ne '58547a669003879e9c51093b48d45f60f1c6d6b3638348cc78ed479b8b42c392') { throw 'Wrong c4 engine' }
$manifest = Get-Content (Join-Path (Split-Path (Split-Path $Exe)) 'BUILD-MANIFEST.json') -Raw | ConvertFrom-Json
if ($manifest.source.commit -ne '344f41aca8b004781dde594fd5270affd7f630c0' -or $manifest.product.product_version -ne '0.1.1') { throw 'Wrong runtime manifest' }
$commands = @($prior.commands | Where-Object { $_.file -notin @('estimate-lowmemory-default.json','plan-huihui-qwen38-flash-next-lowmemory-v1-fresh.jsonl','plan-huihui-qwen38-flash-next-ud-q4-k-xl-v1-fresh.jsonl') } | ForEach-Object { [pscustomobject]@{ file=$_.file; command=$_.command; localappdata=$(if ($_.localappdata -like '*local-empty') { $empty } else { $profile }); exit_code=$_.exit_code } })
$id = 'huihui-qwen38-flash-next-lowmemory-v1'
$commands += [pscustomobject]@{ file='estimate-lowmemory-default.json'; command=@('bubo.exe','models','estimate','--model',$id,'--json'); localappdata=$profile; exit_code=0 }
$commands += [pscustomobject]@{ file="plan-$id-fresh.jsonl"; command=@('bubo.exe','prepare','model','--catalogue-id',$id,'--output',(Join-Path $Scratch 'unpublished'),'--plan','--jsonl'); localappdata=$empty; exit_code=4 }
$commands += [pscustomobject]@{ file='plan-huihui-qwen38-flash-next-ud-q4-k-xl-v1-fresh.jsonl'; command=@('bubo.exe','prepare','model','--catalogue-id','huihui-qwen38-flash-next-ud-q4-k-xl-v1','--output',$PlanRoot,'--plan','--jsonl'); localappdata=$empty; exit_code=0 }
$captured = @()
if ($Only.Count) {
    foreach ($name in $Only) { if ($name -notin $commands.file) { throw "Unknown capture $name" } }
    $commands = @($commands | Where-Object { $_.file -in $Only })
    $captured = @($prior.commands | Where-Object { $_.file -notin $Only })
}
foreach ($command in $commands) {
    if ((Test-Path (Join-Path $GuardDir 'hold-gaming')) -or (Test-Path (Join-Path $GuardDir 'hold-timing.txt'))) { throw 'Owner hold present' }
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Exe
    foreach ($arg in ($command.command | Select-Object -Skip 1)) { $info.ArgumentList.Add($arg) }
    $info.WorkingDirectory = $Scratch
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.WindowStyle = 'Hidden'
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($name in @($info.Environment.Keys)) { if ($name -match '^(BUBO_|GUFO_)') { $info.Environment.Remove($name) | Out-Null } }
    foreach ($name in @('LOCALAPPDATA','APPDATA')) { $info.Environment[$name] = $command.localappdata }
    foreach ($name in @('TEMP','TMP')) { $info.Environment[$name] = $Scratch }
    $info.Environment['PATH'] = "$(Split-Path $Exe);C:\Windows\System32;C:\Windows"
    $process = [Diagnostics.Process]::Start($info)
    try { $process.PriorityClass = 'BelowNormal' } catch { if (-not $process.HasExited) { throw } }
    $raw = Join-Path $Scratch $command.file
    $stream = [IO.File]::Create($raw)
    try {
        $outTask = $process.StandardOutput.BaseStream.CopyToAsync($stream)
        $errTask = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $outTask.Wait()
    } finally { $stream.Dispose() }
    [IO.File]::WriteAllText("$raw.stderr.txt", $errTask.Result)
    if ($process.ExitCode -ne $command.exit_code) { throw "$($command.file): exit $($process.ExitCode), expected $($command.exit_code)" }
    foreach ($line in [IO.File]::ReadAllLines($raw)) { if ($line.Trim()) { $null = $line | ConvertFrom-Json } }
    Copy-Item -LiteralPath $raw -Destination (Join-Path $dest $command.file)
    $captured += [pscustomobject]@{ file=$command.file; command=$command.command; localappdata=$command.localappdata; exit_code=$process.ExitCode; stderr=$errTask.Result; stdout_bytes=(Get-Item $raw).Length; stdout_sha256=(Hash $raw) }
    "$($command.file): exit $($process.ExitCode), $((Get-Item $raw).Length) bytes, $(Hash $raw)"
}
# S1's explicitly derived scan-warning fixture; never edit its output by hand.
$basePath = Join-Path $dest 'models-list-gui.json'
$contractPath = Join-Path $PSScriptRoot '..\..\..\tests\cli\contract\valid\models.json'
$listing = Get-Content $basePath -Raw | ConvertFrom-Json
$warnings = Get-Content $contractPath -Raw | ConvertFrom-Json
$listing.payload.scan_warnings = $warnings.payload.scan_warnings
$listing.payload.scanned = $warnings.payload.scanned
$warningPath = Join-Path $dest 'models-with-warnings.json'
[IO.File]::WriteAllText($warningPath, ($listing | ConvertTo-Json -Depth 100 -Compress) + "`n", [Text.UTF8Encoding]::new($false))
$derived = [ordered]@{ purpose='S20: candidate 4 rows with CLI contract scan warnings; derived fixture, not raw CLI stdout.'; _source='fixture'; base=@{ file='gui/tests/scenarios/recordings/e23/models-list-gui.json'; sha256=(Hash $basePath); kept='everything except payload.scan_warnings and payload.scanned' }; copied=@{ file='tests/cli/contract/valid/models.json'; sha256=(Hash $contractPath); fields=@('payload.scan_warnings','payload.scanned') }; fixture_bytes=(Get-Item $warningPath).Length; fixture_sha256=(Hash $warningPath) }
[IO.File]::WriteAllText((Join-Path $dest 'models-with-warnings.metadata.json'), ($derived | ConvertTo-Json -Depth 12) + "`n")
$metadata = [ordered]@{ captured_at=(Get-Date -Format o); candidate_commit=$manifest.source.commit; executable=$Exe; executable_sha256=(Hash $Exe); engine_sha256=(Hash (Join-Path (Split-Path $Exe) 'bubo-engine.dll')); product_version=$manifest.product.product_version; environment=@{ LOCALAPPDATA=$profile; APPDATA=$profile; TEMP=$Scratch; TMP=$Scratch; PATH="$(Split-Path $Exe);C:\Windows\System32;C:\Windows"; 'BUBO_*/GUFO_*'='removed' }; commands=$captured; note='Candidate 4 CPU-only raw stdout. No server, engine load, GPU or network. Existing E21 model roots read only. Isolated copied saved-model pointer. Unpublished prepare uses --plan and refuses before acquisition. Running-server recordings remain historical.' }
[IO.File]::WriteAllText((Join-Path $dest 'metadata.json'), ($metadata | ConvertTo-Json -Depth 14) + "`n")
foreach ($c in $captured) {
    $path = Join-Path $dest $c.file
    if ((Get-Item $path).Length -ne $c.stdout_bytes -or (Hash $path) -ne $c.stdout_sha256) { throw "Capture integrity failed: $($c.file)" }
}
"PASS: $($captured.Count) raw captures and one derived warnings fixture verified."
