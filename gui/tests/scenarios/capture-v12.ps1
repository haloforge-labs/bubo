# V12 integration: reuse the E21/E23 command inventories and E23 raw-byte recorder.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PreparedDefault,
    [Parameter(Mandatory)][string]$GuardDir,
    [Parameter(Mandatory)][string]$RuntimeDir,
    [Parameter(Mandatory)][string]$DependenciesDir,
    [Parameter(Mandatory)][string]$Exe,
    [Parameter(Mandatory)][string]$Scratch,
    [string]$Destination = (Join-Path $PSScriptRoot 'recordings\v12'),
    [string[]]$Only = @()
)
$ErrorActionPreference = 'Stop'
$dest = $Destination
New-Item -ItemType Directory -Force $dest, $Scratch | Out-Null
$profile = Join-Path $Scratch 'local'
$empty = Join-Path $Scratch 'local-empty'
New-Item -ItemType Directory -Force (Join-Path $profile 'Bubo'), $empty | Out-Null
Copy-Item $PreparedDefault (Join-Path $profile 'Bubo\prepared-default.json')
function Hash([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Wait-Window {
    if (-not (Test-Path (Join-Path $GuardDir 'WINDOW-OPEN.txt')) -or
           -not (Test-Path (Join-Path $GuardDir 'ARMS-DONE.txt')) -or
           (Test-Path (Join-Path $GuardDir 'hold-timing.txt')) -or (Test-Path (Join-Path $GuardDir 'hold-gaming.txt')) -or
           (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory -lt 6291456) {
        throw 'Recording requires WINDOW-OPEN, ARMS-DONE, no holds and 6 GiB free RAM.'
    }
}
$inventory = [ordered]@{}
foreach ($generation in @('e21','e23')) {
    $prior = Get-Content (Join-Path $PSScriptRoot "recordings\$generation\metadata.json") -Raw | ConvertFrom-Json
    foreach ($command in $prior.commands) { $inventory[$command.file] = $command }
}
# Retire plans for rows no longer offered. Installed historical packages remain
# real local imports; their path-based estimates and readiness captures stay.
foreach ($key in @($inventory.Keys)) {
    if ($key -match '^plan-(gemma4-|diffusiongemma-)') { $inventory.Remove($key) }
}
$catalogue = Get-Content (Join-Path $PSScriptRoot '..\..\..\src\cli\model_catalogue.json') -Raw | ConvertFrom-Json
foreach ($entry in $catalogue.entries | Where-Object { $_.architecture -in @('gemma4','diffusion_gemma') }) {
    $id = $entry.id
    $inventory["plan-$id-fresh.jsonl"] = [pscustomobject]@{ file="plan-$id-fresh.jsonl"; command=@('bubo.exe','prepare','model','--catalogue-id',$id,'--plan','--jsonl'); localappdata=$empty; exit_code=$(if ($entry.download_enabled) { 0 } else { 4 }) }
    $inventory["estimate-$id-default.json"] = [pscustomobject]@{ file="estimate-$id-default.json"; command=@('bubo.exe','models','estimate','--model',$id,'--json'); localappdata=$empty; exit_code=0 }
}
# Retired packages remain importable by path. Preserve their real installed
# paths for the no-verdict scenario; new QAT rows have their own fresh plans.
$inventory.Remove('plan-hauhau-rootB-present.jsonl')
foreach($command in $inventory.Values) {
    if($command.file -like 'estimate-hauhau-*') {
        $command.command=@($command.command | ForEach-Object {
            if($_ -like '*models\gemma4-26b-a4b-uncensored-hauhaucs-qat-q4-k-m') {
                Join-Path $_ 'Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf'
            } else { $_ }
        })
    }
    if($command.file -like 'estimate-importpkg-*') {
        $command.command=@($command.command | ForEach-Object {
            if($_ -like '*gemma4-26b-a4b-stock-unsloth-ud-q4-k-m') { Join-Path $_ 'gemma-4-26B-A4B-it-UD-Q4_K_M.gguf' } else { $_ }
        })
    }
}
$warningsCommand=@($inventory['models-list-gui.json'].command)+@('--scan-folder','C:\')
$inventory['models-with-warnings.json']=[pscustomobject]@{ file='models-with-warnings.json'; command=$warningsCommand; localappdata=$profile; exit_code=0 }
$inventory['plan-runtime-fresh.jsonl'] = [pscustomobject]@{ file='plan-runtime-fresh.jsonl'; command=@('bubo.exe','prepare','runtime','--runtime-dir',(Join-Path $Scratch 'runtime-fresh'),'--jsonl'); localappdata=$empty; exit_code=3 }
$lowID='huihui-qwen38-flash-next-lowmemory-v1'
$inventory["plan-$lowID-fresh.jsonl"] = [pscustomobject]@{ file="plan-$lowID-fresh.jsonl"; command=@('bubo.exe','prepare','model','--catalogue-id',$lowID,'--output',(Join-Path $Scratch 'plan-models'),'--plan','--jsonl'); localappdata=$empty; exit_code=0 }
$inventory['plan-lowmemory-short.jsonl'] = [pscustomobject]@{ file='plan-lowmemory-short.jsonl'; command=@('bubo.exe','prepare','model','--catalogue-id',$lowID,'--output',(Join-Path $Scratch 'lower-memory-plan'),'--plan','--jsonl'); localappdata=$empty; exit_code=0 }
$inventory['estimate-flash-catalogue-id.json'] = [pscustomobject]@{ file='estimate-flash-catalogue-id.json'; command=@('bubo.exe','models','estimate','--model','huihui-qwen38-flash-next-ud-q4-k-xl-v1','--json'); localappdata=$profile; exit_code=0 }
foreach ($command in @($inventory.Values)) {
    if ($command.file -notlike 'estimate-*-c262144-s*.json') { continue }
    $file = $command.file.Replace('c262144','c200000')
    $arguments = @($command.command | ForEach-Object { if ($_ -eq '262144') { '200000' } else { $_ } })
    $inventory[$file] = [pscustomobject]@{ file=$file; command=$arguments; localappdata=$command.localappdata; exit_code=$command.exit_code }
}
$inventory['plan-runtime-ready.jsonl'] = [pscustomobject]@{ file='plan-runtime-ready.jsonl'; command=@('bubo.exe','prepare','runtime','--runtime-dir',$RuntimeDir,'--jsonl'); localappdata=$empty; exit_code=0 }
$refreshRoot = Join-Path $Scratch 'runtime-payload-without-metadata'
# Reuse all 263 real payload files through hard links in this job's scratch
# directory. No marker or notices are supplied: this needs metadata refresh.
$published = Get-Content (Join-Path $PSScriptRoot '..\..\..\tests\prepare\fixtures\published-v0.1.0.json') -Raw | ConvertFrom-Json
foreach ($line in ($published.runtime_members -split "`n" | Select-Object -Skip 1)) {
    if (-not $line.Trim()) { continue }
    $fields = $line -split "`t"
    $target = Join-Path $refreshRoot $fields[2]
    $source = Join-Path $RuntimeDir $fields[2]
    if (-not (Test-Path -LiteralPath $target)) {
        New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
        New-Item -ItemType HardLink -Path $target -Target $source | Out-Null
    }
}
$inventory['plan-runtime-refresh.jsonl'] = [pscustomobject]@{ file='plan-runtime-refresh.jsonl'; command=@('bubo.exe','prepare','runtime','--runtime-dir',$refreshRoot,'--jsonl'); localappdata=$empty; exit_code=3 }
$captures = @()
if ($Only.Count) {
    $old = Get-Content (Join-Path $dest 'metadata.json') -Raw | ConvertFrom-Json
    $captures = @($old.commands | Where-Object { $_.file -notin $Only })
}
foreach ($command in $inventory.Values) {
    if ($Only.Count -and $command.file -notin $Only) { continue }
    Wait-Window
    $arguments = @($command.command | Select-Object -Skip 1)
    # All prepare invocations are previews; acquisition is never authorized.
    if ($arguments[0] -eq 'prepare' -and '--plan' -notin $arguments -and $command.file -notlike 'plan-runtime-*.jsonl') { throw 'Non-preview prepare command' }
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Exe
    foreach ($arg in $arguments) { $info.ArgumentList.Add($arg) }
    $info.WorkingDirectory = $Scratch
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.WindowStyle = 'Hidden'
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.RedirectStandardInput = $true
    foreach ($name in @($info.Environment.Keys)) { if ($name -match '^(BUBO_|GUFO_)') { $info.Environment.Remove($name) | Out-Null } }
    $local = if ($command.localappdata -like '*local-empty') { $empty } else { $profile }
    $info.Environment['LOCALAPPDATA'] = $local
    $info.Environment['APPDATA'] = $local
    $info.Environment['TEMP'] = $Scratch
    $info.Environment['TMP'] = $Scratch
    $deps = $DependenciesDir
    $info.Environment['PATH'] = "$(Split-Path $Exe);$deps;C:\Windows\System32;C:\Windows"
    # Existing SDK runtime is discovered read-only; the engine DLL still comes
    # from beside the freshly built CPU executable (bootstrap's EnginePath).
    $runtime = $RuntimeDir
    $info.Environment['BUBO_RUNTIME_DIR'] = $runtime
    $started = Get-Date -Format o
    $process = [Diagnostics.Process]::Start($info)
    # Runtime has no --plan option. Its machine preview requires consent;
    # closed stdin declines before any acquisition or metadata refresh.
    if ($command.file -like 'plan-runtime-*.jsonl') { $process.StandardInput.Close() }
    $raw = Join-Path $dest $command.file
    $stream = [IO.File]::Create($raw)
    try {
        $outTask = $process.StandardOutput.BaseStream.CopyToAsync($stream)
        $errTask = $process.StandardError.ReadToEndAsync()
        while (-not $process.WaitForExit(2000)) {
            if ((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory -lt 3145728) {
                $process.Kill($true)
                throw 'Stopped own recording below 3 GiB free RAM'
            }
        }
        $outTask.Wait()
    } finally { $stream.Dispose() }
    [IO.File]::WriteAllText((Join-Path $Scratch "$($command.file).stderr.txt"), $errTask.Result)
    if ($process.ExitCode -ne $command.exit_code -and -not ($command.file -eq 'plan-runtime-refresh.jsonl' -and $process.ExitCode -eq 0)) { throw "$($command.file): exit $($process.ExitCode), expected $($command.exit_code); $($errTask.Result)" }
    if ($raw -notlike '*.txt') { foreach ($line in [IO.File]::ReadAllLines($raw)) { if ($line.Trim()) { $null = $line | ConvertFrom-Json } } }
    $captures += [ordered]@{ file=$command.file; command=$command.command; localappdata=$local; exit_code=$process.ExitCode; stderr=$errTask.Result; started=$started; finished=(Get-Date -Format o); stdout_bytes=(Get-Item $raw).Length; stdout_sha256=(Hash $raw) }
    Write-Host "$($command.file): $((Get-Item $raw).Length) bytes"
}
# The warning is a real CLI drive-root refusal, with real scanned model rows.
$warningPath = Join-Path $dest 'models-with-warnings.json'
$derived = [ordered]@{ _source='recorded'; file='models-with-warnings.json'; command=$warningsCommand; fixture_bytes=(Get-Item $warningPath).Length; fixture_sha256=(Hash $warningPath) }
[IO.File]::WriteAllText((Join-Path $dest 'models-with-warnings.metadata.json'), ($derived | ConvertTo-Json -Depth 12).Replace("`r`n", "`n") + "`n")
$metadata = [ordered]@{ captured_at=(Get-Date -Format o); candidate_commit=(& git -C (Join-Path $PSScriptRoot '..\..\..') rev-parse HEAD); gui_commit=(& git -C (Join-Path $PSScriptRoot '..\..\..') rev-parse HEAD); executable=$Exe; executable_sha256=(Hash $Exe); engine_sha256=(Hash (Join-Path (Split-Path $Exe) 'bubo-engine.dll')); environment=@{ PATH=$info.Environment['PATH']; BUBO_RUNTIME_DIR=$runtime; 'BUBO_*/GUFO_*'='removed except the recorded runtime discovery path' }; commands=$captures; note='CPU-only final integrated CLI. E21/E23 inventory and isolated saved pointer. Existing model roots read only; retired Gemma packages estimated by their main file paths. No server, GUI, GPU or acquisition. Historical running captures remain historical.' }
[IO.File]::WriteAllText((Join-Path $dest 'metadata.json'), ($metadata | ConvertTo-Json -Depth 14).Replace("`r`n", "`n") + "`n")
Write-Host "PASS: $($captures.Count) raw captures including the recorded scan warning"
