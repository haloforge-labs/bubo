param([Parameter(Mandatory)][string]$GuardDir, [switch]$Race, [Parameter(Mandatory)][string]$EvidenceDir)
$ErrorActionPreference = 'Stop'
$env:CGO_ENABLED = '1' # existing supervisor/fake-CLI proof uses the race runtime
$env:GOFLAGS = '-mod=readonly'
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
if ((go version) -ne 'go version go1.27.2 windows/amd64') { throw 'Use Go 1.27.2.' }
New-Item -ItemType Directory -Force -Path $EvidenceDir | Out-Null
$guiRoot = Split-Path $PSScriptRoot
$failures = [System.Collections.Generic.List[string]]::new()
$utf8 = [System.Text.UTF8Encoding]::new($false)
function Save-Lines([string]$Name, [object[]]$Lines) {
    [System.IO.File]::WriteAllText((Join-Path $EvidenceDir $Name), (($Lines | ForEach-Object { [string]$_ }) -join "`n") + "`n", $utf8)
}
foreach ($module in @('supervisor', 'wails', 'tests', 'updater')) {
    Push-Location (Join-Path $guiRoot $module)
    try {
        $output = @(& go vet ./... 2>&1)
        $code = $LASTEXITCODE
        Save-Lines "$module-vet.log" $output
        $packages = @(& go list ./...)
        Write-Output "$module vet: packages=$($packages.Count), failed=$([int]($code -ne 0))"
        if ($code -ne 0) { $failures.Add("$module vet"); $output | Write-Output }
    } finally { Pop-Location }
}
Push-Location $PSScriptRoot
try {
    $output = @(& go run ./scenarios 2>&1)
    Save-Lines 'scenarios.log' $output
    $output | Write-Output
    if ($LASTEXITCODE -ne 0) { $failures.Add('scenario generator') }
} finally { Pop-Location }
foreach ($module in @('supervisor', 'wails', 'tests')) {
    if ($Race -and (Test-Path -LiteralPath (Join-Path $GuardDir 'gpu-harness.lock'))) {
        throw 'Race proof deferred: GPU harness lock exists; correctness runs may use this script without -Race.'
    }
    Push-Location (Join-Path $guiRoot $module)
    try {
        $testArgs = @('test', '-json', '-count=1', '-timeout', '180s')
        if ($Race) { $testArgs += '-race' }
        $output = @(& go @testArgs './...' 2>&1)
        $code = $LASTEXITCODE
        Save-Lines "$module-tests.jsonl" $output
        $events = @($output | ForEach-Object { try { ([string]$_) | ConvertFrom-Json -ErrorAction Stop } catch {} })
        $passed = @($events | Where-Object { $_.Test -and $_.Action -eq 'pass' }).Count
        $failed = @($events | Where-Object { $_.Test -and $_.Action -eq 'fail' }).Count
        $skipped = @($events | Where-Object { $_.Test -and $_.Action -eq 'skip' }).Count
        $complete = @($events | Where-Object { $_.Test -and $_.Action -in @('pass', 'fail', 'skip') } | Select-Object -ExpandProperty Test)
        $incomplete = @($events | Where-Object { $_.Action -eq 'run' -and $_.Test -notin $complete }).Count
        Write-Output "$module tests: passed=$passed failed=$failed skipped=$skipped incomplete=$incomplete race=$Race"
        if ($code -ne 0) {
            $failures.Add("$module tests")
            $names = @($events | Where-Object { $_.Action -eq 'fail' -and $_.Test } | Select-Object -ExpandProperty Test)
            $events | Where-Object { $_.Action -eq 'output' -and ($_.Test -in $names -or $_.Output -match '^(FAIL\s|panic:|# )') } | ForEach-Object { Write-Output $_.Output }
        }
    } finally { Pop-Location }
}
if ($failures.Count) { throw "Failed U4 proof: $($failures -join ', ')" }
Write-Output 'U4 proof: passed; no visible GUI, model or real inference server launched; updater core unchanged (lane C proof retained).'
