param([Parameter(Mandatory)][string]$GuardDir, [Parameter(Mandatory)][string]$EvidenceDir)
$ErrorActionPreference = 'Stop'
$env:CGO_ENABLED = '1' # the supervisor's fake CLI is race-instrumented
$env:GOFLAGS = '-mod=readonly'
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
# Keep Go/npm caches in their default locations.
New-Item -ItemType Directory -Force -Path $EvidenceDir | Out-Null
$EvidenceDir = (Resolve-Path -LiteralPath $EvidenceDir).Path
$guiRoot = Split-Path $PSScriptRoot
$failures = [System.Collections.Generic.List[string]]::new()
function Fail([string]$Message) { $failures.Add($Message); Write-Output "FAIL: $Message" }
function Wait-TimingOwner {
    if ((Test-Path (Join-Path $GuardDir 'hold-timing.txt')) -or (Test-Path (Join-Path $GuardDir 'hold-gaming.txt')) -or
        -not (Test-Path (Join-Path $GuardDir 'WINDOW-OPEN.txt'))) {
        throw 'The executable test window is closed.'
    }
    if ((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory -lt 6291456) { throw 'Tests need 6 GiB free RAM.' }
    $lock = (Join-Path $GuardDir 'gpu-harness.lock')
    while ((Test-Path -LiteralPath $lock) -and ((Get-Content -LiteralPath $lock -Raw) -match '(?i)timing')) {
        Write-Output 'Waiting for the GPU timing owner before race/browser work.'
        Start-Sleep -Seconds 5
    }
}
function GeneratedHashes {
    $files = @(
        (Join-Path $guiRoot 'supervisor/contract.schema.json'),
        (Join-Path $guiRoot 'supervisor/view_data_generated_windows.go')
    )
    foreach ($folder in @('wails/frontend/wailsjs','wails/frontend/src/scenarios','wails/frontend/src/generated')) {
        $files += @(Get-ChildItem -LiteralPath (Join-Path $guiRoot $folder) -Recurse -File | Select-Object -ExpandProperty FullName)
    }
    $files += @(Get-ChildItem -Path (Join-Path $PSScriptRoot 'scenarios/recordings/*/scenario.json') -File | Select-Object -ExpandProperty FullName)
    $hashes = @{}
    foreach ($file in $files) { $hashes[$file] = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash }
    return $hashes
}
$generatedBefore = GeneratedHashes
if ((go version) -ne 'go version go1.27.2 windows/amd64') { throw 'Use Go 1.27.2.' }
go version
$env:BUBO_FAKE_FIXTURES = Join-Path (Split-Path $guiRoot) 'tests/cli/contract'
$env:BUBO_FAKE_DIR = Join-Path $EvidenceDir 'fakecli-state'
New-Item -ItemType Directory -Force -Path $env:BUBO_FAKE_DIR | Out-Null
$env:BUBO_EVIDENCE_DIR = Join-Path $EvidenceDir 'artifacts'
New-Item -ItemType Directory -Force -Path $env:BUBO_EVIDENCE_DIR | Out-Null
$fakeExe = Join-Path $EvidenceDir 'fakecli.exe'
Push-Location $PSScriptRoot
try {
    Wait-TimingOwner
    go build -race -o $fakeExe ./fakecli
    if ($LASTEXITCODE -ne 0) { throw 'Fake CLI build failed.' }
} finally { Pop-Location }
$env:BUBO_FAKE_EXE = $fakeExe
$modules = @(Get-ChildItem -Path (Join-Path $guiRoot '*/go.mod') | Sort-Object FullName)
Write-Output "Go modules: $($modules.Count) discovered"
foreach ($mod in $modules) {
    $module = Split-Path (Split-Path $mod.FullName) -Leaf
    Push-Location (Split-Path $mod.FullName)
    try {
        $deps = @(go list -deps ./... 2>&1)
        if ($LASTEXITCODE -ne 0) { Fail "$module dependency listing failed" }
        $fyne = @($deps | Where-Object { $_ -match '^fyne\.io/' })
        if ($fyne.Count) { Fail "$module dependencies: $($fyne.Count) fyne.io/ packages" }
        else { Write-Output "$module dependency guard: 1 passed, 0 failed" }
        $deps | Set-Content -LiteralPath (Join-Path $EvidenceDir "$module-deps.log")
        go vet ./... 2>&1 | Tee-Object -FilePath (Join-Path $EvidenceDir "$module-vet.log")
        if ($LASTEXITCODE -ne 0) { Fail "$module vet failed" }
        else { Write-Output "$module vet: 1 passed, 0 failed" }
        $log = Join-Path $EvidenceDir "$module-tests.jsonl"
        Wait-TimingOwner
        # The supervisor race suite alone takes ~185 s while other lanes' CPU work
        # runs (2026-10-15 F3: timed out at 180 s with 0 assertion failures). The
        # cap only stops a hung run; every test must still pass.
        go test -race -json -count=1 -timeout 600s ./... 2>&1 | Set-Content -LiteralPath $log
        $testExit = $LASTEXITCODE
        $events = @(Get-Content -LiteralPath $log | ForEach-Object {
            try { $_ | ConvertFrom-Json -ErrorAction Stop } catch { Write-Output $_ }
        })
        $passed = @($events | Where-Object { $_.Action -eq 'pass' -and $_.Test }).Count
        $failed = @($events | Where-Object { $_.Action -eq 'fail' -and $_.Test }).Count
        $skipped = @($events | Where-Object { $_.Action -eq 'skip' -and $_.Test }).Count
        $completed = @($events | Where-Object { $_.Test -and $_.Action -in @('pass','fail','skip') } | Select-Object -ExpandProperty Test)
        $incomplete = @($events | Where-Object { $_.Action -eq 'run' -and $_.Test -notin $completed }).Count
        Write-Output "$module tests (including subtests): $passed passed, $failed failed, $skipped skipped, $incomplete incomplete"
        if ($testExit -ne 0) {
            $failedNames = @($events | Where-Object { $_.Action -eq 'fail' -and $_.Test } | Select-Object -ExpandProperty Test)
            $events | Where-Object { $_.Action -eq 'output' -and ($_.Test -in $failedNames -or $_.Output -match '^(FAIL\s|panic:|# )') } | ForEach-Object { Write-Output $_.Output }
            Fail "$module tests failed (exit $testExit); see $log"
        }
    } finally { Pop-Location }
}
$viewFiles = Get-ChildItem -LiteralPath (Join-Path $guiRoot 'supervisor') -Filter 'view*.go'
$untyped = @($viewFiles | Select-String -Pattern 'map\[string\]any|interface\s*\{\s*\}')
if ($untyped.Count) { $untyped | Write-Output; Fail 'View contains untyped maps or interfaces' }
else { Write-Output 'View type guard: 1 passed, 0 failed' }
Push-Location (Join-Path $guiRoot 'supervisor')
try {
    $copyHash = (Get-FileHash -LiteralPath 'view_data_generated_windows.go' -Algorithm SHA256).Hash
    go generate ./...
    if ($LASTEXITCODE -ne 0 -or $copyHash -ne (Get-FileHash -LiteralPath 'view_data_generated_windows.go' -Algorithm SHA256).Hash) { Fail 'Generated View copies are stale' }
    else { Write-Output 'View copy generation guard: 1 passed, 0 failed' }
} finally { Pop-Location }
Push-Location $PSScriptRoot
try {
    go run ./scenarios 2>&1 | Tee-Object -FilePath (Join-Path $EvidenceDir 'scenarios.log')
    if ($LASTEXITCODE -ne 0) { Fail 'Scenario generation failed' }
} finally { Pop-Location }
Push-Location (Join-Path $guiRoot 'wails')
try {
    $wails = Join-Path (go env GOPATH) 'bin/wails.exe'
    if (-not (Test-Path -LiteralPath $wails)) { throw 'Install github.com/wailsapp/wails/v2/cmd/wails@v2.16.0.' }
    function BindingHashes {
        $hashes = @{}
        Get-ChildItem -LiteralPath 'frontend/wailsjs' -File -Recurse | ForEach-Object {
            $hashes[$_.FullName] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        }
        return $hashes
    }
    $before = BindingHashes
    & $wails generate module -nocolour 2>&1 | Set-Content -LiteralPath (Join-Path $EvidenceDir 'bindings.log')
    if ($LASTEXITCODE -ne 0) { Fail 'Wails binding generation failed' }
    $after = BindingHashes
    if ($before.Count -ne $after.Count -or @($after.Keys | Where-Object { $before[$_] -ne $after[$_] }).Count) { Fail 'Wails generation changed bindings' }
    git diff --check -- frontend/wailsjs
    if ($LASTEXITCODE -ne 0) { Fail 'Generated bindings have whitespace errors' }
    else { Write-Output 'Binding whitespace guard: 1 passed, 0 failed' }
} finally { Pop-Location }
$frontend = Join-Path $guiRoot 'wails/frontend'
if (Test-Path -LiteralPath (Join-Path $frontend 'package.json')) {
    Push-Location $frontend
    try {
        foreach ($step in @('gen:types','routes','typecheck','lint','knip','format:check','test')) {
            if ($step -eq 'test') { Wait-TimingOwner }
            $stepArgs = if ($step -eq 'test') { @('--','--maxWorkers=2','--no-file-parallelism') } else { @() }
            npm run $step @stepArgs 2>&1 | Tee-Object -FilePath (Join-Path $EvidenceDir ("frontend-$step.log" -replace ':', '-'))
            if ($LASTEXITCODE -ne 0) { Fail "frontend $step failed" }
            else { Write-Output "frontend ${step}: 1 passed, 0 failed" }
        }
    } finally { Pop-Location }
} else { Fail 'Production frontend/package.json is missing' }
$generatedAfter = GeneratedHashes
$changed = @($generatedAfter.Keys | Where-Object { $generatedBefore[$_] -ne $generatedAfter[$_] })
$removed = @($generatedBefore.Keys | Where-Object { -not $generatedAfter.ContainsKey($_) })
if ($changed.Count -or $removed.Count) { $changed | Write-Output; $removed | Write-Output; Fail 'Generated code changed' }
else { Write-Output "Generation guard: $($generatedAfter.Count) files, 0 changed, 0 removed" }
Push-Location (Split-Path $guiRoot)
try {
    git diff --check -- gui/supervisor/contract.schema.json gui/supervisor/view_data_generated_windows.go gui/wails/frontend/wailsjs gui/wails/frontend/src/generated gui/wails/frontend/src/scenarios gui/tests/scenarios/recordings
    if ($LASTEXITCODE -ne 0) { Fail 'Generated artifacts have whitespace errors' }
    else { Write-Output 'Generated whitespace: 1 passed, 0 failed (idempotence checked against entry hashes)' }
} finally { Pop-Location }
Write-Output "repository gate: $($failures.Count) failed checks"
Remove-Item -LiteralPath $fakeExe -Force
if ($failures.Count) { throw ($failures -join "`n") }
