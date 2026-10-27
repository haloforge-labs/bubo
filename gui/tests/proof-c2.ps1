param([Parameter(Mandatory)][string]$EvidenceDir)
$ErrorActionPreference = 'Stop'
$env:CGO_ENABLED = '1' # The existing supervisor TestMain builds its fake CLI with -race.
$env:GOFLAGS = '-mod=readonly'
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
if ((go version) -ne 'go version go1.27.2 windows/amd64') { throw 'Use Go 1.27.2.' }
New-Item -ItemType Directory -Force -Path $EvidenceDir | Out-Null
$EvidenceDir = (Resolve-Path -LiteralPath $EvidenceDir).Path
$guiRoot = Split-Path $PSScriptRoot
$repoRoot = Split-Path $guiRoot
$failures = [System.Collections.Generic.List[string]]::new()
$utf8 = [System.Text.UTF8Encoding]::new($false)
function Save-Lines([string]$Name, [object[]]$Lines) {
    [IO.File]::WriteAllText((Join-Path $EvidenceDir $Name), (($Lines | ForEach-Object { [string]$_ }) -join "`n") + "`n", $utf8)
}
function Generated-Hashes {
    $files = @(
        (Join-Path $guiRoot 'supervisor/contract.schema.json'),
        (Join-Path $guiRoot 'supervisor/view_data_generated_windows.go')
    )
    $files += @(Get-ChildItem -LiteralPath (Join-Path $guiRoot 'wails/frontend/wailsjs') -Recurse -File | Select-Object -ExpandProperty FullName)
    $files += @(Get-ChildItem -LiteralPath (Join-Path $guiRoot 'wails/frontend/src/scenarios') -Recurse -File | Select-Object -ExpandProperty FullName)
    $files += @(Get-ChildItem -Path (Join-Path $PSScriptRoot 'scenarios/recordings/*/scenario.json') -File | Select-Object -ExpandProperty FullName)
    $hashes = @{}
    foreach ($file in $files) { $hashes[$file] = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash }
    return $hashes
}
$before = Generated-Hashes
foreach ($module in Get-ChildItem -Path (Join-Path $guiRoot '*/go.mod') | Sort-Object FullName) {
    $name = Split-Path (Split-Path $module.FullName) -Leaf
    Push-Location (Split-Path $module.FullName)
    try {
        $output = @(& go vet ./... 2>&1)
        $code = $LASTEXITCODE
        Save-Lines "$name-vet.log" $output
        $packages = @(& go list ./...)
        Write-Output "$name vet: packages=$($packages.Count), failed=$([int]($code -ne 0))"
        if ($code -ne 0) { $failures.Add("$name vet"); $output | Write-Output }
        $output = @(& go test -json -count=1 -timeout 180s ./... 2>&1)
        $code = $LASTEXITCODE
        Save-Lines "$name-tests.jsonl" $output
        $events = @($output | ForEach-Object { try { ([string]$_) | ConvertFrom-Json -ErrorAction Stop } catch {} })
        $passed = @($events | Where-Object { $_.Test -and $_.Action -eq 'pass' }).Count
        $failed = @($events | Where-Object { $_.Test -and $_.Action -eq 'fail' }).Count
        $skipped = @($events | Where-Object { $_.Test -and $_.Action -eq 'skip' }).Count
        $completed = @($events | Where-Object { $_.Test -and $_.Action -in @('pass','fail','skip') } | Select-Object -ExpandProperty Test)
        $incomplete = @($events | Where-Object { $_.Action -eq 'run' -and $_.Test -notin $completed }).Count
        Write-Output "$name tests: passed=$passed failed=$failed skipped=$skipped incomplete=$incomplete"
        if ($code -ne 0) {
            $failures.Add("$name tests")
            $names = @($events | Where-Object { $_.Test -and $_.Action -eq 'fail' } | Select-Object -ExpandProperty Test)
            $events | Where-Object { $_.Action -eq 'output' -and ($_.Test -in $names -or $_.Output -match '^(FAIL\s|panic:|# )') } | ForEach-Object { Write-Output $_.Output }
        }
    } finally { Pop-Location }
}
Push-Location (Join-Path $guiRoot 'supervisor')
try {
    $output = @(& go generate ./... 2>&1)
    Save-Lines 'generate.log' $output
    if ($LASTEXITCODE -ne 0) { $failures.Add('go generate'); $output | Write-Output }
} finally { Pop-Location }
Push-Location $PSScriptRoot
try {
    $output = @(& go run ./scenarios 2>&1)
    Save-Lines 'scenarios.log' $output
    $output | Write-Output
    if ($LASTEXITCODE -ne 0) { $failures.Add('scenario generator') }
} finally { Pop-Location }
Push-Location (Join-Path $guiRoot 'wails')
try {
    $wails = Join-Path (go env GOPATH) 'bin/wails.exe'
    $output = @(& $wails generate module -nocolour 2>&1)
    Save-Lines 'bindings.log' $output
    if ($LASTEXITCODE -ne 0) { $failures.Add('Wails generation'); $output | Write-Output }
} finally { Pop-Location }
$after = Generated-Hashes
$changed = @($after.Keys | Where-Object { $before[$_] -ne $after[$_] })
if ($before.Count -ne $after.Count -or $changed.Count) { $failures.Add('generation drift'); $changed | Write-Output }
Write-Output "generation: files=$($after.Count), changed=$($changed.Count), removed=$(@($before.Keys | Where-Object { -not $after.ContainsKey($_) }).Count)"
Push-Location $repoRoot
try {
    foreach ($script in @('schema_test.ps1','docs_schema_test.ps1')) {
        $output = @(& pwsh -NoProfile -File "tests/cli/contract/$script" 2>&1)
        Save-Lines "$script.log" $output
        $output | Write-Output
        if ($LASTEXITCODE -ne 0) { $failures.Add($script) }
    }
    $output = @(& python tests/cli/contract/check_docs.py --self-test 2>&1)
    Save-Lines 'check-docs.log' $output
    $output | Write-Output
    if ($LASTEXITCODE -ne 0) { $failures.Add('check_docs.py') }
} finally { Pop-Location }
Write-Output "C2 proof: failed checks=$($failures.Count)"
if ($failures.Count) { throw "Failed C2 proof: $($failures -join ', ')" }
