param(
    [Parameter(Mandatory=$true)][string]$EvidenceDir,
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$env:PATH = 'C:\bubo\tools\go-1.27.2\go\bin;' + $env:PATH
$env:CGO_ENABLED = '0'
[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'BelowNormal'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$evidence = [System.IO.Path]::GetFullPath($EvidenceDir)
[System.IO.Directory]::CreateDirectory($evidence) | Out-Null
function Save-LF([string]$Path, [object[]]$Lines) {
    [System.IO.File]::WriteAllText($Path, (($Lines | ForEach-Object { [string]$_ }) -join "`n") + "`n", $utf8)
}
Push-Location $PSScriptRoot
try {
    $vet = @(& go vet -tags 'production,wv2runtime.error' ./... 2>&1)
    $vetRC = $LASTEXITCODE
    Save-LF (Join-Path $evidence 'vet.txt') $vet
    if ($vetRC -ne 0) { throw "go vet failed ($vetRC): $vet" }
    $tests = @(& go test -tags 'production,wv2runtime.error' -json -count=1 -timeout 60s ./... 2>&1)
    $testRC = $LASTEXITCODE
    Save-LF (Join-Path $evidence 'tests.jsonl') $tests
    $events = @($tests | Where-Object { ([string]$_).StartsWith('{') } | ForEach-Object { ([string]$_) | ConvertFrom-Json })
    $pass = @($events | Where-Object { $_.Action -eq 'pass' -and $_.Test }).Count
    $fail = @($events | Where-Object { $_.Action -eq 'fail' -and $_.Test }).Count
    $skip = @($events | Where-Object { $_.Action -eq 'skip' -and $_.Test }).Count
    $counts = "wails vet: 1 passed, 0 failed; tests: $pass passed, $fail failed, $skip skipped; CGO_ENABLED=0; no real GUI launched"
    Save-LF (Join-Path $evidence 'counts.txt') @($counts)
    Write-Output $counts
    if ($testRC -ne 0 -or $fail -ne 0) { throw "go test failed ($testRC)" }
    if (!$SkipBuild) {
        & (Join-Path $PSScriptRoot 'build.ps1') -OutputDir $evidence
        if ($LASTEXITCODE -ne 0) { throw 'Production build failed' }
    }
} finally { Pop-Location }
