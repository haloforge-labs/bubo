param([string]$Go = 'go')

$ErrorActionPreference = 'Stop'
$savedCGO = $env:CGO_ENABLED
$scratch = $null
Push-Location $PSScriptRoot
try {
    $env:CGO_ENABLED = '0'
    & $Go vet ./...
    if ($LASTEXITCODE -ne 0) { throw 'go vet failed' }
    $packages = @(& $Go list ./...)
    if ($LASTEXITCODE -ne 0) { throw 'go list failed' }
    Write-Output "go vet: PASS; packages=$($packages.Count)"

    $modules = @(& $Go list -m all)
    if ($LASTEXITCODE -ne 0) { throw 'go list -m failed' }
    $unexpected = @($modules | Where-Object { $_ -ne 'bubo/gui/updater' -and $_ -notmatch '^golang\.org/x/sys v' })
    if ($modules.Count -ne 2 -or $unexpected.Count -ne 0) { throw "Unexpected modules: $unexpected" }
    Write-Output 'dependency guard: PASS; external modules=1 (golang.org/x/sys)'

    $lines = @(& $Go test -count=1 -json ./...)
    $testExit = $LASTEXITCODE
    $events = @($lines | ForEach-Object { $_ | ConvertFrom-Json })
    $passed = @($events | Where-Object { $_.Action -eq 'pass' -and $_.Test })
    $top = @($passed | Where-Object { $_.Test -notmatch '/' })
    $failed = @($events | Where-Object { $_.Action -eq 'fail' -and $_.Test })
    $skipped = @($events | Where-Object { $_.Action -eq 'skip' -and $_.Test })
    $events | Where-Object { $_.Output -match '^helper proof:' } | ForEach-Object { Write-Output $_.Output.TrimEnd() }
    $top | ForEach-Object { Write-Output "PASS $($_.Test)" }
    if ($testExit -ne 0) {
        $events | Where-Object { $_.Output } | ForEach-Object { Write-Output $_.Output.TrimEnd() }
        throw 'go test failed'
    }
    $required = @(
        'TestARealUpdateSucceedsHealthPassesShippedHelperOutsideInstall',
        'TestBNewGUIFailsHealthRollbackOldFolderByteIdentical',
        'TestCLockedFileRetriesThenRollbackOldFolderIntact',
        'TestDSHAMismatchNothingInstalledTypedError',
        'TestESpaceCheckCountsExistingPartialFiles',
        'TestFResumeAfterInterruptedDownload',
        'TestGUnicodeAndSpaceInstallPath',
        'TestHManifestFromNonAllowlistedHostRefused'
    )
    foreach ($name in $required) {
        if ($name -notin $top.Test) { throw "Required case did not pass: $name" }
    }
    if ($skipped.Count -ne 0) { throw 'Updater proof may not skip tests' }
    Write-Output "go test: PASS; required=8/8; top-level=$($top.Count); subtests=$($passed.Count - $top.Count); failed=$($failed.Count); skipped=$($skipped.Count)"

    $tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    $scratch = Join-Path $tempRoot ('bubo-updater-build-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $scratch | Out-Null
    $helper = Join-Path $scratch 'bubo-update.exe'
    & $Go build '-ldflags=-H=windowsgui' -o $helper ./cmd/bubo-update
    if ($LASTEXITCODE -ne 0) { throw 'hidden helper build failed' }
    Write-Output "hidden helper build: PASS; bytes=$((Get-Item -LiteralPath $helper).Length); CGO_ENABLED=0"
}
finally {
    if ($scratch -and (Test-Path -LiteralPath $scratch)) {
        $resolved = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $scratch).Path)
        if ([System.IO.Path]::GetDirectoryName($resolved) -ne $tempRoot.TrimEnd('\') -or [System.IO.Path]::GetFileName($resolved) -notmatch '^bubo-updater-build-[a-f0-9]{32}$') {
            throw "Refusing cleanup outside the verified proof directory: $resolved"
        }
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
    $env:CGO_ENABLED = $savedCGO
    Pop-Location
}
