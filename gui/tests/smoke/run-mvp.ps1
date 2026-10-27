# The user owns GPU/build locks and production stop/restore. This entry
# never starts a window, stops production, scans a port table or loads a model
# in -Fake mode. Real mode must be queued after plan review.
[CmdletBinding()]
param(
    [string]$Package,
    [string]$RuntimePackage,
    [string]$Manifest,
    [Parameter(Mandatory)][string]$Out,
    [switch]$Fake,
    [ValidateRange(1,65535)][int]$Port = 18120,
    [string]$AllowLockOwner,
    [string]$StopFile,
    # Real mode only: run step 1 (facts, cold status, CLI estimate; no model
    # load, no GPU work) and the evidence step, without the user locks.
    [switch]$Step1Only
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
# Obtain the running date before generating timestamps or timestamped paths.
$runDate = Get-Date
if ($Port -ge 8081 -and $Port -le 8086) { throw 'Ports 8081-8086 are forbidden; no port was touched.' }
if (-not [IO.Path]::IsPathFullyQualified($Out)) { throw '-Out must be absolute.' }
$Out = [IO.Path]::GetFullPath($Out)
if (Test-Path -LiteralPath $Out) { throw '-Out must be a new folder for retained evidence.' }
$testsRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$repoRoot = [IO.Path]::GetFullPath((Join-Path $testsRoot '../..'))
if ($Fake -and $Step1Only) { throw '-Step1Only is a real-CLI check; -Fake rehearses every step.' }
if ($Fake -and ($Package -or $RuntimePackage -or $Manifest)) { throw '-Fake supplies its own executable/manifest and cannot use a real package.' }
if ($Package -and $RuntimePackage) { throw '-Package and -RuntimePackage are alternatives; supply only one.' }
$runtimeProvenance = $null
$qualification = $null
if (-not $Fake) {
    if ($RuntimePackage) {
        if (-not [IO.Path]::IsPathFullyQualified($RuntimePackage)) { throw '-RuntimePackage must name an absolute runtime folder.' }
    } elseif (-not $Package -or -not [IO.Path]::IsPathFullyQualified($Package)) { throw '-Package must name the assembled K-b package folder, or supply -RuntimePackage.' }
    if (-not $Manifest -or -not [IO.Path]::IsPathFullyQualified($Manifest)) { throw '-Manifest must name the existing prepared model manifest.' }
    if (-not $AllowLockOwner -and -not $Step1Only) { throw '-AllowLockOwner must name the exact user-held GPU/build lock owner line.' }
    $Manifest = (Resolve-Path -LiteralPath $Manifest).Path
    $packageRoot = if ($RuntimePackage) {
        $RuntimePackage = (Resolve-Path -LiteralPath $RuntimePackage).Path
        $RuntimePackage
    } else {
        $Package = (Resolve-Path -LiteralPath $Package).Path
        $Package
    }
    $cli = Join-Path $packageRoot 'bin/bubo.exe'
    $buildManifest = Join-Path $packageRoot 'BUILD-MANIFEST.json'
    if (-not (Test-Path -LiteralPath $cli -PathType Leaf)) { throw "Missing package CLI: $cli" }
    if (-not (Test-Path -LiteralPath $buildManifest -PathType Leaf)) { throw "Missing package provenance: $buildManifest" }
    $metadata = Get-Content -LiteralPath $buildManifest -Raw | ConvertFrom-Json
    if ($RuntimePackage) {
        # This is the narrower pre-package gate, not a second release-package
        # checker: the runtime need not contain a GUI or updater helper.
        # Integration runtimes (C:\bubo\runtimes\bubo-int-*) copy a base runtime and
        # replace only the CLI and engine. Their MANIFEST-int.txt names the source
        # commit and SHA256 of those two files; BUILD-MANIFEST.json still describes
        # the base, so the int file is the provenance whenever it is present.
        $intManifest = Join-Path $RuntimePackage 'MANIFEST-int.txt'
        $expected = @{}
        if (Test-Path -LiteralPath $intManifest -PathType Leaf) {
            $provenanceFile = $intManifest
            $intText = Get-Content -LiteralPath $intManifest -Raw
            $header = [regex]::Match($intText, '(?m)^(\S+) = .+? built from .+? \(([0-9a-fA-F]{40}) ')
            if (-not $header.Success -or $header.Groups[1].Value -ne (Split-Path -Leaf $RuntimePackage)) { throw 'Runtime MANIFEST-int.txt must name this folder and a full source commit.' }
            $sourceCommit = $header.Groups[2].Value.ToLowerInvariant()
            foreach ($relative in @('bin/bubo.exe','bin/bubo-engine.dll')) {
                $hits = [regex]::Matches($intText, "(?m)^$([regex]::Escape((Split-Path -Leaf $relative))) sha256 ([0-9a-fA-F]{64})\s*$")
                if ($hits.Count -ne 1) { throw "Runtime MANIFEST-int.txt must contain exactly one valid SHA256 for $relative" }
                $expected[$relative] = $hits[0].Groups[1].Value.ToLowerInvariant()
            }
        } else {
            $provenanceFile = $buildManifest
            if ($metadata.source.commit -notmatch '^[0-9a-fA-F]{40}$') { throw 'Runtime BUILD-MANIFEST must declare a full source commit.' }
            $sourceCommit = $metadata.source.commit
            foreach ($relative in @('bin/bubo.exe','bin/bubo-engine.dll')) {
                $rows = @($metadata.files | Where-Object { $_.path.Replace('\','/') -eq $relative })
                if ($rows.Count -ne 1 -or $rows[0].sha256 -notmatch '^[0-9a-fA-F]{64}$') { throw "Runtime BUILD-MANIFEST must contain exactly one valid SHA256 for $relative" }
                $expected[$relative] = $rows[0].sha256.ToLowerInvariant()
            }
        }
        $runtimeProvenance = [ordered]@{ folder=$RuntimePackage; source_commit=$sourceCommit; provenance_file=(Split-Path -Leaf $provenanceFile); provenance_sha256=(Get-FileHash -LiteralPath $provenanceFile -Algorithm SHA256).Hash.ToLowerInvariant(); build_manifest_sha256=(Get-FileHash -LiteralPath $buildManifest -Algorithm SHA256).Hash.ToLowerInvariant(); files=@() }
        foreach ($relative in @('bin/bubo.exe','bin/bubo-engine.dll')) {
            $binary = Join-Path $RuntimePackage $relative
            if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) { throw "Missing runtime binary: $binary" }
            $actual = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -ne $expected[$relative]) { throw "Runtime package SHA256 mismatch for ${relative}: $($runtimeProvenance.provenance_file)=$($expected[$relative]) actual=$actual; no CLI was started." }
            $runtimeProvenance.files += [ordered]@{ path=$relative; sha256=$actual }
        }
        $qualification = 'pre-package smoke (CLI + supervisor on the GPU runtime; GUI exe not involved)'
        Write-Output $qualification
    } else {
        if ($null -ne $metadata.PSObject.Properties['qualification']) { throw "Unqualified package: $($metadata.qualification); structure-only Ka output is refused." }
        $checkoutCommit = (& git -C $repoRoot rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or $metadata.gui.commit -ne $checkoutCommit) { throw 'Run from the final checkout used to build the K-b GUI; supervisor/View source must match the package.' }
        $sourceChanges = @(& git -C $repoRoot status --porcelain --untracked-files=all -- gui/supervisor gui/tests/smoke gui/tests/go.mod gui/tests/go.sum)
        if ($LASTEXITCODE -ne 0 -or $sourceChanges.Count -ne 0) { throw 'Commit the supervisor/View/driver sources before a real run; uncommitted source cannot qualify the packaged build.' }
        # The established checker owns package layout, hash and PE rules. Do not
        # make another package validator/fact list inside this smoke entry.
        $checker = Join-Path $PSScriptRoot '../../../tools/release/check-package.ps1'
        if (-not (Get-Command $checker).Parameters.ContainsKey('RequireGUI')) { throw 'Ka package validator is not integrated in this checkout; merge Ka/D3 before the real run.' }
        & $checker -PackageFolder $Package -RequireGUI -Go 'go'
        if ($LASTEXITCODE -ne 0) { throw 'K-b package validation failed.' }
    }
}
$toolsDir = Join-Path $Out 'tools'
$resultDir = Join-Path $Out 'result'
$null = New-Item -ItemType Directory -Path $toolsDir -Force
$driver = Join-Path $toolsDir 'bubo-smoke.exe'
$oldEnv = @{}
foreach ($name in @('Path','CGO_ENABLED','GOFLAGS','BUBO_FAKE_DIR','BUBO_FAKE_MODE','BUBO_FAKE_FIXTURES','BUBO_FAKE_HTTP','BUBO_FAKE_WATCH_MS','BUBO_FAKE_MVP_RECORDING','BUBO_FAKE_MVP_FAULT','BUBO_FAKE_REQUEST_MS')) { $oldEnv[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
foreach ($existingFake in @(Get-ChildItem Env: | Where-Object { $_.Name.StartsWith('BUBO_FAKE_',[StringComparison]::OrdinalIgnoreCase) })) { $oldEnv[$existingFake.Name] = $existingFake.Value }
$self = [Diagnostics.Process]::GetCurrentProcess()
$oldPriority = $self.PriorityClass
$child = $null
$exitCode = 1
try {
    $self.PriorityClass = 'BelowNormal'
    $env:CGO_ENABLED = '0'
    $env:GOFLAGS = '-mod=readonly'
    Push-Location $testsRoot
    try {
        & go build -o $driver ./smoke
        if ($LASTEXITCODE -ne 0) { throw 'Go smoke build failed.' }
        if ($Fake) {
            $cli = Join-Path $toolsDir 'bubo-fake.exe'
            & go build -o $cli ./fakecli
            if ($LASTEXITCODE -ne 0) { throw 'Go fake CLI build failed.' }
        }
    } finally { Pop-Location }
    $arguments = @('--mvp','--exe',$cli,'--out',$resultDir,'--port',"$Port",'--load-timeout','900s','--poll-interval','250ms')
    if (-not $StopFile) { $StopFile = Join-Path $Out 'STOP' }
    $arguments += @('--stop-file',$StopFile)
    if ($Fake) {
        foreach ($name in $oldEnv.Keys) { if ($name.StartsWith('BUBO_FAKE_',[StringComparison]::OrdinalIgnoreCase)) { [Environment]::SetEnvironmentVariable($name,$null,'Process') } }
        $env:BUBO_FAKE_DIR = Join-Path $Out 'fake-state'
        $null = New-Item -ItemType Directory -Path $env:BUBO_FAKE_DIR
        $Manifest = Join-Path $env:BUBO_FAKE_DIR 'bubo-package.json'
        '{}' | Set-Content -LiteralPath $Manifest -Encoding utf8NoBOM
        $env:BUBO_FAKE_MODE = 'normal'
        $env:BUBO_FAKE_FIXTURES = Join-Path $repoRoot 'tests/cli/contract'
        $env:BUBO_FAKE_HTTP = 'mvp'
        $env:BUBO_FAKE_WATCH_MS = '500'
        $env:BUBO_FAKE_REQUEST_MS = '2800'
        $env:BUBO_FAKE_MVP_FAULT = $null
        $env:BUBO_FAKE_MVP_RECORDING = Join-Path $testsRoot 'scenarios/recordings/b2/owner-budget-status.jsonl'
        $arguments += @('--fake','--keep-env-prefix','BUBO_FAKE_','--proc-patterns','mvp-rehearsal-none.exe','--lock-path',(Join-Path $env:BUBO_FAKE_DIR 'gpu.lock'),'--build-lock-path',(Join-Path $env:BUBO_FAKE_DIR 'build.lock'),'--load-timeout','20s','--request-timeout','15s','--stop-timeout','5s','--query-timeout','10s')
    } elseif ($Step1Only) {
        $arguments += @('--package',$packageRoot,'--mvp-step1-only')
    } else {
        # The driver's package argument checks only the CLI's location within
        # its root. Release qualification belongs to the entry's selected gate.
        $arguments += @('--package',$packageRoot,'--require-locks','--allow-lock-owner',$AllowLockOwner)
    }
    $arguments += @('--manifest',$Manifest)
    $entry = [ordered]@{ started_at=$runDate.ToString('o'); script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash; driver_sha256=(Get-FileHash -LiteralPath $driver -Algorithm SHA256).Hash; revision=(& git -C $repoRoot rev-parse HEAD); fake=[bool]$Fake; package=$Package; cli=$cli; port=$Port; stop_file=$StopFile; arguments=$arguments }
    if ($RuntimePackage) { $entry.qualification = $qualification; $entry.runtime_package = $runtimeProvenance }
    $entry | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Out 'entry.json') -Encoding utf8NoBOM
    # ProcessStartInfo.ArgumentList uses .NET's Windows argument quoting.
    $start = [Diagnostics.ProcessStartInfo]::new($driver)
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.WindowStyle = 'Hidden'
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($argument in $arguments) { $start.ArgumentList.Add($argument) }
    # Windows passes BelowNormal on to children. A real run starts the driver,
    # and through it the CLI server, at Normal as the GUI does in production;
    # only the builds and fake rehearsals stay BelowNormal.
    if (-not $Fake) { $self.PriorityClass = 'Normal' }
    $child = [Diagnostics.Process]::Start($start)
    @{pid=$child.Id; process_creation_time=$child.StartTime.ToUniversalTime().ToFileTimeUtc().ToString()} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'driver-pid.json') -Encoding utf8NoBOM
    $stdout = $child.StandardOutput.ReadToEndAsync()
    $stderr = $child.StandardError.ReadToEndAsync()
    while (-not $child.WaitForExit(1000)) { }
    $stdout.Result | Set-Content -LiteralPath (Join-Path $Out 'driver.stdout.txt') -Encoding utf8NoBOM
    $stderr.Result | Set-Content -LiteralPath (Join-Path $Out 'driver.stderr.txt') -Encoding utf8NoBOM
    Write-Output $stdout.Result
    if ($stderr.Result) { Write-Output $stderr.Result }
    $exitCode = $child.ExitCode
    $report = Get-Content -LiteralPath (Join-Path $resultDir 'report.json') -Raw | ConvertFrom-Json
    if ($RuntimePackage) {
        # The entry owns verified input provenance; retain it in the final
        # result too, including a driver refusal. Never alter its verdict.
        $report | Add-Member -NotePropertyName qualification -NotePropertyValue $qualification
        $report | Add-Member -NotePropertyName runtime_package -NotePropertyValue $runtimeProvenance
        $report | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath (Join-Path $resultDir 'report.json') -Encoding utf8NoBOM
        Write-Output $qualification
    }
    $steps = @($report.steps | Where-Object { $null -ne $_ })
    $checks = @($steps | Where-Object { $null -ne $_.PSObject.Properties['checks'] } | ForEach-Object { $_.checks })
    $failedChecks = @($checks | Where-Object { -not $_.ok })
    Write-Output ("COUNTS: steps={0} checks={1} passed={2} failed={3}; verdict={4}; script_sha256={5}" -f $steps.Count,$checks.Count,($checks.Count-$failedChecks.Count),$failedChecks.Count,$report.overall,$entry.script_sha256)
    if ($exitCode -eq 0 -and ($steps.Count -ne 6 -or $failedChecks.Count -ne 0)) { throw 'Driver result/count disagreement.' }
} finally {
    if ($null -ne $child -and -not $child.HasExited) {
        # Ask the owned driver to use normal supervisor shutdown. The user
        # must inspect driver-pid.json if it does not honor the stop request.
        [IO.File]::WriteAllText($StopFile,'entry interrupted')
        if (-not $child.WaitForExit(60000)) { Write-Warning "Owned driver PID $($child.Id) is still running; evidence and locks must be retained by the user." }
    }
    if ($null -eq $child -or $child.HasExited) {
        # Delete exact task-owned build outputs only; retain all run evidence.
        foreach ($ownedFile in @($driver, (Join-Path $toolsDir 'bubo-fake.exe'))) {
            if (Test-Path -LiteralPath $ownedFile -PathType Leaf) { Remove-Item -LiteralPath $ownedFile -Force }
        }
        if ((Test-Path -LiteralPath $toolsDir) -and @(Get-ChildItem -LiteralPath $toolsDir -Force).Count -eq 0) { Remove-Item -LiteralPath $toolsDir }
    }
    foreach ($name in $oldEnv.Keys) { [Environment]::SetEnvironmentVariable($name,$oldEnv[$name],'Process') }
    $self.PriorityClass = $oldPriority
    if ($null -ne $child) { $child.Dispose() }
}
exit $exitCode
