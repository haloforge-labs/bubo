param([string]$Runtime = "$PSScriptRoot\..\..\tools\windows\vc-runtime.ps1",
      [string]$Root = "$PSScriptRoot\..\..", [string]$Case = 'all',
      [string]$NativeSource = "$PSScriptRoot\..\..\src\cli\prepare\vc_runtime.cpp")
$ErrorActionPreference = 'Stop'
[Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'BelowNormal'
. $Runtime
if ($Case -in @('all', 'native', 'registry-view')) {
  $parameters = (Get-Command Test-BuboVcRuntime).ScriptBlock.Ast.Body.ParamBlock.Parameters
  $registryDefault = $parameters[0].DefaultValue.ScriptBlock.GetScriptBlock()
  $fileDefault = $parameters[1].DefaultValue.ScriptBlock.GetScriptBlock()
  if ($Case -in @('all', 'native')) {
    # Invoke each actual default independently, even if registry short-circuits.
    $result = & $registryDefault
    if ($result -isnot [bool]) { throw 'native registry probe did not return bool' }
    Write-Output "native PowerShell registry64 probe bool=$result"
    foreach ($name in @('vcruntime140.dll', 'msvcp140.dll')) {
      $result = & $fileDefault $name
      if ($result -isnot [bool]) { throw 'native System32 probe did not return bool' }
      Write-Output "native PowerShell System32 $name probe bool=$result"
    }
  }
  $source = [IO.File]::ReadAllText($NativeSource)
  if (!$source.Contains('RRF_RT_REG_DWORD | RRF_SUBKEY_WOW6464KEY')) { throw 'C++ explicit registry64 view missing' }
  # Run the actual default registry script against an in-memory registry API.
  # This observes view, hive, path, default value and disposal without writes.
  function Open-FakeBase($Hive, $View) {
    if ($Hive -ne [Microsoft.Win32.RegistryHive]::LocalMachine -or
        $View -ne [Microsoft.Win32.RegistryView]::Registry64) { throw 'wrong registry view/hive' }
    $script:viewObserved = $true
    $base = [pscustomobject]@{}
    $base | Add-Member ScriptMethod OpenSubKey {
      param($Path, $Writable)
      if ($Path -ne 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -or $Writable) { throw 'wrong key/access' }
      if ($script:registryState -eq 'absent') { return $null }
      $key = [pscustomobject]@{}
      $key | Add-Member ScriptMethod GetValue {
        param($Name, $Default)
        if ($Name -ne 'Installed' -or $Default -ne 0) { throw 'wrong value/default' }
        if ($script:registryState -eq 'missing-value') { return $Default }
        if ($script:registryState -eq 'installed') { return 1 }
        return 0
      }
      $key | Add-Member ScriptMethod Dispose { $script:keyDisposed = $true }
      return $key
    }
    $base | Add-Member ScriptMethod Dispose { $script:baseDisposed = $true }
    return $base
  }
  class BuboFakeRegistry {
    static [object] OpenBaseKey([Microsoft.Win32.RegistryHive]$Hive, [Microsoft.Win32.RegistryView]$View) {
      return (Open-FakeBase $Hive $View)
    }
  }
  $fake = [scriptblock]::Create($registryDefault.ToString().Replace('[Microsoft.Win32.RegistryKey]::OpenBaseKey', '[BuboFakeRegistry]::OpenBaseKey'))
  foreach ($state in @('installed', 'zero', 'absent', 'missing-value')) {
    $script:registryState = $state
    $script:viewObserved = $false; $script:baseDisposed = $false; $script:keyDisposed = $false
    $result = & $fake
    Write-Output "fake registry64 $state bool=$result"
    if (!$script:viewObserved -or !$script:baseDisposed -or
        ($state -ne 'absent' -and !$script:keyDisposed) -or
        $result -isnot [bool] -or $result -ne ($state -eq 'installed')) { throw "registry64/default/disposal regression: $state" }
  }
}
foreach ($kind in @('registry', 'missing', 'files', 'partial')) {
  if ($Case -ne 'all' -and $Case -ne $kind) { continue }
  $registry = { $kind -eq 'registry' }
  $file = { param($name) $kind -eq 'files' -or ($kind -eq 'partial' -and $name -eq 'vcruntime140.dll') }
  $expected = $kind -in @('registry', 'files')
  if ((Test-BuboVcRuntime -RegistryProbe $registry -FileProbe $file) -ne $expected) { throw "detector $kind" }
  # Capture host output independently of the bool return.
  $output = Confirm-BuboVcRuntime -RegistryProbe $registry -FileProbe $file 6>&1
  $result = $output[-1]
  Write-Output "$kind PowerShell detector: $result"
  $output | Write-Output
  if ($result -ne $expected) { throw "confirmation $kind" }
  if (-not $expected -and "$output" -notmatch 'https://aka.ms/vs/17/release/vc_redist.x64.exe') { throw 'missing Microsoft link' }
}
if ($Case -ne 'all' -and $Case -ne 'launchers') { exit 0 }
$scratch = Join-Path (Get-Location) ('crt-start-' + [Guid]::NewGuid().ToString('N'))
$windows = Join-Path $scratch 'tools\windows'
[IO.Directory]::CreateDirectory($windows) | Out-Null
try {
  Copy-Item -LiteralPath (Join-Path $Root 'start.cmd') -Destination $scratch
  # An executable returns to its caller; a batch mock needs CALL to do so.
  # Adapt only the isolated dispatch, leaving the production status line intact.
  $cmdPath = Join-Path $scratch 'start.cmd'
  $cmdSource = [IO.File]::ReadAllText($cmdPath)
  if (!$cmdSource.Contains('powershell -NoProfile')) { throw 'start.cmd dispatch missing' }
  [IO.File]::WriteAllText($cmdPath, $cmdSource.Replace('powershell -NoProfile', 'call powershell -NoProfile'))
  foreach ($launcher in @('launcher.ps1', 'start.ps1')) {
    Copy-Item -LiteralPath (Join-Path $Root "tools\windows\$launcher") -Destination $windows
  }
  # The native registry/file probes are replaced only in this isolated mock.
  $mockRuntime = [IO.File]::ReadAllText($Runtime) + "`nfunction Test-BuboVcRuntime { return `$false }`n"
  [IO.File]::WriteAllText((Join-Path $windows 'vc-runtime.ps1'), $mockRuntime)
  Set-Content -LiteralPath (Join-Path $windows 'inventory.ps1') -Value "throw 'launcher reached inventory without prerequisite'"
  $powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
  # start.cmd resolves this mock command first; it invokes the actual launcher
  # under Windows PowerShell with fake probes and cannot reach an engine.
  Set-Content -LiteralPath (Join-Path $scratch 'powershell.cmd') -Value "@echo off`r`n`"$powershell`" %*`r`nexit /b %errorlevel%"
  $oldPath = $env:PATH
  $env:PATH = "$scratch;$oldPath"
  Push-Location -LiteralPath $scratch
  try {
    $output = & cmd.exe /d /c start.cmd -DryRun 2>&1
    $rc = $LASTEXITCODE
  } finally { Pop-Location; $env:PATH = $oldPath }
  Write-Output "start.cmd mock rc=$rc"
  $output | Write-Output
  if ($rc -ne 4 -or "$output" -notmatch 'https://aka.ms/vs/17/release/vc_redist.x64.exe') { throw 'start.cmd missing-runtime refusal' }
  $output = & $powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $windows 'start.ps1') -DryRun 2>&1
  $rc = $LASTEXITCODE
  Write-Output "start.ps1 mock rc=$rc"
  $output | Write-Output
  if ($rc -ne 4 -or "$output" -notmatch 'https://aka.ms/vs/17/release/vc_redist.x64.exe') { throw 'start.ps1 missing-runtime refusal' }
  # Exercise the real start.cmd forwarding and launcher with a remembered plan.
  # An empty placeholder executable is never invoked under -DryRun.
  [IO.File]::WriteAllText((Join-Path $windows 'vc-runtime.ps1'),
    'function Confirm-BuboVcRuntime { return $true }')
  [IO.File]::WriteAllText((Join-Path $windows 'inventory.ps1'),
    '$GufoRoot = Split-Path (Split-Path $PSScriptRoot); function Get-GufoEnvSwitches { return @() }')
  [IO.File]::WriteAllText((Join-Path $scratch 'bubo.exe'), '')
  $savedLocal = $env:LOCALAPPDATA
  $env:LOCALAPPDATA = Join-Path $scratch 'settings'
  $settingsDirectory = Join-Path $env:LOCALAPPDATA 'gufo'
  [IO.Directory]::CreateDirectory($settingsDirectory) | Out-Null
  $plan = @{ Title = 'Fixture model'; Args = @('serve', '--port', '8080', 'llm', '--model', 'fixture.gguf')
    Env = @{}; Port = 8080; Host = '127.0.0.1'; ServedName = 'fixture' }
  @{ Last = $plan; Features = @{} } | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath (Join-Path $settingsDirectory 'launcher.json') -Encoding UTF8
  $cacheFolder = Join-Path $scratch 'cache folder'
  $env:PATH = "$scratch;$oldPath"
  Push-Location -LiteralPath $scratch
  try {
    $output = & cmd.exe /d /c start.cmd -Last -DryRun -Bin $scratch -CacheDisk $cacheFolder 2>&1
    if ($LASTEXITCODE -ne 0 -or
        "$output" -notmatch ('--cache-disk "' + [regex]::Escape($cacheFolder) + '"') -or
        "$output" -notmatch 'disk cache') { throw 'start.cmd disk cache forwarding/quoting/summary' }
    $trailingFolder = $cacheFolder + '\'
    $output = & cmd.exe /d /c start.cmd -Last -DryRun -Bin $scratch -CacheDisk $trailingFolder 2>&1
    if ($LASTEXITCODE -ne 0 -or
        "$output" -notmatch ('--cache-disk "' + [regex]::Escape($trailingFolder + '\') + '"')) { throw 'trailing separator quoting' }
    $output = & cmd.exe /d /c start.cmd -Last -DryRun -Bin $scratch 2>&1
    if ($LASTEXITCODE -ne 0 -or "$output" -match '--cache-disk') { throw 'disk cache must default off' }
    $output = & cmd.exe /d /c start.cmd -Last -DryRun -Bin $scratch -CacheDisk '   ' 2>&1
    if ($LASTEXITCODE -eq 0 -or "$output" -notmatch 'non-empty valid folder path') { throw 'blank cache folder accepted' }
  } finally {
    Pop-Location
    $env:PATH = $oldPath
    $env:LOCALAPPDATA = $savedLocal
  }
  Write-Output 'PASS: start.cmd and start.ps1 use prerequisite detector before inventory/engine'
} finally { Remove-Item -LiteralPath $scratch -Recurse -Force }
