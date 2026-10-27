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
$bin = Join-Path $scratch 'bin'
[IO.Directory]::CreateDirectory($windows) | Out-Null
[IO.Directory]::CreateDirectory($bin) | Out-Null
$powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
try {
  # The package's start.cmd: the VC++ check, then the app if it is there,
  # else bin\bubo.exe serve with the same arguments.
  Copy-Item -LiteralPath (Join-Path $Root 'start.cmd') -Destination $scratch
  # Mock executables write their arguments to <name>.ran next to themselves
  # and exit 7. The app mock is a window program, so no console opens.
  $mock = 'public static class P { public static int Main(string[] a) { System.IO.File.WriteAllText(System.IO.Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, System.IO.Path.GetFileNameWithoutExtension(System.Environment.GetCommandLineArgs()[0]) + ".ran"), string.Join("|", a)); return 7; } }'
  foreach ($target in @(@((Join-Path $bin 'bubo.exe'), 'ConsoleApplication'),
                        @((Join-Path $scratch 'app-mock\Bubo.Control.Web.exe'), 'WindowsApplication'))) {
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target[0])) | Out-Null
    & $powershell -NoProfile -Command "Add-Type -TypeDefinition '$mock' -OutputAssembly '$($target[0])' -OutputType $($target[1])"
    if ($LASTEXITCODE -or -not (Test-Path -LiteralPath $target[0])) { throw "mock executable $($target[0])" }
  }
  $ran = Join-Path $bin 'bubo.ran'
  $appRan = Join-Path $bin 'Bubo.Control.Web.ran'
  $run = {
    Push-Location -LiteralPath $scratch
    # .\ because NoDefaultCurrentDirectoryInExePath stops cmd searching here.
    try { $script:output = '' | & cmd.exe /d /c .\start.cmd --port 1 'a b' 2>&1; $script:rc = $LASTEXITCODE }
    finally { Pop-Location }
    Write-Output "start.cmd rc=$script:rc"
    $script:output | Write-Output
  }
  # 1. VC++ missing: the link, exit 4, and nothing starts.
  $mockRuntime = [IO.File]::ReadAllText($Runtime) + "`nfunction Test-BuboVcRuntime { return `$false }`n"
  [IO.File]::WriteAllText((Join-Path $windows 'vc-runtime.ps1'), $mockRuntime)
  & $run
  if ($rc -ne 4 -or "$output" -notmatch 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -or
      (Test-Path -LiteralPath $ran)) { throw 'start.cmd missing-runtime refusal' }
  # 2. VC++ present, no app: bin\bubo.exe serve with the same arguments, and
  #    its exit code.
  [IO.File]::WriteAllText((Join-Path $windows 'vc-runtime.ps1'), $mockRuntime.Replace('{ return $false }', '{ return $true }'))
  & $run
  if ($rc -ne 7 -or -not (Test-Path -LiteralPath $ran) -or
      [IO.File]::ReadAllText($ran) -ne 'serve|--port|1|a b') { throw 'start.cmd without the app must run bubo serve with the arguments' }
  Remove-Item -LiteralPath $ran
  # 3. VC++ present, the app is there: the app starts detached, start.cmd
  #    exits 0 and bubo.exe does not run.
  Copy-Item -LiteralPath (Join-Path $scratch 'app-mock\Bubo.Control.Web.exe') -Destination $bin
  & $run
  $deadline = (Get-Date).AddSeconds(15)
  while (-not (Test-Path -LiteralPath $appRan) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
  if ($rc -ne 0 -or -not (Test-Path -LiteralPath $appRan) -or (Test-Path -LiteralPath $ran)) { throw 'start.cmd must open the app when it is there' }
  # The package no longer carries the old console launcher.
  $package = [IO.File]::ReadAllText((Join-Path $Root 'tools\windows\package.ps1'))
  if ($package -match 'launcher\.ps1|inventory\.ps1') { throw 'package.ps1 still ships launcher.ps1 or inventory.ps1' }
  # start.ps1 (repository tool, not in the package) keeps its own check.
  foreach ($launcher in @('launcher.ps1', 'start.ps1')) {
    Copy-Item -LiteralPath (Join-Path $Root "tools\windows\$launcher") -Destination $windows
  }
  [IO.File]::WriteAllText((Join-Path $windows 'vc-runtime.ps1'), $mockRuntime)
  Set-Content -LiteralPath (Join-Path $windows 'inventory.ps1') -Value "throw 'launcher reached inventory without prerequisite'"
  $output = & $powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $windows 'start.ps1') -DryRun 2>&1
  $rc = $LASTEXITCODE
  Write-Output "start.ps1 mock rc=$rc"
  $output | Write-Output
  if ($rc -ne 4 -or "$output" -notmatch 'https://aka.ms/vs/17/release/vc_redist.x64.exe') { throw 'start.ps1 missing-runtime refusal' }
  Write-Output 'PASS: start.cmd checks VC++ first, then opens the app or runs bubo serve; start.ps1 checks VC++'
} finally { Remove-Item -LiteralPath $scratch -Recurse -Force }
