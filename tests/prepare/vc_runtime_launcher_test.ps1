param([string]$Runtime = "$PSScriptRoot\..\..\tools\windows\vc-runtime.ps1",
      [string]$Root = "$PSScriptRoot\..\..", [string]$Case = 'all')
$ErrorActionPreference = 'Stop'
[Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'BelowNormal'
. $Runtime
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
  Write-Output 'PASS: start.cmd and start.ps1 use prerequisite detector before inventory/engine'
} finally { Remove-Item -LiteralPath $scratch -Recurse -Force }
