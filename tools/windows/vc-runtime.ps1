# Read-only VC++ v14 x64 prerequisite check. Probes can be injected by CPU tests.
function Test-BuboVcRuntime {
  param(
    [scriptblock]$RegistryProbe = {
      $base = $null; $key = $null
      try {
        $base = [Microsoft.Win32.RegistryKey]::OpenBaseKey(
          [Microsoft.Win32.RegistryHive]::LocalMachine,
          [Microsoft.Win32.RegistryView]::Registry64)
        $key = $base.OpenSubKey('SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', $false)
        return $key -and $key.GetValue('Installed', 0) -eq 1
      } catch { return $false }
      finally {
        if ($key) { $key.Dispose() }
        if ($base) { $base.Dispose() }
      }
    },
    [scriptblock]$FileProbe = {
      param($Name)
      # Sysnative reaches native System32 even from 32-bit Windows PowerShell.
      $directory = if ([Environment]::Is64BitProcess) { 'System32' } else { 'Sysnative' }
      Test-Path -LiteralPath (Join-Path (Join-Path $env:SystemRoot $directory) $Name) -PathType Leaf
    }
  )
  if ((& $RegistryProbe) -or ((& $FileProbe 'vcruntime140.dll') -and (& $FileProbe 'msvcp140.dll'))) {
    return $true
  }
  return $false
}

function Confirm-BuboVcRuntime {
  param([scriptblock]$RegistryProbe, [scriptblock]$FileProbe)
  $probes = @{}
  if ($RegistryProbe) { $probes.RegistryProbe = $RegistryProbe }
  if ($FileProbe) { $probes.FileProbe = $FileProbe }
  if (Test-BuboVcRuntime @probes) { return $true }
  Write-Host 'Microsoft VC++ v14 x64 Redistributable is required (not shipped). Install it from Microsoft, then run bubo again:'
  Write-Host 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
  return $false
}
