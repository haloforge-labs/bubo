param([string]$ScratchDirectory = "$PSScriptRoot\..\..\build\pe-import-tests")
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\pe-imports.ps1"
if (Test-Path -LiteralPath $ScratchDirectory) { throw "Test directory already exists: $ScratchDirectory" }
$scratch = [IO.Path]::GetFullPath($ScratchDirectory)
New-Item -ItemType Directory -Path $scratch | Out-Null
function New-PeFixture([string]$File, [string]$Import, [string]$Delay, [bool]$Pe32 = $false) {
  $b = [byte[]]::new(0x600)
  function U16($p, [uint16]$v) { [BitConverter]::GetBytes($v).CopyTo($b, $p) }
  function U32($p, [uint32]$v) { [BitConverter]::GetBytes($v).CopyTo($b, $p) }
  U16 0 0x5a4d; U32 0x3c 0x80; U32 0x80 0x4550; U16 0x86 1
  $optionalSize = if ($Pe32) { 224 } else { 240 }
  U16 0x94 $optionalSize
  $optional = 0x98
  U16 $optional $(if ($Pe32) { 0x10b } else { 0x20b })
  U32 ($optional + 60) 0x200
  $dir = $optional + $(if ($Pe32) { 96 } else { 112 })
  U32 ($dir - 4) 16
  $section = $optional + $optionalSize
  U32 ($section + 8) 0x400; U32 ($section + 12) 0x1000
  U32 ($section + 16) 0x400; U32 ($section + 20) 0x200
  if ($Import) {
    U32 ($dir + 8) 0x1000; U32 ($dir + 12) 40
    U32 0x20c 0x1100; U32 0x210 0x1180
    [Text.Encoding]::ASCII.GetBytes($Import).CopyTo($b, 0x300)
  }
  if ($Delay) {
    U32 ($dir + 104) 0x1040; U32 ($dir + 108) 64
    U32 0x240 1; U32 0x244 0x1140
    [Text.Encoding]::ASCII.GetBytes($Delay).CopyTo($b, 0x340)
  }
  [IO.File]::WriteAllBytes($File, $b)
}
try {
  $root = Join-Path $scratch 'root.exe'
  $child = Join-Path $scratch 'child.dll'
  New-PeFixture $root 'child.dll' 'api-ms-win-core-test-l1-1-0.dll'
  New-PeFixture $child 'root.exe' '' $true
  $closure = Get-PeClosure $root
  if ($closure.Nodes.Count -ne 2 -or $closure.Edges.Count -ne 3 -or
      @($closure.Edges | Where-Object Kind -EQ delay).Count -ne 1 -or
      @($closure.Edges | Where-Object Resolution -EQ 'api-set').Count -ne 1) {
    throw 'PE32/PE32+, recursive cycle or delay import regression'
  }
  & pwsh -NoProfile -File "$PSScriptRoot\check-package.ps1" -PackageFolder $scratch
  if ($LASTEXITCODE -ne 0) { throw 'Valid fixture scan failed' }
  $deny = Get-Content -LiteralPath "$PSScriptRoot\nonredist-imports.json" -Raw | ConvertFrom-Json
  foreach ($name in @($deny | ForEach-Object { $_.Replace('*', 'test.dll') })) {
    New-PeFixture $root 'child.dll' $name
    & pwsh -NoProfile -File "$PSScriptRoot\check-package.ps1" -PackageFolder $scratch
    if ($LASTEXITCODE -eq 0) { throw "Deny-list accepted $name delay import" }
  }
  New-PeFixture $root 'libomp140.x86_64.dll' ''
  & pwsh -NoProfile -File "$PSScriptRoot\check-package.ps1" -PackageFolder $scratch
  if ($LASTEXITCODE -eq 0) { throw 'Deny-list accepted normal import' }
  [IO.File]::WriteAllBytes($root, [byte[]]@(0x4d, 0x5a))
  $rejected = $false
  try { Get-PeImports $root | Out-Null } catch { $rejected = $true }
  if (-not $rejected) { throw 'Truncated PE accepted' }
  New-PeFixture $root 'child.dll' ''
  $bad = [IO.File]::ReadAllBytes($root)
  [BitConverter]::GetBytes([uint32]0x800000).CopyTo($bad, 0x20c)
  [IO.File]::WriteAllBytes($root, $bad)
  $rejected = $false
  try { Get-PeImports $root | Out-Null } catch { $rejected = $true }
  if (-not $rejected) { throw 'Unmapped name RVA accepted' }
  Write-Output 'PASS: PE32/PE32+, normal/delay imports, cycle, API-set, deny-list controls, truncated image'
} finally {
  # Only remove the directory this invocation created, at its verified path.
  if ([IO.Path]::GetFullPath($ScratchDirectory) -ne $scratch) { throw 'Scratch path changed' }
  Remove-Item -LiteralPath $scratch -Recurse -Force
}
