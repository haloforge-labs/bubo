param([string]$Gate = "$PSScriptRoot\..\..\tools\release\check-package.ps1",
      [string]$Case = 'all')
$ErrorActionPreference = 'Stop'
[Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'BelowNormal'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$scratch = Join-Path (Get-Location) ('crt-zip-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($scratch) | Out-Null
try {
  $variants = @('vcruntime140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
    'vcruntime140_1.dll', 'concrt140.dll', 'vccorlib140.dll', 'vcomp140.dll',
    'MSVCP140_ATOMIC_WAIT.dll', 'Microsoft.VC143.CRT/anything.txt',
    'libomp140.x86_64.dll', 'ucrtbase.dll', 'api-ms-win-crt-runtime-l1-1-0.dll',
    'concrete.txt', 'concrt-notes.txt', 'vcompat.md', 'clean')
  foreach ($variant in $variants) {
    if ($Case -ne 'all' -and $Case -ne $variant) { continue }
    $zip = Join-Path $scratch (($variant -replace '[/\\]', '_') + '.zip')
    $archive = [IO.Compression.ZipFile]::Open($zip, 'Create')
    try {
      $entry = $archive.CreateEntry('bubo/bin/' + $(if ($variant -eq 'clean') { 'libcurl.dll' } else { $variant }))
      $stream = $entry.Open()
      try { $stream.WriteByte(0) } finally { $stream.Dispose() }
    } finally { $archive.Dispose() }
    $output = & pwsh -NoProfile -File $Gate -PackageZip $zip 2>&1
    $rc = $LASTEXITCODE
    Write-Output "ZIP $variant rc=$rc"
    $output | Write-Output
    if ($variant -in @('clean', 'concrete.txt', 'concrt-notes.txt', 'vcompat.md')) {
      if ($rc -ne 0 -or "$output" -notmatch 'PASS: release ZIP') { throw 'clean zip must pass' }
    } elseif ($rc -ne 1 -or "$output" -notmatch 'FAIL:.*(prerequisite|denied shipped)') {
      throw "zip containing $variant must fail the shipped-file gate"
    }
  }
  # Synthetic PE images are parsed by the real gate, never executed.
  function New-PeFixture([string]$Path, [string]$Import = '', [string]$Delay = '') {
    $bytes = [byte[]]::new(0x600)
    function U16($offset, [uint16]$value) { [BitConverter]::GetBytes($value).CopyTo($bytes, $offset) }
    function U32($offset, [uint32]$value) { [BitConverter]::GetBytes($value).CopyTo($bytes, $offset) }
    U16 0 0x5a4d; U32 0x3c 0x80; U32 0x80 0x4550; U16 0x86 1
    U16 0x94 240; U16 0x98 0x20b; U32 0xd4 0x200; U32 0x104 16
    U32 0x190 0x400; U32 0x194 0x1000; U32 0x198 0x400; U32 0x19c 0x200
    if ($Import) {
      U32 0x110 0x1000; U32 0x114 40; U32 0x20c 0x1100; U32 0x210 0x1180
      [Text.Encoding]::ASCII.GetBytes($Import).CopyTo($bytes, 0x300)
    }
    if ($Delay) {
      U32 0x170 0x1040; U32 0x174 64; U32 0x240 1; U32 0x244 0x1140
      [Text.Encoding]::ASCII.GetBytes($Delay).CopyTo($bytes, 0x340)
    }
    [IO.File]::WriteAllBytes($Path, $bytes)
  }
  foreach ($kind in @('folder-clean', 'folder-crt', 'folder-parent', 'folder-amd')) {
    if ($Case -ne 'all' -and $Case -ne $kind) { continue }
    $parent = if ($kind -eq 'folder-parent') { 'Microsoft.VC143.CRT-parent/vcruntime140.dll' } else { $kind }
    $folder = Join-Path $scratch "$parent/package"
    [IO.Directory]::CreateDirectory($folder) | Out-Null
    New-PeFixture (Join-Path $folder 'fixture.exe')
    [IO.File]::WriteAllText((Join-Path $folder 'concrete.txt'), 'unrelated')
    [IO.File]::WriteAllText((Join-Path $folder 'vcompat.md'), 'unrelated')
    if ($kind -eq 'folder-crt') { New-PeFixture (Join-Path $folder 'vcruntime140.dll') }
    if ($kind -eq 'folder-amd') {
      $data = Join-Path $folder 'bin\rocblas\library'
      [IO.Directory]::CreateDirectory($data) | Out-Null
      [IO.File]::WriteAllText((Join-Path $data 'payload.dat'), 'AMD payload fixture')
    }
    $output = & pwsh -NoProfile -File $Gate -PackageFolder $folder 2>&1
    $rc = $LASTEXITCODE
    Write-Output "FOLDER $kind rc=$rc"
    $output | Write-Output
    if ($kind -eq 'folder-crt') {
      if ($rc -ne 1 -or "$output" -notmatch 'FAIL:.*vcruntime140.dll: VC\+\+ prerequisite') { throw 'folder CRT must fail' }
    } elseif ($kind -eq 'folder-amd') {
      if ($rc -ne 1 -or "$output" -notmatch 'AMD payload is acquired separately') { throw 'folder AMD payload must fail' }
    } elseif ($rc -ne 0 -or "$output" -notmatch 'PASS: release PE') { throw "$kind must pass" }
  }
  # Release CRT/UCRT system imports stay distinct from shipped files.
  $deny = Get-Content -LiteralPath (Join-Path (Split-Path $Gate) 'nonredist-imports.json') -Raw | ConvertFrom-Json
  foreach ($name in @('msvcp140.dll','msvcp140_1.dll','msvcp140_2.dll',
      'vcruntime140.dll','vcruntime140_1.dll','concrt140.dll','vccorlib140.dll','vcomp140.dll',
      'ucrtbase.dll','api-ms-win-crt-runtime-l1-1-0.dll')) {
    if (@($deny | Where-Object { $name -like $_ }).Count) { throw "release system import denied: $name" }
  }
  foreach ($name in @('vcomp140.dll', 'ucrtbase.dll', 'api-ms-win-crt-runtime-l1-1-0.dll', 'vcomp140d.dll', 'libiomp5md.dll')) {
    if ($Case -ne 'all' -and $Case -ne "import-$name") { continue }
    $folder = Join-Path $scratch "import-$name"
    [IO.Directory]::CreateDirectory($folder) | Out-Null
    New-PeFixture (Join-Path $folder 'fixture.exe') $name $name
    $output = & pwsh -NoProfile -File $Gate -PackageFolder $folder 2>&1
    $rc = $LASTEXITCODE
    Write-Output "IMPORT normal+delay $name rc=$rc"
    $output | Write-Output
    $forbidden = $name -in @('vcomp140d.dll', 'libiomp5md.dll')
    if ($forbidden) {
      if ($rc -ne 1 -or "$output" -notmatch 'denied binary/import') { throw "debug/OpenMP import accepted: $name" }
    } elseif ($rc -ne 0) { throw "release system import rejected: $name" }
  }
  Write-Output 'PASS: CRT/UCRT ZIP and folder inventory; relative paths; normal/delay system imports; debug/OpenMP deny-list'
} finally { Remove-Item -LiteralPath $scratch -Recurse -Force }
