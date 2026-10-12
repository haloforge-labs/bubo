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
    'libomp140.x86_64.dll', 'clean')
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
    if ($variant -eq 'clean') {
      if ($rc -ne 0 -or "$output" -notmatch 'PASS: release ZIP') { throw 'clean zip must pass' }
    } elseif ($rc -ne 1 -or "$output" -notmatch 'FAIL:.*(prerequisite|denied shipped)') {
      throw "zip containing $variant must fail the shipped-file gate"
    }
  }
  # Release CRT imports remain distinct from shipped files; debug and libomp remain denied.
  $deny = Get-Content -LiteralPath "$PSScriptRoot\..\..\tools\release\nonredist-imports.json" -Raw | ConvertFrom-Json
  foreach ($name in @('msvcp140.dll','msvcp140_1.dll','msvcp140_2.dll',
      'vcruntime140.dll','vcruntime140_1.dll','concrt140.dll','vccorlib140.dll','vcomp140.dll')) {
    if (@($deny | Where-Object { $name -like $_ }).Count) { throw "release system import denied: $name" }
  }
  Write-Output 'PASS: CRT ZIP inventory; release system imports allowed; debug/OpenMP deny-list retained'
} finally { Remove-Item -LiteralPath $scratch -Recurse -Force }
