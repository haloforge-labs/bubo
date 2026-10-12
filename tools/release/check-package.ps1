# Release inventory and PE scan. CRT prerequisites may be system imports only.
[CmdletBinding(DefaultParameterSetName='Folder')]
param(
  [Parameter(Mandatory, ParameterSetName='Folder')][string]$PackageFolder,
  [Parameter(Mandatory, ParameterSetName='Zip')][string]$PackageZip
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\pe-imports.ps1"
$deny = Get-Content -LiteralPath "$PSScriptRoot\nonredist-imports.json" -Raw | ConvertFrom-Json
$shippedDeny = @(Get-Content -LiteralPath "$PSScriptRoot\nonredist-shipped.json" -Raw | ConvertFrom-Json)
$runtimeRows = Get-Content -LiteralPath "$PSScriptRoot\runtime-members.tsv" | Select-Object -Skip 1
$amdFiles = @($runtimeRows | ForEach-Object { ($_ -split "`t")[2] })
$errors = [Collections.Generic.List[string]]::new()
function Test-ShippedPath([string]$Path) {
  $parts = $Path -split '[\\/]'
  foreach ($part in $parts) {
    foreach ($pattern in $shippedDeny) {
      if ($part -like $pattern) { $errors.Add("${Path}: VC++ prerequisite must not be shipped ($pattern)") }
    }
  }
  $name = $parts[-1]
  if ($name -in $amdFiles -or $Path -match '[\\/]bin[\\/](rocblas|hipblaslt)[\\/]') {
    $errors.Add("${Path}: AMD payload is acquired separately by bubo prepare runtime")
  }
  foreach ($pattern in $deny) {
    if ($name -like $pattern) { $errors.Add("${Path}: denied shipped file ($pattern)") }
  }
}
if ($PackageZip) {
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $archive = [IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $PackageZip).Path)
  try {
    foreach ($entry in $archive.Entries) { Test-ShippedPath $entry.FullName }
    if ($errors.Count) { $errors | ForEach-Object { Write-Output "FAIL: $_" }; exit 1 }
    Write-Output "PASS: release ZIP shipped-file gate ($($archive.Entries.Count) entries)"
  } finally { $archive.Dispose() }
  exit 0
}
$packageRoot = (Resolve-Path -LiteralPath $PackageFolder).Path
$files = @(Get-ChildItem -LiteralPath $packageRoot -Recurse -File |
  Where-Object { $_.Extension -in @('.exe', '.dll') })
if (-not $files.Count) { throw "No release binaries in $PackageFolder" }
foreach ($file in Get-ChildItem -LiteralPath $PackageFolder -Recurse -File) {
  Test-ShippedPath ([IO.Path]::GetRelativePath($packageRoot, $file.FullName))
}
foreach ($file in $files) {
  if ($file.Name -in @('bubo.exe', 'bubo-engine.dll')) {
    $imageText = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($file.FullName))
    if ($imageText -match 'Run(Model|Runtime|All)ForTesting|RuntimeReadyErrorForTesting|LoadModelPackageForTesting|test_loopback|origin_override|synthetic_disclosure|http,https') {
      $errors.Add("$($file.FullName): fixture override symbol in release binary")
    }
  }
  foreach ($name in @($file.Name) + @(Get-PeImports $file.FullName | ForEach-Object Name)) {
    foreach ($pattern in $deny) {
      if ($name -like $pattern) { $errors.Add("$($file.FullName): denied binary/import $name ($pattern)") }
    }
  }
}
if ($errors.Count) { $errors | ForEach-Object { Write-Output "FAIL: $_" }; exit 1 }
Write-Output "PASS: release PE import deny-list ($($files.Count) binaries; normal and delay imports)"
