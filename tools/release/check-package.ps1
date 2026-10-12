# Release binary scan: reject non-redistributable and OpenMP imports in Bubo.
param([Parameter(Mandatory)][string]$PackageFolder)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\pe-imports.ps1"
$deny = Get-Content -LiteralPath "$PSScriptRoot\nonredist-imports.json" -Raw | ConvertFrom-Json
$files = @(Get-ChildItem -LiteralPath $PackageFolder -Recurse -File |
  Where-Object { $_.Extension -in @('.exe', '.dll') })
if (-not $files.Count) { throw "No release binaries in $PackageFolder" }
$errors = [Collections.Generic.List[string]]::new()
$runtimeRows = Get-Content -LiteralPath "$PSScriptRoot\runtime-members.tsv" | Select-Object -Skip 1
$amdFiles = @($runtimeRows | ForEach-Object { ($_ -split "`t")[2] })
foreach ($file in Get-ChildItem -LiteralPath $PackageFolder -Recurse -File) {
  if ($file.Name -in $amdFiles -or $file.FullName -match '[\\/]bin[\\/](rocblas|hipblaslt)[\\/]') {
    $errors.Add("$($file.FullName): AMD payload is acquired separately by bubo prepare runtime")
  }
}
foreach ($file in $files) {
  if ($file.Name -in @('bubo.exe', 'bubo-engine.dll')) {
    $imageText = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($file.FullName))
    if ($imageText -match 'Run(Model|Runtime|All)ForTesting|test_loopback|origin_override') {
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
