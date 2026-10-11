# Release binary scan: reject non-redistributable and OpenMP imports in Bubo.
param([Parameter(Mandatory)][string]$PackageFolder)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\pe-imports.ps1"
$deny = Get-Content -LiteralPath "$PSScriptRoot\nonredist-imports.json" -Raw | ConvertFrom-Json
$files = @(Get-ChildItem -LiteralPath $PackageFolder -Recurse -File |
  Where-Object { $_.Extension -in @('.exe', '.dll') })
if (-not $files.Count) { throw "No release binaries in $PackageFolder" }
$errors = [Collections.Generic.List[string]]::new()
foreach ($file in $files) {
  foreach ($name in @($file.Name) + @(Get-PeImports $file.FullName | ForEach-Object Name)) {
    foreach ($pattern in $deny) {
      if ($name -like $pattern) { $errors.Add("$($file.FullName): denied binary/import $name ($pattern)") }
    }
  }
}
if ($errors.Count) { $errors | ForEach-Object { Write-Output "FAIL: $_" }; exit 1 }
Write-Output "PASS: release PE import deny-list ($($files.Count) binaries; normal and delay imports)"
