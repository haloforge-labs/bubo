# Execute the actual packaging closure predicate against independent edge cases.
$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path "$PSScriptRoot\..\..").Path
$source = Get-Content -LiteralPath "$repository\tools\windows\package.ps1" -Raw
$match = [regex]::Match($source, '\$missing = @\(\$closure\.Edges \| Where-Object (\{[\s\S]*?\})\s*\)')
if (-not $match.Success) { throw 'Missing production closure predicate anchor' }
$predicate = [ScriptBlock]::Create($match.Groups[1].Value.Substring(1, $match.Groups[1].Value.Length - 2))
$amdRuntimeDlls = @('amdhip64_7.dll')
foreach ($case in @(
  @('bubo.exe', 'amdhip64_7.dll', 'missing', 1),
  @('bubo-engine.dll', 'amdhip64_7.dll', 'missing', 0),
  @('bubo-engine.dll', 'unreviewed-amd.dll', 'missing', 1),
  @('bubo-engine.dll', 'libcurl.dll', 'missing', 1),
  @('bubo.exe', 'libcurl.dll', 'missing', 1),
  @('bubo.exe', 'libcurl.dll', 'file', 0))) {
  $entry = $case[0]
  $edge = [pscustomobject]@{ From = 'C:\fixture\binary.dll'; Name = $case[1]; Resolution = $case[2] }
  $actual = @($edge | Where-Object $predicate).Count
  if ($actual -ne $case[3]) { throw "Wrong closure refusal: $($case -join ',') => $actual" }
}
Write-Output 'PASS: actual package closure requires bootstrap and non-AMD dependencies; only known AMD imports on the lazy engine may be absent'
