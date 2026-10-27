param([Parameter(Mandatory=$true)][string]$EvidenceDir)
$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$evidence = [System.IO.Path]::GetFullPath($EvidenceDir)
$exe = Join-Path $evidence 'Bubo.Control.Web.exe'
if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw 'Build the exact executable before scanning.' }
function Save-LF([string]$Name,[object[]]$Lines) {
    [System.IO.File]::WriteAllText((Join-Path $evidence $Name), (($Lines | ForEach-Object { [string]$_ }) -join "`n") + "`n", $utf8)
}
$platform = Get-ChildItem -LiteralPath 'C:\ProgramData\Microsoft\Windows Defender\Platform' -Directory | Sort-Object Name -Descending | Select-Object -First 1
$scanner = Join-Path $platform.FullName 'MpCmdRun.exe'
$engine = Get-MpComputerStatus
Save-LF 'defender-engine.json' @(($engine | Select-Object AMEngineVersion,AMProductVersion,AntivirusSignatureVersion,AntivirusEnabled,RealTimeProtectionEnabled | ConvertTo-Json -Compress))
$started = Get-Date
$scan = @(& $scanner -Scan -ScanType 3 -File $exe 2>&1)
$rc = $LASTEXITCODE
Save-LF 'defender-scan.txt' $scan
Save-LF 'defender-rc.txt' @("MpCmdRun rc: $rc")
# Protection History is read with the Defender API. No Windows Security window
# is opened. Match only a path resource, not unrelated machine detections.
$pathPattern = '(?i)(?:^|[:_])' + [Regex]::Escape($exe) + '(?:$|;)'
$detections = @(Get-MpThreatDetection | Where-Object { @($_.Resources | Where-Object { [string]$_ -match $pathPattern }).Count -gt 0 })
$history = [ordered]@{checked=(Get-Date).ToString('o'); source='Get-MpThreatDetection (Protection History API, read only)'; exact_file=$exe; matching_detections=$detections.Count; detections=$detections}
Save-LF 'protection-history.json' @(($history | ConvertTo-Json -Depth 8 -Compress))
$text = $scan -join "`n"
$pass = $rc -eq 0 -and $text -match 'found no threats' -and $detections.Count -eq 0 -and $engine.AntivirusEnabled
$verdict = [ordered]@{verdict=$(if($pass){'PASS for local exact-file scan'}else{'NOT PROVEN / investigate'}); rc=$rc; scanner_states_no_threats=($text -match 'found no threats'); matching_history_detections=$detections.Count; bytes=(Get-Item -LiteralPath $exe).Length; sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash; scan_started=$started.ToString('o'); scan_finished=(Get-Date).ToString('o')}
Save-LF 'defender-verdict.json' @(($verdict | ConvertTo-Json -Compress))
Write-Output ($verdict | ConvertTo-Json -Compress)
if (!$pass) { throw 'Defender gate was not proven. Read scan and Protection History evidence.' }
