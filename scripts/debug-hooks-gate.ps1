param(
 [Parameter(Mandatory)][string]$RuntimeDir,
 [Parameter(Mandatory)][string]$BaseRuntimeDir,
 [Parameter(Mandatory)][string]$ModelPath,
 [Parameter(Mandatory)][string]$MtpModelPath,
 [Parameter(Mandatory)][string]$OutputDirectory,
 [string]$Python = 'python',
 [int]$Context = 8192,
 [int]$MaxScoreTokens = 8192,
 [int]$StartupTimeoutSeconds = 600
)
# Manual GPU gate. Never run during a timing/build exclusion window.
$ErrorActionPreference = 'Stop'
Get-Date
$null = New-Item -ItemType Directory -Force -Path $OutputDirectory
$helper = Join-Path $PSScriptRoot 'debug-hooks-gate.py'
foreach ($path in @($ModelPath, $MtpModelPath, $helper)) {
 if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $path" }
}
$oldDebug = $env:GUFO_DEBUG_HOOKS
$env:GUFO_DEBUG_HOOKS = '0'
try {
 foreach ($phase in @('score', 'candidate', 'base')) {
  # Refuse to take over an occupied port; never kill an unrelated listener.
  $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 18101)
  try { $probe.Start() } finally { $probe.Stop() }
  $runtime = if ($phase -eq 'base') { $BaseRuntimeDir } else { $RuntimeDir }
  $exe = Join-Path $runtime 'bubo.exe'
  if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing executable: $exe" }
  $arguments = @('serve', 'llm', '--model', ('"' + $ModelPath + '"'),
   '--mtp-model', ('"' + $MtpModelPath + '"'), '--speculative', 'mtp',
   '--host', '127.0.0.1', '--port', '18101', '--context', "$Context",
   '--sessions', '1', '--think', 'off', '--served-model-name', 'debug-gate')
  if ($phase -eq 'score') { $arguments += @('--debug-hooks', '--max-score-tokens', "$MaxScoreTokens") }
  Get-Date
  $process = Start-Process -FilePath $exe -WorkingDirectory $runtime -ArgumentList $arguments `
   -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputDirectory "$phase.stdout.txt") `
   -RedirectStandardError (Join-Path $OutputDirectory "$phase.stderr.txt")
  try {
   $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
   do {
    if ($process.HasExited) { throw "$phase server exited: $($process.ExitCode)" }
    try {
     $ready = Invoke-RestMethod -Uri 'http://127.0.0.1:18101/ready' -TimeoutSec 5
     if ($ready.status -eq 'ready') { break }
    } catch { }
    if ([DateTime]::UtcNow -gt $deadline) { throw "$phase startup timed out" }
    Start-Sleep -Seconds 1
   } while ($true)
   Get-Date
   & $Python $helper $phase --url 'http://127.0.0.1:18101' --out $OutputDirectory
   if ($LASTEXITCODE -ne 0) { throw "$phase HTTP gate failed: $LASTEXITCODE" }
  } finally {
   # Only this script's own child is stopped. No process-name cleanup.
   if (!$process.HasExited) { Stop-Process -Id $process.Id; $process.WaitForExit() }
  }
 }
 Get-Date
 & $Python $helper compare --out $OutputDirectory
 if ($LASTEXITCODE -ne 0) { throw 'HTTP byte gate failed; inspect saved bodies' }
} finally { $env:GUFO_DEBUG_HOOKS = $oldDebug }
