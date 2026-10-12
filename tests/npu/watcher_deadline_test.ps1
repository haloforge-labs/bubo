# CPU/mock only: the test process is the mock host; no engine/device or OS query.
param([Parameter(Mandatory)][string]$Root)
$repo=Split-Path (Split-Path $PSScriptRoot)
. "$repo\tools\npu\logging.ps1"
$run=Job-Path (Join-Path $Root ([guid]::NewGuid().ToString()))
$null=New-Item -ItemType Directory -Path $run
Flush-Text "$run\context.json" '{"mode":"MOCK","stage_id":"deadline"}'
$psi=[Diagnostics.ProcessStartInfo]::new((Get-Process -Id $PID).Path)
$psi.UseShellExecute=$false;$psi.CreateNoWindow=$true
foreach($a in @('-NoProfile','-NonInteractive','-File',"$repo\tools\npu\watcher.ps1",'-Session',$run,'-Mock','-CallTimeoutMs','8000','-ProgressDeadline','1000')){$psi.ArgumentList.Add($a)}
$watcher=[Diagnostics.Process]::Start($psi)
try{
    $seq=1
    function Beat([string]$Phase){
        Flush-Text "$run\heartbeat.txt" "seq=$seq`npid=$PID`nphase=$Phase`ncompleted=1`nmono_ms=0`nprogress_ms=0`n"
    }
    Beat 'fake-call'
    # A legal blocking call exceeds the old 3 s heartbeat and the shortened
    # host-start deadline. A mock host remains alive with no counter updates.
    Start-Sleep -Seconds 5
    if(Test-Path "$run\abort.flag"){throw 'False heartbeat abort during legal 5 s call'}
    $seq++
    Flush-Text "$run\host-result.txt" 'MOCK done'
    Beat 'done'
    if(-not $watcher.WaitForExit(10000) -or $watcher.ExitCode -ne 0){throw 'Watcher did not complete normally'}
    if(Test-Path "$run\abort.flag"){throw 'False host-start-deadline after normal completion'}
    if((Get-Content "$run\watcher.jsonl" -Raw) -notmatch 'normal-completion'){throw 'Missing normal-completion'}
    Write-Output 'PASS: 5 s blocking mock call with timeout 8000 + native phase margin; normal completion past host-start deadline; no abort.flag'
}finally{Flush-Text "$run\stop-watcher.flag" 'test cleanup';$null=$watcher.WaitForExit(10000)}
