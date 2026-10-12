param([Parameter(Mandatory)][string]$Root)
$repo=Split-Path (Split-Path $PSScriptRoot)
. "$repo\tools\npu\logging.ps1"
$script:channelMode='absent'
function Get-WinEvent {
    param($ListLog,$FilterHashtable,$ErrorAction)
    if($ListLog){
        if($script:channelMode -eq 'disabled'){return @{IsEnabled=$false}}
        $id=if($script:channelMode -eq 'absent'){'NoMatchingLogsFound'}else{'UnauthorizedAccess'}
    }else{$id='NoMatchingEventsFound'}
    $record=[Management.Automation.ErrorRecord]::new([Exception]::new('MOCK event query'),$id,[Management.Automation.ErrorCategory]::ObjectNotFound,$null)
    throw $record
}
foreach($mode in 'absent','disabled','denied'){
    $script:channelMode=$mode
    $s=Logging-Snapshot (Get-Date).AddMinutes(-10) (Get-Date)
    if($mode -eq 'denied'){
        if($s.query_errors.Count -ne 1){throw 'Access denial must remain fatal evidence error'}
    }else{
        if($s.query_errors.Count -or $s.channel_status.Count -ne 1 -or $s.channel_status[0].status -ne "channel $mode"){throw 'Optional absent/disabled channel was fatal or unrecorded'}
    }
}
Write-Output 'PASS: optional Kernel-WHEA/Operational absent/disabled recorded without query error; access-denied remains fail closed; OS event queries=0'
