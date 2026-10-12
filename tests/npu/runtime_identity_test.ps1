param([Parameter(Mandatory)][string]$Root)
$repo=Split-Path (Split-Path $PSScriptRoot)
. "$repo\tools\npu\logging.ps1"
. "$repo\tools\npu\runtime-identity.ps1"
$run=Job-Path (Join-Path $Root ([guid]::NewGuid().ToString()))
$null=New-Item -ItemType Directory -Path $run
foreach($name in 'bubo.exe','bubo-engine.dll','bubo-npu-watchdog.exe'){Flush-Text "$run\$name" "MOCK bytes $name"}
$packaged=@{};$lines=@()
foreach($name in 'bubo.exe','bubo-engine.dll','bubo-npu-watchdog.exe'){
    $f=File-Fact "$run\$name";$packaged["bin/$name"]=@{sha256=$f.sha256}
    $lines+="$($f.sha256)  $($f.size)  $name"
}
Flush-Text "$run\qualification.json" (@{build_mode='ON';status='CPU-PREPARED-HARDWARE-UNQUALIFIED';packaged=$packaged}|ConvertTo-Json -Depth 8)
Flush-Text "$run\manifest.txt" ($lines -join "`n")
foreach($manifest in "$run\qualification.json","$run\manifest.txt"){
    if(@(Compare-RuntimeManifest "$run\bubo.exe" $manifest).Count -ne 3){throw 'Positive manifest failed'}
    foreach($name in 'bubo.exe','bubo-engine.dll','bubo-npu-watchdog.exe'){
        Flush-Text "$run\$name" 'drifted bytes'
        $refused=$false
        try{$null=Compare-RuntimeManifest "$run\bubo.exe" $manifest}catch{$refused=$_.Exception.Message -match 'hash mismatch'}
        if(-not $refused){throw "Accepted binary drift $name"}
        Flush-Text "$run\$name" "MOCK bytes $name"
    }
}
Assert-BridgeArguments @('--npu-prefill','on')
foreach($arg in '--fabric-peak','--fabric-peak=1','--FABRIC-PEAK'){
    $refused=$false;try{Assert-BridgeArguments @($arg)}catch{$refused=$_.Exception.Message -match 'forbidden'}
    if(-not $refused){throw 'Fabric peak admitted'}
}
$parents=Disposition-ParentFacts -Mock
if($parents.Count -ne 5 -or @($parents|Where-Object{-not $_.match}).Count){throw 'Missing parent hashes'}
# Replace only file metadata reads to exercise servicing drift without OS I/O.
function File-Fact([string]$Path){return @{path=$Path;sha256=('0'*64)}}
if(@(Disposition-ParentFacts|Where-Object{-not $_.match}).Count -ne 5){throw 'Parent drift not detected'}
Write-Output 'PASS: reviewed JSON/text manifest matches three binaries; each drift refused; fabric-peak forms forbidden; five parent hashes and drift recorded; module/device calls=0'
