# Gufo mini launcher (start.cmd): finds the models on this machine with their
# drafts and vision sidecars, then picks a model and its features with single
# number keys. Everything that is exact and measured faster is on by default;
# the Latin draft vocabulary is the one opt-in (recommended for English use).
#
#   start.cmd            menu
#   start.cmd -Last      start the previous choice again
#   start.cmd -DryRun    print the gufo command instead of starting it
#   start.cmd -Bin DIR   use gufo.exe from DIR
#
# Choices are remembered in %LOCALAPPDATA%\gufo\launcher.json. The full
# question-by-question launcher is tools\windows\start.ps1.
param(
  [switch]$Last,
  [switch]$DryRun,
  [string]$Bin = "",
  # Testing: key presses to replay instead of reading the keyboard (E = Enter).
  [string]$Keys = ""
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "inventory.ps1")
$invariant = [Globalization.CultureInfo]::InvariantCulture

# --- gufo.exe: a source build, or the prebuilt package's bin folder ---------------
$exe = $null
foreach ($dir in @($Bin, (Join-Path $GufoRoot "build\release"), (Join-Path $GufoRoot "bin"), $GufoRoot)) {
  if ($dir -and (Test-Path (Join-Path $dir "gufo.exe"))) { $exe = Join-Path $dir "gufo.exe"; break }
}
if (-not $exe) {
  Write-Host "gufo.exe not found. Build it with tools\windows\build.ps1, or use the prebuilt package." -ForegroundColor Yellow
  exit 1
}

# hipBLASLt's kernel files have long names; past Windows' 260-character path
# limit it crashes on the first matrix multiply, so say so up front.
$longest = Get-ChildItem (Join-Path (Split-Path $exe) "hipblaslt"), (Join-Path (Split-Path $exe) "rocblas") -Recurse -File -ErrorAction SilentlyContinue |
  Sort-Object { $_.FullName.Length } -Descending | Select-Object -First 1
if ($longest -and $longest.FullName.Length -ge 260) {
  Write-Host ("This folder is too deep: GPU kernel files reach {0} characters, past Windows' 260 limit, and gufo would crash." -f $longest.FullName.Length) -ForegroundColor Red
  Write-Host ("Move it to a shorter path, e.g. C:\gufo (at most {0} characters for the folder)." -f (259 - ($longest.FullName.Length - (Split-Path $exe).Length))) -ForegroundColor Red
  exit 1
}

# --- Remembered choices ----------------------------------------------------------
$settingsPath = Join-Path $env:LOCALAPPDATA "gufo\launcher.json"
$settings = @{ Last = $null; Features = @{} }
try {
  if (Test-Path $settingsPath) {
    $saved = Get-Content $settingsPath -Raw | ConvertFrom-Json
    if ($saved.Last) { $settings.Last = $saved.Last }
    if ($saved.Features) { $saved.Features.PSObject.Properties | ForEach-Object { $settings.Features[$_.Name] = $_.Value } }
  }
} catch { }
function Save-Settings {
  try {
    New-Item -ItemType Directory -Force (Split-Path $settingsPath) | Out-Null
    $settings | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $settingsPath
  } catch { }
}

function Get-PortOwner([int]$Port) {
  $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $listener) { return $null }
  $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
  if ($process) { return "$($process.ProcessName) (PID $($process.Id))" }
  "PID $($listener.OwningProcess)"
}

function Start-Plan($Plan) {
  $gufoArgs = @($Plan.Args)
  $port = [int]$Plan.Port
  $owner = Get-PortOwner $port
  if ($owner) {
    for ($free = $port + 1; $free -lt $port + 20 -and (Get-PortOwner $free); $free++) { }
    Write-Host "port $port is taken by $owner, using $free" -ForegroundColor Yellow
    $gufoArgs[[array]::IndexOf($gufoArgs, "--port") + 1] = "$free"
    $port = $free
  }
  $running = @(Get-Process -Name "gufo" -ErrorAction SilentlyContinue)
  if ($running.Count -gt 0 -and -not $DryRun) {
    Write-Host ("gufo is already running (PID {0}) and holds GPU memory; close it first for the full memory." -f ($running.Id -join ", ")) -ForegroundColor Yellow
  }
  # Diagnostic switches from the shell must not leak into a normal start.
  foreach ($variable in Get-GufoEnvSwitches) { Remove-Item "Env:$($variable.Name)" -ErrorAction SilentlyContinue }
  foreach ($property in $Plan.Env.PSObject.Properties) { Set-Item "Env:$($property.Name)" $property.Value }
  $shown = ($gufoArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join " "
  $envText = ($Plan.Env.PSObject.Properties | ForEach-Object { "set $($_.Name)=$($_.Value) && " }) -join ""
  Write-Host ""
  Write-Host $Plan.Title -ForegroundColor Cyan
  Write-Host "$envText`"$exe`" $shown" -ForegroundColor DarkGray
  if ($DryRun) { return }
  Write-Host ("OpenAI API: http://{0}:{1}/v1  model `"{2}`"   (ready at event=listening; Ctrl+C stops)" -f
    $(if ($Plan.Host -eq "0.0.0.0") { "<this-pc>" } else { "127.0.0.1" }), $port, $Plan.ServedName) -ForegroundColor Green
  Write-Host ""
  & $exe @gufoArgs
}

if ($Last) {
  if (-not $settings.Last) { Write-Host "nothing started yet; run start.cmd without -Last" -ForegroundColor Yellow; exit 1 }
  Start-Plan $settings.Last
  exit $LASTEXITCODE
}

# --- What is on this machine ---------------------------------------------------------
Write-Host "Looking for models..." -ForegroundColor DarkGray
$all = @(Find-GufoGgufs)
$memory = Get-GufoMemory
$familyOrder = @{ "fn" = 0; "27b" = 1; "a3b" = 2 }
$targets = @($all | Where-Object { $_.Kind -eq "target" -and $_.MissingShards.Count -eq 0 } |
  Sort-Object @{ Expression = { $familyOrder[$_.Family] } }, Name, @{ Expression = "Bytes"; Descending = $true })

# Per target: the draft, the vision sidecar and which features apply.
function Get-Option($Target) {
  $info = Get-GufoFamilyInfo $Target.Family
  $draft = $null; $draftKind = ""
  switch ($Target.Family) {
    "fn" { $draft = @(Get-GufoCompanions $Target $all "mtp" | ForEach-Object { $_.Record }) | Select-Object -First 1; if ($draft) { $draftKind = "MTP" } }
    "27b" { $draft = @(Get-GufoCompanions $Target $all "dflash" | ForEach-Object { $_.Record }) | Select-Object -First 1; if ($draft) { $draftKind = "DFlash2" } }
    "a3b" {
      if ($Target.HasMtp) { $draftKind = "MTP" }
      else {
        $draft = @(Get-GufoCompanions $Target $all "dflash" | ForEach-Object { $_.Record }) | Select-Object -First 1
        if ($draft) { $draftKind = "DFlash2" }
      }
    }
  }
  $vision = @(Get-GufoCompanions $Target $all "mmproj" | ForEach-Object { $_.Record }) | Select-Object -First 1
  if (-not $vision -and $Target.Family -eq "a3b") {
    # Fine-tunes of Qwen3.6-35B-A3B keep its vision tower: any BF16 Qwen3.6 sidecar.
    $vision = @($all | Where-Object { $_.Kind -eq "mmproj" -and $_.Embedding -eq $Target.Embedding -and
      [int](Get-HeaderValue $_.Header "general.file_type" 32) -eq 32 }) | Select-Object -First 1
  }
  $limit = if ($Target.NativeContext -gt 0) { $Target.NativeContext } else { 262144 }
  $contexts = @($GufoKnownContexts | Where-Object { $_ -le $limit })
  $context = $contexts[0]
  foreach ($candidate in $contexts) {
    if ($candidate -gt $info.DefaultContext) { break }
    if ((Get-GufoFit (Get-GufoMemoryEstimate $Target $candidate 1 $draft) $memory) -ne "fits" -and $candidate -ne $contexts[0]) { break }
    $context = $candidate
  }
  [pscustomobject]@{ Target = $Target; Info = $info; Draft = $draft; DraftKind = $draftKind; Vision = $vision
    Contexts = $contexts; Context = $context; Key = "$($Target.Family)|$($Target.FileName)" }
}

$options = @($targets | ForEach-Object { Get-Option $_ })
$gpu = if ($memory.GpuName) { "$($memory.GpuName), $($memory.DedicatedGiB) GB GPU memory" } else { "no Radeon GPU found" }

function Show-Models {
  Clear-Host
  Write-Host "  gufo  -  $gpu" -ForegroundColor Cyan
  Write-Host ""
  if ($options.Count -eq 0) { Write-Host "  no model gufo can serve was found" -ForegroundColor Yellow }
  for ($i = 0; $i -lt [math]::Min(9, $options.Count); $i++) {
    $o = $options[$i]
    $tags = @()
    if ($o.DraftKind) { $tags += $o.DraftKind }
    if ($o.Vision) { $tags += "vision" }
    $need = Get-GufoMemoryEstimate $o.Target $o.Context 1 $o.Draft
    $fit = Get-GufoFit $need $memory
    $color = if ($fit -eq "fits") { "White" } else { "Yellow" }
    Write-Host ("  {0}  {1,-26} {2,-12} {3,9}  {4,-14} {5}" -f ($i + 1), $o.Target.Name, $o.Target.Quant,
      (Format-GufoGiB ($o.Target.Bytes / 1GB)), ($tags -join " + "), $(if ($fit -eq "fits") { "" } else { $fit })) -ForegroundColor $color
  }
  $missing = @("fn", "27b", "a3b") | Where-Object { $f = $_; -not ($targets | Where-Object { $_.Family -eq $f }) }
  if ($missing) {
    Write-Host ""
    Write-Host "  Not on this machine (pip install -U huggingface_hub, then):" -ForegroundColor DarkGray
    foreach ($f in $missing) { $i = Get-GufoFamilyInfo $f; Write-Host ("    {0,-20} {1}" -f $i.Title, $i.Download) -ForegroundColor DarkGray }
  }
  Write-Host ""
  $lastText = if ($settings.Last) { "Enter = start the last one ($($settings.Last.Title))   " } else { "" }
  Write-Host "  ${lastText}Q = quit" -ForegroundColor DarkGray
}

# Feature switches, by family. Lookup, survival and the draft vocabulary ride on
# speculative decoding; 27B has no survival or draft-vocab switch.
function Get-Features($o) {
  $saved = $settings.Features[$o.Key]
  $f = [ordered]@{ Spec = [bool]$o.DraftKind; Lookup = [bool]$o.DraftKind; Survival = $o.Target.Family -ne "27b" -and [bool]$o.DraftKind
    Vocab = $false; Vision = [bool]$o.Vision; Think = $true; Context = $o.Context; Lan = $false }
  if ($saved) {
    foreach ($name in @($f.Keys)) { if ($null -ne $saved.$name) { $f[$name] = $saved.$name } }
  }
  if (-not $o.DraftKind) { $f.Spec = $false }
  if (-not $o.Vision) { $f.Vision = $false }
  if ($o.Contexts -notcontains [int]$f.Context) { $f.Context = $o.Context }
  $f
}

function Show-Features($o, $f) {
  Clear-Host
  Write-Host ("  {0} {1}" -f $o.Target.Name, $o.Target.Quant) -ForegroundColor Cyan
  Write-Host ("  {0}" -f $o.Target.Path) -ForegroundColor DarkGray
  Write-Host ""
  $box = { param($on, $available) if (-not $available) { "[-]" } elseif ($on) { "[x]" } else { "[ ]" } }
  $spec = [bool]$o.DraftKind; $rides = $spec -and $f.Spec; $family = $o.Target.Family
  $rows = @(
    @("1", (& $box $f.Spec $spec), "Speculative decoding", $(if ($spec) { "$($o.DraftKind)$(if ($o.Draft) { ': ' + $o.Draft.FileName })" } else { "no draft found: $($o.Info.DraftDownload)" }), $spec),
    @("2", (& $box $f.Lookup $rides), "Prompt lookup", "drafts copied from the context; big win on code edits", $rides),
    @("3", (& $box $f.Survival ($rides -and $family -ne "27b")), "Survival stopping", "stop a draft chain once it is unlikely to be accepted", ($rides -and $family -ne "27b")),
    @("4", (& $box $f.Vocab ($rides -and $family -ne "27b")), "Latin draft vocabulary", "RECOMMENDED for English: ~5% faster, output identical", ($rides -and $family -ne "27b")),
    @("5", (& $box $f.Vision ([bool]$o.Vision)), "Images (vision)", $(if ($o.Vision) { $o.Vision.FileName } else { "no matching mmproj-BF16.gguf found" }), [bool]$o.Vision),
    @("6", (& $box $f.Think $true), "Thinking", "reasoning before the answer (clients can still turn it off per request)", $true),
    @("7", "   ", "Context", ("{0}K tokens per request  (press 7 to change)" -f ([int]$f.Context / 1024)), $true),
    @("8", (& $box $f.Lan $true), "Open to the network", $(if ($f.Lan) { "0.0.0.0, asks for an API key" } else { "this PC only (127.0.0.1)" }), $true))
  foreach ($r in $rows) {
    $color = if (-not $r[4]) { "DarkGray" } elseif ($r[0] -eq "4" -and -not $f.Vocab) { "Yellow" } else { "White" }
    Write-Host ("  {0} {1} {2,-24} {3}" -f $r[0], $r[1], $r[2], $r[3]) -ForegroundColor $color
  }
  $need = Get-GufoMemoryEstimate $o.Target ([int]$f.Context) 1 $(if ($f.Spec) { $o.Draft } else { $null })
  Write-Host ""
  Write-Host ("  ~{0} of GPU memory: {1}" -f (Format-GufoGiB $need), (Get-GufoFit $need $memory)) -ForegroundColor DarkGray
  Write-Host "  Enter = start   B = back   D = show the command   Q = quit" -ForegroundColor DarkGray
}

function New-Plan($o, $f) {
  $t = $o.Target; $family = $t.Family
  $sampling = if ($f.Think) {
    $temp = if ($null -ne $t.Sampler.Temperature) { $t.Sampler.Temperature } else { 1.0 }
    $topP = if ($null -ne $t.Sampler.TopP) { $t.Sampler.TopP } else { 0.95 }
    $topK = if ($null -ne $t.Sampler.TopK) { $t.Sampler.TopK } else { 20 }
    @("--temperature", $temp.ToString($invariant), "--top-p", $topP.ToString($invariant), "--top-k", "$topK", "--min-p", "0")
  } else { @("--temperature", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0") }
  $hostName = if ($f.Lan) { "0.0.0.0" } else { "127.0.0.1" }
  $server = @("serve", "--host", $hostName, "--port", "8080", "--sessions", "1")
  if ($f.Vision) { $server += @("--max-request-bytes", "33554432") }   # base64 images next to a long chat
  $llm = @("llm", "--model", $t.Path, "--served-model-name", $o.Info.ServedName, "--context", "$($f.Context)",
    "--max-output-bytes", "8388608", "--think", $(if ($f.Think) { "on" } else { "off" })) + $sampling
  $vars = [ordered]@{}
  $labels = @()
  if ($f.Spec -and $o.DraftKind) {
    $rest = @()
    switch ($family) {
      "fn" { $llm += @("--speculative", "mtp", "--mtp-model", $o.Draft.Path) }
      "27b" {
        $llm += @("--speculative", "dflash2", "--dflash-model", $o.Draft.Path)
        # DFlash2 keeps its 3-draft cap; lookup copies run up to 7.
        if ($f.Lookup) { $llm += @("--draft-tokens", "7"); $vars["GUFO_QWEN27_LOOKUP"] = "3" } else { $llm += @("--draft-tokens", "3") }
      }
      "a3b" {
        if ($o.DraftKind -eq "MTP") { $llm += @("--speculative", "mtp", "--draft-tokens", "6") }
        else { $llm += @("--speculative", "dflash2", "--dflash-model", $o.Draft.Path, "--draft-tokens", "7", "--min-draft-tokens", "1") }
      }
    }
    $labels += $o.DraftKind
    if ($family -ne "27b") {
      if ($f.Survival) { $llm += @("--mtp-policy", "survival"); $labels += "survival" }
      if ($f.Vocab) { $llm += @("--mtp-draft-vocab", "latin"); $labels += "latin vocab" }
      if ($f.Lookup) { $llm += "--prompt-lookup" }
    }
    if ($f.Lookup) { $labels += "lookup" }
  } else { $labels += "no speculation" }
  if ($f.Vision) { $llm += @("--mmproj", $o.Vision.Path); $labels += "vision" }
  if ($f.Lan) {
    $key = Read-Host "  API key clients must send (Enter = none: anyone on the network can use the server)"
    if ($key) { $server += @("--api-key", $key) }
  }
  [pscustomobject]@{
    Title = "$($t.Name) $($t.Quant): $($labels -join ', '), $([int]$f.Context / 1024)K context, thinking $(if ($f.Think) { 'on' } else { 'off' })"
    Args = @($server + $llm); Env = [pscustomobject]$vars; Port = 8080; Host = $hostName; ServedName = $o.Info.ServedName
  }
}

$script:queue = [System.Collections.Generic.Queue[char]]::new([char[]]$Keys)
function Read-Key {
  if ($script:queue.Count -gt 0) {
    $c = $script:queue.Dequeue()
    if ($c -eq "E") { return [pscustomobject]@{ Key = "Enter"; KeyChar = [char]13 } }
    return [pscustomobject]@{ Key = [string]$c; KeyChar = $c }
  }
  if ($Keys) { exit 0 }
  [Console]::ReadKey($true)
}

while ($true) {
  Show-Models
  $key = Read-Key
  if ($key.Key -eq "Q" -or $key.Key -eq "Escape") { exit 0 }
  if ($key.Key -eq "Enter" -and $settings.Last) { Start-Plan $settings.Last; exit $LASTEXITCODE }
  $n = 0
  if (-not [int]::TryParse([string]$key.KeyChar, [ref]$n) -or $n -lt 1 -or $n -gt [math]::Min(9, $options.Count)) { continue }
  $o = $options[$n - 1]
  $f = Get-Features $o
  $back = $false
  while (-not $back) {
    Show-Features $o $f
    $key = Read-Key
    switch -regex ([string]$key.KeyChar) {
      "1" { if ($o.DraftKind) { $f.Spec = -not $f.Spec } }
      "2" { $f.Lookup = -not $f.Lookup }
      "3" { if ($o.Target.Family -ne "27b") { $f.Survival = -not $f.Survival } }
      "4" { if ($o.Target.Family -ne "27b") { $f.Vocab = -not $f.Vocab } }
      "5" { if ($o.Vision) { $f.Vision = -not $f.Vision } }
      "6" { $f.Think = -not $f.Think }
      "7" { $i = [array]::IndexOf($o.Contexts, [int]$f.Context); $f.Context = $o.Contexts[($i + 1) % $o.Contexts.Count] }
      "8" { $f.Lan = -not $f.Lan }
      "[bB]" { $back = $true }
      "[qQ]" { exit 0 }
      "[dD]" { $saveDry = $DryRun; $DryRun = $true; Start-Plan (New-Plan $o $f); $DryRun = $saveDry; Write-Host "  (any key)" -ForegroundColor DarkGray; [void](Read-Key) }
    }
    if ($key.Key -eq "Enter") {
      $plan = New-Plan $o $f
      $settings.Features[$o.Key] = [pscustomobject]$f
      if (-not $DryRun) { $settings.Last = $plan; Save-Settings }
      Start-Plan $plan
      exit $LASTEXITCODE
    }
  }
}
