# Interactive launcher for the Gufo server on Windows. It finds the models on
# this machine (Hugging Face cache, models folders, LM Studio), lists the ones
# gufo can serve with what fits in GPU memory, and asks only for what matters;
# Enter keeps the recommended value everywhere.
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\start.ps1            # pick and start
#   powershell -ExecutionPolicy Bypass -File tools\windows\start.ps1 -Last      # start the previous choice again
#   powershell -ExecutionPolicy Bypass -File tools\windows\start.ps1 -List      # what is here, and what fits
#   ... -DryRun          print the gufo command instead of starting it
#   ... -ModelDir DIR    also search DIR for GGUF files (remembered)
#   ... -Bin DIR         use gufo.exe from DIR (default: build\release)
#
# Choices are remembered in %LOCALAPPDATA%\gufo\start.json (never the API key).
param(
  [switch]$Last,
  [switch]$List,
  [switch]$DryRun,
  [string[]]$ModelDir = @(),
  [string]$Bin = ""
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\inventory.ps1"
$invariant = [Globalization.CultureInfo]::InvariantCulture

function Ask([string]$Question, [string]$Default) {
  $answer = Read-Host ("{0} [{1}]" -f $Question, $Default)
  if ([string]::IsNullOrWhiteSpace($answer)) { return $Default }
  $answer.Trim()
}
function Ask-Choice([string]$Question, [string[]]$Valid, [string]$Default) {
  while ($true) {
    $answer = Ask $Question $Default
    $match = $Valid | Where-Object { $_ -eq $answer } | Select-Object -First 1
    if ($match) { return $match }
    Write-Host "  please answer one of: $($Valid -join ', ')" -ForegroundColor Yellow
  }
}
function Ask-YesNo([string]$Question, [bool]$Default) {
  $answer = Ask-Choice $Question @("y", "n", "yes", "no") $(if ($Default) { "y" } else { "n" })
  $answer.StartsWith("y")
}
function Ask-Number([string]$Question, [int]$Default, [int]$Min, [int]$Max) {
  while ($true) {
    $answer = Ask $Question "$Default"
    $value = 0
    if ([int]::TryParse(($answer -replace "(?i)k$", ""), [ref]$value)) {
      if ($answer -match "(?i)k$") { $value *= 1024 }
      if ($value -ge $Min -and $value -le $Max) { return $value }
    }
    Write-Host "  please enter a number from $Min to $Max" -ForegroundColor Yellow
  }
}
function Format-Arg([string]$Value) {
  if ($Value -match '[\s"]' -or $Value -eq "") { return '"' + ($Value -replace '"', '\"') + '"' }
  $Value
}
function Format-Context([int]$Tokens) {
  if ($Tokens % 1024 -eq 0) { return "{0}K" -f ($Tokens / 1024) }
  "$Tokens"
}
function Get-PortOwner([int]$Port) {
  $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $listener) { return $null }
  $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
  if ($process) { return "$($process.ProcessName) (PID $($process.Id))" }
  "PID $($listener.OwningProcess)"
}
function Get-FreePort([int]$Port) {
  for ($candidate = $Port; $candidate -lt $Port + 20; $candidate++) {
    if (-not (Get-PortOwner $candidate)) { return $candidate }
  }
  $Port
}
function Get-Source($Record) {
  if ($Record.Repo) { return $Record.Repo }
  $Record.Path | Split-Path
}

# --- Remembered settings ----------------------------------------------------
$settingsFile = Join-Path $env:LOCALAPPDATA "gufo\start.json"
$settings = [pscustomobject]@{ ModelDirs = @(); Last = $null }
if (Test-Path $settingsFile) {
  try {
    $loaded = Get-Content $settingsFile -Raw | ConvertFrom-Json
    if ($loaded.ModelDirs) { $settings.ModelDirs = @($loaded.ModelDirs) }
    if ($loaded.Last) { $settings.Last = $loaded.Last }
  } catch {
    Write-Host "ignoring unreadable $settingsFile" -ForegroundColor Yellow
  }
}
function Save-Settings {
  $directory = Split-Path $settingsFile
  if (-not (Test-Path $directory)) { $null = New-Item -ItemType Directory -Path $directory }
  $settings | ConvertTo-Json -Depth 6 | Set-Content -Path $settingsFile -Encoding UTF8
}
foreach ($dir in $ModelDir) {
  if (-not (Test-Path -LiteralPath $dir -PathType Container)) { throw "-ModelDir $dir is not a folder" }
  $full = (Resolve-Path -LiteralPath $dir).Path
  if ($settings.ModelDirs -notcontains $full) {
    $settings.ModelDirs = @($settings.ModelDirs) + $full
    Save-Settings
    Write-Host "will also search $full from now on"
  }
}

# --- The build --------------------------------------------------------------
if (-not $Bin) { $Bin = Join-Path $GufoRoot "build\release" }
$exe = Join-Path $Bin "gufo.exe"
if (-not $List -and -not (Test-Path $exe)) {
  Write-Host "gufo.exe not found in $Bin." -ForegroundColor Yellow
  Write-Host "Build it with: powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1"
  Write-Host "(tools\windows\check.ps1 lists anything the build still needs)"
  if ($DryRun -or -not (Ask-YesNo "Build now? It takes a while" $false)) { exit 1 }
  & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build.ps1")
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $exe)) { throw "the build failed; see the output above" }
}
# A build older than the checked-out source is the usual "fixed upstream,
# still broken here" trap after a git pull.
if ((Test-Path $exe) -and (Get-Command git -ErrorAction SilentlyContinue) -and (Test-Path (Join-Path $GufoRoot ".git"))) {
  $headTime = & git -C $GufoRoot log -1 --format=%ct -- src compat cmake CMakeLists.txt CMakePresets.json 2>$null
  if ($headTime) {
    $head = [DateTimeOffset]::FromUnixTimeSeconds([int64]$headTime).UtcDateTime
    $built = (Get-Item $exe).LastWriteTimeUtc
    if ($head -gt $built.AddMinutes(1)) {
      Write-Host ("note: gufo.exe was built {0:yyyy-MM-dd HH:mm}, before the last source commit ({1:yyyy-MM-dd HH:mm}); rebuild with tools\windows\build.ps1" -f $built.ToLocalTime(), $head.ToLocalTime()) -ForegroundColor Yellow
    }
  }
}

# --- Launch (shared by -Last and the menu) ------------------------------------
function Start-Gufo($Plan) {
  $gufoArgs = @($Plan.Args)
  $port = [int]$Plan.Port
  $owner = Get-PortOwner $port
  if ($owner) {
    $free = Get-FreePort ($port + 1)
    Write-Host "port $port is taken by $owner" -ForegroundColor Yellow
    if (-not (Ask-YesNo "Use port $free instead?" $true)) { exit 1 }
    $index = [array]::IndexOf($gufoArgs, "--port")
    $gufoArgs[$index + 1] = "$free"
    $port = $free
  }
  $running = @(Get-Process -Name "gufo", "gufo-a3b" -ErrorAction SilentlyContinue)
  foreach ($process in $running) {
    Write-Host ("gufo is already running: {0} (PID {1}, {2:N1} GiB) and holds GPU memory" -f $process.ProcessName, $process.Id, ($process.WorkingSet64 / 1GB)) -ForegroundColor Yellow
  }
  if ($running.Count -gt 0 -and -not (Ask-YesNo "Start another server anyway?" $false)) { exit 1 }
  if ($Plan.NeedsApiKey -and -not $DryRun) {
    Write-Host "Serving to the local network: Windows may ask once to let gufo.exe through the firewall." -ForegroundColor Yellow
    $key = Read-Host "API key clients must send (Enter = none, anyone on the network can use the server)"
    if ($key) { $gufoArgs = @($gufoArgs[0]) + @("--api-key", $key) + @($gufoArgs | Select-Object -Skip 1) }
  }

  # Variables the server reads. The plan's own are set for this run only;
  # stray GUFO_* / A3B_* switches left in the shell change what runs.
  $planEnv = @{}
  if ($Plan.Env) { foreach ($property in $Plan.Env.PSObject.Properties) { $planEnv[$property.Name] = [string]$property.Value } }
  $stray = @(Get-GufoEnvSwitches | Where-Object { -not $planEnv.ContainsKey($_.Name) })
  $clear = @()
  if ($stray.Count -gt 0) {
    Write-Host "These variables in this shell change how gufo runs:" -ForegroundColor Yellow
    $stray | ForEach-Object { Write-Host "  $($_.Name)=$($_.Value)" -ForegroundColor Yellow }
    if (Ask-YesNo "Ignore them for this run?" $true) { $clear = @($stray | ForEach-Object { $_.Name }) }
  }

  $display = (@("&", (Format-Arg $Plan.Exe)) + @($gufoArgs | ForEach-Object {
    if ($_ -eq $key -and $key) { "<api-key>" } else { Format-Arg $_ } })) -join " "
  $envLine = ($planEnv.GetEnumerator() | ForEach-Object { "`$env:$($_.Key) = `"$($_.Value)`"; " }) -join ""
  Write-Host ""
  Write-Host "$envLine$display" -ForegroundColor DarkGray
  if ($DryRun) { return }

  $saved = @{}
  foreach ($name in @($planEnv.Keys) + $clear) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, "Process") }
  try {
    foreach ($name in $clear) { [Environment]::SetEnvironmentVariable($name, $null, "Process") }
    foreach ($name in $planEnv.Keys) { [Environment]::SetEnvironmentVariable($name, $planEnv[$name], "Process") }
    $address = if ($Plan.Host -eq "0.0.0.0") { "http://<this-pc>:$port/v1" } else { "http://127.0.0.1:$port/v1" }
    Write-Host ("Loading {0}. Ready when the log shows event=load_completed; then the API is {1}, model id `"{2}`". Ctrl+C stops it." -f $Plan.Title, $address, $Plan.ServedName) -ForegroundColor Cyan
    if ($Plan.Log) {
      $logDir = Join-Path $GufoRoot "logs"
      if (-not (Test-Path $logDir)) { $null = New-Item -ItemType Directory -Path $logDir }
      $logFile = Join-Path $logDir ("gufo-{0}-{1:yyyyMMdd-HHmmss}.log" -f $Plan.ServedName, (Get-Date))
      Write-Host "log: $logFile" -ForegroundColor Cyan
      $writer = New-Object System.IO.StreamWriter($logFile, $false, (New-Object System.Text.UTF8Encoding($false)))
      $ErrorActionPreference = "Continue"
      try {
        & $Plan.Exe @gufoArgs 2>&1 | ForEach-Object {
          $line = "$_"
          Write-Host $line
          $writer.WriteLine($line)
          $writer.Flush()
          if ($line -match "event=load_completed") {
            Write-Host "READY: $address (model id `"$($Plan.ServedName)`")" -ForegroundColor Green
          }
        }
      } finally { $writer.Dispose() }
    } else {
      & $Plan.Exe @gufoArgs
    }
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
      Write-Host "gufo.exe exited with code $LASTEXITCODE; tools\windows\check.ps1 checks the setup" -ForegroundColor Red
    }
  } finally {
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], "Process") }
  }
}

if ($Last) {
  $plan = $settings.Last
  if (-not $plan) { throw "no previous launch remembered yet; run start.ps1 without -Last" }
  $missing = @(@($plan.Files) + $plan.Exe | Where-Object { $_ -and -not (Test-Path -LiteralPath $_) })
  if ($missing.Count -gt 0) { throw "the previous launch used files that are gone: $($missing -join ', ')" }
  Write-Host "Previous launch: $($plan.Title) - $($plan.Summary)"
  Start-Gufo $plan
  exit $LASTEXITCODE
}

# --- What is on this machine ----------------------------------------------------
Write-Host "Looking for models..." -ForegroundColor DarkGray
$all = @(Find-GufoGgufs $settings.ModelDirs)
$memory = Get-GufoMemory
$familyOrder = @{ "fn" = 0; "27b" = 1; "a3b" = 2 }
$targets = @($all | Where-Object { $_.Kind -eq "target" } |
  Sort-Object @{ Expression = { $familyOrder[$_.Family] } }, Name, @{ Expression = "Bytes"; Descending = $true })

# Per target: its companions, its speed modes and its recommended context.
function Get-TargetPlanBase($Target) {
  $info = Get-GufoFamilyInfo $Target.Family
  $drafts = @()
  $modes = @()
  $vision = $null
  $note = ""
  $plain = @{ Label = "no speculation (plain decoding)"; Spec = @(); Env = @{}; Draft = $false }
  switch ($Target.Family) {
    "fn" {
      $drafts = @(Get-GufoCompanions $Target $all "mtp" | ForEach-Object { $_.Record })
      $vision = @(Get-GufoCompanions $Target $all "mmproj" | ForEach-Object { $_.Record }) | Select-Object -First 1
      if ($drafts.Count -gt 0) {
        $modes += @{ Label = "MTP + prompt lookup + Latin draft vocab + survival (fastest measured)"; Draft = $true; Env = @{}
          Spec = @("--speculative", "mtp", "--mtp-model", "{draft}", "--prompt-lookup", "--mtp-draft-vocab", "latin", "--mtp-policy", "survival") }
        $modes += @{ Label = "MTP, adaptive, up to 7 drafts (upstream default)"; Draft = $true; Env = @{}
          Spec = @("--speculative", "mtp", "--mtp-model", "{draft}") }
        $modes += @{ Label = "MTP, at most 3 drafts"; Draft = $true; Env = @{}
          Spec = @("--speculative", "mtp", "--mtp-model", "{draft}", "--draft-tokens", "3") }
      } else { $note = "no MTP draft file found, so decode is plain and much slower: $($info.DraftDownload)" }
    }
    "27b" {
      $drafts = @(Get-GufoCompanions $Target $all "dflash" | ForEach-Object { $_.Record })
      $vision = @(Get-GufoCompanions $Target $all "mmproj" | ForEach-Object { $_.Record }) | Select-Object -First 1
      if ($drafts.Count -gt 0) {
        # Prompt lookup for 27B is a GUFO_QWEN27_LOOKUP switch today:
        # DFlash2 keeps its 3-draft cap, copies run up to --draft-tokens.
        $modes += @{ Label = "DFlash2, at most 3 drafts + prompt lookup (fastest measured)"; Draft = $true
          Env = @{ GUFO_QWEN27_LOOKUP = "3" }
          Spec = @("--speculative", "dflash2", "--dflash-model", "{draft}", "--draft-tokens", "7") }
        $modes += @{ Label = "DFlash2, at most 3 drafts"; Draft = $true; Env = @{}
          Spec = @("--speculative", "dflash2", "--dflash-model", "{draft}", "--draft-tokens", "3") }
        $modes += @{ Label = "DFlash2, adaptive, up to 7 drafts (upstream default)"; Draft = $true; Env = @{}
          Spec = @("--speculative", "dflash2", "--dflash-model", "{draft}") }
      } else { $note = "no DFlash2 draft found, so decode is plain (about half the speed): $($info.DraftDownload)" }
    }
    "a3b" {
      $drafts = @(Get-GufoCompanions $Target $all "dflash" | ForEach-Object { $_.Record })
      $vision = @(Get-GufoCompanions $Target $all "mmproj" | ForEach-Object { $_.Record }) | Select-Object -First 1
      $extras = @("--mtp-policy", "survival", "--mtp-draft-vocab", "latin")
      if ($Target.HasMtp) {
        $modes += @{ Label = "MTP 6 + survival + Latin draft vocab + prompt lookup (fastest measured)"; Draft = $false; Env = @{}
          Spec = @("--speculative", "mtp", "--draft-tokens", "6") + $extras + "--prompt-lookup" }
        $modes += @{ Label = "MTP 6 + survival + Latin draft vocab"; Draft = $false; Env = @{}
          Spec = @("--speculative", "mtp", "--draft-tokens", "6") + $extras }
      }
      if ($drafts.Count -gt 0) {
        $modes += @{ Label = "DFlash2 7 + survival + Latin draft vocab + prompt lookup"; Draft = $true; Env = @{}
          Spec = @("--speculative", "dflash2", "--dflash-model", "{draft}", "--draft-tokens", "7") + $extras + "--prompt-lookup" }
      }
      if (-not $Target.HasMtp -and $drafts.Count -eq 0) {
        $note = "this GGUF has no MTP layer and no DFlash2 draft was found, so it decodes plainly; the -MTP repo has one: $($info.DraftDownload)"
      }
    }
  }
  $modes += $plain
  # The largest context up to the model's recommendation that fits the carve-out.
  $contextLimit = if ($Target.NativeContext -gt 0) { $Target.NativeContext } else { 262144 }
  $contexts = @($GufoKnownContexts | Where-Object { $_ -le $contextLimit })
  if ($contexts.Count -eq 0) { $contexts = @($contextLimit) }
  $draft = if ($drafts.Count -gt 0 -and $modes[0].Draft) { $drafts[0] } else { $null }
  $context = $contexts[0]
  foreach ($candidate in $contexts) {
    if ($candidate -gt $info.DefaultContext) { break }
    $need = Get-GufoMemoryEstimate $Target $candidate 1 $draft
    if ((Get-GufoFit $need $memory) -ne "fits" -and $candidate -ne $contexts[0]) { break }
    $context = $candidate
  }
  @{ Info = $info; Drafts = $drafts; Modes = $modes; Vision = $vision; Note = $note; Contexts = $contexts; Context = $context }
}

$gpuLine = if ($memory.GpuName) { "$($memory.GpuName), $($memory.DedicatedGiB) GB dedicated GPU memory, $($memory.RamGiB) GB RAM visible to Windows" } else { "no Radeon GPU found" }
Write-Host ""
Write-Host "Gufo on Windows - $gpuLine" -ForegroundColor Cyan
if ($memory.GpuName -and $memory.DedicatedGiB -lt 64) {
  Write-Host "  tip: raise Variable Graphics Memory (AMD Software > Performance > Tuning); 96 GB is what Flash-Next wants" -ForegroundColor Yellow
}
Write-Host ""

$rows = @()
$index = 0
foreach ($target in $targets) {
  $base = Get-TargetPlanBase $target
  $draft = if ($base.Drafts.Count -gt 0 -and $base.Modes[0].Draft) { $base.Drafts[0] } else { $null }
  $need = Get-GufoMemoryEstimate $target $base.Context 1 $draft
  $fit = Get-GufoFit $need $memory
  $features = @()
  if ($base.Modes.Count -gt 1) { $features += ($base.Modes[0].Label -split "[ ,]")[0] } else { $features += "no draft" }
  if ($base.Vision) { $features += "vision" }
  $problem = ""
  if ($target.MissingShards.Count -gt 0) { $problem = "incomplete download: shard(s) $($target.MissingShards -join ', ') missing" }
  elseif ($fit -eq "too big") { $problem = "needs ~$(Format-GufoGiB $need), more than this machine has" }
  $label = if ($problem) { "  -)" } else { $index++; "{0,3})" -f $index }
  $fitText = switch ($fit) { "fits" { "fits" } "spills" { "spills past the carve-out (slow load)" } default { $fit } }
  $color = if ($problem) { "DarkGray" } elseif ($fit -eq "fits") { "White" } else { "Yellow" }
  Write-Host ("{0} {1,-22} {2,-11} {3,9}  {4,-14} {5,5} ctx ~{6}  {7}" -f $label, $target.Name, $target.Quant,
    (Format-GufoGiB ($target.Bytes / 1GB)), ($features -join "+"), (Format-Context $base.Context), (Format-GufoGiB $need),
    $(if ($problem) { $problem } else { $fitText })) -ForegroundColor $color
  Write-Host ("      {0}" -f (Get-Source $target)) -ForegroundColor DarkGray
  if (-not $problem) { $rows += @{ Target = $target; Base = $base } }
}
if ($targets.Count -eq 0) { Write-Host "  no model gufo can serve was found" -ForegroundColor Yellow }

$missingFamilies = @("fn", "27b", "a3b") | Where-Object { $family = $_; -not ($targets | Where-Object { $_.Family -eq $family }) }
if ($missingFamilies) {
  Write-Host ""
  Write-Host "Not on this machine (pip install -U huggingface_hub, then):" -ForegroundColor DarkGray
  foreach ($family in $missingFamilies) {
    $info = Get-GufoFamilyInfo $family
    Write-Host ("  {0,-20} {1}" -f $info.Title, $info.Download) -ForegroundColor DarkGray
  }
}
$others = @($all | Where-Object { $_.Kind -eq "other" })
if ($List) {
  if ($others.Count -gt 0) {
    Write-Host ""
    Write-Host "Other GGUF files (not servable here):" -ForegroundColor DarkGray
    $others | ForEach-Object { Write-Host "  $($_.FileName): $($_.Note)" -ForegroundColor DarkGray }
  }
  foreach ($row in $rows) {
    if ($row.Base.Note) { Write-Host "  $($row.Target.Name) $($row.Target.Quant): $($row.Base.Note)" -ForegroundColor Yellow }
  }
  exit 0
}

# --- Pick a model ------------------------------------------------------------------
Write-Host ""
$valid = @(for ($i = 1; $i -le $rows.Count; $i++) { "$i" }) + @("p", "q")
$prompt = "Model number"
if ($settings.Last) {
  Write-Host "  l) the previous launch: $($settings.Last.Title) - $($settings.Last.Summary)"
  $valid += "l"
}
Write-Host "  p) add a folder (or a .gguf path) to search    q) quit"
while ($true) {
  $default = if ($settings.Last) { "l" } elseif ($rows.Count -gt 0) { "1" } else { "p" }
  $choice = (Ask-Choice $prompt $valid $default).ToLowerInvariant()
  if ($choice -eq "q") { exit 0 }
  if ($choice -eq "l") {
    $plan = $settings.Last
    $missing = @(@($plan.Files) + $plan.Exe | Where-Object { $_ -and -not (Test-Path -LiteralPath $_) })
    if ($missing.Count -gt 0) { Write-Host "the previous launch used files that are gone: $($missing -join ', ')" -ForegroundColor Yellow; continue }
    Start-Gufo $plan
    exit $LASTEXITCODE
  }
  if ($choice -eq "p") {
    $path = (Read-Host "Folder or .gguf file").Trim('" ')
    if (-not $path -or -not (Test-Path -LiteralPath $path)) { Write-Host "  not found" -ForegroundColor Yellow; continue }
    $folder = if (Test-Path -LiteralPath $path -PathType Leaf) { Split-Path (Resolve-Path -LiteralPath $path).Path } else { (Resolve-Path -LiteralPath $path).Path }
    & $PSCommandPath -ModelDir $folder -Bin $Bin
    exit $LASTEXITCODE
  }
  $row = $rows[[int]$choice - 1]
  break
}
$target = $row.Target
$base = $row.Base
$info = $base.Info

# --- Speed mode -------------------------------------------------------------------
Write-Host ""
Write-Host "Speed mode for $($target.Name) $($target.Quant):"
for ($i = 0; $i -lt $base.Modes.Count; $i++) { Write-Host ("  {0}) {1}" -f ($i + 1), $base.Modes[$i].Label) }
if ($base.Note) { Write-Host "  note: $($base.Note)" -ForegroundColor Yellow }
$mode = $base.Modes[0]
if ($base.Modes.Count -gt 1) {
  $mode = $base.Modes[[int](Ask-Choice "Mode" @(1..$base.Modes.Count | ForEach-Object { "$_" }) "1") - 1]
}

# --- Settings, with recommended defaults ----------------------------------------------
$state = @{
  Context = $base.Context
  Sessions = 1
  Think = "on"
  Port = Get-FreePort 8080
  Host = "127.0.0.1"
  Draft = $(if ($mode.Draft) { $base.Drafts[0] } else { $null })
  Log = $false
  Extra = ""
}
$canSessions = $target.Family -eq "27b"
$samplerOn = @(
  $(if ($null -ne $target.Sampler.Temperature) { $target.Sampler.Temperature } else { 1.0 }),
  $(if ($null -ne $target.Sampler.TopP) { $target.Sampler.TopP } else { 0.95 }),
  $(if ($null -ne $target.Sampler.TopK) { $target.Sampler.TopK } else { 20 }))

function Show-Plan {
  $need = Get-GufoMemoryEstimate $target $state.Context $state.Sessions $state.Draft
  $fit = Get-GufoFit $need $memory
  $sampler = if ($state.Think -eq "on") { "temp {0}, top-p {1}, top-k {2} (the model's own)" -f $samplerOn[0], $samplerOn[1], $samplerOn[2] } else { "temp 0.7, top-p 0.8, top-k 20 (Qwen non-thinking)" }
  Write-Host ""
  Write-Host ("  Model     {0} {1}" -f $target.Name, $target.Quant)
  Write-Host ("            {0}" -f $target.Path) -ForegroundColor DarkGray
  Write-Host ("  Speed     {0}" -f $mode.Label)
  if ($state.Draft) { Write-Host ("            draft {0}" -f $state.Draft.Path) -ForegroundColor DarkGray }
  $sessionText = if ($state.Sessions -gt 1) { "$($state.Sessions) sessions, each" } else { "1 session," }
  Write-Host ("  Context   {0} tokens, {1} ~{2} of {3} GB dedicated: {4}" -f $state.Context, $sessionText, (Format-GufoGiB $need), $memory.DedicatedGiB, $fit) -ForegroundColor $(if ($fit -eq "fits") { "White" } else { "Yellow" })
  Write-Host ("  Thinking  {0}; sampler {1}; clients may override both per request" -f $state.Think, $sampler)
  if ($target.Family -ne "a3b") {
    $visionText = if ($base.Vision) { "images accepted ($($base.Vision.FileName))" } else { "text only (no matching mmproj-BF16.gguf found)" }
    Write-Host ("  Vision    {0}" -f $visionText)
  }
  $address = if ($state.Host -eq "0.0.0.0") { "http://<this-pc>:$($state.Port)/v1 (local network; asks for an API key)" } else { "http://127.0.0.1:$($state.Port)/v1 (this PC only)" }
  Write-Host ("  Server    {0}, model id `"{1}`"" -f $address, $info.ServedName)
  Write-Host ("  Log       {0}" -f $(if ($state.Log) { "console + logs\gufo-$($info.ServedName)-<time>.log" } else { "console only" }))
  if ($state.Extra) { Write-Host ("  Extra     {0}" -f $state.Extra) }
}

function Edit-Plan {
  Write-Host ""
  Write-Host "Context per session (memory is an estimate):"
  $options = @($base.Contexts)
  for ($i = 0; $i -lt $options.Count; $i++) {
    $need = Get-GufoMemoryEstimate $target $options[$i] $state.Sessions $state.Draft
    Write-Host ("  {0}) {1,6} tokens  ~{2}  {3}" -f ($i + 1), (Format-Context $options[$i]), (Format-GufoGiB $need), (Get-GufoFit $need $memory))
  }
  $current = [array]::IndexOf($options, [int]$state.Context) + 1
  $answer = Ask "Pick 1-$($options.Count) or type a token count" $(if ($current -gt 0) { "$current" } else { "$($state.Context)" })
  $value = 0
  if ([int]::TryParse($answer, [ref]$value)) {
    if ($value -ge 1 -and $value -le $options.Count) { $state.Context = $options[$value - 1] }
    elseif ($value -ge 1024) { $state.Context = [math]::Min($value, [math]::Max($target.NativeContext, 1024)) }
  }
  if ($canSessions) {
    $state.Sessions = Ask-Number "Concurrent sessions (1-8; each holds its own KV cache)" $state.Sessions 1 8
  }
  $state.Think = Ask-Choice "Thinking on or off by default" @("on", "off") $state.Think
  if ($state.Draft -and $base.Drafts.Count -gt 1) {
    Write-Host "Draft file:"
    for ($i = 0; $i -lt $base.Drafts.Count; $i++) { Write-Host ("  {0}) {1}" -f ($i + 1), $base.Drafts[$i].Path) }
    $current = [array]::IndexOf($base.Drafts, $state.Draft) + 1
    $state.Draft = $base.Drafts[[int](Ask-Choice "Draft" @(1..$base.Drafts.Count | ForEach-Object { "$_" }) "$current") - 1]
  }
  $state.Port = Ask-Number "Port" $state.Port 1 65535
  $network = Ask-Choice "Who may connect: 1) this PC only  2) the local network" @("1", "2") $(if ($state.Host -eq "0.0.0.0") { "2" } else { "1" })
  $state.Host = if ($network -eq "2") { "0.0.0.0" } else { "127.0.0.1" }
  $state.Log = Ask-YesNo "Also save the log to a file" $state.Log
  $extra = Read-Host "Extra gufo options, e.g. --max-tokens 32768 (Enter = $(if ($state.Extra) { $state.Extra } else { 'none' }); - clears)"
  if ($extra -eq "-") { $state.Extra = "" } elseif ($extra) { $state.Extra = $extra.Trim() }
}

while ($true) {
  Show-Plan
  $next = Ask-Choice "Enter = start, c = change settings, d = print the command only, q = quit" @("s", "c", "d", "q") "s"
  if ($next -eq "q") { exit 0 }
  if ($next -eq "c") { Edit-Plan; continue }
  if ($next -eq "d") { $DryRun = $true }
  $need = Get-GufoMemoryEstimate $target $state.Context $state.Sessions $state.Draft
  if ((Get-GufoFit $need $memory) -eq "too big" -and -not (Ask-YesNo "~$(Format-GufoGiB $need) is more than this machine has; start anyway?" $false)) { continue }
  break
}

# --- The command ---------------------------------------------------------------------
$toString = { param($value) if ($value -is [single] -or $value -is [double]) { $value.ToString($invariant) } else { "$value" } }
$sampling = if ($state.Think -eq "on") {
  @("--temperature", (& $toString $samplerOn[0]), "--top-p", (& $toString $samplerOn[1]), "--top-k", (& $toString $samplerOn[2]), "--min-p", "0")
} else {
  @("--temperature", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0")
}
$server = @("serve", "--host", $state.Host, "--port", "$($state.Port)", "--sessions", "$($state.Sessions)")
if ($base.Vision) { $server += @("--max-request-bytes", "33554432") }   # base64 images next to a long chat
$llm = @("llm", "--model", $target.Path, "--served-model-name", $info.ServedName,
  "--context", "$($state.Context)", "--max-output-bytes", "8388608", "--think", $state.Think) + $sampling
$llm += @($mode.Spec | ForEach-Object { if ($_ -eq "{draft}") { $state.Draft.Path } else { $_ } })
if ($base.Vision) { $llm += @("--mmproj", $base.Vision.Path) }
if ($state.Sessions -gt 1) { $llm += @("--max-pending", "16", "--max-pending-per-client", "16") }
if ($state.Extra) {
  $llm += @([regex]::Matches($state.Extra, '"[^"]*"|\S+') | ForEach-Object { $_.Value.Trim('"') })
}
$files = @($target.Path)
if ($state.Draft) { $files += $state.Draft.Path }
if ($base.Vision) { $files += $base.Vision.Path }
$plan = [pscustomobject]@{
  Title = "$($target.Name) $($target.Quant)"
  Summary = "$($mode.Label), $(Format-Context $state.Context) ctx, thinking $($state.Think), port $($state.Port)"
  Exe = $exe
  Args = @($server + $llm)
  Env = [pscustomobject]$mode.Env
  Files = $files
  Port = $state.Port
  Host = $state.Host
  NeedsApiKey = $state.Host -eq "0.0.0.0"
  Log = $state.Log
  ServedName = $info.ServedName
}
if (-not $DryRun) {
  $settings.Last = $plan
  Save-Settings
}
Start-Gufo $plan
exit $LASTEXITCODE
