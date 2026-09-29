# Checks everything Gufo needs on Windows, from the build toolchain to the
# models, and says how to fix what is missing. Read-only: installs and
# changes nothing.
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\check.ps1
#
# Sections: build prerequisites, the build itself (runtime files, whether
# HIP sees the GPU, whether it is older than the source), GPU memory, the
# models on this machine (what fits, which drafts and vision sidecars are
# missing), and runtime traps (a busy port, a server already running, stray
# GUFO_* switches). -ModelDir adds folders to the model search.
param(
  [string]$Rocm = "C:\TheRock\build",
  [string]$Vcpkg = "C:\vcpkg",
  [string]$Ninja = "C:\tools\ninja\ninja.exe",
  [string]$Bin = "",
  [string[]]$ModelDir = @()
)
. "$PSScriptRoot\inventory.ps1"
$validatedRock = "10.0.0"
$tarball = "therock-dist-windows-gfx1151-$validatedRock.tar.gz"
$failures = 0
$next = @()

function Report([string]$name, [bool]$ok, [string]$detail, [string]$fix = "") {
  if ($ok) {
    Write-Host ("  ok    {0,-22} {1}" -f $name, $detail)
  } else {
    Write-Host ("  FAIL  {0,-22} {1}" -f $name, $detail) -ForegroundColor Red
    if ($fix) { Write-Host ("        {0,-22} fix: {1}" -f "", $fix) -ForegroundColor Yellow }
    $script:failures++
  }
}
function Note([string]$name, [string]$detail) {
  Write-Host ("  note  {0,-22} {1}" -f $name, $detail) -ForegroundColor Yellow
}
function Section([string]$title) {
  Write-Host ""
  Write-Host $title -ForegroundColor Cyan
}

Write-Host "Gufo on Windows: checking the setup"

# --- Build prerequisites ---------------------------------------------------------
Section "Build prerequisites"
$os = Get-CimInstance Win32_OperatingSystem
Report "Windows x64" ($os.OSArchitecture -match "64" -and [int]$os.BuildNumber -ge 22000) `
  "$($os.Caption) build $($os.BuildNumber)" "Windows 11 x64 is required"

$versionFile = Join-Path $Rocm ".info\version"
if (Test-Path $versionFile) {
  $version = (Get-Content $versionFile -TotalCount 1).Trim()
  Report "TheRock ROCm" $true "$version at $Rocm"
  if ($version -ne $validatedRock) {
    Note "TheRock version" "validated on $validatedRock; other releases may round kernels differently"
  }
  Report "TheRock clang" (Test-Path "$Rocm\lib\llvm\bin\clang++.exe") "$Rocm\lib\llvm\bin\clang++.exe" `
    "re-extract $tarball"
} else {
  Report "TheRock ROCm" $false "not found at $Rocm" `
    "extract https://stable.repo.amd.com/rocm/core/tarball/$tarball to $Rocm"
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = if (Test-Path $vswhere) {
  & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
Report "VS C++ build tools" ([bool]$vs) ($(if ($vs) { $vs } else { "not found" })) `
  "install Visual Studio Build Tools with 'Desktop development with C++'"

$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmake) {
  $cmakeVersion = [version]((& cmake --version | Select-Object -First 1) -replace "[^0-9.]", "")
  Report "CMake" ($cmakeVersion -ge [version]"3.21") "$cmakeVersion" "install CMake 3.21 or newer"
} else {
  Report "CMake" $false "not on PATH" "install CMake 3.21 or newer and add it to PATH"
}
$ninjaExe = if (Test-Path $Ninja) { $Ninja } else { (Get-Command ninja -ErrorAction SilentlyContinue).Source }
Report "Ninja" ([bool]$ninjaExe) ($(if ($ninjaExe) { $ninjaExe } else { "not found" })) `
  "put ninja.exe on PATH or at $Ninja"

if (Test-Path "$Vcpkg\vcpkg.exe") {
  Report "vcpkg" $true $Vcpkg
  $missing = @("icu", "curl", "openssl", "libpng", "libjpeg-turbo") |
    Where-Object { -not (Test-Path "$Vcpkg\installed\x64-windows\share\$_") }
  Report "vcpkg packages" ($missing.Count -eq 0) `
    ($(if ($missing.Count) { "missing: $($missing -join ', ')" } else { "icu curl openssl libpng libjpeg-turbo" })) `
    "$Vcpkg\vcpkg.exe install icu curl openssl libpng libjpeg-turbo --triplet x64-windows"
} else {
  Report "vcpkg" $false "not found at $Vcpkg" "clone https://github.com/microsoft/vcpkg to $Vcpkg and run bootstrap-vcpkg.bat"
}
$prerequisiteFailures = $failures

# --- The build ---------------------------------------------------------------------
Section "Build"
if (-not $Bin) { $Bin = Join-Path $GufoRoot "build\release" }
$exe = Join-Path $Bin "gufo.exe"
$built = $false
if (Test-Path $exe) {
  $built = $true
  $builtAt = (Get-Item $exe).LastWriteTime
  Report "gufo.exe" $true ("{0} (built {1:yyyy-MM-dd HH:mm})" -f $exe, $builtAt)
  # A build older than the checked-out source is the usual "fixed upstream,
  # still broken here" trap after a git pull.
  if ((Get-Command git -ErrorAction SilentlyContinue) -and (Test-Path (Join-Path $GufoRoot ".git"))) {
    $branch = & git -C $GufoRoot rev-parse --abbrev-ref HEAD 2>$null
    $commit = & git -C $GufoRoot log -1 --format="%h %cd" --date=format:"%Y-%m-%d %H:%M" 2>$null
    if ($commit) { Report "source" $true "$branch at $commit" }
    $sourceTime = & git -C $GufoRoot log -1 --format=%ct -- src compat cmake CMakeLists.txt CMakePresets.json 2>$null
    if ($sourceTime) {
      $sourceAt = [DateTimeOffset]::FromUnixTimeSeconds([int64]$sourceTime).LocalDateTime
      if ($sourceAt -gt $builtAt.AddMinutes(1)) {
        Note "build age" ("the source changed {0:yyyy-MM-dd HH:mm}, after this build; rebuild with tools\windows\build.ps1" -f $sourceAt)
        $next += "rebuild: powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1"
      }
    }
    $dirty = @(& git -C $GufoRoot status --porcelain -- src compat cmake CMakeLists.txt 2>$null)
    if ($dirty.Count -gt 0) { Note "source" "$($dirty.Count) uncommitted source change(s); the build may not match any commit" }
  }
  # build.ps1 copies these next to gufo.exe; without them it fails to start.
  $runtime = @("amdhip64_7.dll", "libhipblaslt.dll", "rocblas.dll", "hipblas.dll", "libcurl.dll", "libssl-3-x64.dll", "icuuc78.dll")
  $missingRuntime = @($runtime | Where-Object { -not (Test-Path (Join-Path $Bin $_)) })
  $missingKernels = @("hipblaslt\library", "rocblas\library") | Where-Object { -not (Test-Path (Join-Path $Bin $_)) }
  Report "runtime files" ($missingRuntime.Count -eq 0 -and @($missingKernels).Count -eq 0) `
    ($(if ($missingRuntime.Count -or @($missingKernels).Count) { "missing: $((@($missingRuntime) + @($missingKernels)) -join ', ')" } else { "ROCm, hipBLASLt/rocBLAS kernels and vcpkg DLLs beside gufo.exe" })) `
    "rerun tools\windows\build.ps1 (it copies them)"
  # Ask the engine itself whether HIP sees a gfx1151 (driver + runtime).
  $probe = @(& $exe diagnose --section gpu 2>&1 | ForEach-Object { "$_" })
  $gpuLine = $probe | Where-Object { $_ -match "^GPU Architecture\s*:" } | Select-Object -First 1
  $supported = [bool]($probe | Where-Object { $_ -match "Compatibility Verdict: \[supported\]" })
  Report "HIP sees the GPU" $supported ($(if ($gpuLine) { ($gpuLine -replace "^GPU Architecture\s*:\s*", "") } else { "gufo diagnose found no GPU" })) `
    "update the AMD Software (Adrenalin) driver; run '$exe diagnose' for details"
} elseif ($prerequisiteFailures -eq 0) {
  Report "gufo.exe" $false "not built yet ($Bin)" "powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1"
  $next += "build: powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1"
} else {
  Report "gufo.exe" $false "not built yet; fix the prerequisites above first"
}

# --- GPU memory ---------------------------------------------------------------------
Section "GPU and memory"
$memory = Get-GufoMemory
if ($memory.GpuName) {
  Report "AMD GPU" $true "$($memory.GpuName), driver $($memory.Driver)"
  if ($memory.DedicatedGiB -gt 0) {
    $detail = "$($memory.DedicatedGiB) GB dedicated, $($memory.RamGiB) GB RAM left to Windows"
    if ($memory.DedicatedGiB -lt 96) {
      Note "dedicated GPU memory" "$detail; 96 GB is recommended for Qwen3.8-Flash-Next at 256K context (AMD Software > Performance > Tuning > Variable Graphics Memory)"
    } else {
      Report "dedicated GPU memory" $true $detail
    }
    if ($memory.RamGiB -gt 0 -and $memory.RamGiB -lt 24) {
      Note "system RAM" "only $($memory.RamGiB) GB left to Windows; the Flash-Next load maps ~100 GB of files, so keep other apps small"
    }
  } else {
    Note "dedicated GPU memory" "size unknown; 96 GB is recommended for Qwen3.8-Flash-Next"
  }
} else {
  Report "AMD GPU" $false "no Radeon adapter found" "Gufo needs a gfx1151 GPU (Ryzen AI Max+ 395 / Radeon 8060S)"
}

# --- Models -------------------------------------------------------------------------
Section "Models (Hugging Face cache, models folders, LM Studio$(if ($ModelDir) { ', -ModelDir' }))"
$all = @(Find-GufoGgufs $ModelDir)
$targets = @($all | Where-Object { $_.Kind -eq "target" })
$servable = 0
$familyOrder = @{ "fn" = 0; "27b" = 1; "a3b" = 2 }
foreach ($target in ($targets | Sort-Object @{ Expression = { $familyOrder[$_.Family] } }, Name, Quant)) {
  $info = Get-GufoFamilyInfo $target.Family
  $label = "$($target.Name) $($target.Quant)"
  if ($target.MissingShards.Count -gt 0) {
    Note $info.Title "$label is incomplete: shard(s) $($target.MissingShards -join ', ') missing; rerun: $($info.Download)"
    continue
  }
  $draftKind = if ($target.Family -eq "fn") { "mtp" } else { "dflash" }
  $drafts = @(Get-GufoCompanions $target $all $draftKind)
  $draft = if ($drafts.Count -gt 0) { $drafts[0].Record } else { $null }
  $context = [math]::Min($info.DefaultContext, $(if ($target.NativeContext -gt 0) { $target.NativeContext } else { 262144 }))
  $need = Get-GufoMemoryEstimate $target $context 1 $draft
  $fit = Get-GufoFit $need $memory
  $extras = @()
  if ($target.Family -eq "fn") { $extras += $(if ($draft) { "MTP" } else { "no MTP draft" }) }
  if ($target.Family -eq "27b") { $extras += $(if ($draft) { "DFlash2" } else { "no DFlash2 draft" }) }
  if ($target.Family -eq "a3b") {
    $extras += $(if ($target.HasMtp) { "MTP inside" } elseif ($draft) { "DFlash2" } else { "no MTP layer" })
  } else {
    $vision = @(Get-GufoCompanions $target $all "mmproj")
    $extras += $(if ($vision.Count -gt 0) { "vision" } else { "no vision sidecar" })
  }
  $detail = "$label ($($extras -join ', ')): ~$(Format-GufoGiB $need) at $([int]($context / 1024))K context"
  switch ($fit) {
    "fits" { Report $info.Title $true "$detail, fits"; $servable++ }
    "spills" { Note $info.Title "$detail, spills past the $($memory.DedicatedGiB) GB carve-out (slow load, slower decode; smaller context helps)"; $servable++ }
    "too big" { Note $info.Title "$detail, more than this machine has" }
    default { Report $info.Title $true $detail; $servable++ }
  }
  if ($target.Family -eq "fn" -and -not $draft) { Note "" "no MTP draft file, so decode is plain and much slower: $($info.DraftDownload)" }
  if ($target.Family -eq "27b" -and -not $draft) { Note "" "no DFlash2 draft, so decode is plain (about half the speed): $($info.DraftDownload)" }
  if ($target.Family -eq "a3b" -and -not $target.HasMtp -and -not $draft) { Note "" "this GGUF has no MTP layer; the -MTP repo has one: $($info.DraftDownload)" }
  if ($target.Family -ne "a3b" -and @(Get-GufoCompanions $target $all "mmproj").Count -eq 0) {
    Note "" "images need the matching mmproj-BF16.gguf from the same Hugging Face repo"
  }
}
foreach ($family in @("fn", "27b", "a3b")) {
  if (-not ($targets | Where-Object { $_.Family -eq $family })) {
    $info = Get-GufoFamilyInfo $family
    Write-Host ("  --    {0,-22} not downloaded: {1}" -f $info.Title, $info.Download) -ForegroundColor DarkGray
  }
}
$unreadable = @($all | Where-Object { -not $_.Header })
foreach ($file in $unreadable) { Note "unreadable GGUF" "$($file.Path) (interrupted download?)" }
$unsupported = @($all | Where-Object { $_.Kind -eq "other" -and $_.Header })
if ($unsupported.Count -gt 0) {
  Write-Host ("  --    {0,-22} {1}" -f "other GGUF files", (($unsupported | ForEach-Object { "$($_.FileName) ($($_.Arch))" }) -join ", ")) -ForegroundColor DarkGray
}
if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
  Note "hf (Hugging Face CLI)" "not on PATH; needed for the downloads above: pip install -U huggingface_hub"
}
if ($servable -eq 0) { $next += "download a model (commands above)" }

# --- Runtime --------------------------------------------------------------------------
Section "Runtime"
$listener = Get-NetTCPConnection -State Listen -LocalPort 8080 -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
  $owner = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
  Note "port 8080" "in use by $(if ($owner) { "$($owner.ProcessName) (PID $($owner.Id))" } else { "PID $($listener.OwningProcess)" }); start.ps1 offers the next free port"
} else {
  Report "port 8080" $true "free"
}
$running = @(Get-Process -Name "gufo", "gufo-a3b" -ErrorAction SilentlyContinue)
foreach ($process in $running) {
  Note "server running" ("{0} (PID {1}) holds {2:N1} GiB; a second server competes for GPU memory" -f $process.ProcessName, $process.Id, ($process.WorkingSet64 / 1GB))
}
$switches = @(Get-GufoEnvSwitches)
if ($switches.Count -gt 0) {
  Note "environment" ("set in this shell and read by gufo: " + (($switches | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ", "))
  Note "" "start.ps1 offers to ignore them; to drop them here: Remove-Item Env:NAME"
} else {
  Report "environment" $true "no GUFO_* / A3B_* switches set"
}

# --- Verdict ------------------------------------------------------------------------------
Write-Host ""
if ($failures -gt 0) {
  Write-Host "$failures problem(s) above; each FAIL line says how to fix it." -ForegroundColor Red
} elseif ($built -and $servable -gt 0) {
  Write-Host "Ready: $servable model(s) can be served." -ForegroundColor Green
  $next += "start: powershell -ExecutionPolicy Bypass -File tools\windows\start.ps1"
} else {
  Write-Host "Nothing is broken, but gufo cannot serve anything yet." -ForegroundColor Yellow
}
foreach ($step in $next) { Write-Host "  next: $step" }
if ($failures -gt 0) { exit 1 }
