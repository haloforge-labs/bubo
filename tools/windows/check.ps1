# Checks the Windows build and run prerequisites of Gufo and says how to fix
# anything missing. Read-only: installs and changes nothing.
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\check.ps1
param(
  [string]$Rocm = "C:\TheRock\build",
  [string]$Vcpkg = "C:\vcpkg",
  [string]$Ninja = "C:\tools\ninja\ninja.exe"
)
$validatedRock = "10.0.0"
$tarball = "therock-dist-windows-gfx1151-$validatedRock.tar.gz"
$failures = 0

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

Write-Host "Gufo Windows prerequisites"

# Operating system.
$os = Get-CimInstance Win32_OperatingSystem
Report "Windows x64" ($os.OSArchitecture -match "64" -and [int]$os.BuildNumber -ge 22000) `
  "$($os.Caption) build $($os.BuildNumber)" "Windows 11 x64 is required"

# TheRock ROCm.
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

# Visual Studio C++ build tools (headers, libraries, linker environment).
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = if (Test-Path $vswhere) {
  & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
Report "VS C++ build tools" ([bool]$vs) ($(if ($vs) { $vs } else { "not found" })) `
  "install Visual Studio Build Tools with 'Desktop development with C++'"

# CMake and Ninja.
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

# vcpkg and the libraries Gufo links.
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

# The GPU and its dedicated memory (the BIOS / Adrenalin carve-out).
$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "Radeon" } | Select-Object -First 1
if ($gpu) {
  Report "AMD GPU" $true "$($gpu.Name), driver $($gpu.DriverVersion)"
  # The registry holds the 64-bit dedicated size (AdapterRAM caps at 4 GB).
  $dedicated = $null
  Get-ChildItem "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}" -ErrorAction SilentlyContinue |
    ForEach-Object {
      $props = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
      if ($props.DriverDesc -eq $gpu.Name -and $props."HardwareInformation.qwMemorySize") {
        $dedicated = [uint64]$props."HardwareInformation.qwMemorySize"
      }
    }
  if ($dedicated) {
    $gib = [math]::Round($dedicated / 1GB)
    if ($gib -lt 96) {
      Note "dedicated GPU memory" "$gib GB; 96 GB is recommended for Qwen3.8-Flash-Next at 256K context (AMD Software > Performance > Tuning > Variable Graphics Memory)"
    } else {
      Report "dedicated GPU memory" $true "$gib GB"
    }
  } else {
    Note "dedicated GPU memory" "size unknown; 96 GB is recommended for Qwen3.8-Flash-Next"
  }
} else {
  Report "AMD GPU" $false "no Radeon adapter found" "Gufo needs a gfx1151 GPU (Ryzen AI Max+ 395 / Radeon 8060S)"
}

# The model download (docs\WINDOWS.md) uses the Hugging Face CLI.
if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
  Note "hf (Hugging Face CLI)" "not on PATH; needed to download models: pip install -U huggingface_hub"
}

# Stray diagnostic switches change the engine's behaviour.
$diagnostics = @(Get-ChildItem Env: | Where-Object { $_.Name -like "GUFO_*" })
if ($diagnostics.Count -gt 0) {
  Note "GUFO_* variables" ("set in this shell: " + (($diagnostics | ForEach-Object { $_.Name }) -join ", "))
}

if ($failures -eq 0) {
  Write-Host "All prerequisites found. Build: powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1" -ForegroundColor Green
} else {
  Write-Host "$failures prerequisite(s) missing." -ForegroundColor Red
  exit 1
}
