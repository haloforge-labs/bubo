# Windows build of Gufo against AMD TheRock ROCm (gfx1151).
#
# Prerequisites (paths overridable by parameter; tools\windows\check.ps1
# checks them all):
#   - TheRock 10.0.0 for Windows gfx1151, extracted to C:\TheRock\build:
#     https://stable.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx1151-10.0.0.tar.gz
#     (SHA-256 1293927b06b3b8d4bd7e0265823fb998bc9e0d83c68f33dcfa5d32663b30ce38)
#   - vcpkg at C:\vcpkg with: icu curl openssl libpng libjpeg-turbo (x64-windows)
#   - Visual Studio Build Tools (MSVC STL + Windows SDK; the compiler is TheRock clang)
#   - CMake 3.21+, and ninja on PATH or at C:\tools\ninja\ninja.exe
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1     # release -> build\release\gufo.exe
#   powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test -Target <target>
param(
  [string]$Preset = "release",
  [string]$Target = "",
  [string]$Rocm = "C:\TheRock\build",
  [string]$Vcpkg = "C:\vcpkg",
  [string]$Ninja = "C:\tools\ninja\ninja.exe",
  [int]$Jobs = 0,
  [switch]$Reconfigure,
  [switch]$KeepGoing,
  [string[]]$CMakeArgs = @()
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path

# The port is validated on one TheRock release; a newer clang can round the
# fused kernels differently (see docs\WINDOWS.md).
$validatedRock = "10.0.0"
$rockVersionFile = Join-Path $Rocm ".info\version"
if (-not (Test-Path $rockVersionFile)) {
  throw "TheRock not found at $Rocm (see docs\WINDOWS.md, or run tools\windows\check.ps1)"
}
$rockVersion = (Get-Content $rockVersionFile -TotalCount 1).Trim()
if ($rockVersion -ne $validatedRock) {
  Write-Warning "TheRock $rockVersion at $Rocm; this port is validated on $validatedRock"
}

# Ninja: the -Ninja path, else the one on PATH (as tools\windows\check.ps1 accepts).
if (-not (Test-Path $Ninja)) {
  $onPath = Get-Command ninja -ErrorAction SilentlyContinue
  if (-not $onPath) { throw "ninja not found at $Ninja or on PATH (see tools\windows\check.ps1)" }
  $Ninja = $onPath.Source
}

# Import the MSVC developer environment (headers, libs, link.exe).
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio C++ build tools not found" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$env:PATH = "$(Split-Path $vswhere);$env:PATH"
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
  if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

$env:HIP_PATH = $Rocm.Replace('\', '/')
$env:HIP_DEVICE_LIB_PATH = "$Rocm/lib/llvm/amdgcn/bitcode".Replace('\', '/')
$env:PATH = "$Rocm\bin;$Rocm\lib\llvm\bin;$(Split-Path $Ninja);$env:PATH"
$clang = "$Rocm\lib\llvm\bin\clang.exe".Replace('\', '/')
$clangxx = "$Rocm\lib\llvm\bin\clang++.exe".Replace('\', '/')

$build = Join-Path $root "build\$Preset"
if ($Reconfigure -and (Test-Path "$build\CMakeCache.txt")) {
  Remove-Item "$build\CMakeCache.txt"
}
if (-not (Test-Path "$build\CMakeCache.txt")) {
  cmake -S $root --preset $Preset `
    "-DCMAKE_MAKE_PROGRAM=$($Ninja.Replace('\', '/'))" `
    "-DCMAKE_C_COMPILER=$clang" `
    "-DCMAKE_CXX_COMPILER=$clangxx" `
    "-DCMAKE_HIP_COMPILER=$clangxx" `
    "-DCMAKE_TOOLCHAIN_FILE=$($Vcpkg.Replace('\', '/'))/scripts/buildsystems/vcpkg.cmake" `
    "-DVCPKG_TARGET_TRIPLET=x64-windows" `
    "-DCMAKE_PREFIX_PATH=$($Rocm.Replace('\', '/'))" `
    "-DCMAKE_LINKER_TYPE=LLD" `
    @CMakeArgs
  if ($LASTEXITCODE -ne 0) { throw "configure failed" }
}

$buildArgs = @("--build", $build)
if ($Target) { $buildArgs += @("--target", $Target) }
if ($Jobs -gt 0) { $buildArgs += @("--parallel", $Jobs) }
if ($KeepGoing) { $buildArgs += @("--", "-k", "0") }
cmake @buildArgs
if ($LASTEXITCODE -ne 0) { throw "build failed" }

# Runtime DLLs next to gufo.exe: ROCm (amdhip64, hipblas, hipblaslt, rocblas
# and their kernel libraries) and vcpkg's (icu, curl, ssl, png, jpeg, zlib).
$bin = $build
$rocmDlls = @("amdhip64_7.dll", "amd_comgr*.dll", "hipblas.dll", "libhipblaslt.dll", "origami.dll", "rocblas.dll", "rocsolver.dll", "rocsparse.dll", "rocm_kpack.dll", "rocm-openblas*.dll", "hiprtc*.dll")
foreach ($pattern in $rocmDlls) {
  Get-ChildItem "$Rocm\bin\$pattern" -ErrorAction SilentlyContinue | Copy-Item -Destination $bin -Force
}
foreach ($dir in @("hipblaslt", "rocblas")) {
  $libraryDir = Join-Path "$Rocm\bin" $dir
  if (Test-Path $libraryDir) {
    Copy-Item $libraryDir -Destination $bin -Recurse -Force
  }
}
Get-ChildItem "$Vcpkg\installed\x64-windows\bin\*.dll" | Copy-Item -Destination $bin -Force
Write-Host "gufo: built $build"
