# Packs the release build into a prebuilt zip for people who do not build:
#   gufo-windows-<version>\
#     start.cmd                    the launcher (number keys)
#     bin\gufo.exe + DLLs          ROCm runtime (gfx1151 kernels), vcpkg libraries,
#                                  and the Visual C++ / OpenMP runtimes (app-local)
#     tools\windows\launcher.ps1, inventory.ps1
#     README.txt, LICENSE, THIRD-PARTY.txt
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\package.ps1 [-Version 2026-10-03] [-Out dist]
param(
  [string]$Version = (Get-Date -Format "yyyy-MM-dd"),
  [string]$Out = ""
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$build = Join-Path $root "build\release"
if (-not $Out) { $Out = Join-Path $root "dist" }
if (-not (Test-Path (Join-Path $build "gufo.exe"))) { throw "build first: tools\windows\build.ps1" }

$name = "gufo-windows-$Version"
$stage = Join-Path $Out $name
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
$bin = Join-Path $stage "bin"
New-Item -ItemType Directory -Force $bin, (Join-Path $stage "tools\windows") | Out-Null

Copy-Item (Join-Path $build "gufo.exe") $bin
Get-ChildItem (Join-Path $build "*.dll") | Copy-Item -Destination $bin
foreach ($dir in "rocblas", "hipblaslt") {
  if (Test-Path (Join-Path $build $dir)) { Copy-Item (Join-Path $build $dir) -Destination $bin -Recurse }
}

# Visual C++ runtime (app-local deployment of the redistributable files) and
# the LLVM OpenMP runtime the ROCm BLAS libraries import.
$redist = Get-ChildItem "${env:ProgramFiles(x86)}\Microsoft Visual Studio", "$env:ProgramFiles\Microsoft Visual Studio" -Recurse -Directory `
  -Filter "Microsoft.VC14*.CRT" -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -match "\\Redist\\MSVC\\[0-9.]+\\x64\\" -and $_.FullName -notmatch "onecore" } |
  Sort-Object FullName -Descending | Select-Object -First 1
if (-not $redist) { throw "Visual C++ redistributable files not found (Visual Studio Build Tools)" }
Get-ChildItem (Join-Path $redist.FullName "*.dll") | Where-Object { $_.Name -match "^(msvcp140|vcruntime140)" } | Copy-Item -Destination $bin
$omp = Get-ChildItem (Join-Path (Split-Path $redist.FullName) "Microsoft.VC14*.OpenMP.LLVM\libomp140.x86_64.dll") -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $omp) { $omp = Get-Item "$env:SystemRoot\System32\libomp140.x86_64.dll" -ErrorAction SilentlyContinue }
if (-not $omp) { throw "libomp140.x86_64.dll not found (install the Visual C++ redistributable)" }
Copy-Item $omp.FullName $bin

Copy-Item (Join-Path $root "start.cmd") $stage
Copy-Item (Join-Path $root "tools\windows\launcher.ps1"), (Join-Path $root "tools\windows\inventory.ps1") (Join-Path $stage "tools\windows")
Copy-Item (Join-Path $root "LICENSE") $stage
Copy-Item (Join-Path $root "tools\windows\prebuilt\README.txt"), (Join-Path $root "tools\windows\prebuilt\THIRD-PARTY.txt") $stage
$commit = (& git -C $root rev-parse --short HEAD 2>$null)
Add-Content (Join-Path $stage "README.txt") "`r`nBuilt from https://github.com/pixmaate/gufo commit $commit ($Version)."

$zip = Join-Path $Out "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
# Windows' bsdtar writes zip; a GNU tar earlier on PATH (Git) does not.
& (Join-Path $env:SystemRoot "System32\tar.exe") -a -c -f $zip -C $Out $name
if ($LASTEXITCODE -ne 0) { throw "zip failed" }
"{0}  {1:N0} MB" -f $zip, ((Get-Item $zip).Length / 1MB)
