# Packs the release build into a prebuilt zip for people who do not build:
#   bubo-<product-version>-windows-x64\
#     start.cmd                    the launcher (number keys)
#     bin\bubo.exe + DLLs          ROCm runtime (gfx1151 kernels), vcpkg libraries,
#                                  and the Visual C++ runtimes (app-local)
#     tools\windows\launcher.ps1, inventory.ps1
#     README.txt, LICENSE, THIRD-PARTY.txt
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\package.ps1 [-Out dist]
param(
  [string]$Out = "",
  [string]$BuildName = "release",
  [string]$Vcpkg = "C:\vcpkg"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$Version = (Get-Content -LiteralPath (Join-Path $root 'tools\release\product_version.txt') -Raw).Trim()
if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+$') { throw 'Invalid tracked product version' }
$build = Join-Path $root "build\$BuildName"
$brand = Get-Content -LiteralPath (Join-Path $build "generated\brand.json") -Raw | ConvertFrom-Json
if ($brand.product -ne 'bubo' -or $brand.exe -ne 'bubo.exe') { throw 'Bubo package requires GUFO_BRAND=bubo' }
if ($brand.product_version -ne $Version) { throw 'Product version changed; reconfigure and rebuild before packaging' }
if (-not (Test-Path -LiteralPath (Join-Path $build 'prepare_gate.exe'))) {
  throw 'Build prepare_gate before packaging; production source evidence is mandatory'
}
& (Join-Path $build 'prepare_gate.exe') --release (Join-Path $root 'tools\release\prepare-evidence')
if ($LASTEXITCODE -ne 0) { throw 'Prepare release pin/evidence gate failed' }
if (-not $Out) { $Out = Join-Path $root "dist" }
$Out = [IO.Path]::GetFullPath($Out)
if (-not (Test-Path (Join-Path $build $brand.exe))) { throw "build first: tools\windows\build.ps1" }

$name = "$($brand.product)-$Version-windows-x64"
$stage = Join-Path $Out $name
if ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($stage)) -ne $Out) { throw 'Package stage escapes output directory' }
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
$bin = Join-Path $stage "bin"
New-Item -ItemType Directory -Force $bin, (Join-Path $stage "tools\windows") | Out-Null

Copy-Item (Join-Path $build $brand.exe) $bin
if (-not (Test-Path -LiteralPath (Join-Path $build 'bubo-engine.dll'))) {
  throw 'Bubo package requires the matching lazy inference DLL; rebuild bubo'
}
# bubo-engine.dll and the vcpkg z.dll are included by the DLL glob below.
# zlib1.dll is not the runtime name in this vcpkg triplet.
# Audited against normal/delay import closure and source/port dynamic loads.
# Keep AMD runtime files: prepare supplies them; this filter removes only the
# unused vcpkg extras. Recheck the closure before dropping any other library.
$unused = @('libssl-3-x64.dll','legacy.dll','icuio*.dll','icutu*.dll',
  'turbojpeg.dll','spng.dll','libwebpdecoder.dll','libwebpdemux.dll','libwebpmux.dll')
Get-ChildItem (Join-Path $build "*.dll") | Where-Object {
  $file = $_
  -not @($unused | Where-Object { $file.Name -like $_ }).Count
} | Copy-Item -Destination $bin
foreach ($dir in "rocblas", "hipblaslt") {
  if (Test-Path (Join-Path $build $dir)) { Copy-Item (Join-Path $build $dir) -Destination $bin -Recurse }
}

# Visual C++ runtime (app-local deployment of the redistributable files).
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio C++ build tools not found' }
$redist = Get-ChildItem -Path (Join-Path $vs 'VC\Redist\MSVC\*\x64\Microsoft.VC14*.CRT') -Directory |
  Sort-Object FullName -Descending | Select-Object -First 1
if (-not $redist) { throw "Visual C++ redistributable files not found (Visual Studio Build Tools)" }
Get-ChildItem (Join-Path $redist.FullName "*.dll") | Where-Object { $_.Name -match "^(msvcp140|vcruntime140)" } | Copy-Item -Destination $bin

Copy-Item (Join-Path $root "start.cmd") $stage
Copy-Item (Join-Path $root "tools\windows\launcher.ps1"), (Join-Path $root "tools\windows\inventory.ps1") (Join-Path $stage "tools\windows")
Copy-Item (Join-Path $root "LICENSE") $stage
Copy-Item (Join-Path $root "tools\windows\prebuilt\README.txt"), (Join-Path $root "tools\windows\prebuilt\THIRD-PARTY.txt") $stage
New-Item -ItemType Directory -Force (Join-Path $stage "licenses") | Out-Null
Copy-Item (Join-Path $Vcpkg "installed\x64-windows\share\libwebp\copyright") (Join-Path $stage "licenses\libwebp.txt")
Copy-Item (Join-Path $Vcpkg "installed\x64-windows\share\zlib\copyright") (Join-Path $stage "licenses\zlib.txt")
$commit = (& git -C $root rev-parse --short HEAD 2>$null)
Add-Content (Join-Path $stage "README.txt") "`r`nBuilt from https://github.com/pixmaate/gufo commit $commit ($Version)."

& "$root\tools\release\check-package.ps1" -PackageFolder $stage
if ($LASTEXITCODE -ne 0) { throw 'Release import deny-list failed' }
. "$root\tools\release\pe-imports.ps1"
foreach ($entry in @($brand.exe, 'bubo-engine.dll')) {
  # The engine is loaded dynamically, so it is a separate closure root.
  $closure = Get-PeClosure (Join-Path $bin $entry)
  $missing = @($closure.Edges | Where-Object {
    $_.Resolution -eq 'missing' -and -not $_.From.StartsWith("$env:SystemRoot\", [StringComparison]::OrdinalIgnoreCase)
  })
  if ($missing.Count) { throw "Package import closure is incomplete: $($missing.Name -join ', ')" }
}
& "$root\tools\release\build-manifest.ps1" -BuildFolder $build -PackageFolder $stage

$zip = Join-Path $Out "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
# Windows' bsdtar writes zip; a GNU tar earlier on PATH (Git) does not.
& (Join-Path $env:SystemRoot "System32\tar.exe") -a -c -f $zip -C $Out $name
if ($LASTEXITCODE -ne 0) { throw "zip failed" }
"{0}  {1:N0} MB" -f $zip, ((Get-Item $zip).Length / 1MB)
