# Packs the release build into a prebuilt zip for people who do not build:
#   bubo-v<product-version>-windows-x64-gfx1151\
#     start.cmd                    the launcher (number keys)
#     bin\bubo.exe + DLLs          lazy engine and vcpkg libraries; VC++ v14 x64 is a prerequisite
#                                  AMD files are acquired by bubo prepare runtime
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

$name = "$($brand.product)-v$Version-windows-x64-gfx1151"
$stage = Join-Path $Out $name
if ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($stage)) -ne $Out) { throw 'Package stage escapes output directory' }
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
$bin = Join-Path $stage "bin"
New-Item -ItemType Directory -Force $bin, (Join-Path $stage "tools\windows") | Out-Null

Copy-Item (Join-Path $build $brand.exe) $bin
if (-not (Test-Path -LiteralPath (Join-Path $build 'bubo-engine.dll'))) {
  throw 'Bubo package requires the matching lazy inference DLL; rebuild bubo'
}
# bubo-engine.dll and the vcpkg z.dll are included by this closed allow-list.
# zlib1.dll is not the runtime name in this vcpkg triplet.
# Audited against normal/delay import closure and source/port dynamic loads.
# Route B: the archive never redistributes AMD runtime DLLs or kernel data.
$bootstrapDlls = @('bubo-engine.dll', 'libcurl.dll', 'libcrypto-3-x64.dll',
  'z.dll', 'icudt78.dll', 'icuin78.dll', 'icuuc78.dll', 'libpng16.dll',
  'jpeg62.dll', 'libwebp.dll', 'libsharpyuv.dll')
foreach ($dependency in $bootstrapDlls) {
  if (-not (Test-Path -LiteralPath (Join-Path $build $dependency) -PathType Leaf)) {
    throw "Missing package dependency: $dependency; rebuild before packaging"
  }
}
Get-ChildItem (Join-Path $build "*.dll") | Where-Object {
  $_.Name -in $bootstrapDlls
} | Copy-Item -Destination $bin

# VC++ v14 x64 Redistributable is a prerequisite, never shipped.

Copy-Item (Join-Path $root "start.cmd") $stage
Copy-Item (Join-Path $root "tools\windows\launcher.ps1"), (Join-Path $root "tools\windows\inventory.ps1"), (Join-Path $root "tools\windows\vc-runtime.ps1") (Join-Path $stage "tools\windows")
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
  # Only the lazy engine may have missing AMD imports before runtime prepare.
  $amdRuntimeDlls = @(Get-Content -LiteralPath (Join-Path $root 'tools\release\runtime-members.tsv') |
    Select-Object -Skip 1 | ForEach-Object { ($_ -split "`t")[2] } |
    Where-Object { $_ -notmatch '/' -and $_ -like '*.dll' })
  $missing = @($closure.Edges | Where-Object {
    $_.Resolution -eq 'missing' -and
    -not $_.From.StartsWith("$env:SystemRoot\", [StringComparison]::OrdinalIgnoreCase) -and
    -not ($entry -eq 'bubo-engine.dll' -and $_.Name -in $amdRuntimeDlls)
  })
  if ($missing.Count) { throw "Package import closure is incomplete: $($missing.Name -join ', ')" }
}
& "$root\tools\release\build-manifest.ps1" -BuildFolder $build -PackageFolder $stage

$zip = Join-Path $Out "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
# Windows' bsdtar writes zip; a GNU tar earlier on PATH (Git) does not.
& (Join-Path $env:SystemRoot "System32\tar.exe") -a -c -f $zip -C $Out $name
if ($LASTEXITCODE -ne 0) { throw "zip failed" }
& "$root\tools\release\check-package.ps1" -PackageZip $zip
if ($LASTEXITCODE -ne 0) { throw 'Release ZIP contents gate failed' }
"{0}  {1:N0} MB" -f $zip, ((Get-Item $zip).Length / 1MB)
