# Record the configured offline toolchain and the exact packaged binary bytes.
param(
  [string]$BuildFolder,
  [Parameter(Mandatory)][string]$PackageFolder,
  [string]$Output = '',
  [string[]]$ExtraFiles = @(),
  [switch]$ExtendExisting,
  [switch]$StructureOnly
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath "$PSScriptRoot\..\..").Path
$package = (Resolve-Path -LiteralPath $PackageFolder).Path
if (-not $Output) { $Output = Join-Path $package 'BUILD-MANIFEST.json' }
function Get-ExtraHashes {
  foreach ($relative in $ExtraFiles) {
    $name = [IO.Path]::GetFullPath((Join-Path $package $relative))
    if (-not $name.StartsWith($package + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "Extra file escapes package: $relative" }
    $file = Get-Item -LiteralPath $name
    if ($file.PSIsContainer) { throw "Extra file is a directory: $relative" }
    [pscustomobject]@{path=[IO.Path]::GetRelativePath($package, $name).Replace('\','/'); size=$file.Length; sha256=(Get-FileHash -LiteralPath $name -Algorithm SHA256).Hash.ToLowerInvariant()}
  }
}
if ($ExtendExisting) {
  $manifest = Get-Content -LiteralPath (Join-Path $package 'BUILD-MANIFEST.json') -Raw | ConvertFrom-Json
  if (-not $manifest.files.Count) { throw 'Runtime manifest has no binary hashes' }
  foreach ($record in $manifest.files) {
    $name = [IO.Path]::GetFullPath((Join-Path $package $record.path))
    if (-not $name.StartsWith($package + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Runtime manifest path escapes package' }
    if ((Get-Item -LiteralPath $name).Length -ne $record.size -or (Get-FileHash -LiteralPath $name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.sha256) { throw "Runtime manifest bytes differ: $($record.path)" }
  }
  $extra = @(Get-ExtraHashes)
  if (@($extra | Where-Object { $_.path -in $manifest.files.path }).Count) { throw 'Extra file already has a manifest entry' }
  $manifest.files = @($manifest.files) + $extra
  $manifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Output -Encoding utf8
  Write-Output "PASS: extended BUILD-MANIFEST.json; preserved=$($manifest.files.Count - $extra.Count) added=$($extra.Count) binary hashes"
  return
}
if (-not $BuildFolder) { throw 'BuildFolder is required unless ExtendExisting is used' }
$build = (Resolve-Path -LiteralPath $BuildFolder).Path
$cache = @{}
foreach ($line in Get-Content -LiteralPath (Join-Path $build 'CMakeCache.txt')) {
  if ($line -match '^([^#/:][^:]*):[^=]+=(.*)$') { $cache[$matches[1]] = $matches[2] }
}
$brand = Get-Content -LiteralPath (Join-Path $build 'generated\brand.json') -Raw | ConvertFrom-Json
$commit = (& git -C $root rev-parse HEAD).Trim()
if ($LASTEXITCODE) { throw 'Cannot identify source commit' }
$short = (& git -C $root rev-parse --short HEAD).Trim()
if ($brand.revision -ne $short) {
  if (-not $StructureOnly) { throw 'Build revision differs from HEAD; reconfigure and rebuild' }
  $commit = (& git -C $root rev-parse "$($brand.revision)^{commit}").Trim()
  if ($LASTEXITCODE) { throw 'Cannot resolve CPU build revision for structure proof' }
  $short = $brand.revision
}
$dirty = [bool](& git -C $root status --porcelain=v1)
$compiler = $cache['CMAKE_CXX_COMPILER']
$clang = (& $compiler -v 2>&1 | ForEach-Object { "$_" }) -join "`n"
if ($LASTEXITCODE) { throw 'Cannot identify compiler' }
$flags = [ordered]@{}
foreach ($key in @($cache.Keys | Sort-Object)) {
  if ($key -match '^(GUFO_|ENGINE_ENABLE_HIP$|BUILD_TESTING$|CMAKE_(BUILD_TYPE|CXX_FLAGS.*|C_FLAGS.*|HIP_.*|LINKER_TYPE|TOOLCHAIN_FILE)$|VCPKG_|.*_VERSION.*$|FIND_PACKAGE_MESSAGE_DETAILS_)') {
    $flags[$key] = $cache[$key]
  }
}
$triplet = $cache['VCPKG_TARGET_TRIPLET']
$installed = $cache['VCPKG_INSTALLED_DIR']
$registry = $cache['Z_VCPKG_ROOT_DIR']
if (-not $registry) { $registry = Split-Path (Split-Path (Split-Path $cache['CMAKE_TOOLCHAIN_FILE'])) }
$registryCommit = (& git -C $registry rev-parse HEAD).Trim()
if ($LASTEXITCODE) { throw 'Cannot identify offline vcpkg registry' }
$registryDirty = [bool](& git -C $registry status --porcelain=v1)
$ports = @()
foreach ($block in ((Get-Content -LiteralPath (Join-Path $installed 'vcpkg\status') -Raw) -split '\r?\n\r?\n')) {
  $fields = @{}
  foreach ($line in ($block -split '\r?\n')) {
    if ($line -match '^([^:]+): (.*)$') { $fields[$matches[1]] = $matches[2] }
  }
  if ($fields['Architecture'] -eq $triplet -and $fields['Status'] -eq 'install ok installed') {
    $ports += [pscustomobject]@{name=$fields['Package']; version=$fields['Version']; port_version=$fields['Port-Version']; feature=$fields['Feature']; dependencies=$fields['Depends']}
  }
}
$files = @(Get-ChildItem -LiteralPath $package -Recurse -File | Where-Object {
  $_.Name -eq $brand.exe -or $_.Extension -eq '.dll'
} | Sort-Object FullName)
$exe = @($files | Where-Object Name -EQ $brand.exe)
if ($exe.Count -ne 1) { throw 'Package must contain exactly one release executable' }
$hashes = @($files | ForEach-Object {
  [pscustomobject]@{path=$_.FullName.Substring($package.Length + 1).Replace('\','/'); size=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$hashes += @(Get-ExtraHashes)
. "$PSScriptRoot\pe-imports.ps1"
$closure = Get-PeClosure $exe[0].FullName @($exe[0].DirectoryName)
# Lazy inference is a separate closure root; preserve linked-port evidence.
$enginePath = Join-Path $exe[0].DirectoryName 'bubo-engine.dll'
if (-not (Test-Path -LiteralPath $enginePath)) { throw 'Package inference DLL missing' }
$engineClosure = Get-PeClosure $enginePath @($exe[0].DirectoryName)
$linkedNames = @(@($closure.Edges) + @($engineClosure.Edges) | ForEach-Object Name | Sort-Object -Unique)
$linkedPorts = @()
foreach ($port in ($ports | Where-Object { $_.version })) {
  $list = Join-Path $installed "vcpkg\info\$($port.name)_$($port.version)_$triplet.list"
  # Port revisions are encoded in the package filename when nonzero.
  if (-not (Test-Path -LiteralPath $list)) {
    $matchesList = @(Get-ChildItem -LiteralPath (Join-Path $installed 'vcpkg\info') -Filter "$($port.name)_*_$triplet.list")
    if ($matchesList.Count -ne 1) { throw "Missing installed file list for $($port.name)" }
    $list = $matchesList[0].FullName
  }
  $dlls = @(Get-Content -LiteralPath $list | Where-Object { $_ -match '/bin/[^/]+\.dll$' -and $_ -notmatch '/debug/' } | ForEach-Object { [IO.Path]::GetFileName($_) })
  if (@($dlls | Where-Object { $_ -in $linkedNames }).Count) { $linkedPorts += $port }
}
$baselineFile = Join-Path $registry 'versions\baseline.json'
$manifest = [ordered]@{
  schema_version = 1
  source = [ordered]@{commit=$commit; short_commit=$short; dirty=$dirty}
  product = $brand
  toolchain = [ordered]@{compiler=$compiler; clang_v=$clang; msvc_tools_version=$cache['BUILD_MANIFEST_MSVC_TOOLS_VERSION']; windows_sdk_version=$cache['BUILD_MANIFEST_WINDOWS_SDK_VERSION']}
  cmake = $flags
  vcpkg = [ordered]@{mode=$(if ($cache['VCPKG_MANIFEST_MODE'] -eq 'ON') {'manifest'} else {'classic'}); triplet=$triplet; registry_commit=$registryCommit; registry_dirty=$registryDirty; manifest_baseline=$cache['VCPKG_BASELINE']; registry_baseline_sha256=(Get-FileHash -LiteralPath $baselineFile -Algorithm SHA256).Hash.ToLowerInvariant(); installed_ports=$ports; linked_ports=$linkedPorts}
  files = $hashes
}
if ($StructureOnly) { $manifest['qualification'] = 'structure only, CPU engine, not shippable' }
$manifest | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath $Output -Encoding utf8
$parsed = Get-Content -LiteralPath $Output -Raw | ConvertFrom-Json
$exeHash = @($parsed.files | Where-Object { [IO.Path]::GetFileName($_.path) -eq $brand.exe })[0].sha256
if ($exeHash -ne (Get-FileHash -LiteralPath $exe[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()) { throw 'Manifest executable hash mismatch' }
Write-Output "PASS: BUILD-MANIFEST.json parses; executable SHA256=$exeHash; $($hashes.Count) binary hashes"
