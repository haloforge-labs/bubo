# Check that a runtime package built at an earlier (gated) commit may ship with
# the release checkout: the runtime commit is an ancestor of the release commit,
# and between them only the GUI and the listed docs and packaging-tool files
# change. Anything else (engine, CLI, tests, CMake, the product version,
# package.ps1) would ship a runtime the gate did not build. Throws with the
# offending paths.
[CmdletBinding()]
param(
  [Parameter(Mandatory)][string]$RuntimeCommit,
  [Parameter(Mandatory)][string]$ReleaseCommit,
  [string]$Repo = "$PSScriptRoot\..\.."
)
$ErrorActionPreference = 'Stop'
foreach ($c in @($RuntimeCommit, $ReleaseCommit)) {
  if ($c -notmatch '^[0-9a-f]{40}$') { throw "Not a full commit: $c" }
}
& git -C $Repo merge-base --is-ancestor $RuntimeCommit $ReleaseCommit
if ($LASTEXITCODE) { throw "Runtime commit $RuntimeCommit is not an ancestor of release commit $ReleaseCommit" }
$changed = @(& git -C $Repo -c core.quotepath=off diff --no-renames --name-only $RuntimeCommit $ReleaseCommit)
if ($LASTEXITCODE) { throw 'Cannot diff the runtime and release commits' }
# The packaging allowlist; everything else must be unchanged.
$allowed = @(
  '^gui/', '^\.gitattributes$', '^docs/CLI-CONTRACT\.md$', '^docs/release/THIRD-PARTY-GUI\.md$',
  '^tools/release/(build-manifest|check-package)\.ps1$', '^tools/release/gui-licenses/',
  '^tools/windows/(assemble-app|proof-app-package|check-runtime-provenance)\.ps1$',
  '^tools/windows/ASSEMBLE-APP\.md$', '^tools/windows/prebuilt/THIRD-PARTY\.txt$'
)
$engine = @($changed | Where-Object { $path = $_; -not @($allowed | Where-Object { $path -match $_ }).Count })
if ($engine.Count) { throw "Release commit changes runtime sources after the runtime commit: $($engine -join ', ')" }
"PASS: runtime $($RuntimeCommit.Substring(0,8)) -> release $($ReleaseCommit.Substring(0,8)): $($changed.Count) changed files, all GUI, docs or packaging tools"
