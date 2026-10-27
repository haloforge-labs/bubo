# Desktop package assembly

`assemble-app.ps1` extends a **clean existing runtime package folder**, without
rebuilding the CLI, finding vcpkg/TheRock, or modifying `package.ps1`. It preserves
existing member bytes except for additive `BUILD-MANIFEST.json` and
`THIRD-PARTY.txt` updates. It does not prepare the AMD runtime or include it in the
archive. A developer folder with an already prepared runtime is not valid input.

Run from the merged release checkout with Go 1.27.2, Node 24.19.0 and Wails
v2.16.0 on PATH. The runtime is the validated package built at the
frozen engine commit F, used byte for byte. The release checkout is F merged with
the GUI branch. Its product version must match the runtime's. The product-version
bump belongs to the engine branch, before F.

```powershell
$commit = (git rev-parse HEAD).Trim()
$short = (git rev-parse --short HEAD).Trim()
$version = (Get-Content tools/release/product_version.txt -Raw).Trim()
$bundle = "C:\path\to\gui-licenses-$short"
$out = "C:\path\to\pkg-kb-$short"
$env:CGO_ENABLED = '0'
tools/release/gui-licenses/generate.ps1 -FrontendDir gui/wails/frontend -Bundle $bundle
Push-Location gui/wails
wails build -skipbindings -m -nosyncgomod -tags production,wv2runtime.error -webview2 error -o Bubo.Control.Web.exe
Pop-Location
tools/windows/assemble-app.ps1 `
  -RuntimePackage "C:\path\to\runtime" `
  -GUIExe gui/wails/build/bin/Bubo.Control.Web.exe -Commit $commit `
  -LicenseBundle $bundle -Out $out -Notes C:\path\to\release-notes.txt `
  -Version $version -Tag "v$version" -MinUpdatableFrom 0.1.0 `
  -RuntimeChanged:$false -RuntimeDownloadBytes 0 -ModelsChanged:$false -ModelsNeedsPrepare ''
$env:BUBO_RELEASE_MANIFEST = "$out\bubo-release.json"
$env:BUBO_RELEASE_ZIP = "$out\bubo-v$version-windows-x64-gfx1151.zip"
Push-Location gui/updater
go test -v -count=1 -run '^TestGeneratedReleasePackage$' .
Pop-Location
```

The Wails build here builds the final frontend after frontend validation. `-skipbindings` avoids
executing the GUI during packaging; the GUI test gate owns binding generation.
Wails 2.16.0 omits VCS stamps in its output, so a prebuilt GUI requires an explicit
`-Commit` declaration from its builder. Any available VCS stamp is checked. The
caller must supply the GUI from the release checkout (HEAD = `-Commit`).
`check-runtime-provenance.ps1` then requires the runtime's source commit F to be
an ancestor of `-Commit`, and the F..`-Commit` diff to touch only `gui/` and the
docs and packaging-tool files allowed by the provenance check. A change to `src/`,
CMake, `package.ps1` or `product_version.txt` after F fails the assembly. The
runtime's members are copied unchanged, and the assembler checks them byte for byte. The helper is built with
`CGO_ENABLED=0` and `-H=windowsgui` from the assembly checkout. Both application
images are inspected as x64 Windows-GUI PE files without execution.

The updater owns the executable layout (`bubo-release -PackageInfo`); assembly,
`check-package.ps1 -RequireGUI`, and staging use it. `-RequireGUI` is opt-in so the
existing CLI packager and its checks remain compatible. The folder gate scans
normal and delay imports for all binaries. The ZIP gate checks shipped names;
the in-package test performs actual extraction, CRC/hash/size/byte-total checks,
path checks, required-image/subsystem checks and build-manifest hash checks.

`build-manifest.ps1 -ExtraFiles` adds arbitrary package-relative binary hashes.
`-ExtendExisting` first verifies every original runtime hash, then appends hashes
without restating the runtime's build/toolchain provenance. The GUI section adds
its source declaration and the runtime ID extracted from the CLI source at the
runtime's recorded revision. That header must exist; there is no fallback ID.

`go run ./cmd/bubo-release` in `gui/updater` can regenerate a manifest from a ZIP.
It accepts `-Version`, `-Tag`, `-Commit`, `-MinUpdatableFrom`, `-Notes`, `-Released`,
`-RuntimeChanged`, `-RuntimeDownloadBytes`, `-ModelsChanged`, `-ModelsNeedsPrepare`,
`-Zip`, `-Out` and `-InstallPathLength`. Version/tag/commit default to packaged
metadata, and mismatches fail. The assembly script runs `date` for `Released`.
The command reports the longest package-relative path and refuses a path budget
that reaches 260 UTF-16 units. No manifest is uploaded or published here.

The assembly script requires every release decision explicitly, with no defaults:

| Parameter | Required value |
|---|---|
| `-RuntimeChanged` | Boolean comparison against the supported prior release; use `:$false` or `:$true` in PowerShell. |
| `-RuntimeDownloadBytes` | Actual engine transfer byte count as a nonnegative 64-bit integer; zero for unchanged runtime. |
| `-ModelsChanged` | Boolean comparison of model payloads against the supported prior release. |
| `-ModelsNeedsPrepare` | Comma-separated catalogue IDs needing preparation; explicitly pass `''` for none. |
| `-Notes` | Path to a nonempty UTF-8 file containing the release-approved, user-facing summary shown on the update card. |

The example's false/zero/empty decisions are valid only after verifying reuse
against the supported prior package. The script forwards each decision explicitly
to `bubo-release`, including false, zero and the empty model list. Do not put build
labels, revisions or qualification instructions in the summary file.

Run the package-independent assembly interface proof with:

```powershell
pwsh -NoProfile -File tools/windows/test-assemble-app.ps1
```

For the structure proof only, `proof-app-package.ps1` derives a CPU seed from an existing
package's member inventory and CPU binaries. Seed files are hard links in
guarded temporary storage, removed at the end. `-StructureOnly` permits the CPU
build's different revision, records its real build revision, and places
`STRUCTURE-ONLY.txt` in the result. It never claims a GPU or release-quality pass.
The legacy-frontend proof GUI and the final-frontend inventory are deliberately
identified in the proof report; regenerate the inventory for the real final GUI.

The licence generator supports `-FrontendRef app-d2-pages` with an installed
node_modules path for structure checks. It reads only the selected lock/metadata/licence files
from that revision into temporary storage and never restores them into the
checkout. The pinned npm scanner supplies licence metadata/texts; the lock's
production flags select the shipped inventory, with a completeness assertion.
The offline go-licenses adapter uses the upstream library and classifier, checks
coverage against `go list -deps` for each binary, and copies module-cache texts.
No source-host HTTP requests are made. Generated notices have short hash-based
names to preserve Windows path headroom. Docs and package notices come from the
same generation run. See generated REVIEW-NOTES for licences needing review.
