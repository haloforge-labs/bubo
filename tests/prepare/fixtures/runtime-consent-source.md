# `bubo prepare` runtime step: consent screen text + install-doc paragraph (release thread, 2026-10-11)
Source: codex-jobs\2026-10-11\release-amd-terms\REPORT-20261011-1753.md. Rule: do NOT claim that no other terms apply. Wording is ASD-STE100 style. Binding text for the runtime step (prefill uses it in a test; the hashes below are full and come from release-amd-path REPORT-20261010-1958; prefill verifies them against the compiled pins).

## Consent screen
```
Bubo needs the AMD ROCm runtime. Bubo does not include AMD files.
Bubo downloads three packages directly from AMD (ROCm 10.0.0 for gfx1151):

  rocm_sdk_core            758,050,981 bytes  SHA-256 066195CB0F7DF02E6009FACB17652DD992D13FAE44D87652E5AB928D5B4F0851
    https://stable.repo.amd.com/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-10.0.0-py3-none-win_amd64.whl
  rocm_sdk_libraries       116,831,755 bytes  SHA-256 BFC7A1BD3B6050FB4D36D8AC2EEA80D3AFD70C5604A29142463D6CB890FC73BF
    https://stable.repo.amd.com/rocm/core/whl-next/rocm-sdk-libraries/rocm_sdk_libraries-10.0.0-py3-none-win_amd64.whl
  rocm_sdk_device_gfx1151  139,346,623 bytes  SHA-256 1AB27EBFE61EA3C3FD2C5C1A0E780AE22ED9B662106CE0AEFDA0A30E3B11577F
    https://stable.repo.amd.com/rocm/core/whl-next/rocm-sdk-device-gfx1151/rocm_sdk_device_gfx1151-10.0.0-py3-none-win_amd64.whl
  Total 1,014,229,359 bytes (967.2 MiB). Bubo checks each hash before it extracts a file.

Files go to: <per-user folder>. Bubo needs no administrator access.

Licences: AMD licenses ROCm per component. Components can have third-party terms.
  AMD licence page: https://rocm.docs.amd.com/en/docs-10.0.0/about/license.html
  Licence texts that come with the packages (COMGR, hipcc) are saved in: <licenses folder>
  Component licences (MIT, Apache-2.0 with LLVM exception, BSD): see <licenses folder>\NOTICE-windows.md
Other AMD terms can apply to these files. Bubo cannot confirm which. If you continue, you accept the terms that apply to the AMD files.

Continue? [y/N]
```
(`--yes` accepts the displayed plan, so the text is printed in the log in that case.)

## Install-doc paragraph
Bubo does not include AMD files. The runtime step of `bubo prepare` downloads AMD's ROCm 10.0.0 packages for `gfx1151` directly from `stable.repo.amd.com`. Before it downloads, it shows the address, size and SHA-256 value of each package, and the licence information. AMD licenses ROCm per component, and components can have third-party terms. Read the [AMD ROCm 10.0.0 licence page](https://rocm.docs.amd.com/en/docs-10.0.0/about/license.html) and the licence texts that Bubo saves in the `licenses` folder. Other AMD terms can apply to these files. If you continue, you accept the terms that apply.

## Owner decision (item f): approved 2026-10-11, "yeah" (via operator)
Route B goes ahead with this disclosure: source URLs, sizes, SHA-256, licence links, "AMD terms may apply; continuing accepts them". No "no other terms" claim. The owner may ask AMD later; this does not hold the release.

## Owner-facing residual risk (item f)
AMD's index has no click-through and the wheels carry no licence field. Neither AMD's Software EULA nor the HIP SDK agreement names these wheels; no permission or ban for third-party installers was found. Risk of stating "no extra terms apply": medium-high, so we do not state it. Recommendation: proceed with the disclosure above; optionally ask AMD later.
