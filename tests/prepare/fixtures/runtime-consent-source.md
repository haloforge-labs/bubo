# `bubo prepare` runtime step: consent screen and install documentation
Source: AMD ROCm 10.0.0 package metadata and licence notices. The consent text discloses that other terms can apply. Tests verify the package hashes against the compiled pins.

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

Licenses: AMD licenses ROCm per component. Components can have third-party terms.
  AMD license page: https://rocm.docs.amd.com/en/docs-10.0.0/about/license.html
  License texts that come with the packages (COMGR, hipcc) are saved in: <licenses folder>
  Component licenses (MIT, Apache-2.0 with LLVM exception, BSD): see <licenses folder>\NOTICE-windows.md
Other AMD terms can apply to these files. Bubo cannot confirm which. If you continue, you accept the terms that apply to the AMD files.

Continue? [y/N]
```
(`--yes` accepts the displayed plan, so the text is printed in the log in that case.)

## Install-doc paragraph
Bubo does not include AMD files. The runtime step of `bubo prepare` downloads AMD's ROCm 10.0.0 packages for `gfx1151` directly from `stable.repo.amd.com`. Before it downloads, it shows the address, size and SHA-256 value of each package, and the licence information. AMD licenses ROCm per component, and components can have third-party terms. Read the [AMD ROCm 10.0.0 licence page](https://rocm.docs.amd.com/en/docs-10.0.0/about/license.html) and the licence texts that Bubo saves in the `licenses` folder. Other AMD terms can apply to these files. If you continue, you accept the terms that apply.

## Disclosure policy
Show source URLs, sizes, SHA-256 values and licence links before acquisition.
State that other AMD terms can apply and that continuing accepts applicable terms.
