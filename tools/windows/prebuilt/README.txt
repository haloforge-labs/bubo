Bubo for Windows - prebuilt
===========================

This archive contains bubo.exe, the matching lazy bubo-engine.dll,
non-AMD runtime dependencies.
Prerequisite: VC++ v14 x64 Redistributable (not shipped).
Bubo does not include AMD files or model weights.

NEEDS
  - Microsoft VC++ v14 x64 Redistributable installed separately:
    https://aka.ms/vs/17/release/vc_redist.x64.exe
    Install the VC++ v14 x64 Redistributable before the PREPARE steps.
    start.cmd and tools\windows\start.ps1 check before invoking bubo.exe;
    they print the link above when the prerequisite is missing.
    Bubo never downloads or installs the redistributable.
  - Windows 11 and an AMD Strix Halo gfx1151 GPU for inference.
  - A current AMD Software: Adrenalin Edition driver installed separately.
  - Local disk space for the prepared runtime and eight-file model package.

PREPARE
  Install the VC++ v14 x64 Redistributable first:
    https://aka.ms/vs/17/release/vc_redist.x64.exe
  There is no checking script entry point for prepare. bubo.exe links the CRT
  dynamically: a direct bubo.exe prepare with the CRT missing shows a Windows
  loader error before main(), so prepare cannot print the installation link.
  Its in-process check is a second line of defence for partially missing CRT.
  Extract the archive to a per-user folder. Open a terminal in bin\ and run:
    .\bubo.exe prepare runtime
    .\bubo.exe prepare model --output "D:\BuboData"
  Bare .\bubo.exe prepare or prepare all selects both steps and displays one
  complete plan before approval. --yes accepts the displayed plan.
  Preparation needs no administrator access, Python, GPU or local model build.
  The runtime step acquires AMD ROCm 10.0.0 for gfx1151 directly from AMD.
  Before transfer it prints the package URLs, sizes, SHA-256 values and licence
  disclosure. Other AMD terms can apply. Continuing accepts applicable terms.
  Full notices are saved in bin\licenses\, including NOTICE-windows.md.
  Keep the runtime wheel cache if offline repair is needed.

  Production model preparation requires approved source pins and model notices.
  An OPEN gate fails before transfer. Public download/provenance qualification
  and the final production HIP package are separate release checks.

EXISTING MODEL
  Place the four exact catalogue-named shards in:
    ROOT\models\huihui-qwen38-flash-next-ud-q4-k-xl-v1\
  Then run prepare model --output ROOT. Existing files are checked and reused.
  --verify forces full SHA-256 checks of all eight model files.
  See bubo prepare --help for the final options and exit codes.

SERVE
  .\bubo.exe serve llm --model "D:\BuboData\models\huihui-qwen38-flash-next-ud-q4-k-xl-v1\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
  The prepared bubo-package.json binds the model sidecars under the default
  profile. --profile stock suppresses twin, draft head and MTP unless positively
  requested; --profile exact forbids those approximation/speculation features.
  --verify-model is an optional full byte check before serving.

TROUBLE
  Configure HTTPS_PROXY=http://host:port if a proxy is required; system PAC
  settings are not read. Keep TLS validation enabled.
  Stop a process holding the runtime DLLs before repair; prepare names holders.
  A corrupt existing model is preserved. Delete the named file and run prepare
  again as instructed by its error message.
