# Shared by start.ps1 and check.ps1 (dot-sourced): finds the GGUF files that
# `gufo serve llm` can run on Windows, reads their headers, pairs each target
# with its drafts and vision sidecar, and estimates its device memory.
# Read-only.

$GufoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$GufoKnownContexts = @(32768, 65536, 131072, 262144)

# Where the downloads usually are: the Hugging Face cache, this checkout's
# models folder (and one beside it), and the LM Studio / llama.cpp caches.
function Get-GufoModelRoots([string[]]$Extra = @()) {
  $hub = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE }
         elseif ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" }
         else { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }
  $roots = @(
    @{ Path = $hub; Hub = $true },
    @{ Path = (Join-Path $GufoRoot "models"); Hub = $false },
    @{ Path = (Join-Path (Split-Path $GufoRoot) "models"); Hub = $false },
    @{ Path = (Join-Path $env:USERPROFILE "models"); Hub = $false },
    @{ Path = (Join-Path $env:USERPROFILE ".lmstudio\models"); Hub = $false },
    @{ Path = (Join-Path $env:USERPROFILE ".cache\lm-studio\models"); Hub = $false },
    @{ Path = (Join-Path $env:LOCALAPPDATA "llama.cpp"); Hub = $false }
  )
  foreach ($dir in $Extra) { if ($dir) { $roots += @{ Path = $dir; Hub = $false } } }
  $seen = @{}
  foreach ($root in $roots) {
    if (-not (Test-Path -LiteralPath $root.Path -PathType Container)) { continue }
    $full = (Resolve-Path -LiteralPath $root.Path).Path.TrimEnd('\')
    if ($seen.ContainsKey($full)) { continue }
    $seen[$full] = $true
    $root.Path = $full
    $root
  }
}

# The key/value header of a GGUF, up to the tokenizer keys (everything the
# launcher needs comes before them). $null when the file is not a readable GGUF.
function Read-GgufHeader([string]$Path) {
  $sizes = @{ 0 = 1; 1 = 1; 2 = 2; 3 = 2; 4 = 4; 5 = 4; 6 = 4; 7 = 1; 10 = 8; 11 = 8; 12 = 8 }
  try {
    $stream = [System.IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
  } catch { return $null }
  try {
    $reader = New-Object System.IO.BinaryReader($stream)
    if ($stream.Length -lt 24 -or $reader.ReadUInt32() -ne 0x46554747) { return $null }
    $null = $reader.ReadUInt32()
    $header = @{ "gguf.tensor_count" = $reader.ReadUInt64() }
    $count = $reader.ReadUInt64()
    for ($i = 0; $i -lt $count -and $i -lt 1000; $i++) {
      $key = [Text.Encoding]::UTF8.GetString($reader.ReadBytes([int]$reader.ReadUInt64()))
      if ($key.StartsWith("tokenizer.")) { break }
      $type = $reader.ReadUInt32()
      if ($type -eq 8) {
        $value = [Text.Encoding]::UTF8.GetString($reader.ReadBytes([int]$reader.ReadUInt64()))
      } elseif ($type -eq 9) {
        $elementType = [int]$reader.ReadUInt32()
        $length = $reader.ReadUInt64()
        if ($elementType -eq 8) {
          for ($j = 0; $j -lt $length; $j++) { $null = $stream.Seek([int64]$reader.ReadUInt64(), 'Current') }
        } elseif ($sizes.ContainsKey($elementType)) {
          $null = $stream.Seek([int64]$length * $sizes[$elementType], 'Current')
        } else { break }
        $value = $length
      } elseif ($sizes.ContainsKey([int]$type)) {
        $bytes = $reader.ReadBytes($sizes[[int]$type])
        $value = switch ($type) {
          0 { $bytes[0] } 1 { [sbyte]$bytes[0] } 2 { [BitConverter]::ToUInt16($bytes, 0) }
          3 { [BitConverter]::ToInt16($bytes, 0) } 4 { [BitConverter]::ToUInt32($bytes, 0) }
          5 { [BitConverter]::ToInt32($bytes, 0) } 6 { [BitConverter]::ToSingle($bytes, 0) }
          7 { $bytes[0] -ne 0 } 10 { [BitConverter]::ToUInt64($bytes, 0) }
          11 { [BitConverter]::ToInt64($bytes, 0) } 12 { [BitConverter]::ToDouble($bytes, 0) }
        }
      } else { break }
      $header[$key] = $value
    }
    return $header
  } catch {
    return $null
  } finally {
    $stream.Dispose()
  }
}

# File size, following Hugging Face cache symlinks (their own Length is 0).
function Get-GgufFileBytes($Item) {
  if ($Item.LinkType -eq "SymbolicLink" -and $Item.Target) {
    $target = @($Item.Target)[0]
    if (-not [System.IO.Path]::IsPathRooted($target)) { $target = Join-Path $Item.DirectoryName $target }
    $resolved = Get-Item -LiteralPath $target -ErrorAction SilentlyContinue
    if ($resolved) { return [int64]$resolved.Length }
    return [int64]0
  }
  [int64]$Item.Length
}

function ConvertTo-GufoMatchName([string]$Name) {
  if (-not $Name) { return "" }
  ($Name.ToLowerInvariant() -replace "[^a-z0-9]", "")
}

function Get-GufoQuantLabel([string]$FileName) {
  $match = [regex]::Match($FileName, "(?i)(UD-)?(IQ\d_[A-Z]+|Q\d_K(_[A-Z]+)?|Q\d_\d|BF16|F16|F32|MXFP4\w*)")
  if ($match.Success) { return $match.Value.ToUpperInvariant() }
  ""
}

# Every GGUF under the roots, one record each (split models once, by their
# first shard). In the Hugging Face cache the newest snapshot holding a file wins.
function Find-GufoGgufs([string[]]$ExtraDirs = @()) {
  $records = @()
  $seenPaths = @{}
  $seenHubFiles = @{}
  foreach ($root in (Get-GufoModelRoots $ExtraDirs)) {
    $items = @()
    if ($root.Hub) {
      foreach ($repo in (Get-ChildItem -LiteralPath $root.Path -Directory -Filter "models--*" -ErrorAction SilentlyContinue)) {
        $repoName = ($repo.Name -replace "^models--", "") -replace "--", "/"
        $snapshots = Get-ChildItem -LiteralPath (Join-Path $repo.FullName "snapshots") -Directory -ErrorAction SilentlyContinue |
          Sort-Object LastWriteTime -Descending
        foreach ($snapshot in $snapshots) {
          foreach ($file in (Get-ChildItem -LiteralPath $snapshot.FullName -Recurse -Depth 3 -Filter "*.gguf" -File -ErrorAction SilentlyContinue)) {
            $relative = $file.FullName.Substring($snapshot.FullName.Length + 1)
            $hubKey = "$repoName|$relative"
            if ($seenHubFiles.ContainsKey($hubKey)) { continue }
            $seenHubFiles[$hubKey] = $true
            $items += @{ File = $file; Repo = $repoName; Snapshot = $snapshot.FullName }
          }
        }
      }
    } else {
      foreach ($file in (Get-ChildItem -LiteralPath $root.Path -Recurse -Depth 4 -Filter "*.gguf" -File -ErrorAction SilentlyContinue)) {
        $items += @{ File = $file; Repo = ""; Snapshot = "" }
      }
    }
    foreach ($item in $items) {
      $file = $item.File
      if ($seenPaths.ContainsKey($file.FullName)) { continue }
      $seenPaths[$file.FullName] = $true
      $bytes = Get-GgufFileBytes $file
      $shardCount = 1
      $missing = @()
      $split = [regex]::Match($file.Name, "^(.*)-(\d{5})-of-(\d{5})\.gguf$")
      if ($split.Success) {
        if ([int]$split.Groups[2].Value -ne 1) { continue }
        $shardCount = [int]$split.Groups[3].Value
        for ($n = 2; $n -le $shardCount; $n++) {
          $shard = Join-Path $file.DirectoryName ("{0}-{1:D5}-of-{2}.gguf" -f $split.Groups[1].Value, $n, $split.Groups[3].Value)
          $shardItem = Get-Item -LiteralPath $shard -ErrorAction SilentlyContinue
          if ($shardItem) { $bytes += Get-GgufFileBytes $shardItem } else { $missing += $n }
        }
      }
      $header = Read-GgufHeader $file.FullName
      $records += (New-GufoRecord $file $item.Repo $item.Snapshot $header $bytes $shardCount $missing)
    }
  }
  $records
}

function Get-HeaderValue($Header, [string]$Key, $Default = $null) {
  if ($Header -and $Header.ContainsKey($Key)) { return $Header[$Key] }
  $Default
}

function New-GufoRecord($File, [string]$Repo, [string]$Snapshot, $Header, [int64]$Bytes, [int]$Shards, $Missing) {
  $arch = [string](Get-HeaderValue $Header "general.architecture" "")
  $type = [string](Get-HeaderValue $Header "general.type" "")
  $name = [string](Get-HeaderValue $Header "general.name" "")
  if (-not $name) { $name = [IO.Path]::GetFileNameWithoutExtension($File.Name) -replace "-\d{5}-of-\d{5}$", "" }
  $record = [pscustomobject]@{
    Path = $File.FullName; FileName = $File.Name; Repo = $Repo; Snapshot = $Snapshot
    Bytes = $Bytes; Shards = $Shards; MissingShards = @($Missing)
    Arch = $arch; Name = $name; Quant = (Get-GufoQuantLabel $File.Name)
    Kind = "other"; Family = ""; Embedding = 0; NativeContext = 0; HasMtp = $false
    MatchName = ""; Header = $Header; Note = ""
    Sampler = @{
      Temperature = Get-HeaderValue $Header "general.sampling.temp" $null
      TopP = Get-HeaderValue $Header "general.sampling.top_p" $null
      TopK = Get-HeaderValue $Header "general.sampling.top_k" $null
    }
  }
  if (-not $Header) {
    $record.Note = "unreadable GGUF header (incomplete download?)"
    return $record
  }
  $embedding = [int](Get-HeaderValue $Header "$arch.embedding_length" 0)
  $record.Embedding = $embedding
  $record.NativeContext = [int](Get-HeaderValue $Header "$arch.context_length" 0)
  $nextn = [int](Get-HeaderValue $Header "$arch.nextn_predict_layers" 0)
  if ($type -eq "mmproj" -or $arch -eq "clip") {
    $record.Kind = "mmproj"
    $record.Embedding = [int](Get-HeaderValue $Header "clip.vision.projection_dim" 0)
    $record.MatchName = ConvertTo-GufoMatchName $name
  } elseif ($arch -eq "dflash") {
    $record.Kind = "dflash"
    $base = Get-HeaderValue $Header "general.base_model.0.name" (Get-HeaderValue $Header "general.base_model.name" (Get-HeaderValue $Header "general.basename" $name))
    $record.MatchName = ConvertTo-GufoMatchName ([string]$base)
    $record.Note = "draft for $base"
  } elseif ($arch -eq "qwen4exp") {
    if ($nextn -gt 0) {
      $record.Kind = "mtp"
      $record.Family = "fn"
    } else {
      $record.Kind = "target"
      $record.Family = "fn"
    }
    $record.MatchName = ConvertTo-GufoMatchName $name
  } elseif ($arch -eq "qwen35" -and $embedding -eq 5120) {
    $record.Kind = "target"
    $record.Family = "27b"
    $record.MatchName = ConvertTo-GufoMatchName $name
  } elseif ($arch -eq "qwen35moe") {
    $record.Kind = "target"
    $record.Family = "a3b"
    $record.HasMtp = $nextn -gt 0
    $record.MatchName = ConvertTo-GufoMatchName $name
  } else {
    $record.Note = if ($arch) { "architecture '$arch' is not served by gufo on Windows" } else { "no architecture in the header" }
  }
  $record
}

# What a family is called and how it is fetched when nothing is on disk.
function Get-GufoFamilyInfo([string]$Family) {
  switch ($Family) {
    "fn" { @{
      Title = "Qwen3.8-Flash-Next"; ServedName = "flash-next"; DefaultContext = 262144
      Download = 'hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf"'
      DraftDownload = 'hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"'
      Overhead = 3.5 } }
    "27b" { @{
      Title = "Qwen3.8-27B"; ServedName = "qwen27b"; DefaultContext = 65536
      Download = 'hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf'
      DraftDownload = 'hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf'
      Overhead = 3.0 } }
    "a3b" { @{
      Title = "Qwen3.6-35B-A3B"; ServedName = "a3b"; DefaultContext = 131072
      Download = 'hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf'
      DraftDownload = 'hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf'
      Overhead = 3.0 } }
  }
}

# Candidates for a target, best first: same download, then made for this
# model by name. Any Flash-Next MTP file fits any Flash-Next target; DFlash2
# drafts and vision sidecars must come with the target or name it, and
# sidecars must be BF16.
function Get-GufoCompanions($Target, $All, [string]$Kind) {
  $candidates = @($All | Where-Object {
    $_.Kind -eq $Kind -and $_.MissingShards.Count -eq 0 -and $(
      if ($Kind -eq "mtp") { $_.Family -eq $Target.Family -and $_.Embedding -eq $Target.Embedding }
      else { $_.Embedding -eq $Target.Embedding })
  })
  $scored = foreach ($candidate in $candidates) {
    $score = 0
    if ($Target.Repo -and $candidate.Repo -eq $Target.Repo) { $score += 100 }
    if ($candidate.MatchName -and $Target.MatchName -and
        ($candidate.MatchName.StartsWith($Target.MatchName) -or $Target.MatchName.StartsWith($candidate.MatchName))) { $score += 50 }
    if ($candidate.FileName -match "(?i)shared") { $score += 10 }
    if ($candidate.Quant -match "Q8_0") { $score += 5 }
    if ($Kind -ne "mtp" -and $score -lt 50) { continue }
    if ($Kind -eq "mmproj" -and [int](Get-HeaderValue $candidate.Header "general.file_type" 32) -ne 32) { continue }
    [pscustomobject]@{ Record = $candidate; Score = $score }
  }
  @($scored | Sort-Object -Property @{ Expression = "Score"; Descending = $true }, @{ Expression = { $_.Record.Path }; Descending = $false })
}

# Rough device memory for one server, in GiB: the weights the GPU holds, the
# KV cache (fp16, full-attention layers only) for every session, the draft,
# and a per-model allowance for recurrent state and scratch. Measured
# reference: Flash-Next UD-Q4_K_XL at 262144 = ~89 GiB.
function Get-GufoMemoryEstimate($Target, [int]$Context, [int]$Sessions = 1, $Draft = $null) {
  $info = Get-GufoFamilyInfo $Target.Family
  $header = $Target.Header
  $arch = $Target.Arch
  $weights = $Target.Bytes / 1GB
  # Flash-Next streams its 26.8 GiB n-gram (PLE) table from disk.
  if ($Target.Family -eq "fn") { $weights = [math]::Max(0, $weights - 26.8) }
  $blocks = [int](Get-HeaderValue $header "$arch.block_count" 0) - [int](Get-HeaderValue $header "$arch.nextn_predict_layers" 0)
  $interval = [int](Get-HeaderValue $header "$arch.full_attention_interval" 1)
  if ($interval -lt 1) { $interval = 1 }
  $kvHeads = [int](Get-HeaderValue $header "$arch.attention.head_count_kv" 0)
  $headBytes = [int](Get-HeaderValue $header "$arch.attention.key_length" 0) + [int](Get-HeaderValue $header "$arch.attention.value_length" 0)
  $kvPerToken = [math]::Floor($blocks / $interval) * $kvHeads * $headBytes * 2
  $kv = $kvPerToken * $Context * $Sessions / 1GB
  $draftGiB = if ($Draft) { $Draft.Bytes / 1GB } else { 0 }
  [math]::Round($weights + $kv + $draftGiB + $info.Overhead, 1)
}

# Dedicated (carve-out) GPU memory and system RAM, in GiB.
function Get-GufoMemory {
  $result = @{ GpuName = ""; Driver = ""; DedicatedGiB = 0; RamGiB = 0 }
  $gpu = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "Radeon" } | Select-Object -First 1
  if ($gpu) {
    $result.GpuName = $gpu.Name
    $result.Driver = $gpu.DriverVersion
    # The registry holds the 64-bit dedicated size (AdapterRAM caps at 4 GB).
    Get-ChildItem "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}" -ErrorAction SilentlyContinue |
      ForEach-Object {
        $props = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
        if ($props.DriverDesc -eq $gpu.Name -and $props."HardwareInformation.qwMemorySize") {
          $result.DedicatedGiB = [math]::Round([uint64]$props."HardwareInformation.qwMemorySize" / 1GB)
        }
      }
  }
  $system = Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue
  if ($system) { $result.RamGiB = [math]::Round($system.TotalPhysicalMemory / 1GB) }
  $result
}

# "fits" inside the carve-out, "spills" into shared memory (works, but the
# load is slow and decode slower), or "too big" for the machine.
function Get-GufoFit([double]$NeedGiB, $Memory) {
  if ($Memory.DedicatedGiB -le 0) { return "unknown" }
  if ($NeedGiB -le $Memory.DedicatedGiB - 1.5) { return "fits" }
  # Beyond the carve-out HIP reaches Windows' shared GPU memory, half of the
  # RAM Windows sees (~110 GiB in all on a 128 GB machine with 96 carved out).
  if ($NeedGiB -le $Memory.DedicatedGiB + $Memory.RamGiB / 2 - 2) { return "spills" }
  "too big"
}

# GUFO_* and A3B_* switches (and HOST / PORT) change what gufo.exe runs.
function Get-GufoEnvSwitches {
  @(Get-ChildItem Env: | Where-Object { $_.Name -like "GUFO_*" -or $_.Name -like "A3B_*" -or $_.Name -eq "HOST" -or $_.Name -eq "PORT" })
}

function Format-GufoGiB([double]$Value) { "{0:N1} GiB" -f $Value }
