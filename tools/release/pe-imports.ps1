# Offline PE32/PE32+ import reader. No executable is loaded and no SDK is needed.
param(
  [string]$Path,
  [string[]]$SearchPath = @(),
  [string]$Json
)
$ErrorActionPreference = 'Stop'

function Get-PeImports([string]$File) {
  $bytes = [IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $File).Path)
  function Assert-Span([long]$Offset, [long]$Size) {
    if ($Offset -lt 0 -or $Size -lt 0 -or $Offset + $Size -gt $bytes.LongLength) {
      throw "Truncated PE: $File at $Offset (+$Size)"
    }
  }
  function Read-U16([long]$Offset) { Assert-Span $Offset 2; [BitConverter]::ToUInt16($bytes, $Offset) }
  function Read-U32([long]$Offset) { Assert-Span $Offset 4; [BitConverter]::ToUInt32($bytes, $Offset) }
  function Read-U64([long]$Offset) { Assert-Span $Offset 8; [BitConverter]::ToUInt64($bytes, $Offset) }
  if ((Read-U16 0) -ne 0x5a4d) { throw "Not an MZ image: $File" }
  $pe = Read-U32 0x3c
  if ((Read-U32 $pe) -ne 0x4550) { throw "Not a PE image: $File" }
  $sectionCount = Read-U16 ($pe + 6)
  $optionalSize = Read-U16 ($pe + 20)
  $optional = $pe + 24
  Assert-Span $optional $optionalSize
  $magic = Read-U16 $optional
  if ($magic -eq 0x20b) {
    if ($optionalSize -lt 112) { throw "Short PE32+ optional header: $File" }
    $directory = $optional + 112
    $directoryCount = Read-U32 ($optional + 108)
    $imageBase = Read-U64 ($optional + 24)
  } elseif ($magic -eq 0x10b) {
    if ($optionalSize -lt 96) { throw "Short PE32 optional header: $File" }
    $directory = $optional + 96
    $directoryCount = Read-U32 ($optional + 92)
    $imageBase = Read-U32 ($optional + 28)
  } else { throw "Unknown optional header: $File" }
  $headersSize = Read-U32 ($optional + 60)
  $sections = @()
  for ($i = 0; $i -lt $sectionCount; $i++) {
    $s = $optional + $optionalSize + $i * 40
    Assert-Span $s 40
    $sections += [pscustomobject]@{
      Rva = Read-U32 ($s + 12); VirtualSize = Read-U32 ($s + 8)
      RawSize = Read-U32 ($s + 16); Raw = Read-U32 ($s + 20)
    }
  }
  function Convert-Rva([long]$Rva, [long]$Size = 1) {
    if ($Rva -lt $headersSize -and $Rva + $Size -le $headersSize) {
      Assert-Span $Rva $Size; return $Rva
    }
    foreach ($s in $sections) {
      if ($Rva -ge $s.Rva -and $Rva -lt $s.Rva + [Math]::Max($s.RawSize, $s.VirtualSize)) {
        $delta = $Rva - $s.Rva
        if ($delta + $Size -gt $s.RawSize) { throw "RVA has no file data: $File at $Rva" }
        $offset = $s.Raw + $delta
        Assert-Span $offset $Size; return $offset
      }
    }
    throw "Unmapped RVA: $File at $Rva"
  }
  function Read-Name([long]$Rva) {
    $start = Convert-Rva $Rva
    $end = $start
    while ($true) {
      [void](Convert-Rva ($Rva + $end - $start))
      if ($bytes[$end] -eq 0) { break }
      if ($end - $start -gt 4096) { throw "Unterminated import name: $File" }
      $end++
    }
    $name = [Text.Encoding]::ASCII.GetString($bytes, $start, $end - $start)
    if ($name -notmatch '^[A-Za-z0-9_.+-]+$') { throw "Unsafe import name in $File" }
    $name
  }
  $imports = [Collections.Generic.List[object]]::new()
  foreach ($kind in @(@{Index=1; Width=20; NameOffset=12; Kind='import'},
                       @{Index=13; Width=32; NameOffset=4; Kind='delay'})) {
    if ($directoryCount -le $kind.Index) { continue }
    $entry = $directory + $kind.Index * 8
    if ($entry + 8 -gt $optional + $optionalSize) { throw "Invalid data directory: $File" }
    $rva = Read-U32 $entry
    $size = Read-U32 ($entry + 4)
    if ($rva -eq 0) { continue }
    $terminated = $false
    for ($pos = 0; $pos + $kind.Width -le $size; $pos += $kind.Width) {
      $offset = Convert-Rva ($rva + $pos) $kind.Width
      $zero = $true
      for ($j = 0; $j -lt $kind.Width; $j += 4) { if ((Read-U32 ($offset + $j)) -ne 0) { $zero = $false } }
      if ($zero) { $terminated = $true; break }
      [long]$nameRva = Read-U32 ($offset + $kind.NameOffset)
      if ($kind.Kind -eq 'delay' -and ((Read-U32 $offset) -band 1) -eq 0) { $nameRva -= $imageBase }
      $imports.Add([pscustomobject]@{ Name = Read-Name $nameRva; Kind = $kind.Kind })
    }
    if (-not $terminated) { throw "Unterminated $($kind.Kind) directory: $File" }
  }
  $imports.ToArray()
}

function Get-PeClosure([string]$RootFile, [string[]]$Directories = @()) {
  $root = (Resolve-Path -LiteralPath $RootFile).Path
  $system = Join-Path $env:SystemRoot 'System32'
  $search = @((Split-Path $root)) + $Directories + @($system)
  $queue = [Collections.Generic.Queue[string]]::new()
  $queue.Enqueue($root)
  $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
  $nodes = [Collections.Generic.List[object]]::new()
  $edges = [Collections.Generic.List[object]]::new()
  while ($queue.Count) {
    $file = $queue.Dequeue()
    if (-not $seen.Add($file)) { continue }
    $imports = @(Get-PeImports $file)
    $nodes.Add([pscustomobject]@{Path=$file; Name=[IO.Path]::GetFileName($file); Imports=$imports})
    foreach ($import in $imports) {
      $resolved = $null
      # API-set names are Windows loader contracts, not redistributable files.
      $type = if ($import.Name -match '^(api|ext)-ms-') { 'api-set' } else { 'missing' }
      if ($type -ne 'api-set') {
        foreach ($dir in $search) {
          $candidate = Join-Path $dir $import.Name
          if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $resolved = (Resolve-Path -LiteralPath $candidate).Path
            $type = if ($resolved.StartsWith($system + '\', [StringComparison]::OrdinalIgnoreCase)) { 'system' } else { 'file' }
            $queue.Enqueue($resolved)
            break
          }
        }
      }
      $edges.Add([pscustomobject]@{From=$file; Name=$import.Name; Kind=$import.Kind; Resolution=$type; Path=$resolved})
    }
  }
  [pscustomobject]@{ Root=$root; SearchPath=$search; Nodes=$nodes.ToArray(); Edges=$edges.ToArray() }
}

if ($MyInvocation.InvocationName -ne '.' -and $Path) {
  $closure = Get-PeClosure $Path $SearchPath
  if ($Json) { $closure | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $Json -Encoding utf8 }
  foreach ($node in $closure.Nodes) {
    Write-Output $node.Path
    foreach ($edge in $closure.Edges | Where-Object From -EQ $node.Path) {
      Write-Output "  $($edge.Kind): $($edge.Name) [$($edge.Resolution)] $($edge.Path)"
    }
  }
}
