# NPU staging and configuration

ParseKnobs requires the immutable release policy capability before interpreting
artifact paths or lane settings. Configuration values cannot enable a denied
profile. Integer and boolean values use strict bounded parsing.

Q8Matrix describes complete CPU Q8_0 storage with 2560 columns and canonical
34-byte blocks. Staging checks byte lengths and offsets, preserves quantized
bits, rejects nonfinite scales and pads short shared-expert rows with zeros.

Linear z selects trailing slices from 16384 rows. Query/gate uses 24 query
heads, two K/V heads and head dimension 256; K/V stays on the GPU.
Shared gate/up matrices must have matching row counts through 1024.

StageQ8 returns independent CPU input bytes. BFP encoding creates the fixed
XRT weight streams; the persistent session provides completion fences.
Only complete 4096-token groups within executor capacity qualify.
