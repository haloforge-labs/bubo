#!/usr/bin/env python3
"""Capture an independent production-shape MiniMax H3 block-0 oracle.

The raw BF16 tensors remain privately held beside the accepted checkpoint.
Only this reproducible teacher and content hashes belong in the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != _SCRIPT_DIRECTORY
]

import numpy
import torch
import torch.nn.functional as functional
from safetensors import safe_open


MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
REFERENCE_REVISION = "8974cc055ea9c02fcd14cc27dfda3e1027c05153"
HIDDEN = 5376
HEADS = 56
HEAD_DIMENSION = 128
INNER = HEADS * HEAD_DIMENSION
FEED_FORWARD = 14336
ROPE_HALF = 48
ROPE_FREQUENCIES = 16
SLOTS = 6
EPSILON = 1.0e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_tensor(path: Path, tensor: torch.Tensor, dtype: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = tensor.detach().contiguous().cpu()
    if dtype == "BF16":
        if contiguous.dtype != torch.bfloat16:
            contiguous = contiguous.to(torch.bfloat16)
        array = contiguous.view(torch.uint16).numpy().astype("<u2", copy=False)
    elif dtype == "U32":
        array = contiguous.to(torch.int64).numpy().astype("<u4", copy=False)
    else:
        raise ValueError(f"unsupported oracle dtype: {dtype}")
    path.write_bytes(memoryview(numpy.ascontiguousarray(array)).tobytes())
    return {
        "dtype": dtype,
        "shape": list(tensor.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


class Checkpoint:
    def __init__(self, transformer_root: Path, device: str):
        self.root = transformer_root
        self.device = device
        self.index = json.loads(
            (transformer_root / "model.safetensors.index.json").read_text()
        )["weight_map"]

    def tensor(self, name: str) -> torch.Tensor:
        shard_name = self.index.get(name)
        if shard_name is None:
            raise KeyError(f"checkpoint tensor is absent: {name}")
        with safe_open(
            self.root / shard_name, framework="pt", device=self.device
        ) as shard:
            return shard.get_tensor(name)


def production_positions() -> torch.Tensor:
    text_rows = 6
    frames = 22
    latent_height = 16
    latent_width = 16
    video_latent_frames = ((frames - 5) // 17) * 5 + 2
    audio_latent_frames = round(frames * 40 / 24)
    frame_rows = latent_height // 2
    frame_columns = latent_width // 2
    positions: list[tuple[float, float, float]] = []
    for row in range(text_rows):
        positions.append((float(row), 0.0, 0.0))

    spatial = [
        (row * 32.0 / frame_rows, column * 32.0 / frame_columns)
        for row in range(frame_rows)
        for column in range(frame_columns)
    ]
    cursor = float(text_rows)
    low_width = spatial[0][1]
    high_width = spatial[frame_columns - 1][1]
    for index in range(audio_latent_frames):
        positions.append((cursor + index, 0.0, low_width))
    for index in range(audio_latent_frames):
        positions.append((cursor + index, 0.0, high_width))

    frames_per_token = (1, 4, 4, 4, 4)
    temporal = cursor
    for index in range(video_latent_frames):
        for height, width in spatial:
            positions.append((temporal, height, width))
        temporal += (5.0 / 3.0) * frames_per_token[index % 5]
    result = torch.tensor(positions, dtype=torch.float32)
    if result.shape != (528, 3):
        raise RuntimeError(f"unexpected production layout: {result.shape}")
    return result


def rms_adaln(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    modulation: torch.Tensor,
    row_map: torch.Tensor,
    shift_slot: int,
    scale_slot: int,
) -> torch.Tensor:
    values = input_tensor.float()
    inverse = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + EPSILON)
    normalized = values * inverse * weight.float()
    selected = modulation[row_map]
    scale = selected[:, scale_slot, :].float()
    shift = selected[:, shift_slot, :].float()
    return (normalized * (1.0 + scale) + shift).to(torch.bfloat16)


def gated_residual(
    residual: torch.Tensor,
    branch: torch.Tensor,
    modulation: torch.Tensor,
    row_map: torch.Tensor,
    slot: int,
) -> torch.Tensor:
    gate = modulation[row_map, slot, :].float()
    return (residual.float() + branch.float() * gate).to(torch.bfloat16)


def qkv_norm_rope(
    qkv: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = qkv.shape[0]
    grouped = qkv.view(rows, HEADS, 3, HEAD_DIMENSION)
    query_values = grouped[:, :, 0, :].float()
    key_values = grouped[:, :, 1, :].float()
    value = grouped[:, :, 2, :].contiguous()
    query_inverse = torch.rsqrt(
        query_values.square().mean(dim=-1, keepdim=True) + EPSILON
    )
    key_inverse = torch.rsqrt(
        key_values.square().mean(dim=-1, keepdim=True) + EPSILON
    )
    query_norm = query_values * query_inverse * query_weight.float()
    key_norm = key_values * key_inverse * key_weight.float()
    cosine = rope_cos.float()[:, None, :]
    sine = rope_sin.float()[:, None, :]

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        first = tensor[:, :, :ROPE_HALF]
        second = tensor[:, :, ROPE_HALF : ROPE_HALF * 2]
        tail = tensor[:, :, ROPE_HALF * 2 :]
        return torch.cat(
            (
                first * cosine - second * sine,
                second * cosine + first * sine,
                tail,
            ),
            dim=-1,
        ).to(torch.bfloat16)

    return rotate(query_norm), rotate(key_norm), value


def full_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    query_heads = query.transpose(0, 1).float()
    key_heads = key.transpose(0, 1).float()
    value_heads = value.transpose(0, 1).float()
    scores = torch.matmul(query_heads, key_heads.transpose(-1, -2))
    scores *= 1.0 / math.sqrt(HEAD_DIMENSION)
    probabilities = torch.softmax(scores, dim=-1)
    return (
        torch.matmul(probabilities, value_heads)
        .transpose(0, 1)
        .contiguous()
        .view(query.shape[0], INNER)
        .to(torch.bfloat16)
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm torch device is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    checkpoint = Checkpoint(
        args.model_root / "FL2VA" / "transformer", args.device
    )
    started = time.monotonic()

    positions = production_positions().to(device)
    inverse_frequency = checkpoint.tensor("rope.inv_freq").float()
    if inverse_frequency.shape != (ROPE_FREQUENCIES,):
        raise RuntimeError(
            f"unexpected inverse-frequency shape: {inverse_frequency.shape}"
        )
    angles = positions[:, :, None] * inverse_frequency[None, None, :]
    rope_cos = torch.cos(angles).reshape(528, ROPE_HALF).to(torch.bfloat16)
    rope_sin = torch.sin(angles).reshape(528, ROPE_HALF).to(torch.bfloat16)

    indices = torch.arange(528 * HIDDEN, device=device, dtype=torch.float32)
    hidden = (
        torch.sin(indices * 0.0013) * 0.35
        + torch.cos(indices * 0.00017) * 0.15
    ).view(528, HIDDEN).to(torch.bfloat16)
    modulation_indices = torch.arange(
        3 * SLOTS * HIDDEN, device=device, dtype=torch.float32
    )
    modulation = (
        torch.sin(modulation_indices * 0.0031) * 0.035
        + torch.cos(modulation_indices * 0.0007) * 0.015
    ).view(3, SLOTS, HIDDEN).to(torch.bfloat16)
    row_map = torch.empty(528, dtype=torch.int64, device=device)
    row_map[:6] = 1
    row_map[6:80] = 2
    row_map[80:] = 0

    prefix = "blocks.0."
    norm1 = checkpoint.tensor(prefix + "norm1.weight")
    norm2 = checkpoint.tensor(prefix + "norm2.weight")
    query_norm_weight = checkpoint.tensor(prefix + "attn.q_norm.weight")
    key_norm_weight = checkpoint.tensor(prefix + "attn.k_norm.weight")
    qkv_weight = checkpoint.tensor(prefix + "attn.qkv_proj.weight")
    output_weight = checkpoint.tensor(prefix + "attn.out_proj.weight")
    fc1_weight = checkpoint.tensor(prefix + "mlp.fc1.weight")
    fc2_weight = checkpoint.tensor(prefix + "mlp.fc2.weight")

    with torch.inference_mode():
        modulation_attention = rms_adaln(
            hidden, norm1, modulation, row_map, 0, 1
        )
        projected_qkv = functional.linear(modulation_attention, qkv_weight)
        query, key, value = qkv_norm_rope(
            projected_qkv,
            query_norm_weight,
            key_norm_weight,
            rope_cos,
            rope_sin,
        )
        attention_heads = full_attention(query, key, value)
        attention_output = functional.linear(attention_heads, output_weight)
        gated_attention = gated_residual(
            hidden, attention_output, modulation, row_map, 2
        )
        modulation_mlp = rms_adaln(
            gated_attention, norm2, modulation, row_map, 3, 4
        )
        fc1 = functional.linear(modulation_mlp, fc1_weight)
        gate, up = fc1.split(FEED_FORWARD, dim=-1)
        activated = (functional.silu(gate.float()) * up.float()).to(
            torch.bfloat16
        )
        mlp_output = functional.linear(activated, fc2_weight)
        block_output = gated_residual(
            gated_attention, mlp_output, modulation, row_map, 5
        )
    torch.cuda.synchronize(device)

    tensors = {
        "hidden.bf16": (hidden, "BF16"),
        "modulation.bf16": (modulation, "BF16"),
        "row_map.u32": (row_map, "U32"),
        "rope_cos.bf16": (rope_cos, "BF16"),
        "rope_sin.bf16": (rope_sin, "BF16"),
        "modulation_attention.bf16": (modulation_attention, "BF16"),
        "attention_output.bf16": (attention_output, "BF16"),
        "modulation_mlp.bf16": (modulation_mlp, "BF16"),
        "block_output.bf16": (block_output, "BF16"),
    }
    files = {
        name: write_tensor(args.output / name, tensor, dtype)
        for name, (tensor, dtype) in tensors.items()
    }
    for name, (tensor, _) in tensors.items():
        if not torch.isfinite(tensor.float()).all():
            raise RuntimeError(f"oracle tensor contains non-finite values: {name}")

    metadata = {
        "schema": "gufo.minimax-h3-dit-block-golden.v1",
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "teacher": {
            "framework": "torch-direct",
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "device": torch.cuda.get_device_name(device),
            "attention": "explicit-fp32-full-sdpa",
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "block": 0,
        "geometry": {
            "text_rows": 6,
            "width": 256,
            "height": 256,
            "frames": 22,
            "audio_rows": 74,
            "video_rows": 448,
            "rows": 528,
        },
        "arithmetic": {
            "weights": "BF16",
            "activations": "BF16 at operation boundaries",
            "norm_and_attention_accumulation": "FP32",
            "epsilon": EPSILON,
        },
        "files": files,
        "elapsed_seconds": time.monotonic() - started,
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
