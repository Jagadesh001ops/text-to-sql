"""Tests for the chunked safetensors loader used on Windows (no GPU needed)."""

import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import run_baseline  # noqa: E402


def _make(tmp_path):
    tensors = {
        "a.weight": torch.randn(37, 11, dtype=torch.bfloat16),   # not a multiple of the chunk size
        "b.bias": torch.randn(5, dtype=torch.bfloat16),
        "c.scalar_like": torch.randn(1, dtype=torch.float32),
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    return path, tensors


def test_loads_every_tensor_exactly_across_many_chunks(tmp_path):
    path, tensors = _make(tmp_path)
    params = {k: torch.zeros_like(v) for k, v in tensors.items()}
    loaded: set[str] = set()
    run_baseline.load_safetensors_into(path, params, loaded, chunk_bytes=16)   # tiny chunks
    assert loaded == set(tensors)
    for k, v in tensors.items():
        assert torch.equal(params[k], v), k


def test_shape_or_dtype_mismatch_is_an_error(tmp_path):
    path, tensors = _make(tmp_path)
    params = {k: torch.zeros_like(v) for k, v in tensors.items()}
    params["a.weight"] = torch.zeros(11, 37, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="a.weight"):
        run_baseline.load_safetensors_into(path, params, set())
    params = {k: torch.zeros_like(v) for k, v in tensors.items()}
    params["b.bias"] = torch.zeros(5, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="b.bias"):
        run_baseline.load_safetensors_into(path, params, set())


def test_truncated_file_is_detected(tmp_path):
    path, tensors = _make(tmp_path)
    path.write_bytes(path.read_bytes()[:-8])
    params = {k: torch.zeros_like(v) for k, v in tensors.items()}
    with pytest.raises(RuntimeError, match="truncated"):
        run_baseline.load_safetensors_into(path, params, set())
