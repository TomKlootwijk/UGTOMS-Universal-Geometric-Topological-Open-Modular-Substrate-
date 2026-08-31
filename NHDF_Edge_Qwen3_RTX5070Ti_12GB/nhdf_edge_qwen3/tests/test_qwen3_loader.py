from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch
from torch import nn

from nhdf_edge.runtime import qwen3_loader


class _TinyModel(nn.Module):
    def __init__(
        self,
        *,
        buffer_device: str | None = None,
        cache_implementation: str | None = None,
        cache_config: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.generation_config = SimpleNamespace(
            cache_implementation=cache_implementation,
            cache_config=cache_config,
        )
        if buffer_device is not None:
            self.rotary_emb = nn.Module()
            self.rotary_emb.register_buffer("inv_freq", torch.ones(2, device=buffer_device))


def _manifest(
    *,
    kv_bits: int = 16,
    packed_bytes: int = 0,
    context_tokens: int = 5,
    workspace_gb: float = 0.0,
    reserve_gb: float = 0.0,
    partial_pack: bool = False,
    validation_status: str = "VALIDATED",
) -> dict[str, object]:
    return {
        "config": {
            "model": {"layers": 2, "kv_heads": 3, "head_dim": 4},
            "target": {
                "default_context_tokens": context_tokens,
                "kv_bits": kv_bits,
                "workspace_gb": workspace_gb,
                "runtime_reserve_gb": reserve_gb,
            },
        },
        "summary": {"packed_bytes": packed_bytes},
        "partial_pack": partial_pack,
        "validation": {"status": validation_status},
    }


def _patch_empty_loader(
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, object],
    *,
    model_factory: Callable[[dict[str, Any]], _TinyModel] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "include_buffers_calls": [],
        "active_include_buffers": None,
        "models_built": 0,
    }

    @contextmanager
    def init_empty_weights(*, include_buffers: bool):
        state["include_buffers_calls"].append(include_buffers)
        state["active_include_buffers"] = include_buffers
        try:
            yield
        finally:
            state["active_include_buffers"] = None

    class AutoConfig:
        @staticmethod
        def from_pretrained(path, *, local_files_only: bool):
            assert path.name == "hf_metadata"
            assert local_files_only
            return {"model_type": "tiny-qwen3-moe"}

    class AutoModelForCausalLM:
        @staticmethod
        def from_config(config, *, dtype: torch.dtype):
            state["models_built"] += 1
            state["torch_dtype"] = dtype
            if model_factory is not None:
                return model_factory(state)
            return _TinyModel()

    class AutoTokenizer:
        pass

    class EmptyReader:
        def __init__(self, root, *, verify_crc: bool):
            state["reader_root"] = root
            state["verify_crc"] = verify_crc
            self.manifest = manifest

        @staticmethod
        def names() -> tuple[()]:
            return ()

        def require_validated(self, *, allow_unvalidated: bool = False) -> None:
            status = self.manifest["validation"]["status"]
            if status != "VALIDATED" and not allow_unvalidated:
                raise RuntimeError(
                    f"refusing to execute pack with validation status {status}"
                )

    monkeypatch.setattr(
        qwen3_loader,
        "_optional_imports",
        lambda: (init_empty_weights, AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    )
    monkeypatch.setattr(qwen3_loader, "PackReader", EmptyReader)
    return state


def test_loader_keeps_config_derived_buffers_out_of_meta_context(monkeypatch, tmp_path) -> None:
    def model_factory(state: dict[str, Any]) -> _TinyModel:
        # Mirrors accelerate's include_buffers behavior closely enough to make
        # this test fail if the loader puts derived RoPE buffers on meta again.
        device = "meta" if state["active_include_buffers"] else "cpu"
        return _TinyModel(buffer_device=device)

    state = _patch_empty_loader(monkeypatch, _manifest(), model_factory=model_factory)

    model = qwen3_loader.load_qwen3_moe(
        tmp_path,
        device="cpu",
        require_cuda_extension=False,
    )

    assert state["include_buffers_calls"] == [False]
    assert not model.rotary_emb.inv_freq.is_meta
    assert model.rotary_emb.inv_freq.device.type == "cpu"


def test_loader_rejects_a_remaining_meta_buffer(monkeypatch, tmp_path) -> None:
    _patch_empty_loader(
        monkeypatch,
        _manifest(),
        model_factory=lambda _state: _TinyModel(buffer_device="meta"),
    )

    with pytest.raises(RuntimeError, match=r"meta parameters or buffers.*rotary_emb\.inv_freq"):
        qwen3_loader.load_qwen3_moe(
            tmp_path,
            device="cpu",
            require_cuda_extension=False,
        )


def test_loader_rejects_partial_pack_before_model_construction(monkeypatch, tmp_path) -> None:
    state = _patch_empty_loader(monkeypatch, _manifest(partial_pack=True))

    with pytest.raises(ValueError, match="cannot load a partial NHDF pack"):
        qwen3_loader.load_qwen3_moe(
            tmp_path,
            device="cpu",
            require_cuda_extension=False,
        )

    assert state["models_built"] == 0


def test_loader_rejects_quality_failed_pack_without_explicit_override(monkeypatch, tmp_path) -> None:
    state = _patch_empty_loader(
        monkeypatch,
        _manifest(validation_status="QUALITY_FAILED"),
    )

    with pytest.raises(RuntimeError, match="validation status QUALITY_FAILED"):
        qwen3_loader.load_qwen3_moe(
            tmp_path,
            device="cpu",
            require_cuda_extension=False,
        )

    assert state["models_built"] == 0


def test_loader_allows_unvalidated_pack_only_with_explicit_override(monkeypatch, tmp_path) -> None:
    _patch_empty_loader(monkeypatch, _manifest(validation_status="UNCALIBRATED"))

    model = qwen3_loader.load_qwen3_moe(
        tmp_path,
        device="cpu",
        require_cuda_extension=False,
        allow_unvalidated=True,
    )

    assert isinstance(model, _TinyModel)


def test_cuda_budget_and_int8_kv_cache_defaults(monkeypatch, tmp_path) -> None:
    manifest = _manifest(
        kv_bits=8,
        packed_bytes=1_000,
        context_tokens=5,
        workspace_gb=0.000001,
        reserve_gb=0.000002,
    )
    _patch_empty_loader(monkeypatch, manifest)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (5_000, 10_000))

    model = qwen3_loader.load_qwen3_moe(
        tmp_path,
        device="cuda",
        require_cuda_extension=False,
    )

    # One quantized token includes HQQ scale/zero metadata; four recent
    # residual-window tokens remain FP16.
    assert model.nhdf_runtime_budget == {
        "free_bytes_before_load": 5_000,
        "total_device_bytes": 10_000,
        "pack_bytes": 1_000,
        "kv_cache_bytes": 435,
        "workspace_bytes": 1_000,
        "reserve_bytes": 2_000,
        "required_bytes": 4_435,
        "context_tokens": 5,
        "kv_bits": 8,
    }
    assert model.config.use_cache is True
    assert model.generation_config.cache_implementation == "quantized"
    assert model.generation_config.cache_config == {
        "backend": "hqq",
        "nbits": 8,
        "axis_key": 0,
        "axis_value": 0,
        "q_group_size": 64,
        "residual_length": 128,
    }


def test_cuda_budget_fails_before_model_construction(monkeypatch, tmp_path) -> None:
    state = _patch_empty_loader(
        monkeypatch,
        _manifest(
            kv_bits=8,
            packed_bytes=1_000,
            context_tokens=5,
            workspace_gb=0.000001,
            reserve_gb=0.000002,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (4_434, 10_000))

    with pytest.raises(
        RuntimeError,
        match=r"insufficient free VRAM.*0\.000 GB free.*0\.000 GB required",
    ):
        qwen3_loader.load_qwen3_moe(
            tmp_path,
            device="cuda",
            require_cuda_extension=False,
        )

    assert state["models_built"] == 0


def test_fp16_kv_cache_preserves_transformers_defaults(monkeypatch, tmp_path) -> None:
    _patch_empty_loader(
        monkeypatch,
        _manifest(kv_bits=16),
        model_factory=lambda _state: _TinyModel(
            cache_implementation="dynamic",
            cache_config={"existing": True},
        ),
    )

    model = qwen3_loader.load_qwen3_moe(
        tmp_path,
        device="cpu",
        require_cuda_extension=False,
    )

    assert model.config.use_cache is True
    assert model.generation_config.cache_implementation == "dynamic"
    assert model.generation_config.cache_config == {"existing": True}
    assert model.nhdf_runtime_budget is None


def test_loader_rejects_unsupported_kv_cache_precision(monkeypatch, tmp_path) -> None:
    _patch_empty_loader(monkeypatch, _manifest(kv_bits=4))

    with pytest.raises(ValueError, match="unsupported runtime KV cache precision: 4 bits"):
        qwen3_loader.load_qwen3_moe(
            tmp_path,
            device="cpu",
            require_cuda_extension=False,
        )
