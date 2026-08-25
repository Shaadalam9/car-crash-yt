"""Hugging Face model loading with an explicit quantisation fallback."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .shared import log


def _device_map(device: str) -> Any:
    if device.startswith("cuda:"):
        return {"": int(device.split(":", 1)[1])}
    return {"": device}


def load_model_with_fallback(
    loader: Callable[..., Any],
    model_name: str,
    *,
    device: str,
    load_in_4bit: bool,
    label: str,
) -> Any:
    import torch

    base: Dict[str, Any] = {
        "device_map": _device_map(device),
        "torch_dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32,
    }
    if not load_in_4bit:
        return loader(model_name, **base)

    try:
        from transformers import BitsAndBytesConfig

        quantisation = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return loader(model_name, quantization_config=quantisation, **base)
    except Exception as exc:
        log(f"Could not load {label} in 4 bit mode: {exc}")
        log(f"Retrying {label} without quantisation")
        return loader(model_name, **base)

