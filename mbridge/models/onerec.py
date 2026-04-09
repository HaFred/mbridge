# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""OpenOneRec dense causal LMs (e.g. OpenOneRec/OneRec-8B-pro).

The Hub checkpoint is Qwen3-shaped: same layer topology and tensor names as
`Qwen3ForCausalLM`; fields like ``vocab_size`` or ``layer_types`` in
`config.json` differ from base `Qwen/Qwen3-8B` but do not change the
Megatron↔HF weight map handled here.

``model_type: "qwen3"`` loads ``Qwen3Bridge``, which supports both fused-TE
and unfused Megatron norms (e.g. ``decoder.layers.*.input_layernorm.weight``).

This module registers alternate ``model_type`` strings for local configs.
"""

from ..core import register_model
from .qwen3 import Qwen3Bridge


@register_model(
    [
        "onerec",
        "one_rec",
        "openonerec",
        "open_onerec",
    ]
)
class OneRecBridge(Qwen3Bridge):
    """Dense OpenOneRec / Qwen3 finetunes; inherits ``Qwen3Bridge`` mappings."""

    pass
