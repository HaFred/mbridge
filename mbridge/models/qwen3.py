# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from ..core import register_model
from .qwen2 import Qwen2Bridge


@register_model("qwen3")
class Qwen3Bridge(Qwen2Bridge):
    """
    Bridge implementation for Qwen3 models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for Qwen3 models, handling the conversion between
    Hugging Face Qwen3 format and Megatron-Core.
    """

    # Megatron may use either (a) TE-fused attention where input RMSNorm weights live on
    # ``self_attention.linear_qkv.layer_norm_weight``, or (b) unfused layers with
    # ``decoder.layers.{i}.input_layernorm.weight``. HF Qwen3 always uses
    # ``model.layers.{i}.input_layernorm.weight``; map both Megatron layouts.
    _ATTENTION_MAPPING = {
        "input_layernorm.weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
        **Qwen2Bridge._ATTENTION_MAPPING,
    }
    _MLP_MAPPING = {
        "post_attention_layernorm.weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        **Qwen2Bridge._MLP_MAPPING,
    }

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name
        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]
        if (
            ".self_attention." in mcore_weights_name
            or "input_layernorm.weight" in mcore_weights_name
        ):
            return self._weight_name_mapping_attention(mcore_weights_name)
        if (
            "mlp" in mcore_weights_name
            or "post_attention_layernorm.weight" in mcore_weights_name
        ):
            return self._weight_name_mapping_mlp(mcore_weights_name)
        return self._weight_name_mapping_other(mcore_weights_name)

    def _build_config(self):
        """
        Build the configuration for Qwen3 models.

        Configures Qwen3-specific parameters such as QK layer normalization.
        Qwen3 uses layer normalization on query and key tensors.

        Returns:
            TransformerConfig: Configuration object for Qwen3 models
        """
        return self._build_base_config(
            # qwen3
            qk_layernorm=True,
        )
