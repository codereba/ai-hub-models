# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os

from qai_hub_models.models._shared.deepseekr1.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    DeepSeekR1Base_Quantizable,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_ENCODINGS = "deepseekr1.encodings" # 这是做quantize以后输出的文件
DEFAULT_ENCODINGS_ZIP = DEFAULT_ENCODINGS + ".zip"

NUM_LAYERS = 28
NUM_SPLITS = 3
NUM_LAYERS_PER_SPLIT = 14

# Hugging face repo name and url
HF_REPO_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Minimum memory (RAM+swap) recommended for export.
MIN_MEMORY_RECOMMENDED = 50

# 手机端运行关键是通过Quantize缩小模型体积，来适配手机的内存容量
class DeepSeekR1_Quantizable(DeepSeekR1Base_Quantizable):
    def __init__(self, huggingface_model_name: str = HF_REPO_NAME, *args, **kwargs):
        super().__init__(
            huggingface_model_name=huggingface_model_name,
            min_memory_recommended=MIN_MEMORY_RECOMMENDED,
            *args,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        aimet_encodings: str | None = "DEFAULT",
        huggingface_model_name: str = HF_REPO_NAME,
    ) -> DeepSeekR1_Quantizable:
        """
        Load a pre-trained deepseek-r1 (1.5B) model from Meta via HuggingFace.

        sequence_length:
            Instantiate with this token sequence length input. A longer
            sequence length means the model is capable of processing more
            tokens at once. This can only be set to greater than one to process
            prompts, since responses are auto-regressive in nature and require
            this to be 1.
        context_length:
            Total context length of model. Longer context length means the
            model is more capable of making longer connections in the input
            prompt. However, it also hurts runtime performance (both time-to-
            first-token and tokens-per-second), so this is a tradeoff that may
            depend on the use case.
        aimet_encodings:
            Path to AIMET quantization encodings file.
        huggingface_model_name:
            Name or URL of the HuggingFace model. Change this if you want to
            change the weights.
        """
        # AIMET 是一种做Quantize的工具
        # if aimet_encodings == "DEFAULT":
        #         aimet_encodings = os.path.join(
        #             CachedWebModelAsset.from_asset_store(
        #                 MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS_ZIP
        #             ).fetch(extract=True),
        #             DEFAULT_ENCODINGS,
        #         )

        return cls(
            aimet_encodings=aimet_encodings,
            sequence_length=sequence_length,
            context_length=context_length,
            huggingface_model_name=huggingface_model_name,
        )

    @staticmethod
    def get_output_names(num_hidden_layers: int = NUM_LAYERS):
        return DeepSeekR1Base_Quantizable.get_output_names(
            num_hidden_layers=num_hidden_layers
        )

    @staticmethod
    def get_input_spec(
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> InputSpec:
        '''
        {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 1536,
  "initializer_range": 0.02,
  "intermediate_size": 8960,
  "max_position_embeddings": 131072,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 12,
  "num_hidden_layers": 28,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000,
  "sliding_window": 4096,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.44.0",
  "use_cache": true,
  "use_mrope": false,
  "use_sliding_window": false,
  "vocab_size": 151936
}
        '''
        return DeepSeekR1Base_Quantizable.get_input_spec(
            num_hidden_layers=NUM_LAYERS,
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=1536,
            num_key_value_heads=2,
            num_attention_heads=12,
        )
