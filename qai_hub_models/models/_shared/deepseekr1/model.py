# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import qai_hub as hub
from qai_hub.public_rest_api import DatasetEntries
from transformers.models.qwen2 import modeling_qwen2

from qai_hub_models.models.common import (
    SampleInputsType,
    SourceModelFormat,
    TargetRuntime,
)
from qai_hub_models.utils.aimet.encodings import map_encodings
from qai_hub_models.utils.huggingface import (
    ensure_has_required_transformer,
    has_model_access,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.system_info import has_recommended_memory

from .model_adaptations import (
    QcDeepSeekR1_apply_rotary_pos_emb,
    SHADynamicCacheNewValueOnly,
    SHADeepSeekR1Attention,
)

from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.aimet.aimet_dummy_model import AimetEncodingLoaderMixin
# from qai_hub_models.utils.quantization_aimet import AIMETQuantizableMixin
# DeepSeekR1 chat template:
# "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}
# {% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}
# {%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}
# {%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}
# {%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}
# {%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}
# {%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}
# {% endif %}"

MIN_TRANFORMER_VERSION = "4.44.0"

# isort: off

# TODO: 10761 remove transformer version check once AIMET
# transformer restriction is uplifted.
ensure_has_required_transformer(MIN_TRANFORMER_VERSION)
from transformers import AutoConfig, AutoTokenizer # noqa: E402
from transformers.modeling_outputs import CausalLMOutputWithPast

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Configs
AIMET_ENCODINGS_PREFIX = "config"
AIMET_CONFIG = "default_config_llama"

DEFAULT_SEQUENCE_LENGTH = 128
DEFAULT_CONTEXT_LENGTH = 4096

DATA_DIR = "data"
USE_CACHED_DATA = True

## Ref: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/blob/main/tokenizer_config.json
'''
   {
      "id": 151643,
      "content": "<｜end▁of▁sentence｜>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 151644,
      "content": "<｜User｜>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151645,
      "content": "<｜Assistant｜>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": false
    },
    {
      "id": 151646,
      "content": "<｜begin▁of▁sentence｜>",
      "normalized": false,
      "lstrip": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    {
      "id": 151647,
      "content": "<|EOT|>",
      "normalized": false,
      "lstrip": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    {
      "id": 151648,
      "content": "<think>",
      "normalized": false,
      "lstrip": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    {
      "id": 151649,
      "content": "</think>",
      "normalized": false,
      "lstrip": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    {
      "id": 151650,
      "content": "<|quad_start|>",
      "normalized": false,
      "lstrip": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    {
      "id": 151651,
      "content": "<|quad_end|>",
      "normalized": false,
      "lstrip": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    {
      "id": 151652,
      "content": "<|vision_start|>",
      "normalized": false,
      "lstrip": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    {
      "id": 151653,
      "content": "<|vision_end|>",
      "normalized": false,
      "lstrip": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    {
      "id": 151654,
      "content": "<|vision_pad|>",
      "normalized": false,
      "lstrip": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    {
      "id": 151655,
      "content": "<|image_pad|>",
      "normalized": false,
      "lstrip": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    {
      "id": 151656,
      "content": "<|video_pad|>",
      "normalized": false,
      "lstrip": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    {
      "id": 151657,
      "content": "<tool_call>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    {
      "id": 151658,
      "content": "</tool_call>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    {
      "id": 151659,
      "content": "<|fim_prefix|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    {
      "id": 151660,
      "content": "<|fim_middle|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    {
      "id": 151661,
      "content": "<|fim_suffix|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    {
      "id": 151662,
      "content": "<|fim_pad|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    {
      "id": 151663,
      "content": "<|repo_name|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    {
      "id": 151664,
      "content": "<|file_sep|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    }
'''
BEGIN_TEXT = "<｜begin▁of▁sentence｜>"
END_TEXT = "<｜end▁of▁sentence｜>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
SYSTEM_ID = "system"
ASSISTANT_ID = "<｜Assistant｜>"
USER_ID = "<｜User｜>"
EOT_ID = "<|EOT|>"
END_TOKENS = {"<|EOT|>", "<|EOT|>", "<｜end▁of▁sentence｜>"}

DEFAULT_PROMPT_CONTEXT = "You are a helpful AI assistant"
DEFAULT_USER_PROMPT = "What do llamas eat? Keep the answer under ten words."


# 这是输入到DeepSeekR1模型的输入模板
def get_input_prompt_with_tags(
    previous_history: str = "",
    system_context_prompt: str = DEFAULT_PROMPT_CONTEXT,
    user_input_prompt: str = DEFAULT_USER_PROMPT,
):
    """
    Get prompt to set context and initialize prompt-processor
    """
    prompt = previous_history
    prompt += "" if len(previous_history) == 0 else "</s>"

    prompt = f"""{BEGIN_TEXT}{START_HEADER}{SYSTEM_ID}{END_HEADER}

{system_context_prompt}
{START_HEADER}{USER_ID}{END_HEADER}

{user_input_prompt}{EOT_ID}{START_HEADER}{ASSISTANT_ID}{END_HEADER}


"""
    return prompt


def onnx_counting(i):
    # Softmax, Softmax_1, Softmax_2, ...
    if i == 0:
        return ""
    else:
        return f"_{i}"

class RopeEmbedding:
    def __init__(self, head_dim=128, max_length=2048, config=None):
        self.cos, self.sin = self.precompute(head_dim, max_length, config)

    def precompute(self, head_dim, max_length, config):
        head_dim = (
            config.head_dim
            if hasattr(config, "head_dim")
            else config.hidden_size // config.num_attention_heads
        )
        kwargs = {
            "max_position_embeddings": config.max_position_embeddings,
            "base": config.rope_theta,
            "config": config,
        }

        if not hasattr(config, "rope_scaling"):
            setattr(config, "rope_scaling", None)

        rope = modeling_qwen2.Qwen2RotaryEmbedding(config)
        dummy_x = torch.Tensor([1.0])
        position_ids = torch.arange(max_length).view(1, -1)
        if hasattr(rope, "_original_forward"):
            embeddings = rope._original_forward(dummy_x, position_ids)
        else:
            embeddings = rope.forward(dummy_x, position_ids)

        # for adapted llama
        emb_size = embeddings[0].size(-1) // 2
        embeddings = [emb[:, :, :emb_size] for emb in embeddings]
        embeddings = [emb.unsqueeze(0) for emb in embeddings]
        return embeddings

    def get_embedding(self, position_ids, dtype=torch.float32):
        """
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, head_sim//2][2]
        """
        cos = self.cos[0, 0, :, :]  # [seq_len, dim]
        sin = self.sin[0, 0, :, :]  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin

def get_tokenizer(hf_repo_name):
    """
    Tokenizer to use for Llama3
    DeepSeekR1使用的是Llama3的FastTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_name, is_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    return tokenizer


def prepare_decoder_attention_mask(
    attention_mask, input_shape, inputs_embeds, past_key_values_length, mask_neg=-50.0
):
    # Copied from transformers.models.bart.modeling_bart._make_causal_mask
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        mask_neg: float = -50.0,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape[0], input_ids_shape[1]
        # mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
        mask = torch.full(
            (tgt_len, tgt_len), torch.tensor(mask_neg, device=device), device=device
        )
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [
                    torch.zeros(
                        tgt_len, past_key_values_length, dtype=dtype, device=device
                    ),
                    mask,
                ],
                dim=-1,
            )
        return mask[None, None, :, :].expand(
            bsz, 1, tgt_len, tgt_len + past_key_values_length
        )

    # Copied from transformers.models.bart.modeling_bart._expand_mask
    def _expand_mask(
        mask: torch.Tensor,
        dtype: torch.dtype,
        mask_neg: float = -50.0,
        tgt_len: int = None,
    ):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = (
            mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        )

        inverted_mask = 1.0 - expanded_mask

        # return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), mask_neg)

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
            mask_neg=mask_neg,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

        expanded_attn_mask = _expand_mask(
            attention_mask,
            inputs_embeds.dtype,
            tgt_len=input_shape[1],
            mask_neg=mask_neg,
        ).to(inputs_embeds.device)

        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def prepare_combined_attention_mask(
    attention_mask,
    input_shape,
    past_key_values_length,
    mask_neg=-50.0,
    dtype=torch.float32,
):
    dummy_embedding = torch.tensor((1.0,)).to(torch.float32)
    new_mask = prepare_decoder_attention_mask(
        attention_mask, input_shape, dummy_embedding, past_key_values_length, mask_neg
    )
    return new_mask.clamp_min(mask_neg).to(dtype)


def get_past_keyval_with_shift(
    past_key_vals: list[torch.Tensor],
    new_key_vals: list[torch.Tensor],
    length: int,
) -> list[torch.Tensor]:
    """
    Clip past key value to feed next iteration
    """
    ret = []
    # Key and Values are concatanated on batch dimension
    for i in range(0, len(past_key_vals), 2):
        n = new_key_vals[i].shape[3]
        m = past_key_vals[i].shape[3]
        remove = n + m - length
        key_cache = torch.cat(
            [past_key_vals[i][:, :, :, remove:], new_key_vals[i]], dim=3
        )
        val_cache = torch.cat(
            [past_key_vals[i + 1][:, :, remove:], new_key_vals[i + 1]], dim=2
        )

        ret.append(key_cache)
        ret.append(val_cache)
    return ret

# 这个函数需要先实现deepseekr1是对qwen2的调优，个别参数有变化，所以使用transformers.modeling_qwen2 pytorch类
def monkey_patch_huggingface_deepseekr1_modeling():

    # modeling_qwen2.QWEN2_ATTENTION_CLASSES["eager"] = SHADeepSeekR1Attention

    def bypass_RotaryEmbedding(self, x, position_ids, *args, **kwargs):
        return position_ids

    # Bypass rotary_emb module
    if hasattr(modeling_qwen2.Qwen2RotaryEmbedding, "_original_forward") == False:
        modeling_qwen2.Qwen2RotaryEmbedding._original_forward = (
            modeling_qwen2.Qwen2RotaryEmbedding.forward
        )
        modeling_qwen2.Qwen2RotaryEmbedding.forward = bypass_RotaryEmbedding
    modeling_qwen2.apply_rotary_pos_emb = QcDeepSeekR1_apply_rotary_pos_emb

    def DeepSeekR1RMSNorm_forward(self, hidden_states):
        # Raise to rank 4
        hidden_states = hidden_states.unsqueeze(0)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (hidden_states * self.weight).squeeze(0)

    # def forward(self, hidden_states):
    #     input_dtype = hidden_states.dtype
    #     hidden_states = hidden_states.to(torch.float32)
    #     variance = hidden_states.pow(2).mean(-1, keepdim=True)
    #     hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    #     return self.weight * hidden_states.to(input_dtype)
    
    modeling_qwen2.Qwen2RMSNorm.forward = DeepSeekR1RMSNorm_forward

def get_past_key_names(
    start: int = 0,
    end: int = 8,
    num_of_past_key_heads: int = 32,
    suffix: str = "",
    bundled_kvcache: bool = True,
):
    past_key_val_name = []

    if bundled_kvcache:
        # Key and Values are concatanated on batch dimension
        for i in range(start, end):
            past_key_val_name += [
                f"past_key_{i}{suffix}",
                f"past_value_{i}{suffix}",
            ]
        return past_key_val_name

    # Key and Values are separate for each head
    for i in range(start, end):
        cache_names = [
            f"past_key_{i}_h{j}{suffix}" for j in range(num_of_past_key_heads)
        ] + [f"past_value_{i}_h{j}{suffix}" for j in range(num_of_past_key_heads)]
        past_key_val_name.extend(cache_names)
    return past_key_val_name

# 这里的网络结构不正确，下一步再匹配deepseekr1
class DeepSeekR1_QuantizableMixin(AimetEncodingLoaderMixin, BaseModel):
    def __init__(self, model, encoding_path, is_token_generator=False):
        AimetEncodingLoaderMixin.__init__(self, model, encoding_path)
        BaseModel.__init__(self)
        self.model = model
        self.split_part = 1
        self.is_token_generator = is_token_generator

    def get_qnn_graph_name(self) -> Optional[str]:
        model_name = "token" if self.is_token_generator else "prompt"
        return f"{model_name}_part{self.split_part}"

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        options = "--max_profiler_iterations 50"
        graph_name = self.get_qnn_graph_name()
        if graph_name is not None:
            options += f" --qnn_options context_enable_graphs={graph_name}"
        return options

    @staticmethod
    def get_output_names(
        start: int = 0,
        end: int = 8,
        past_key_val_heads: int = 32,
        bundled_kvcache: bool = True,
        output_name: str = "",
    ) -> list[str]:
        # Clipped hidden layers are named same as first part for all parts
        # Eventually, each split should have respective names.
        # layer_start, layer_end = get_hidden_layer_range_from_split(split_part=split_part, model_split_map=model_split_map)

        output_list = [output_name if output_name else f"layers_{end - 1}_add_out_0"]
        output_list += get_past_key_names(
            start,
            end,
            num_of_past_key_heads=past_key_val_heads,
            bundled_kvcache=bundled_kvcache,
            suffix="_out",
        )
        return output_list

    def _sample_inputs_impl(
        self, input_spec: Optional[InputSpec] = None
    ) -> SampleInputsType:
        data = self.get_calibration_data(input_spec=input_spec)
        for key, val in data.items():
            data[key] = [val.detach().numpy()]
        return data

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """
        Source model format preferred for conversion on AI Hub.
        """
        return SourceModelFormat.TORCHSCRIPT

class DeepSeekR1Base_Quantizable(DeepSeekR1_QuantizableMixin, ABC):
    def __init__(
        self,
        huggingface_model_name: str,
        min_memory_recommended: int,
        aimet_encodings: str,
        sequence_length: int,
        context_length: int,
        load_pretrained: bool = True,
        _make_small_for_debugging: bool = False,  # construct a small and incorrect network
    ):
        """
        This is an abstract base class of all deepseek r1 models.

        Parameters
        ----------

        huggingface_model_name:
            Name of the HuggingFace model. Subclasses should provide a default
            for this.
        min_memory_recommended:
            Minimum recommended memory in GB for running export.
        aimet_encodings:
            AIMET encodings file.
        sequence_length:
            Input sequence length (in tokens).
        context_length:
            Total context length (in tokens).
        load_pretrained:
            Load a pre-trained model as opposed to a randomly initialized.
        """

        # import deepseekr1
        self.huggingface_model_name = huggingface_model_name

        # Ensure User has access to model,
        # otherwise point to instructions to get access and error out.
        # has_model_access(self.huggingface_model_name)

        # Ensure User has recommended memory,
        # otherwise, provide warning to user and recommend to increase swap-space as a work-around.
        has_recommended_memory(min_memory_recommended)

        self.llm_config = self._llm_config(
            _make_small_for_debugging=_make_small_for_debugging
        )

        # TODO: Make this into a context manager
        monkey_patch_huggingface_deepseekr1_modeling()

        if load_pretrained:
            model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(
                self.huggingface_model_name,
                config=self.llm_config,
                ignore_mismatched_sizes=_make_small_for_debugging,
            )
        else:
            model = modeling_qwen2.Qwen2ForCausalLM(self.llm_config)
        model.eval()

        os.environ["TOKENIZERS_PARALLELISM"] = "0"

        for name, module in model.named_modules():
            if hasattr(module, "prepare_conv"):
                module.prepare_conv()
            if hasattr(module, "prepare_sha"):
                module.prepare_sha()

        super().__init__(model, aimet_encodings)

        self.sequence_length = sequence_length
        self.context_length = context_length
        self.tokenizer = get_tokenizer(self.huggingface_model_name)

    def _llm_config(self, _make_small_for_debugging: bool = False):
        """
        Construct and return a HuggingFace LLM config.
        """
        llm_config = AutoConfig.from_pretrained(
            self.huggingface_model_name, trust_remote_code=True
        )
        if _make_small_for_debugging:
            llm_config.num_hidden_layers = 8
            llm_config.num_attention_heads = 4
            llm_config.num_key_value_heads = 2
            llm_config.vocab_size = 13
            embed_dim = 8
            llm_config.head_dim = embed_dim * 2
            llm_config.hidden_size = llm_config.num_attention_heads * embed_dim * 2
        llm_config._attn_implementation = "eager"
        llm_config._attn_implementation_internal = "eager"

        return llm_config

    @abstractmethod
    def from_pretrained(
        cls,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        aimet_encodings: str | None = "DEFAULT",
    ) -> DeepSeekR1Base_Quantizable:
        pass

    @staticmethod
    def get_output_names(num_hidden_layers: int):
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids_cos,
        position_ids_sin,
        *past_key_values,
    ):
        kv_cache = SHADynamicCacheNewValueOnly()
        for layer_idx, (k, v) in enumerate(
            zip(past_key_values[::2], past_key_values[1::2])
        ):
            k_split = [k[i : i + 1] for i in range(self.llm_config.num_key_value_heads)]
            v_split = [v[i : i + 1] for i in range(self.llm_config.num_key_value_heads)]
            kv_cache.update(k_split, v_split, layer_idx, {})

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=[position_ids_cos, position_ids_sin],
            past_key_values=kv_cache,
        )

    #     CausalLMOutputWithPast
    # loss: Optional[torch.FloatTensor] = None
    # logits: torch.FloatTensor = None
    # past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
        # out_cache = out["past_key_values"]
        out_cache = out.past_key_values
        flat_output_past_key_values = []
        for layer in range(len(out_cache)):
            k = torch.cat(out_cache.key_cache[layer], dim=0) if out_cache.key_cache[layer].shape[0] > 1 else out_cache.key_cache[layer]
            v = torch.cat(out_cache.value_cache[layer], dim=0) if out_cache.value_cache[layer].shape[0] > 1 else out_cache.value_cache[layer]
            flat_output_past_key_values += [k, v]

        return [out.logits] + flat_output_past_key_values

    def get_qnn_graph_name(self) -> Optional[str]:
        # Graph name of splits is determined by export script
        return None

    @staticmethod
    def get_input_spec(
        num_hidden_layers: int,
        sequence_length: int,
        context_length: int,
        hidden_size: int,
        num_key_value_heads: int,
        num_attention_heads: int,
    ) -> InputSpec:
        embed_dim = hidden_size // num_attention_heads // 2
        input_spec = {
            "input_ids": ((1, sequence_length), "int32"),
            "attention_mask": (
                (1, 1, sequence_length, context_length),
                "float32",
            ),
            # These are half the length of the hidden size per head because
            # each cos/sin are applied to a half-sliced copy of the hidden size
            # and then concatenated.
            "position_ids_cos": (
                (1, 1, sequence_length, embed_dim),
                "float32",
            ),
            "position_ids_sin": (
                (1, 1, sequence_length, embed_dim),
                "float32",
            ),
        }

        # TODO: We could support sequence_length == CONTEXT_LENGTH, but the
        # KV cache input needs to be removed.
        assert (
            sequence_length < context_length
        ), "It is currently not supported to set input sequence length to the same as or longer than context length. There should be no KV cache input at all in such case."

        for layer in range(num_hidden_layers):
            past_k_name = f"past_key_{layer}_in"
            input_spec[past_k_name] = (
                (
                    num_key_value_heads,
                    1,
                    embed_dim * 2,
                    context_length - sequence_length,
                ),
                "float32",
            )

            past_v_name = f"past_value_{layer}_in"
            input_spec[past_v_name] = (
                (
                    num_key_value_heads,
                    1,
                    context_length - sequence_length,
                    embed_dim * 2,
                ),
                "float32",
            )
        return input_spec

    def _use_zip_file(self) -> bool:
        """
        Should the return of convert_to_hub_source_model be zipped.
        """
        return False

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        # No calibration data needed
        return None

    def _adapt_aimet_encodings(
        self, src_encodings_path, dst_encodings_path, onnx_model_path
    ):
        """
        Adapt encodings from AIMET Pro to vanilla onnx export.

        Works for the new 3.0 and 3.1 encodings.
        """
        import onnx

        with open(src_encodings_path) as f:
            encodings = json.load(f)

        model = onnx.load(onnx_model_path)

        model_input_names = {}
        for node in model.graph.node:
            model_input_names[node.name] = node.input

        model_names = (
            {o for x in model.graph.node for o in x.output}
            | {x.name for x in model.graph.input}
            | {x.name for x in model.graph.output}
        )
        model_param_names = {x.name for x in model.graph.initializer}

        uses_lists = isinstance(encodings["activation_encodings"], list)
        if uses_lists:
            # Convert encodings to dictionaries for faster look-ups
            encodings["activation_encodings"] = {
                v["name"]: v for v in encodings["activation_encodings"]
            }
            encodings["param_encodings"] = {
                v["name"]: v for v in encodings["param_encodings"]
            }

        enc_names = set(encodings["activation_encodings"].keys())
        enc_param_names = set(encodings["param_encodings"].keys())

        new_encodings = {
            "activation_encodings": {},
            "excluded_layers": [],
            "param_encodings": {},
            "quantizer_args": encodings["quantizer_args"],
            "version": encodings["version"],
        }

        all_names = model_param_names | model_names
        num_attention_heads = self.llm_config.num_attention_heads
        num_key_value_heads = self.llm_config.num_key_value_heads
        mapping, rev_mapping, known_unused = map_encodings(
            [
                (
                    r"/model_layers_(\d+)_input_layernorm_Mul_1/Mul_output_0",
                    "/model/model/layers.{0}/input_layernorm/Mul_1_output_0",
                ),
                (
                    r"/model_layers_(\d+)_self_attn_q_proj_conv_Conv/Conv_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/q_proj_sha.{i}/Conv_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Mul/Mul_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Mul{onnx_counting(i * 4)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Mul_2/Mul_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Mul{onnx_counting(2 + i * 4)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Mul_1/Mul_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Mul{onnx_counting(1 + i * 4)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Mul_3/Mul_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Mul{onnx_counting(3 + i * 4)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Sub/Sub_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Sub{onnx_counting(i)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Add/Add_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Add{onnx_counting(i)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_k_proj_conv_Conv/Conv_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/k_proj_sha.{i}/Conv_output_0"
                        for i in range(num_key_value_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Mul_4/Mul_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Mul{onnx_counting(num_attention_heads * 4 + i * 4)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Mul_6/Mul_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Mul{onnx_counting(num_attention_heads * 4 + 2 + i * 4)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Mul_5/Mul_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Mul{onnx_counting(num_attention_heads * 4 + 1 + i * 4)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Mul_7/Mul_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Mul{onnx_counting(num_attention_heads * 4 + 3 + i * 4)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Sub_1/Sub_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Sub{onnx_counting(num_attention_heads + i)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Add_1/Add_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Add{onnx_counting(num_attention_heads + i)}_output_0"
                        for i in range(num_key_value_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_v_proj_conv_Conv/Conv_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/v_proj_sha.{i}/Conv_output_0"
                        for i in range(num_key_value_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_MatMul/MatMul_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/MatMul{onnx_counting(i)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Div/Div_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Div{onnx_counting(i)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Add_2/Add_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Add{onnx_counting(num_attention_heads + num_key_value_heads + i)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_Softmax/Softmax_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/Softmax{onnx_counting(i)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_MatMul_1/MatMul_output_0",
                    [
                        f"/model/model/layers.{{0}}/self_attn/MatMul{onnx_counting(num_attention_heads + i)}_output_0"
                        for i in range(num_attention_heads)
                    ],
                ),
                (
                    r"/model_layers_(\d+)_self_attn_o_proj_conv_Conv/Conv_output_0",
                    "/model/model/layers.{0}/self_attn/o_proj_conv/Conv_output_0",
                ),
                (
                    r"/model_layers_(\d+)_Add/Add_output_0",
                    "/model/model/layers.{0}/Add_output_0",
                ),
                (
                    r"/model_layers_(\d+)_post_attention_layernorm_Mul_1/Mul_output_0",
                    "/model/model/layers.{0}/post_attention_layernorm/Mul_1_output_0",
                ),
                (
                    r"/model_layers_(\d+)_mlp_gate_proj_conv_Conv/Conv_output_0",
                    "/model/model/layers.{0}/mlp/gate_proj/MatMul_output_0",
                ),
                (
                    r"/model_layers_(\d+)_mlp_act_fn_Sigmoid/Sigmoid_output_0",
                    "/model/model/layers.{0}/mlp/act_fn/Sigmoid_output_0",
                ),
                (
                    r"/model_layers_(\d+)_mlp_act_fn_Mul/Mul_output_0",
                    "/model/model/layers.{0}/mlp/act_fn/Mul_output_0",
                ),
                (
                    r"/model_layers_(\d+)_mlp_up_proj_conv_Conv/Conv_output_0",
                    "/model/model/layers.{0}/mlp/up_proj/MatMul_output_0",
                ),
                (
                    r"/model_layers_(\d+)_mlp_Mul/Mul_output_0",
                    "/model/model/layers.{0}/mlp/Mul_output_0",
                ),
                (
                    r"/model_layers_(\d+)_mlp_down_proj_conv_Conv/Conv_output_0",
                    "/model/model/layers.{0}/mlp/down_proj/MatMul_output_0",
                ),
                (
                    r"/model_layers_(\d+)_Add_1/Add_output_0",
                    "/model/model/layers.{0}/Add_1_output_0",
                ),
                ("/model_norm_Mul_1/Mul_output_0", "/model/model/norm/Mul_1_output_0"),
                ("/lm_head_conv_Conv/Conv_output_0", "/model/lm_head/MatMul_output_0"),
                (r"(.*)", "{0}"),
            ],
            enc_names,
            all_names,
            src_encodings=encodings["activation_encodings"],
            dst_encodings=new_encodings["activation_encodings"],
        )

        def split_weights(
            src_encodings,
            dst_encodings,
            src_name,
            dst_name,
            dst_pattern_index,
            num_patterns,
            groups,
        ):
            if src_name in src_encodings:
                src_entry = src_encodings[src_name]
                dst_entry = deepcopy(src_entry)
                # Slice it!
                if isinstance(dst_entry, dict):
                    dst_entry["name"] = dst_name
                    for key in ["scale", "offset", "per_block_int_scale"]:
                        n = len(dst_entry[key]) // num_patterns
                        dst_entry[key] = dst_entry[key][
                            dst_pattern_index * n : (dst_pattern_index + 1) * n
                        ]

                    # dst_encodings.append(dst_entry)
                    dst_encodings[dst_name] = dst_entry
                else:
                    n = len(dst_entry) // num_patterns
                    dst_entry = dst_entry[
                        dst_pattern_index * n : (dst_pattern_index + 1) * n
                    ]
                    dst_encodings[dst_name] = dst_entry

        # These parameters are stored as activations
        param_mapping, rev_param_mapping, param_known_unused = map_encodings(
            [
                (
                    r"model_layers_(\d+)_(input|post_attention)_layernorm_weight",
                    "model.model.layers.{0}.{1}_layernorm.weight",
                ),
                (r"model_norm_weight", "model.model.norm.weight"),
            ],
            enc_names,
            all_names,
            src_encodings=encodings["activation_encodings"],
            dst_encodings=new_encodings["param_encodings"],
        )

        # Process weight mappings
        param_mapping, rev_param_mapping, param_known_unused = map_encodings(
            [
                ("model_embed_tokens_Gather.weight", "model.model.embed_tokens.weight"),
                (
                    r"model_layers_(\d+)_self_attn_(k|v)_proj_conv_Conv.weight",
                    (
                        (
                            [
                                f"model.model.layers.{{0}}.self_attn.{{1}}_proj_sha.{i}.weight"
                                for i in range(num_key_value_heads)
                            ]
                        ),
                        split_weights,
                    ),
                ),
                (
                    r"model_layers_(\d+)_self_attn_q_proj_conv_Conv.weight",
                    (
                        (
                            [
                                f"model.model.layers.{{0}}.self_attn.q_proj_sha.{i}.weight"
                                for i in range(num_attention_heads)
                            ]
                        ),
                        split_weights,
                    ),
                ),
                (
                    r"model_layers_(\d+)_self_attn_o_proj_conv_Conv.weight",
                    "model.model.layers.{0}.self_attn.o_proj_conv.weight",
                ),
                (
                    r"model_layers_(\d+)_mlp_(gate|up|down)_proj_conv_Conv.weight",
                    ("/model/model/layers.{0}/mlp/{1}_proj/MatMul", 1),
                ),
                (r"lm_head_conv_Conv.weight", ("/model/lm_head/MatMul", 1)),
            ],
            enc_param_names,
            all_names,
            model_input_names,
            src_encodings=encodings["param_encodings"],
            dst_encodings=new_encodings["param_encodings"],
        )

        # This is needed for subtle reasons.
        # Gather ops require weights and output range to be the same, so that
        # it can be implemented as a memory look-up. Therefore, AIMET does not
        # store the output activation. However, since we may split the model
        # right after this op, it could lead the input to the second part
        # without activation encodings.
        embed_a_name = "/model/model/embed_tokens/Gather_output_0"
        embed_w_name = "model.model.embed_tokens.weight"
        new_encodings["activation_encodings"][embed_a_name] = new_encodings[
            "param_encodings"
        ][embed_w_name]
        if uses_lists:
            new_encodings["activation_encodings"][embed_a_name]["name"] = embed_a_name

        # Fill in "zero" encodings for RMSNorm internals. If these are not
        # collapsed before runtime, it should result in catastophic numerical
        # results (which is good, since it is better to catch this bug instead
        # of getting a slightly worse model, which can be hard to detect).
        zero_keys = []
        for layer in range(self.llm_config.num_hidden_layers):
            for sec in ["input", "post_attention"]:
                zero_keys += [
                    f"/model/model/layers.{layer}/{sec}_layernorm/Pow_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/ReduceMean_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/Add_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/Sqrt_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/Div_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/Mul_output_0",
                ]

        zero_keys += [
            "/model/model/norm/Pow_output_0",
            "/model/model/norm/ReduceMean_output_0",
            "/model/model/norm/Add_output_0",
            "/model/model/norm/Sqrt_output_0",
            "/model/model/norm/Div_output_0",
            "/model/model/norm/Mul_output_0",
        ]

        for key in zero_keys:
            if uses_lists:
                # aimet format 1.0
                zero_entry = {
                    "bw": 16,
                    "dtype": "INT",
                    "enc_type": "PER_TENSOR",
                    "is_sym": False,
                    "name": key,
                    "offset": [0],
                    "scale": [1e-20],
                }
            else:
                # aimet format 0.x
                zero_entry = [
                    {
                        "bitwidth": 16,
                        "dtype": "int",
                        "is_symmetric": "False",
                        "max": 0.0,
                        "min": 0.0,
                        "offset": 0,
                        "scale": 1e-20,
                    }
                ]
            new_encodings["activation_encodings"][key] = zero_entry

        changes = True
        while changes:
            changes = False
            for node in model.graph.node:
                if node.output[0] in new_encodings["activation_encodings"]:
                    continue

                if node.op_type in {
                    "Concat",
                    "Split",
                    "Transpose",
                    "Cast",
                    "Reshape",
                    "Slice",
                }:
                    if node.input[0] in new_encodings["activation_encodings"]:
                        for output_name in node.output:
                            dst_entry = deepcopy(
                                new_encodings["activation_encodings"][node.input[0]]
                            )
                            if isinstance(dst_entry, dict):
                                dst_entry["name"] = output_name
                            new_encodings["activation_encodings"][
                                output_name
                            ] = dst_entry
                            enc_names.add(output_name)
                            changes = True

        if uses_lists:
            # convert back
            new_encodings["activation_encodings"] = list(
                new_encodings["activation_encodings"].values()
            )
            new_encodings["param_encodings"] = list(
                new_encodings["param_encodings"].values()
            )

        with open(dst_encodings_path, "w") as write_file:
            json.dump(new_encodings, write_file, indent=4, sort_keys=True)

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if not input_spec:
            input_specs = self.get_input_spec(
                sequence_length=self.sequence_length,
                context_length=self.context_length,
            )
        input_prompt = DEFAULT_USER_PROMPT
        input_prompt_processed = get_input_prompt_with_tags(
            user_input_prompt=input_prompt
        )
        input_tokens = self.tokenizer(
            input_prompt_processed,
            return_tensors="pt",
            padding="max_length",
            max_length=self.context_length,
        )
        num_tokens = min(
            torch.sum(input_tokens["attention_mask"]).item(), self.sequence_length
        )
        input_ids = input_tokens["input_ids"].type(torch.int32)[
            :, -self.sequence_length :
        ]

        padding_size = self.sequence_length - num_tokens
        position_ids = [0] * (padding_size) + list(
            range(0, self.sequence_length - padding_size)
        )
        position_ids = (
            torch.Tensor(position_ids).type(torch.long).reshape(1, self.sequence_length)
        )
        position_ids = (
            torch.Tensor(position_ids).type(torch.long).reshape(1, self.sequence_length)
        )
        rope_embedding = RopeEmbedding(
            max_length=self.context_length, config=self.llm_config
        )
        position_ids_cos, position_ids_sin = rope_embedding.get_embedding(position_ids)
        attention_mask = torch.zeros((1, self.context_length))
        attention_mask[:, -num_tokens:] = 1.0
        cm_attention_masks = prepare_combined_attention_mask(
            attention_mask=attention_mask,
            input_shape=(1, self.sequence_length),
            past_key_values_length=self.context_length - self.sequence_length,
        )

        input_dict = {
            "input_ids": [input_ids.detach().numpy()],
            "attention_mask": [cm_attention_masks.detach().numpy()],
            "position_ids_cos": [position_ids_cos.detach().numpy()],
            "position_ids_sin": [position_ids_sin.detach().numpy()],
        }

        # Populate the rest with zeros (KV cache input)
        for k, (shape, _) in input_specs.items():
            if k.startswith("past_"):
                input_dict[k] = [np.zeros(shape, dtype=np.float32)]

        return input_dict