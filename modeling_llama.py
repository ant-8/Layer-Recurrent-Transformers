from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM, LlamaAttention, LlamaRMSNorm, LlamaMLP, eager_attention_forward, apply_rotary_pos_emb, rotate_half, repeat_kv, LlamaRotaryEmbedding
from transformers.activations import ACT2FN
from transformers.models.mpt.modeling_mpt import build_mpt_alibi_tensor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.processing_utils import Unpack
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2LMHeadModel
import torch
import torch.nn as nn
from transformers import LogitsProcessorList, StoppingCriteriaList
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from positional_encodings import FIRE, CoPE, build_alibi_slopes
'''def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, loop_num=0, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Introduce loop-specific offset
    loop_offset = loop_num * 0.1  # Scaling factor for loop-specific adjustment
    cos_loop = cos + loop_offset
    sin_loop = sin + loop_offset

    # Apply rotary position embedding with loop-specific offset
    q_embed = (q * cos_loop) + (rotate_half(q) * sin_loop)
    k_embed = (k * cos_loop) + (rotate_half(k) * sin_loop)

    return q_embed, k_embed'''

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super(AdaptiveLayerNorm, self).__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))

        self.adaptive_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.adaptive_beta = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized_x = (x - mean) / (std + self.eps)

        adaptive_gamma = self.adaptive_gamma(x)
        adaptive_beta = self.adaptive_beta(x)
        return adaptive_gamma * normalized_x + adaptive_beta

def eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout= 0.0,
    position_bias = None,
    hidden_states=None,
    cope=None,
    **kwargs,
):
    batch_size, seq_length = hidden_states.shape[:2]
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    query_length = seq_length#if past_key_value is None else seq_length + past_key_value[0].shape[2]
    if position_bias is not None:
        #print(position_bias)
        if len(position_bias.shape) != 3:
            raise ValueError(f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias.shape)}")
        key_length = key_states.shape[-2]

        position_bias_query_index = max(0, position_bias.size(1) - query_length)
        position_bias_key_index = max(0, position_bias.size(2) - key_length)

        position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]

        attn_weights = attn_weights + position_bias
    
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if cope is not None:
        attn_weights += cope(query, attn_weights)
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class LlamaMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        if config.intermediate_size_map is not None:
            self.intermediate_size = int(config.intermediate_size_map[layer_idx] * config.hidden_size)
        else:
            self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, loop_num):
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        down_proj = self.down_proj(self.act_fn(gate_out) * up_out)

        return down_proj

class LlamaAttention(nn.Module):
    #Multi-headed attention from 'Attention Is All You Need' paper

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.attn_scaling or self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        if self.config.use_head_scale:
            self.scale_param = torch.nn.Parameter(torch.full((config.num_attention_heads,), 1.6, dtype=torch.bfloat16))
            self.scale_param.register_hook(self._clamp_hook)
        
        self.cope = CoPE(self.config.max_position_embeddings, self.head_dim) if self.config.positional_encoding=="cope" else None
        self.fire = FIRE(num_heads=self.config.num_attention_heads) if self.config.positional_encoding=="fire" else None
        

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )

        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def _clamp_hook(self, grad):
        with torch.no_grad():
            self.scale_param.data.clamp_(min=1)
    
    def query_scale(
        self,
        query_states
    ):
        bsz, num_heads, q_len, head_dim = query_states.shape
        origin_scale = 1.0 / math.sqrt(head_dim)
        scale_factor = self.scale_param * origin_scale
        scale_factor = scale_factor.unsqueeze(-1).unsqueeze(-1).to(query_states.dtype)
        query_states = query_states * scale_factor
        return query_states
    
    def forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings = None,
        past_key_value = None,
        cache_position = None,
        loop_encoding_kv = None,
        loop_num = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        if position_embeddings is not None and self.config.positional_encoding=="rope":
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if loop_encoding_kv is not None:
                query_states = query_states + loop_encoding_kv
                key_states = key_states + loop_encoding_kv

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            if position_embeddings is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            else:
                cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        if self.config.use_head_scale:
            query_states = self.query_scale(query_states)

        alibi_slopes=None
        if (self.config.positional_encoding=="alibi" and self.config._attn_implementation=="flash_attention_2"):
            alibi_slopes=build_alibi_slopes(self.config.num_attention_heads, device=query_states.device)

        if (self.config._attn_implementation=="eager"):
            module = self
            query, key, value = query_states, key_states, value_states
            scaling = 1 if self.config.use_head_scale else self.scaling
            dropout=0.0 if not self.training else self.attention_dropout
            position_bias=position_embeddings if self.config.positional_encoding=="alibi" else None
            cope=self.cope
            
            batch_size, seq_length = hidden_states.shape[:2]
            key_states = repeat_kv(key, module.num_key_value_groups)
            value_states = repeat_kv(value, module.num_key_value_groups)
        
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            query_length = seq_length#if past_key_value is None else seq_length + past_key_value[0].shape[2]
            if position_bias is not None:
                #print(position_bias)
                if len(position_bias.shape) != 3:
                    raise ValueError(f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias.shape)}")
                key_length = key_states.shape[-2]
        
                position_bias_query_index = max(0, position_bias.size(1) - query_length)
                position_bias_key_index = max(0, position_bias.size(2) - key_length)
        
                position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]
        
                attn_weights = attn_weights + position_bias
            
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
        
            if self.cope is not None:
                attn_weights += self.cope(query, attn_weights)

            if self.fire is not None:
                fire_bias = self.fire(query_states)
                attn_weights = attn_weights + fire_bias
            
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                hidden_states=hidden_states,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config, layer_idx)

        ln_impl = LlamaRMSNorm
        if config.loop_map is not None and config.loop_map[layer_idx] > 1 and config.use_adaptive_layer_norm:
            ln_impl = AdaptiveLayerNorm
        
        self.input_layernorm = ln_impl(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ln_impl(config.hidden_size, eps=config.rms_norm_eps)
        
        self.has_multiple_loops = config.loop_map is not None and config.loop_map[layer_idx] > 1
        if self.config.use_loop_encoding and self.has_multiple_loops:
            self.embed_loop_nums = nn.Embedding(config.loop_map[layer_idx], config.hidden_size)
            self.weight_loop_nums = nn.Embedding(config.loop_map[layer_idx], 1)

    def forward(
        self,
        hidden_states,
        residual_weight,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: Optional[bool] = False,
        use_cache = False,
        cache_position = None,
        position_embeddings = None,
        loop_num = 0,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        loop_weight = 1
        if self.config.use_loop_encoding and self.has_multiple_loops:
            loop_id = torch.Tensor([loop_num]).to(torch.int).to(hidden_states.device)
            loop_weight = self.weight_loop_nums(loop_id)
            loop_encoding = self.embed_loop_nums(loop_id)
            hidden_states = hidden_states + loop_encoding

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            loop_num=loop_num,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, loop_num)
        hidden_states = residual + (hidden_states * residual_weight * loop_weight)
        #if residual_weight < 1: print(residual_weight)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

class LoopedLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size) if config.positional_encoding=="ape" else None

        if (self.config.use_recurrent_embeds):
            self.adapter = torch.nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        
        for i in range(len(self.layers)):
            self.layers[i] = LlamaDecoderLayer(config, layer_idx=i)
        self.post_init()

    def forward(
        self,
        loop_weight = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        position_embeddings=None
        if self.config.positional_encoding=="rope":
            position_embeddings=self.rotary_emb(hidden_states, position_ids)
        elif self.config.positional_encoding=="alibi" and not self.config._attn_implementation=="flash_attention_2":
            position_embeddings = build_mpt_alibi_tensor(self.config.num_attention_heads, max(input_ids.shape[-1], self.config.max_position_embeddings), device=hidden_states.device)
        elif self.config.positional_encoding=="ape":
            input_shape = input_ids.size()
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
            position_embeddings = self.embed_positions(position_ids)
            hidden_states = inputs_embeds + position_embeddings
            position_embeddings=None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_layers = self.layers[: self.config.num_hidden_layers]

        prev_step_hidden_states = hidden_states
        initial_input_states = hidden_states

        def forward_layer(layer, residual_weight=1, j=0):
            nonlocal hidden_states, all_self_attns, all_hidden_states
            x = hidden_states
            if self.config.sequential_looping == False and self.config.use_recurrent_embeds and j > 0:
                x = x + prev_step_hidden_states
    
            layer_outputs = decoder_layer(
                x,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                residual_weight=residual_weight,
                loop_num=j,
                **flash_attn_kwargs,
            )
    
            hidden_states = layer_outputs[0]
    
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        step = 0
        if self.config.sequential_looping:
            for j in range(self.config.num_loops):
                if self.config.use_recurrent_embeds and j > 0:
                    #combined_states = torch.cat((hidden_states, initial_input_states), dim=-1)
                    #hidden_states = self.adapter(combined_states)
                    hidden_states = hidden_states + initial_input_states


                for i in range(len(all_layers)):
                    decoder_layer = all_layers[i]
                    forward_layer(decoder_layer, 1, j)
        else:
            num_steps = sum(self.config.loop_map)
            for i in range(len(all_layers)):
                decoder_layer = all_layers[i]
                n = self.config.loop_map[i]
                
                for j in range(n):
                    '''if j > 0 and loop_weight is not None:
                        residual_weight = min(1, loop_weight / (2**(j-1)))
                    else:
                        residual_weight = 1'''
                    #residual_weight = loop_weight * (((num_steps-step) + 1)/(num_steps + 1))
                    step += 1
                    forward_layer(decoder_layer, j=j)
                prev_step_hidden_states = hidden_states

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

class LoopedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LoopedLlamaModel(config)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        cache_position = None,
        num_logits_to_keep = 0,
        loop_weight = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            loop_weight=loop_weight,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        do_sample: bool = False,
    ):
        device = input_ids.device
        generated = input_ids
        max_length = input_ids.shape[1] + max_new_tokens

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self(input_ids=generated)
            logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)

            if do_sample:
                # Sample from the distribution
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append the new token to the generated sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if all sequences reach the end-of-sequence token
            if (next_token == self.config.eos_token_id).all():
                break

        return generated

class LlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        for i in range(len(self.layers)):
            self.layers[i] = LlamaDecoderLayer(config, layer_idx=i)

class LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head.weight = self.model.embed_tokens.weight