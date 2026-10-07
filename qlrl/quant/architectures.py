'''
Where each supported architecture keeps its layers. Written by Claude Code on 2026-10-07.

RTN and GPTQ only need to find the weight matrices, which is the same for every model.
SmoothQuant and AWQ also need to know which operation produces each matrix's input
(a norm or another linear), because they fold a scale s into that producer. That wiring
is not stored anywhere in the weights; it lives in each model's forward() code. So it is
written out by hand here, one function per method and one branch per architecture.
To support a new architecture, add a branch to each function below.

Supported: "llama" (Llama 2/3) and "gpt2" (GPT-2 and models trained with its config).

Version 2 of SmoothQuant and AWQ (option 3, input_scales.py) did not fold s and needed none
of the per-architecture maps below; it was rolled back on 2026-10-07, so smoothquant.py and
awq.py use these maps again. decoder_layers and is_quantized_module, at the end of this
file, are generic and used by every method.

Two layout differences the helpers below hide:
  - nn.Linear stores its weight as [out_features, in_features]; GPT-2's Conv1D stores the
    transpose, [in_features, out_features]. get_weight / set_weight always use [out, in].
  - Llama's RMSNorm has only a gain g; GPT-2's LayerNorm computes (x - mean) / std * g + b,
    so dividing its output by s means dividing both g and b by s.
'''
import torch
from transformers.pytorch_utils import Conv1D


def get_weight(module: torch.nn.Module) -> torch.Tensor:
    """Return the weight as [out_features, in_features], whatever the layer type."""
    if isinstance(module, Conv1D):
        return module.weight.T
    return module.weight


def set_weight(module: torch.nn.Module, weight: torch.Tensor) -> None:
    """Write a [out_features, in_features] weight back in the layer's own layout and dtype."""
    if isinstance(module, Conv1D):
        weight = weight.T
    module.weight.copy_(weight.to(module.weight.dtype))


def divide_output_channels(module: torch.nn.Module, scales: torch.Tensor) -> None:
    """Make a norm or linear output y / s instead of y, one s per output channel."""
    if isinstance(module, (torch.nn.Linear, Conv1D)):
        # Output channel j of a linear is row j of its [out, in] weight.
        set_weight(module, get_weight(module).float() / scales.unsqueeze(1))
    else:
        # A norm's output channel j is multiplied by its gain g_j.
        module.weight.copy_((module.weight.float() / scales).to(module.weight.dtype))
    # Any bias is added to output channel j, so it is divided too.
    if getattr(module, "bias", None) is not None:
        module.bias.copy_((module.bias.float() / scales).to(module.bias.dtype))


def multiply_input_channels(module: torch.nn.Module, scales: torch.Tensor) -> None:
    """Multiply input column j of a linear's weight by s_j (its bias is unaffected)."""
    set_weight(module, get_weight(module).float() * scales)


def smoothing_pairs(layer: torch.nn.Module, model_type: str) -> list[tuple]:
    """Return (norm, linears reading the norm's output) for one decoder layer."""
    if model_type == "llama":
        attention, mlp = layer.self_attn, layer.mlp
        return [
            (layer.input_layernorm, [attention.q_proj, attention.k_proj, attention.v_proj]),
            (layer.post_attention_layernorm, [mlp.gate_proj, mlp.up_proj]),
        ]
    # GPT-2: c_attn computes q, k and v with one matrix; c_fc is the MLP's first matrix.
    return [
        (layer.ln_1, [layer.attn.c_attn]),
        (layer.ln_2, [layer.mlp.c_fc]),
    ]


def awq_scale_groups(layer: torch.nn.Module, model_type: str, args: tuple, kwargs: dict) -> list[tuple]:
    """Return (producer, linears reading its output, block to inspect) for one decoder layer."""
    # args / kwargs are what the decoder layer itself receives (masks, positions); the
    # attention block needs some of them to run on its own.
    if model_type == "llama":
        attention, mlp = layer.self_attn, layer.mlp

        def llama_attention(x):
            return attention(hidden_states=x, **kwargs)[0]

        qkv = [attention.q_proj, attention.k_proj, attention.v_proj]
        groups = [(layer.input_layernorm, qkv, llama_attention)]
        # v -> o only when shapes match; with grouped-query attention (Llama 3) v is narrower.
        if attention.v_proj.out_features == attention.o_proj.in_features:
            groups.append((attention.v_proj, [attention.o_proj], attention.o_proj))
        groups.append((layer.post_attention_layernorm, [mlp.gate_proj, mlp.up_proj], mlp))
        groups.append((mlp.up_proj, [mlp.down_proj], mlp.down_proj))
        return groups

    # GPT-2 calls each block as block(x, past_key_values, attention_mask, ...), so the
    # attention mask is the second of the extra positional arguments.
    def gpt2_attention(x):
        return layer.attn(x, attention_mask=args[1])[0]

    # Not scaled for GPT-2, to keep the code simple:
    #   c_attn -> attn.c_proj: only the v third of c_attn's output feeds c_proj.
    #   c_fc -> mlp.c_proj: GELU sits between them, and GELU(x / s) != GELU(x) / s.
    return [
        (layer.ln_1, [layer.attn.c_attn], gpt2_attention),
        (layer.ln_2, [layer.mlp.c_fc], layer.mlp),
    ]


def awq_clip_targets(layer: torch.nn.Module, model_type: str) -> list[torch.nn.Module]:
    """Return the matrices AWQ clips: every one except those producing q and k."""
    # q and k errors pass through the attention softmax, which a matrix's own output does
    # not measure, so the reference code does not clip them.
    if model_type == "llama":
        attention, mlp = layer.self_attn, layer.mlp
        return [attention.v_proj, attention.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj]
    # GPT-2's c_attn holds q, k and v in one matrix, so it is skipped whole.
    return [layer.attn.c_proj, layer.mlp.c_fc, layer.mlp.c_proj]


def decoder_layers(model: torch.nn.Module) -> tuple[torch.nn.ModuleList, str]:
    """Return the stack of repeated decoder layers and its name, e.g. 'model.layers'."""
    # Every decoder-only transformer keeps its N identical layers in one nn.ModuleList, and
    # it is the longest list in the model. Llama: model.layers; GPT-2: transformer.h.
    lists = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.ModuleList)
    ]
    name, layers = max(lists, key=lambda item: len(item[1]))
    return layers, name


def is_quantized_module(name: str, module: torch.nn.Module) -> bool:
    """True for every linear inside the repeated layers; never for lm_head or embeddings."""
    # Modules inside a ModuleList have a number in their name (model.layers.3.mlp.up_proj);
    # lm_head, which papers keep in 16 bits, sits outside the layers and has none.
    inside_layers = any(part.isdigit() for part in name.split("."))
    return inside_layers and isinstance(module, (torch.nn.Linear, Conv1D))
