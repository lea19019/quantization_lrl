'''
Where things live inside GPT-2 and Llama 3.1 8B. Description only, no code yet.
Written by Claude Code on 2026-10-15 at Adrian's request, from what the archived code
(branch archive/before-reset) had to know about each model.

• Why this file exists
  RTN and GPTQ only need to find every weight matrix and round it. AWQ also
  needs to know what produces each matrix's input (a norm or another linear), because it
  moves a scale s into that producer. That wiring is not stored in the weights; it lives in
  each model's forward() code, so it has to be written down by hand. For Llama only, that is
  a handful of lines; this file may end up being just those lines, or folded into awq.py.

• GPT-2 small (124M): what your RTN and GPTQ were written for
  - 12 decoder layers in model.transformer.h; hidden size 768.
  - Per layer: ln_1 -> attn.c_attn (768 -> 2304, q, k and v in ONE matrix) -> attention
    -> attn.c_proj (768 -> 768); ln_2 -> mlp.c_fc (768 -> 3072) -> GELU -> mlp.c_proj
    (3072 -> 768).
  - Its linears are Conv1D, which stores the weight TRANSPOSED: [in, out]. That is why your
    rtn.py checks isinstance(module, Conv1D) and uses param.T.
  - LayerNorm: (x - mean) / std * g + b, so it has a gain g and a bias b.

• Llama 3.1 8B: what we are moving to
  - 32 decoder layers in model.model.layers; hidden size 4096; vocabulary 128,256.
  - Per layer, attention block:
        input_layernorm (RMSNorm)
          -> self_attn.q_proj  [4096, 4096]
          -> self_attn.k_proj  [1024, 4096]   8 key/value heads, not 32 (grouped-query
          -> self_attn.v_proj  [1024, 4096]   attention), so k and v are 4x narrower
          -> attention -> self_attn.o_proj [4096, 4096]
  - Per layer, MLP block:
        post_attention_layernorm (RMSNorm)
          -> mlp.gate_proj [14336, 4096] and mlp.up_proj [14336, 4096]
          -> SiLU(gate) * up -> mlp.down_proj [4096, 14336]
  - 7 linears per layer, 224 in total. They are nn.Linear, stored as [out, in]: no
    transpose, unlike GPT-2.
  - RMSNorm: x / rms(x) * g. Only a gain g, no mean and no bias.
  - Not quantized (papers keep them in 16 bits): embed_tokens, lm_head, the norms.

• What AWQ needs from this (Llama only)
  - "Who produces the input of which linears", where s can be folded in for free:
        input_layernorm          -> q_proj, k_proj, v_proj
        post_attention_layernorm -> gate_proj, up_proj
        up_proj                  -> down_proj
        v_proj -> o_proj is skipped: v is narrower than o's input under grouped-query
        attention, so the shapes do not line up.
  - Folding s into an RMSNorm means dividing its g by s. Folding it into a linear means
    dividing that linear's output rows by s.

• Testing on a CPU
  The full model is 16 GB in bf16 and slow on a CPU. The archived code built a tiny Llama
  with the same structure and the real tokenizer: load the config with AutoConfig, set
  hidden_size = 64, intermediate_size = 128, num_hidden_layers = 1,
  num_attention_heads = 4, num_key_value_heads = 2, and create it with
  AutoModelForCausalLM.from_config. Its weights are random, so its output is nonsense, but
  every name and shape relationship above holds. That is enough to check the code runs and
  that the rounding behaves; the real numbers then come from a GPU job on the full model.
'''
