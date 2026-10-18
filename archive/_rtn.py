'''
By: Adrian Castillo - Sep 14, 2026

Still to validate:
- A constant group stays unchanged.
- Every rounded value lies on its group's 16-value grid at 4 bits.
- A non-constant weight moves by at most half a grid spacing.
- The original model stays unchanged after quantization.
- The quantized model produces finite logits.


1. Select a weight matrix to quantize.
2. Take one row.
3. Split it into 128-weight groups.
4. For each group, find its smallest and largest values.
5. Build that group’s allowed-value grid from those endpoints and the chosen bit width.
6. Replace every weight with its closest grid value.
7. Put the rows back into a matrix of exactly the same shape.
'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from qlrl.quant.utils.generate import generate, load_or_create_quantized_model


def rtn( model: PreTrainedModel, bits: int = 4 ) -> PreTrainedModel:
    from copy import deepcopy
    from transformers.pytorch_utils import Conv1D

    quantized_model = deepcopy(model)
    number_of_levels = 2 ** bits
    # names = "\n".join( f"{name}, {param.ndim}" for name, param in model.named_parameters())
    # print(names)

    with torch.no_grad():
        for name, module in quantized_model.named_modules():
            # Quantize only linear layers: Conv1D in GPT-2, nn.Linear in Llama.
            # Embeddings and LayerNorms are other module types, so they are skipped here.
            if not isinstance(module, (torch.nn.Linear, Conv1D)):
                continue
            # lm_head is linear but stays in full precision. In GPT-2 it is the same
            # tensor as wte, so quantizing it would quantize the embeddings too.
            if name == "lm_head":
                continue
            param = module.weight

            is_conv1d = isinstance(module, Conv1D)

            if is_conv1d:
                matrix = param.T
            else:
                matrix = param

            G = matrix.reshape(matrix.shape[0], matrix.shape[-1]//128, 128)
            group_min = G.amin(dim=-1, keepdim=True)
            group_max = G.amax(dim=-1, keepdim=True)
            spacer = (group_max - group_min) / (number_of_levels - 1)
            spacer = torch.where(spacer == 0, 1.0, spacer)
            q = torch.round( (G - group_min )/ spacer) # What was this?
            q = torch.clamp(q, 0, number_of_levels - 1)
            rounded_m = q * spacer + group_min
            reshaped_matrix = rounded_m.reshape(matrix.shape)

           
            if is_conv1d: param.copy_(reshaped_matrix.T)
            else: param.copy_(reshaped_matrix)
    return quantized_model


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    output_folder = "models/gpt2-rtn-4bit"
    # rtn(model)
    quantized_model = load_or_create_quantized_model(model, tokenizer, output_folder, rtn)
    text = "The capital of France is"

    print("Prompt:", repr(text))
    print("Full-precision continuation:", repr(generate(model, tokenizer, text, 20)))
    print(
        "RTN 4-bit continuation:",
        repr(generate(quantized_model, tokenizer, text, 20)),
    )

    print(sum(not torch.equal(p, q) for p, q in zip(model.parameters(), quantized_model.parameters())))

    # logits, input_ids = forward_pass(model, tokenizer, text) # What is exactly logits and input_ids?
    # print(tokenizer.convert_ids_to_tokens(input_ids[0]))  # the subword pieces

    # probs = torch.softmax(logits[0, -1], dim=-1)  # scores for the token after "is"
    # top = torch.topk(probs, 5)
    # for p, i in zip(top.values, top.indices):
    #     print(f"{p:.3f}  {tokenizer.decode(i)!r}")

    # gold = gold_logprobs(logits, input_ids)
    # print(gold, "ppl =", perplexity(gold))
    # print(repr(generate(model, tokenizer, text, max_new_tokens=10)))


if __name__ == "__main__":
    main()
