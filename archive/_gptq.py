'''
By: Adrian Castillo - Sep 26, 2026

• GPTQ roadmap:
  1. Copy the model and collect calibration inputs for each weight matrix.
  2. For one matrix W, compute inverse-Hessian information from its inputs X.
  3. Quantize one input column at a time using the RTN grid.
  4. Freeze that quantized column in Q.
  5. Update the remaining floating-point columns of W to compensate for its rounding error.
  6. Repeat until every column is frozen, then replace W with Q.
  7. Later: add group size 128, block updates, and Cholesky for speed/stability.

• GPTQ algorithm for one weight matrix W:

  1. Collect calibration inputs X for this layer.
  2. Compute H_inverse = inverse(X @ X.T + damping * I).
  3. Set work = copy(W) and Q = empty matrix.
  4. For each input column j:
     a. Round work[:, j] with the RTN quantization grid; save it in Q[:, j].
     b. error = work[:, j] - Q[:, j].
     c. Divide error by H_inverse[j, j].
     d. Subtract that scaled error from work[:, j:] using H_inverse[j, j:],
        so later columns compensate for the rounding error.
     e. Update the remaining H_inverse for the next column.
  5. Replace W with Q.
'''
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.pytorch_utils import Conv1D

from qlrl.quant.utils.calibration import load_c4, sample_windows
# Uncomment together with the commented lines at the end of main().
from qlrl.quant.utils.generate import generate, load_or_create_quantized_model


def gptq(model: PreTrainedModel, H: dict[str, torch.Tensor], bits: int = 4) -> PreTrainedModel:
    # Llama 3.1 8B is quantized in place: a copy would cost another 16 GB of GPU memory.
    quantized_model = model if model.config.model_type == "llama" else deepcopy(model)
    number_of_levels = 2 ** bits

    with torch.no_grad():
        for name, module in quantized_model.named_modules():
            # Quantize only linear layers: Conv1D in GPT-2, nn.Linear in Llama.
            if not isinstance(module, (torch.nn.Linear, Conv1D)): continue
            # lm_head stays in full precision (it is the same tensor as wte in GPT-2).
            if name == "lm_head": continue
            param = module.weight
            is_conv1d = isinstance(module, Conv1D)

            if is_conv1d: matrix = param.T
            else: matrix = param

            # matrix: [out, in] (Conv1D already transposed).
            layer_H = H[name]
            damping = 0.01 * torch.diag(layer_H).mean()
            I = torch.eye(layer_H.shape[0], device=layer_H.device)
            L = torch.linalg.cholesky(layer_H + damping * I)
            inv_H = torch.cholesky_inverse(L)
            U = torch.linalg.cholesky(inv_H, upper=True)
            # float32 for the GPTQ math even when the model is bf16; copy_ casts back.
            work = matrix.float().clone()
            Q = torch.zeros_like(work)

            for start in range(0, matrix.shape[1], 128):
                group = work[:, start:start + 128]
                group_min = group.amin(dim=-1, keepdim=True)
                group_max = group.amax(dim=-1, keepdim=True)
                spacer = (group_max - group_min) / (number_of_levels - 1)
                spacer = torch.where(spacer == 0, 1.0, spacer)

                end = start + 128
                scaled_errors = torch.zeros(matrix.shape[0], 128, device=work.device)
                for col_idx in range(start, end):
                    column = work[:, col_idx:col_idx + 1]
                    q = torch.round((column - group_min) / spacer)
                    q = torch.clamp(q, 0, number_of_levels - 1)
                    Q[:, col_idx:col_idx + 1] = q * spacer + group_min

                    error = work[:, col_idx:col_idx + 1] - Q[:, col_idx:col_idx + 1]
                    scaled_error = error / U[col_idx, col_idx]
                    work[:, col_idx+1:end] -= scaled_error * U[col_idx, col_idx+1:end]
                    scaled_errors[:, col_idx - start:col_idx - start + 1] = scaled_error

                work[:, end:] -= scaled_errors @ U[start:end, end:]

            if is_conv1d: param.copy_(Q.T)
            else: param.copy_(Q)

    return quantized_model


def get_calibration(model: PreTrainedModel, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    # input_ids: [N, L]. Returns { linear: H [in, in] } over every calibration token.
    H: dict[str, torch.Tensor] = {}
    handles = []

    def make_save_input(name):
        def save_input(_module, args):
            x = args[0].detach()
            # Turn [batch, sequence_length, input_features]
            # into [input_features, calibration_tokens]
            X = x.reshape(-1, x.shape[-1]).T.float()
            # H is a sum over tokens, so each window adds its own X X^T.
            if name not in H: H[name] = X @ X.T     # first window: start the sum
            else: H[name] += X @ X.T                # later windows: add to it
        return save_input

    for name, module in model.named_modules():
        # The same layers gptq() rounds: every linear except lm_head (GPT-2 and Llama).
        if isinstance(module, (torch.nn.Linear, Conv1D)) and name != "lm_head":
            handle = module.register_forward_pre_hook(make_save_input(name))
            handles.append(handle)

    with torch.no_grad():
        for i in range(input_ids.shape[0]):
            # use_cache=False: otherwise a key/value cache is built and kept for nothing.
            model(input_ids[i:i + 1].to(model.device), use_cache=False)  # one window: [1, L]

    [handle.remove() for handle in handles]

    return H


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    text = "The capital of France is"

    # Small enough for the login node; the full size (128 x 1024) is a SLURM job.
    input_ids = sample_windows(tokenizer, load_c4(), n=16, length=512, seed=0)
    H = get_calibration(model, input_ids)

    output_folder = "models/gpt2-gptq-4bit"
    quantized_model = load_or_create_quantized_model(
        model, tokenizer, output_folder, lambda m: gptq(m, H)
    )

    print("Prompt:", repr(text))
    print("Full-precision continuation:", repr(generate(model, tokenizer, text, 20)))
    print(
        "GPTQ 4-bit continuation:",
        repr(generate(quantized_model, tokenizer, text, 20)),
    )

    print(sum(not torch.equal(p, q) for p, q in zip(model.parameters(), quantized_model.parameters())))

if __name__ == "__main__":
    main()
