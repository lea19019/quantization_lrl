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

from qlrl.calibration.c4 import load_c4, sample_windows
# Uncomment together with the commented lines at the end of main().
from qlrl.quant.utils import generate, load_or_create_quantized_model


def gptq_matrix(matrix: torch.Tensor, H: torch.Tensor, bits: int = 4) -> torch.Tensor:
    # matrix: [out, in] (Conv1D already transposed). Returns Q, the same shape, in float32.
    number_of_levels = 2 ** bits
    damping = 0.01 * torch.diag(H).mean()
    I = torch.eye(H.shape[0], device=H.device)
    L = torch.linalg.cholesky(H + damping * I)
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

    return Q


def gptq(model: PreTrainedModel, input_ids: torch.Tensor, bits: int = 4) -> PreTrainedModel:
    # Block by block: calibrate a block, quantize its linears, then pass the inputs through
    # the quantized block so the next block's H sees the quantized model's outputs.
    # Llama 3.1 8B is quantized in place: a copy would cost another 16 GB of GPU memory.
    quantized_model = model if model.config.model_type == "llama" else deepcopy(model)
    inputs, args, kwargs = catch_block_inputs(quantized_model, input_ids)
    blocks = get_blocks(quantized_model)

    with torch.no_grad():
        # Every linear except lm_head lives inside a block (GPT-2 and Llama).
        for index, block in enumerate(blocks):
            H = get_calibration(block, inputs, args, kwargs)
            for name, module in block.named_modules():
                # Quantize only linear layers: Conv1D in GPT-2, nn.Linear in Llama.
                if not isinstance(module, (torch.nn.Linear, Conv1D)): continue
                param = module.weight
                is_conv1d = isinstance(module, Conv1D)

                if is_conv1d: matrix = param.T
                else: matrix = param

                Q = gptq_matrix(matrix, H[name], bits)
                if is_conv1d: param.copy_(Q.T)
                else: param.copy_(Q)

            del H  # free this block's H's before the next block
            inputs = run_block(block, inputs, args, kwargs)
            print(f"GPTQ: quantized block {index + 1}/{len(blocks)}", flush=True)

    return quantized_model


def get_calibration(
        block: torch.nn.Module, inputs: torch.Tensor, args: tuple, kwargs: dict
    ) -> dict[str, torch.Tensor]:
    # inputs: [N, L, hidden], what this block receives. Returns { linear in the block: H }.
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

    for name, module in block.named_modules():
        # Linear layers: Conv1D in GPT-2, nn.Linear in Llama.
        if isinstance(module, (torch.nn.Linear, Conv1D)):
            handle = module.register_forward_pre_hook(make_save_input(name))
            handles.append(handle)

    run_block(block, inputs, args, kwargs)  # the outputs are not needed here

    [handle.remove() for handle in handles]

    return H


def get_blocks(model: PreTrainedModel) -> list[torch.nn.Module]:
    # GPT-2: model.transformer.h. Llama: model.model.layers.
    name = "transformer.h" if hasattr(model, "transformer") else "model.layers"
    return list(model.get_submodule(name).children())


def catch_block_inputs(
        model: PreTrainedModel, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, tuple, dict]:
    # Returns hidden states [N, L, hidden] plus block 0's other arguments (causal mask,
    # Llama's rotary cos/sin). Every window has length L, so those are the same for all.
    inputs, other_args, kwargs = [], [], {}

    def catch(_module, args, block_kwargs):
        inputs.append(args[0])
        other_args[:] = args[1:]
        kwargs.update(block_kwargs)

    handle = get_blocks(model)[0].register_forward_pre_hook(catch, with_kwargs=True)
    with torch.no_grad():
        for i in range(input_ids.shape[0]):
            # use_cache=False: otherwise a key/value cache is passed in and keeps growing.
            model(input_ids[i:i + 1].to(model.device), use_cache=False)
    handle.remove()
    return torch.cat(inputs), tuple(other_args), kwargs


def run_block(block: torch.nn.Module, inputs: torch.Tensor, args: tuple, kwargs: dict) -> torch.Tensor:
    # One window at a time, as in get_calibration. inputs and outputs: [N, L, hidden].
    outputs = []
    with torch.no_grad():
        for i in range(inputs.shape[0]):
            out = block(inputs[i:i + 1], *args, **kwargs)
            outputs.append(out[0] if isinstance(out, tuple) else out)
    return torch.cat(outputs)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    text = "The capital of France is"

    # Small enough for the login node; the full size (128 x 1024) is a SLURM job.
    input_ids = sample_windows(tokenizer, load_c4(), n=16, length=512, seed=0)

    output_folder = "models/gpt2-gptq-4bit"
    quantized_model = load_or_create_quantized_model(
        model, tokenizer, output_folder, lambda m: gptq(m, input_ids)
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
