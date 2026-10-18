'''
Entry point for quantization: python -m qlrl.quantization configs/quantize.yaml

Loads the model and the C4 calibration windows named in the YAML, then runs each method
that has a section under `methods:`; a method with no section is skipped. Each method
writes models/<model>-<method>-<bits>bit[-seed<seed>]/ with the quantized model
(SmoothQuant excepted, see smoothquant.py) and a manifest.json.
'''
import json
import subprocess
import sys
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from qlrl.quant import (
    AWQ,
    GPTQ,
    RTN,
    Quantizer,
    SmoothQuant,
    load_c4,
    load_wikitext2,
    sample_windows,
    wikitext2_perplexity,
)


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        config["model"], dtype=getattr(torch, config["dtype"])
    ).eval()
    model.to(device)  # type: ignore[arg-type]  # transformers 5.x stub bug

    calibration = config["calibration"]
    seed = calibration["seed"]
    input_ids = sample_windows(tokenizer, load_c4(), calibration["n"], calibration["length"], seed)
    methods = config["methods"]
    output = Path(config["output"]) / Path(config["model"]).name

    # Full precision: the reference every method's perplexity is compared with.
    if config.get("perplexity"):
        finish(None, model, f"{output}-full-precision", config, tokenizer)

    if methods.get("rtn"):
        rtn = RTN(**methods["rtn"])
        rtn.calibrate(model, input_ids)
        finish(rtn, rtn.quantize(model), f"{output}-rtn-{rtn.bits}bit", config, tokenizer)

    if methods.get("gptq"):
        gptq = GPTQ(**methods["gptq"])
        gptq.calibrate(model, input_ids)
        folder = f"{output}-gptq-{gptq.bits}bit-seed{seed}"
        finish(gptq, gptq.quantize(model), folder, config, tokenizer)

    if methods.get("awq"):
        awq = AWQ(**methods["awq"])
        awq.calibrate(model, input_ids)
        folder = f"{output}-awq-{awq.bits}bit-seed{seed}"
        finish(awq, awq.quantize(model), folder, config, tokenizer)

    if methods.get("smoothquant"):
        smoothquant = SmoothQuant(**methods["smoothquant"])
        smoothquant.calibrate(model, input_ids)
        folder = f"{output}-smoothquant-w{smoothquant.bits}a{smoothquant.act_bits}-seed{seed}"
        finish(smoothquant, smoothquant.quantize(model), folder, config, tokenizer)


def finish(
        quantizer: Quantizer | None, quantized_model: PreTrainedModel, folder: str, config: dict,
        tokenizer: transformers.PreTrainedTokenizerBase,
    ) -> None:
    """Save the quantized model, check its WikiText-2 perplexity if asked, write the manifest.

    quantizer None is the full-precision model: nothing is saved, only its manifest.
    """
    Path(folder).mkdir(parents=True, exist_ok=True)
    if quantizer: quantizer.save(quantized_model, tokenizer, folder)
    manifest = {
        "config": config,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if config.get("perplexity"):
        settings = config["perplexity"]
        ids = load_wikitext2(tokenizer)
        ppl, windows = wikitext2_perplexity(
            quantized_model, ids, settings["length"], settings["max_windows"]
        )
        manifest["wikitext2_perplexity"] = {"value": ppl, "windows": windows}
        print(f"{Path(folder).name}: WikiText-2 perplexity {ppl:.3f} over {windows} windows")
    (Path(folder) / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
