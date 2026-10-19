'''
Diagnostic: is GPTQ's extra damage tied to the BOS token? GPTQ and AWQ calibrate on random
C4 windows, which almost never start with BOS; every evaluation sentence does.
Scores tokens 2..n of 300 FLORES devtest sentences per language twice: after BOS (as in
the evaluation) and with no BOS. Prints negative log p per word, per model.
Run: python scripts/check_bos.py
'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qlrl.data.dataset import ParallelData

MODELS = {"full": "meta-llama/Llama-3.1-8B",
          "rtn": "models/Llama-3.1-8B-rtn-4bit",
          "gptq": "models/Llama-3.1-8B-gptq-4bit-seed0",
          "awq": "models/Llama-3.1-8B-awq-4bit-seed0",
          "gptq_bos": "models/bos/Llama-3.1-8B-gptq-4bit-seed0",
          "awq_bos": "models/bos/Llama-3.1-8B-awq-4bit-seed0"}
tokenizer = AutoTokenizer.from_pretrained(MODELS["full"])
LANGUAGES = ["yor", "amh", "fra", "swh"]
test = ParallelData().split("test", LANGUAGES, ["flores"])
texts = {"eng": test.filter(lambda r: r["lang"] == "yor")["en"][:300]}
for lang in LANGUAGES:
    texts[lang] = test.filter(lambda r, lang=lang: r["lang"] == lang)["tgt"][:300]


def nll_per_word(model, sentences, bos):
    total, words = 0.0, 0
    for s in sentences:
        ids = tokenizer(s, add_special_tokens=False).input_ids
        ids = ([tokenizer.bos_token_id] if bos else []) + ids
        ids = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            logp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
        gold = logp[torch.arange(ids.shape[1] - 1), ids[0, 1:]]
        total -= gold[int(bos):].sum().item()  # same tokens both ways: text tokens 2..n
        words += len(s.split())
    return total / words


for name, path in MODELS.items():
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval().to("cuda")
    for lang, sentences in texts.items():
        with_bos = nll_per_word(model, sentences, True)
        without = nll_per_word(model, sentences, False)
        print(f"{name:8} {lang}  with BOS {with_bos:6.3f}   without BOS {without:6.3f}", flush=True)
    del model
