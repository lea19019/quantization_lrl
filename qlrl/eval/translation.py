'''
Generating translations: the test sets, the prompt, the length cap and greedy decoding.
'''
import math

import torch

from qlrl.data.dataset import ParallelData


def load_data(config: dict) -> list[dict]:
    """Return one dict per language: its test set's source, dev rows, dev examples, test rows."""
    data = ParallelData(config["data"])
    test_sets = []
    for lang in config["languages"]:
        source = config["test_sets"][lang]
        test = list(data.split("test", [lang], [source]))
        dev = list(data.split("dev", [lang], [source]))
        test_sets.append({"lang": lang, "source": source, "dev": dev,
                          "examples": dev[:config["examples"]],
                          "rows": test[:config["max_sentences"] or len(test)]})
    return test_sets


def get_prompts(examples: list[tuple[str, str]], sentences: list[str], src: str, tgt: str):
    """Return one prompt per sentence: the example pairs, then the sentence and an open tag."""
    shots = "\n".join(f"<{src}>{s}</{src}> <{tgt}>{t}</{tgt}>" for s, t in examples)
    return [f"{shots}\n<{src}>{sentence}</{src}> <{tgt}>" for sentence in sentences]


def token_ratio(tokenizer, pairs: list[tuple[str, str]]) -> float:
    """Return the mean over (source, target) pairs of target tokens / source tokens."""
    ratios = []
    for source, target in pairs:
        source_tokens = len(tokenizer(source, add_special_tokens=False).input_ids)
        target_tokens = len(tokenizer(target, add_special_tokens=False).input_ids)
        ratios.append(target_tokens / source_tokens)
    return sum(ratios) / len(ratios)


def generation_caps(tokenizer, sources: list[str], ratio: float) -> list[int]:
    """Return, per source sentence, the most new tokens its translation may take."""
    # 2x the source tokens, scaled by how many more tokens the target language needs for the
    # same meaning (token_ratio on the dev set). A flat 2x cut every English -> Amharic output:
    # Amharic needs ~7.6x the tokens of English, and cutting inside a character left U+FFFD.
    caps = []
    for source in sources:
        source_tokens = len(tokenizer(source, add_special_tokens=False).input_ids)
        caps.append(math.ceil(2 * source_tokens * ratio))
    return caps


def translate(model, tokenizer, prompts: list[str], caps: list[int], tgt: str, batch_size):
    """Return the greedy continuation of each prompt, cut at </tgt> and at its cap in tokens."""
    stop = f"</{tgt}>"
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True)
        batch = batch.to(model.device)
        batch_caps = caps[i:i + batch_size]
        with torch.no_grad():
            ids = model.generate(**batch, max_new_tokens=max(batch_caps), do_sample=False,
                                 stop_strings=[stop], tokenizer=tokenizer,
                                 pad_token_id=tokenizer.pad_token_id)
        new = ids[:, batch.input_ids.shape[1]:]  # [batch, new tokens], prompt dropped
        for row, cap in zip(new, batch_caps):
            text = tokenizer.decode(row[:cap], skip_special_tokens=True)
            outputs.append(text.split(stop)[0].strip())
    return outputs
