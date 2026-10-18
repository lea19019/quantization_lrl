'''
Give the Spanish-X corpora an English side: translate their Spanish side with Qwen3-32B.

AmericasNLP, MayanV and the Kichwa lexicon pair the language with Spanish. English is the
project's anchor, so their Spanish side is machine-translated into English and the
Indigenous side stays human-written ("synthetic source", as in back-translation).

Prompt: three Spanish/English pairs from FLORES+ dev, then the sentence, in tags:
    <spa>{example}</spa> <eng>{example}</eng>   (x3)
    <spa>{sentence}</spa>
The answer is started with <eng>, so the model writes only the translation; it is cut at
</eng>. Greedy decoding. Qwen3's "thinking" mode is off.

Output: <translated>/<name>.en, line i the translation of line i of the Spanish file, and
manifest.json. Lines over MAX_CHARACTERS are whole documents (81 Nahuatl train lines, up
to 24,050 characters) or long dictionary entries (34 Kichwa), not sentences; they stay empty.
Ran once as Slurm job 14131729; prepare.py runs it again only if the config has `translate:`.
'''
# Cluster rules: docs/workflow/BYU_ORC_AGENTS.md (copy of https://rc.byu.edu/documentation/BYU_ORC_AGENTS.md)
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from qlrl.data.sources import flores, spanish_sides

MAX_CHARACTERS = 1000


def build_prompt(
        tokenizer: PreTrainedTokenizerBase, examples: list[tuple[str, str]], spanish: str
    ) -> str:
    """Return the chat prompt for one sentence, ending in <eng>."""
    lines = ["Translate Spanish to English."]
    lines += [f"<spa>{es}</spa> <eng>{en}</eng>" for es, en in examples]
    lines.append(f"<spa>{spanish}</spa>")
    messages = [{"role": "user", "content": "\n".join(lines)}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return prompt + "<eng>"


def translate_sentences(
        model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerBase,
        sentences: list[str], examples: list[tuple[str, str]], batch_size: int,
    ) -> dict[str, str]:
    """Return {Spanish: English} for every sentence."""
    english = {}
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        prompts = [build_prompt(tokenizer, examples, sentence) for sentence in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        longest = max(len(tokenizer(sentence)["input_ids"]) for sentence in batch)
        with torch.no_grad():
            output = model.generate(
                **inputs, do_sample=False, max_new_tokens=2 * longest + 20,
                stop_strings=["</eng>"], tokenizer=tokenizer,
            )
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        answers = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for sentence, answer in zip(batch, answers, strict=True):
            english[sentence] = " ".join(answer.split("</eng>")[0].split())
        print(f"{start + len(batch)} / {len(sentences)}", flush=True)
    return english


def translate(config: dict, test: bool) -> None:
    """Translate every Spanish side named by the config and write the .en files."""
    model_name, batch_size = config["translate"]["model"], config["translate"]["batch_size"]
    spanish = spanish_sides(config["external"], config["languages"])
    # Each distinct sentence once (the AmericasNLP test sets share one Spanish side),
    # longest first so the batch that needs the most memory runs first.
    distinct = {line for lines in spanish.values() for line in lines if line}
    sentences = sorted((s for s in distinct if len(s) <= MAX_CHARACTERS), key=len, reverse=True)
    if test:  # a small model and 8 sentences spread from longest to shortest
        model_name, sentences = "Qwen/Qwen3-1.7B", sentences[:: len(sentences) // 8][:8]
    # The first three FLORES+ dev pairs, as (Spanish, English).
    examples = [(es, en) for en, es in flores(config["external"], "spa_Latn", "dev")[:3]]

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
    english = translate_sentences(model, tokenizer, sentences, examples, batch_size)

    output = config["translated"] + ("_test" if test else "")
    os.makedirs(output, exist_ok=True)
    for name, lines in spanish.items():
        with open(f"{output}/{name}.en", "w", encoding="utf-8") as handle:
            handle.writelines(english.get(line, "") + "\n" for line in lines)
    manifest = {
        "model": model_name, "batch_size": batch_size, "sentences": len(sentences),
        "skipped_longer_than_max_characters": len(distinct) - len(sentences),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    with open(f"{output}/manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
