'''
Log-probabilities of given text: the human reference after the translation prompt (forced
decoding) and bare sentences (perplexity). No generation, so no decoding noise.
'''
import math

import torch

from qlrl.quant.utils.perplexity import gold_logprobs


def text_logprob(model, tokenizer, prefix: str, text: str) -> tuple[float, int]:
    """Return (summed log p in nats, token count) of the tokens of `text` given `prefix`."""
    # Tokenized apart and joined, so `text` keeps its own tokens: in one string Llama merges
    # the prompt's ">" with the first letter of the reference into a single token.
    prefix_ids = tokenizer(prefix).input_ids
    text_ids = tokenizer(text, add_special_tokens=False).input_ids
    ids = torch.tensor([prefix_ids + text_ids], device=model.device)  # [1, seq]
    with torch.no_grad():
        logits = model(ids).logits.float()  # [1, seq, vocab]; bf16 logits lose precision
    gold = gold_logprobs(logits, ids)  # [seq - 1]; gold[t - 1] scores token t
    return gold[len(prefix_ids) - 1:].sum().item(), len(text_ids)


def referenceLogProb_eval(model, tokenizer, prefixes: list[str], texts: list[str]):
    """Return, per text, its log p given its prefix, with token, word and byte counts."""
    rows = []
    for prefix, text in zip(prefixes, texts):
        logprob, tokens = text_logprob(model, tokenizer, prefix, text)
        rows.append({"logprob": logprob, "tokens": tokens,
                     "words": len(text.split()), "bytes": len(text.encode())})
    return rows


def perplexity_eval(model, tokenizer, texts: list[str]) -> list[dict]:
    """Return referenceLogProb_eval rows for bare sentences, after only a BOS token."""
    # Llama's tokenizer adds BOS itself; GPT-2's does not, so give it one to predict from.
    adds_bos = tokenizer("a").input_ids[0] == tokenizer.bos_token_id
    prefix = "" if adds_bos else tokenizer.bos_token
    return referenceLogProb_eval(model, tokenizer, [prefix] * len(texts), texts)


def summarize(rows: list[dict]) -> dict:
    """Return perplexity per token and negative log p per word and per byte, over all rows."""
    logprob = sum(row["logprob"] for row in rows)
    return {"perplexity": math.exp(-logprob / sum(row["tokens"] for row in rows)),
            "NLL per word": -logprob / sum(row["words"] for row in rows),
            "NLL per byte": -logprob / sum(row["bytes"] for row in rows)}
