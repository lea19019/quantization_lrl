'''CPU tests of the from-scratch pipeline: tokenizer, token files, schedule, training and resume.'''
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from tokenizers import Regex, pre_tokenizers

from qlrl.pretrain import Pretraining, TokenData
from qlrl.pretrain.tokenizer import build, sampling_rates, save
from qlrl.pretrain.train import learning_rate

CONFIG = yaml.safe_load(Path("configs/pretrain/tokenizer.yaml").read_text())
TEST = yaml.safe_load(Path("configs/pretrain/llama_test.yaml").read_text())
TEXTS = ["The cat sat on the mat.", "Le chat est assis sur le tapis.", "नमस्ते दुनिया",
         "ကမ္ဘာ မင်္ဂလာပါ", "我们在这里。", "Ẹ kú àárọ̀, ẹ ṣé o.", "Sawubona mhlaba 2026!"] * 20


def split(text: str) -> list[str]:
    """Return the pieces the configured pattern cuts a text into, before BPE."""
    splitter = pre_tokenizers.Split(Regex(CONFIG["pattern"]), behavior="isolated")
    return [piece for piece, _ in splitter.pre_tokenize_str(text)]


def test_pattern_keeps_combining_marks_inside_words():
    assert split("नमस्ते दुनिया") == ["नमस्ते", " दुनिया"]
    assert split("Ẹ kú àárọ̀") == ["Ẹ", " kú", " àárọ̀"]


def tiny_tokenizer(folder: Path) -> None:
    """Train a 300-token tokenizer on TEXTS and save it to folder."""
    tokenizer, trainer = build(CONFIG, 300)
    tokenizer.train_from_iterator([TEXTS], trainer)
    save(tokenizer, folder, {"config": CONFIG})


def test_tokenizer_round_trip_and_special_ids(tmp_path):
    from transformers import AutoTokenizer
    tiny_tokenizer(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    assert [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id] == [0, 1, 2]
    for text in TEXTS[:7] + ["unseen ქართული ✓"]:
        ids = tokenizer(text).input_ids
        assert ids[0] == 0  # <s> first, as the evaluation prompts expect
        assert tokenizer.decode(ids, skip_special_tokens=True) == text


def test_sampling_rates_follow_alpha():
    chars = {"big": 1000, "small": 10}
    natural = sampling_rates(chars, 1.0, 101)
    assert natural["big"] == pytest.approx(0.1) and natural["small"] == pytest.approx(0.1)
    flat = sampling_rates(chars, 0.0, 20)  # 10 characters each: all of small, 1% of big
    assert flat["small"] == 1.0 and flat["big"] == pytest.approx(0.01)


def write_tokens(root: Path, sizes: dict[str, int]) -> np.ndarray:
    """Write one corpus per language with sizes[lang] tokens; return the joined stream."""
    stream, start = [], 0
    for lang, size in sorted(sizes.items()):
        folder = root / "parallel" / lang
        folder.mkdir(parents=True)
        ids = np.arange(start, start + size, dtype=np.uint16) % 300
        ids.tofile(folder / "train.bin")
        ids[:50].tofile(folder / "valid.bin")
        (folder / "manifest.json").write_text(json.dumps({"lang": lang}))
        stream.append(ids)
        start += size
    return np.concatenate(stream)


def test_windows_cover_the_stream_once_across_files(tmp_path):
    stream = write_tokens(tmp_path, {"aaa": 700, "bbb": 5, "ccc": 1300})
    data = TokenData(str(tmp_path), seq_len=99, seed=0)
    assert data.windows == len(stream) // 100
    windows = data.batch(list(range(data.windows))).numpy()
    starts = sorted(data.order * 100)
    assert np.array_equal(np.concatenate([stream[s:s + 100] for s in starts]),
                          stream[: data.windows * 100])
    assert sorted(windows[:, 0].tolist()) == sorted(stream[starts].tolist())
    assert data.tokens() == {"aaa": 700, "bbb": 5, "ccc": 1300}
    assert set(data.valid) == {"aaa", "bbb", "ccc"}


def test_schedule_warms_up_holds_and_cools_down():
    settings = {"warmup_steps": 10, "cooldown_fraction": 0.2}
    rates = [learning_rate(step, 100, settings) for step in range(100)]
    assert rates[0] == pytest.approx(0.1) and rates[9] == 1.0
    assert all(rate == 1.0 for rate in rates[10:80])
    assert all(a > b for a, b in pairwise(rates[80:])) and rates[-1] < 0.3


def tiny_run(root: Path, name: str, max_steps: int | None) -> Pretraining:
    """Return a run of the test config on the tiny tokenizer and token files under root,
    writing under root / name, after it trained up to max_steps (None: the whole pass)."""
    config = json.loads(json.dumps(TEST))
    config["model"]["vocab_size"] = 300
    config["train"] |= {"seq_len": 31, "batch_tokens": 248, "micro_batch": 2, "warmup_steps": 2,
                        "max_steps": max_steps, "eval_every": 3, "save_every": 2,
                        "log_every": 2}
    config |= {"tokenizer": str(root / "tokenizer"), "tokens": str(root / "tokens")}
    for key in ("checkpoints", "output", "results"):
        config[key] = str(root / name / key)
    run = Pretraining(config)
    run.run()
    return run


def test_training_lowers_loss_and_resume_matches(tmp_path):
    tiny_tokenizer(tmp_path / "tokenizer")
    write_tokens(tmp_path / "tokens", {"aaa": 2000, "bbb": 2000})
    whole = tiny_run(tmp_path, "whole", None)
    history = [json.loads(line) for line in whole.history.read_text().splitlines()]
    losses = [entry["train_loss"] for entry in history if "train_loss" in entry]
    assert losses[-1] < losses[0]
    assert history[0]["step"] == 0 and "valid_loss" in history[0]
    half = tiny_run(tmp_path, "split", 4)  # stops at step 4 and saves resume.pt
    assert (half.folder / "resume.pt").exists()
    resumed = tiny_run(tmp_path, "split", None)
    for (name, a), b in zip(whole.raw.state_dict().items(), resumed.raw.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6), name
