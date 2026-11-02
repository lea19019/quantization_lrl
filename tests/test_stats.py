'''CPU tests of qlrl/stats.py: bootstrap counts, chrF++ from statistics, damage, drops.'''
import numpy as np
import sacrebleu

from qlrl.analysis.stats import chrf, damage, drop, resample_counts

HYPOTHESES = ["the cat sat on the mat", "a dog ran", "it is raining today", "hello there"]
REFERENCES = ["the cat sat on a mat", "the dog runs", "it rains today", "hello"]
SETTINGS = {"level": 0.95, "floor_standard_errors": 3}


def run(logprob: list[float]) -> dict:
    """Return a run with the same per-sentence log-probs on both sides."""
    side = {"logprob": np.array(logprob), "words": np.array([2.0, 3.0, 4.0, 5.0]),
            "bytes": np.array([10.0, 15.0, 20.0, 25.0])}
    return {"lang": side, "eng": side}


def test_counts_are_resamples_of_n_sentences():
    counts = resample_counts(4, 50, seed=0)
    assert counts.shape == (51, 4) and (counts[0] == 1).all()
    assert (counts.sum(axis=1) == 4).all()


def test_chrf_from_statistics_equals_sacrebleu():
    found = chrf(np.ones((1, 4)), HYPOTHESES, REFERENCES)[0]
    expected = sacrebleu.CHRF(word_order=2).corpus_score(HYPOTHESES, [REFERENCES]).score
    assert abs(found - expected) < 1e-9


def test_damage_is_a_ratio_of_sums_averaged_over_seeds():
    full = run([-10.0, -20.0, -30.0, -40.0])
    worse = run([-11.0, -22.0, -33.0, -44.0])  # 10 nats lost over 70 bytes
    row = damage(full, [worse, full], resample_counts(4, 20, seed=0), "bytes", 0.95)
    assert abs(row["damage"] - 5 / 70) < 1e-12 and row["gap"] == 0
    row = damage(full, [full], resample_counts(4, 20, seed=0), "bytes", 0.95)
    assert row["damage"] == row["gap low"] == row["gap high"] == 0


def test_drops_noise_floor_and_loss():
    scores = chrf(resample_counts(4, 50, seed=0), HYPOTHESES, REFERENCES)
    assert drop(scores, [scores, scores], 0.0, SETTINGS)["reading"] == "noise"
    full = np.array([20.0, 19.0, 21.0, 20.5, 19.5])
    assert drop(full, [full - 5], 19.0, SETTINGS)["reading"] == "at floor"
    row = drop(full, [full - 5], 0.0, SETTINGS)
    assert row["reading"] == "loss" and row["skill %"] == 25.0
