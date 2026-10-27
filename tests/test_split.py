'''CPU tests of the shared held-out split (split.hold_out_shared).'''
from qlrl.data.split import core_lines, decontaminate, coverage_of, hold_out, hold_out_shared, normalize, usable

# Letters only: normalize() drops digits. 5,000 distinct verses "Verse aaa.", "Verse baa.", ...
VERSES = [f"Verse {chr(97 + i % 26)}{chr(97 + i // 26 % 26)}{chr(97 + i // 676)}." for i in range(5000)]


def language(verses: list[str], tag: str) -> list[tuple[str, str]]:
    """Return one language's pairs: each English verse with a made-up target side."""
    return [(en, f"{tag} {en}") for en in verses]


def shared(pairs: list, size: int, seed: int = 0, core: set | None = None,
           protected: set | None = None) -> tuple:
    """hold_out_shared with a 10% cap and, by default, no core and nothing protected."""
    return hold_out_shared(pairs, size, 0.1, seed, core or set(), protected or set())


def held(split: list) -> set[str]:
    return {normalize(en) for en, _ in split}


def test_languages_hold_out_the_same_verses():
    big = shared(language(VERSES, "big"), 100)
    other = shared(language(VERSES[::-1], "other"), 100)
    assert held(big[1]) == held(other[1]) and held(big[2]) == held(other[2])
    assert len(big[1]) == len(big[2]) == 100


def test_a_verse_is_dev_or_test_in_every_language():
    big = shared(language(VERSES, "big"), 100)
    small = shared(language(VERSES[:500], "small"), 100)
    assert len(small[1]) == len(small[2]) == 50  # 10% of 500 pairs each
    assert not held(small[1]) & held(big[2]) and not held(small[2]) & held(big[1])


def test_small_language_keeps_most_of_its_data():
    train, dev, test = shared(language(VERSES[:300], "tiny"), 1000)
    assert len(dev) == len(test) == 30 and len(train) == 240


def test_duplicates_stay_in_train_and_nothing_is_lost():
    pairs = language(VERSES[:1000], "x") + language(VERSES[:200], "copy") + [("29.", "29.")]
    train, dev, test = shared(pairs, 50)
    assert len(train) + len(dev) + len(test) == len(pairs)
    assert ("29.", "29.") in train
    assert len(held(dev) | held(test)) == 100  # one pair per held-out verse


def test_seeded_and_different_from_another_seed():
    pairs = language(VERSES, "big")
    assert shared(pairs, 100) == shared(pairs, 100)
    assert held(shared(pairs, 100, seed=1)[1]) != held(shared(pairs, 100)[1])


def test_old_split_unchanged_for_phase_one():
    pairs = language(VERSES[:100], "x")
    train, dev, test = hold_out(pairs, 10, 0)
    assert len(dev) == len(test) == 10 and len(train) == 80


def test_core_is_shared_and_big_languages_spare_small_ones():
    big = language(VERSES, "big")
    small = language(VERSES[4000:], "small")  # a small language with only the last 1,000 verses
    others = [language(VERSES[4000:], f"o{i}") for i in range(3)]  # ... which many languages have
    coverage, protected = coverage_of([big, small] + others, 2000)
    core = core_lines(coverage, 40, 0)
    assert len(core) == 80 and core <= held(small) and protected == held(small)
    big_split = shared(big, 100, core=core, protected=protected)
    small_split = shared(small, 100, core=core, protected=protected)
    assert core <= held(big_split[1]) | held(big_split[2])
    assert core <= held(small_split[1]) | held(small_split[2])
    # outside the core, nobody holds out the small language's lines, not even itself
    assert not ((held(big_split[1]) | held(big_split[2])) - core) & held(small)
    assert held(small_split[1]) | held(small_split[2]) == core


def test_bare_verse_numbers_and_misaligned_pairs_are_not_held_out():
    pairs = language(VERSES[:1000], "x")
    pairs[5] = (pairs[5][0], "40.")
    pairs[6] = (pairs[6][0], pairs[6][1] * 4)
    assert 5 not in usable(pairs) and 6 not in usable(pairs) and 7 in usable(pairs)
    train, dev, test = shared(pairs, 1000)
    assert pairs[5] in train and pairs[6] in train


def test_held_out_sentence_inside_a_longer_line_is_dropped_only_when_asked():
    seen = {normalize("Reime porãpa che ra'y?")}
    rows = [{"en": "He asked.", "tgt": "Añeno ijykére ha apoko hese. Reime porãpa che ra'y?"},
            {"en": "He went.", "tgt": "Oho."}]
    assert decontaminate(rows, seen) == rows  # phase 1: whole sides only
    assert decontaminate(rows, seen, sentences=True) == rows[1:]
