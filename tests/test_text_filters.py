'''CPU tests of qlrl/data/text_filters.py and the validation split of freetext.py.'''
from collections import Counter

import numpy as np

from qlrl.data import split
from qlrl.data.freetext import in_validation
from qlrl.data.text_filters import (
    WORD,
    Cleaner,
    blocked,
    contaminated,
    is_prose,
    normalize,
    script_of,
    skeleton,
)

CEBUANO = "Kaliwatan sa langaw ang {}. Una ning gihulagway ni {} ni adtong {}."


class FakeGlotLID:
    """Labels a line English if it contains "the", else Quechua, always with probability 0.95."""

    def predict(self, lines):
        labels = [["__label__eng_Latn" if " the " in f" {line} " else "__label__quy_Latn"]
                  for line in lines]
        return labels, np.full((len(lines), 1), 0.95)


def cleaner(prose_only: bool) -> Cleaner:
    """Return a Cleaner with the fake GlotLID and one held-out sentence, no model loaded."""
    cleaner = Cleaner.__new__(Cleaner)
    cleaner.model, cleaner.threshold, cleaner.labels = FakeGlotLID(), 0.9, {"quy_Latn"}
    cleaner.judged_chars, cleaner.read_chars = 20, 200
    cleaner.held_out = {normalize("Allinllachu kanki, wawqey.")}
    cleaner.prose_only, cleaner.min_chars, cleaner.counts = prose_only, 80, Counter()
    return cleaner


def test_blocked_sites_subdomains_and_words():
    sites, words = ["jw.org", "bible.is"], ["biblia"]
    assert blocked("https://www.jw.org/qu/", sites, words)
    assert blocked("https://live.bible.is/x", sites, words)
    assert blocked("https://example.com/biblia/1", sites, words)
    assert not blocked("https://qu.wikipedia.org/wiki/Inti", sites, words)


def test_headings_and_list_items_are_not_prose():
    assert is_prose("Inti nisqaqa huk quyllurmi.", 80)
    assert is_prose("x" * 80, 80)  # long running text without punctuation (Thai, Lao)
    assert not is_prose("Kaypipas qhaway ", 80)
    assert not is_prose(" Ojos del Salado", 80)
    assert not is_prose(" Muyuqin (duodeno)", 80)
    assert is_prose('Paymi nirqa: "Arí."', 80)


def test_contamination_matches_a_sentence_inside_a_line():
    held_out = {normalize("29 And he went.")}
    assert contaminated("Intro text here. 29. and he went", held_out)
    assert not contaminated("And he stayed.", held_out)
    assert contaminated("北京很大。他走了。东京也很大。", {normalize("他走了。")})


def test_fast_normalize_equals_split_normalize():
    for text in ["29. And he went!", "Don't  STOP\tnow", "Ñat'ikuna (visceras)", "Ẹ kú àárọ̀ 2024",
                 "ሰላም ነው።", "привет, мир", "你好，世界", "สวัสดี ครับ", "नमस्ते दुनिया"]:
        assert normalize(text) == split.normalize(text), text
    every = "".join(chr(code) for code in range(0x30000) if not 0xD800 <= code < 0xE000)
    assert normalize(every) == split.normalize(every)


def test_template_copies_share_a_skeleton_and_articles_do_not():
    frequent = {"kaliwatan", "sa", "langaw", "ang", "una", "ning", "gihulagway", "ni", "adtong"}
    first = skeleton(CEBUANO.format("Culicoides dampfi", "Root", "1937"), frequent)
    name = "Wirth ug Blanton"
    second = skeleton(CEBUANO.format("Culicoides daedaloides", name, "1959"), frequent)
    assert first == second
    assert first != skeleton("Ang langaw sa Panama dako kaayo ug daghan.", frequent)


def test_unspaced_scripts_are_split_into_characters():
    frequent = set("的是在了和有")
    first = skeleton("北京是中国的首都，有很多人。", frequent)
    assert first != skeleton("东京是日本的首都，在本州岛。", frequent)
    assert WORD.findall("abc 北京") == ["abc", "北", "京"]


def test_clean_drops_other_language_contaminated_and_non_prose_lines():
    text = ("Inti nisqaqa huk quyllurmi.\n\nThis is the English line.\n"
            "Allinllachu kanki, wawqey.\n\nKaypipas qhaway \n Ojos del Salado")
    found = cleaner(prose_only=True)
    assert found.clean(text) == "Inti nisqaqa huk quyllurmi."
    assert found.counts == {"lines_not_prose": 2, "lines_other_language": 1,
                            "lines_contaminated": 1}


def test_fineweb_keeps_short_lines_and_does_not_judge_them():
    assert cleaner(prose_only=False).clean("Inti\nKilla\nthe end") == "Inti\nKilla\nthe end"


def test_validation_split_is_seeded_and_about_the_fraction():
    config = {"seed": 0}
    chosen = [in_validation(str(i), config, 0.01) for i in range(20000)]
    assert chosen == [in_validation(str(i), config, 0.01) for i in range(20000)]
    assert 100 < sum(chosen) < 300
    assert chosen != [in_validation(str(i), {"seed": 1}, 0.01) for i in range(20000)]


def test_chinese_articles_go_to_their_script():
    chars = {"hans": set("们这会"), "hant": set("們這會")}
    assert script_of("我们在这里开会。", chars, 3) == "hans"
    assert script_of("我們在這裡開會。", chars, 3) == "hant"
    assert script_of("北京是首都。", chars, 3) is None  # no telling characters
    assert script_of("我们們这這", chars, 3) is None  # mixed
