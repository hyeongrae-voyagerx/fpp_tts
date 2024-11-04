""" adapted from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
    1. "english_cleaners" for English text
    2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
         the Unidecode library (https://pypi.python.org/pypi/Unidecode)
    3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
         the symbols in symbols.py to match your data).
"""

import re
import jamotools
from .abbreviations import normalize_abbreviations
from .acronyms import normalize_acronyms, spell_acronyms
from .datestime import normalize_datestime
from .letters_and_numbers import normalize_letters_and_numbers
from .numerical import normalize_numbers
from .unidecoder import unidecoder
from .kr_text_processing import replace_idioms, convert_eng, replace_idioms, convert_num, to_jamo
from .jp_text_processing import jp_num2kanji, jp_punc2punc
import cutlet
import unicodedata
import fugashi

kunrei = cutlet.Cutlet(use_foreign_spelling=False, system="kunrei")
hepburn = cutlet.Cutlet(use_foreign_spelling=False, system="hepburn")
nihon = cutlet.Cutlet(use_foreign_spelling=False, system="nihon")

def romaji_kana(text, katsu):
    romaji = katsu.romaji(text)
    tokens = katsu.tagger(cutlet.normalize_text(text))
    romaji_tokens = katsu.romaji_tokens(tokens)
    romaji_2, kana = [], []
    for romaji_token, token in zip(romaji_tokens, tokens):
        romaji_2.append(romaji_token.surface)
        k = token.feature.kana
        if k is not None and k != "":
            kana.append(k)
        else:
            kana.append(token.surface)
        if romaji_token.space:
            romaji_2.append(" ")
            kana.append(" ")
    romaji_2, kana = "".join(romaji_2).strip(), "".join(kana).strip()

    assert(romaji == romaji_2)
    assert(len(romaji.split(" ")), len(kana.split(" ")))

    return romaji, kana


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")


def expand_abbreviations(text):
    return normalize_abbreviations(text)


def expand_numbers(text):
    return normalize_numbers(text)


def expand_acronyms(text):
    return normalize_acronyms(text)


def expand_datestime(text):
    return normalize_datestime(text)


def expand_letters_and_numbers(text):
    return normalize_letters_and_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def separate_acronyms(text):
    text = re.sub(r"([0-9]+)([a-zA-Z]+)", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z]+)([0-9]+)", r"\1 \2", text)
    return text


def convert_to_ascii(text):
    return unidecoder(text)


def basic_cleaners(text):
    """Basic pipeline that collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, with number and abbreviation expansion."""
    # text = convert_to_ascii(text)
    text = jamotools.split_syllables(text, jamo_type="JAMO")
    text = lowercase(text)

    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners_v2(text):
    text = convert_to_ascii(text)
    text = expand_datestime(text)
    text = expand_letters_and_numbers(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = spell_acronyms(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    # compatibility with basic_english symbol set
    text = re.sub(r"/+", " ", text)
    return text


def korean_cleaners(text):
    text = lowercase(text)
    text = replace_idioms(text, True)
    text = convert_eng(text)
    text = replace_idioms(text, False)
    text = convert_num(text)
    text = to_jamo(text)
    text = collapse_whitespace(text)

    return text

def japanese_hepburn(text):
    text = lowercase(text)
    # text = convert_eng(text)
    # text = convert_num(text)
    # text = to_jamo(text)
    text, kana = romaji_kana(text, nihon)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text, kana

def japanese_kunrei(text):
    text = lowercase(text)
    # text = convert_eng(text)
    # text = convert_num(text)
    # text = to_jamo(text)
    text, kana = romaji_kana(text, nihon)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text, kana

def japanese_nihon(text):
    text = unicodedata.normalize('NFKC', text)
    text = lowercase(text)
    text = text.replace('、', ",")
    # text = convert_eng(text)
    # text = to_jamo(text)
    text = jp_num2kanji(text)
    text, kana = romaji_kana(text, nihon)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text, kana

def japanese_kana(text):
    text = unicodedata.normalize('NFKC', text)
    text = lowercase(text)
    text = text.replace('、', ",")
    # text = convert_eng(text)
    # text = to_jamo(text)
    text = jp_num2kanji(text)
    _, kana = romaji_kana(text, nihon)
    kana = jp_punc2punc(kana)
    kana = collapse_whitespace(kana)
    return kana
