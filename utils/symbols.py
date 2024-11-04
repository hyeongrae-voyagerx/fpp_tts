""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """
from .cmudict import valid_symbols


# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in valid_symbols]


def get_symbols(symbol_set="english_basic"):
    if symbol_set == "english_basic":
        _pad = "_"
        _punctuation = "!'(),.:;? "
        _special = "-"
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == "english_basic_lowercase":
        _pad = "_"
        _punctuation = "!'\",.:;? "
        _math = "#%&*+-/[]()"
        _special = "_—"
        _letters = "abcdefghijklmnopqrstuvwxyz"
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == "english_expanded":
        _punctuation = "!'\",.:;? "
        _math = "#%&*+-/[]()"
        _special = "_@©°½—₩€$"
        _accented = "áçéêëñöøćž"
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        symbols = (
            list(_punctuation + _math + _special + _accented + _letters) + _arpabet
        )
    elif symbol_set == "hangul_jamo":
        _pad = "_"
        _punctuation = "!'\",.:;? "
        _math = "#%&*+-/[]()"
        _special = "_—"
        _letters = "abcdefghijklmnopqrstuvwxyz"
        _jamo_leads = "".join([chr(_) for _ in range(0x1100, 0x1113)])
        _jamo_vowels = "".join([chr(_) for _ in range(0x1161, 0x1176)])
        _jamo_tails = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])
        symbols = (
            list(
                _pad
                + _punctuation
                + _math
                + _special
                + _jamo_leads
                + _jamo_vowels
                + _jamo_tails
                + _letters
            )
        )
    elif symbol_set == "japanese_kana":
        _pad = "_"
        _punctuation = "!'\",.:;? "
        _math = "#%&*+-/[]()"
        _special = "_—"
        _letters = "abcdefghijklmnopqrstuvwxyz"
        # Hiragana range: U+3040 to U+309F
        _hiragana = "".join([chr(_) for _ in range(0x3040, 0x30A0)])
        # Katakana range: U+30A0 to U+30FF
        _katakana = "".join([chr(_) for _ in range(0x30A0, 0x3100)])
        symbols = (
            list(
                _pad
                + _punctuation
                + _math
                + _special
                + _hiragana
                + _katakana
                + _letters
            )
        )
    else:
        raise Exception("{} symbol set does not exist".format(symbol_set))

    return symbols


def get_pad_idx(symbol_set="english_basic"):
    if symbol_set in {"english_basic", "english_basic_lowercase"}:
        return 0
    else:
        raise Exception("{} symbol set not used yet".format(symbol_set))
