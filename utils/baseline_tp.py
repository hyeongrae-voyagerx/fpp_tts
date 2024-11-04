_pad = "_"
_punctuation = "!'\",.:;? "
_math = "#%&*+-/[]()"
_special = "_—"
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
    )
)

symbol_to_id = {symbol: i for i, symbol in enumerate(symbols)}