from .korean_syllables import krsyl, syllist

_pad = '_'
_punc = ";:,.!?¡¿—-…«»'“”~() "
_ko_pron_sos = "<SOS>"
_ko_pron_eos = "<EOS>"

_jamo_leads = "".join([chr(_) for _ in range(0x1100, 0x1113)])
_jamo_vowels = "".join([chr(_) for _ in range(0x1161, 0x1176)])
_jamo_tails = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])
_kor_characters = _jamo_leads + _jamo_vowels + _jamo_tails + "|"

_kor_syllables_list = syllist

_cmu_characters = [
    'AA', 'AE', 'AH',
    'AO', 'AW', 'AY',
    'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
    'F', 'G', 'HH', 'IH', 'IY',
    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY',
    'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
    'V', 'W', 'Y', 'Z', 'ZH'
]


_ko_pron_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
_ko_pron_characters = list(_ko_pron_letters) + list(_letters_ipa)

punctuation_marks = ['.', '?', '!', ',', ' ']
_ko_pron_char_with_puncs = [y + p for y in _ko_pron_characters for p in punctuation_marks]
_ko_pron_characters_enchanced_puncs = _ko_pron_characters + _ko_pron_char_with_puncs
    
lang_to_symbols = {
    'common': [_pad] + list(_punc),
    'ko_KR': list(_kor_characters), 
    'ko_KR_spaces' : list(_kor_characters),
    'en_US': _cmu_characters, 
    
    'ko_pron': _ko_pron_characters_enchanced_puncs,
    'ko_pron_with_tone': list(_ko_pron_letters) + ['…'],
    'ko_pronwiht_tone_interspersed': list(_ko_pron_letters) + ['…', '|'],
    'ko_pron_with_tone_and_pos': list(_ko_pron_letters) + ['…'],
    'ko_pron_with_tone_and_pos_word_size': list(_ko_pron_letters) + ['…'],
    'ko_pron_with_tone_and_pos_word_size_eos_sos': list(_ko_pron_letters) + ['…'] + [_ko_pron_sos],

    'ko_KR_syllables': _kor_syllables_list,

    'raw_korean_with_tone_and_pos_word_size': list(_kor_characters),
    'raw_korean_with_tone_and_pos_word_size_with_sos': list(_kor_characters) + [_ko_pron_sos],
    'raw_korean_with_tone_and_pos_word_size_with_eos_and_sos': list(_kor_characters) + [_ko_pron_sos, _ko_pron_eos],
    
    'ko_KR_sep_interword': list(_kor_characters)
}

def lang_to_dict(lang):
    symbol_lang = lang_to_symbols['common'] + lang_to_symbols[lang]
    dict_lang = {s: i for i, s in enumerate(symbol_lang)}
    return dict_lang

def lang_to_dict_inverse(lang):
    symbol_lang = lang_to_symbols['common'] + lang_to_symbols[lang]
    dict_lang = {i: s for i, s in enumerate(symbol_lang)}
    return dict_lang

def symbol_len(lang): # "ko_KR_spaces" -> 89
    symbol_lang = lang_to_symbols['common'] + lang_to_symbols[lang]
    return len(symbol_lang)
