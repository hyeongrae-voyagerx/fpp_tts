import re
import os
import cmudict
from jamo import j2h, h2j


def adjust(arpabets):
    '''Modify arpabets so that it fits our processes'''
    string = " " + " ".join(arpabets) + " $"
    string = re.sub("\d", "", string)
    string = string.replace(" T S ", " TS ")
    string = string.replace(" D Z ", " DZ ")
    string = string.replace(" AW ER ", " AWER ")
    string = string.replace(" IH R $", " IH ER ")
    string = string.replace(" EH R $", " EH ER ")
    string = string.replace(" $", "")

    return string.strip("$ ").split()


def num_to_hangeul(num, sino=True):
    """Process a string looking like arabic number.
    num: string. Consists of [0-9,]. e.g., 12,345
    sino: boolean. If True, sino-Korean numerals, i.e., 일, 이, .. are considered.
        Otherwise, pure Korean ones in their modifying forms such as 한, 두, ... are returned.

    >>> process_num("123,456,789", sino=True)
    일억이천삼백사십오만육천칠백팔십구

    >>> process_num("123,456,789", sino=False)
    일억이천삼백사십오만육천칠백여든아홉
    """
    num = re.sub(",", "", num)

    if num == "0":
        return "영"
    if not sino and num == "20":
        return "스무"

    digits = "123456789"
    names = "일이삼사오육칠팔구"
    digit2name = {d: n for d, n in zip(digits, names)}

    modifiers = "한 두 세 네 다섯 여섯 일곱 여덟 아홉"
    decimals = "열 스물 서른 마흔 쉰 예순 일흔 여든 아흔"
    digit2mod = {d: mod for d, mod in zip(digits, modifiers.split())}
    digit2dec = {d: dec for d, dec in zip(digits, decimals.split())}

    spelledout = []
    for i, digit in enumerate(num):
        i = len(num) - i - 1
        if sino:
            if i == 0:
                name = digit2name.get(digit, "")
            elif i == 1:
                name = digit2name.get(digit, "") + "십"
                name = name.replace("일십", "십")
        else:
            if i == 0:
                name = digit2mod.get(digit, "")
            elif i == 1:
                name = digit2dec.get(digit, "")
        if digit == '0':
            if i % 4 == 0:
                last_three = spelledout[-min(3, len(spelledout)):]
                if "".join(last_three) == "":
                    spelledout.append("")
                    continue
            else:
                spelledout.append("")
                continue
        if i == 2:
            name = digit2name.get(digit, "") + "백"
            name = name.replace("일백", "백")
        elif i == 3:
            name = digit2name.get(digit, "") + "천"
            name = name.replace("일천", "천")
        elif i == 4:
            name = digit2name.get(digit, "") + "만"
            name = name.replace("일만", "만")
        elif i == 5:
            name = digit2name.get(digit, "") + "십"
            name = name.replace("일십", "십")
        elif i == 6:
            name = digit2name.get(digit, "") + "백"
            name = name.replace("일백", "백")
        elif i == 7:
            name = digit2name.get(digit, "") + "천"
            name = name.replace("일천", "천")
        elif i == 8:
            name = digit2name.get(digit, "") + "억"
        elif i == 9:
            name = digit2name.get(digit, "") + "십"
        elif i == 10:
            name = digit2name.get(digit, "") + "백"
        elif i == 11:
            name = digit2name.get(digit, "") + "천"
        elif i == 12:
            name = digit2name.get(digit, "") + "조"
        elif i == 13:
            name = digit2name.get(digit, "") + "십"
        elif i == 14:
            name = digit2name.get(digit, "") + "백"
        elif i == 15:
            name = digit2name.get(digit, "") + "천"
        spelledout.append(name)

    return "".join(elem for elem in spelledout)


def to_choseong(arpabet):
    '''Arpabet to choseong or onset'''
    d = \
        {'B': 'ᄇ',
         'CH': 'ᄎ',
         'D': 'ᄃ',
         'DH': 'ᄃ',
         'DZ': 'ᄌ',
         'F': 'ᄑ',
         'G': 'ᄀ',
         'HH': 'ᄒ',
         'JH': 'ᄌ',
         'K': 'ᄏ',
         'L': 'ᄅ',
         'M': 'ᄆ',
         'N': 'ᄂ',
         'NG': 'ᄋ',
         'P': 'ᄑ',
         'R': 'ᄅ',
         'S': 'ᄉ',
         'SH': 'ᄉ',
         'T': 'ᄐ',
         'TH': 'ᄉ',
         'TS': 'ᄎ',
         'V': 'ᄇ',
         'W': 'W',
         'Y': 'Y',
         'Z': 'ᄌ',
         'ZH': 'ᄌ'}

    return d.get(arpabet, arpabet)


def to_jungseong(arpabet):
    '''Arpabet to jungseong or vowel'''
    d = \
        {'AA': 'ᅡ',
         'AE': 'ᅢ',
         'AH': 'ᅥ',
         'AO': 'ᅩ',
         'AW': 'ᅡ우',
         'AWER': "ᅡ워",
         'AY': 'ᅡ이',
         'EH': 'ᅦ',
         'ER': 'ᅥ',
         'EY': 'ᅦ이',
         'IH': 'ᅵ',
         'IY': 'ᅵ',
         'OW': 'ᅩ',
         'OY': 'ᅩ이',
         'UH': 'ᅮ',
         'UW': 'ᅮ'}
    return d.get(arpabet, arpabet)


def to_jongseong(arpabet):
    '''Arpabet to jongseong or coda'''
    d = \
        {'B': 'ᆸ',
         'CH': 'ᆾ',
         'D': 'ᆮ',
         'DH': 'ᆮ',
         'F': 'ᇁ',
         'G': 'ᆨ',
         'HH': 'ᇂ',
         'JH': 'ᆽ',
         'K': 'ᆨ',
         'L': 'ᆯ',
         'M': 'ᆷ',
         'N': 'ᆫ',
         'NG': 'ᆼ',
         'P': 'ᆸ',
         'R': 'ᆯ',
         'S': 'ᆺ',
         'SH': 'ᆺ',
         'T': 'ᆺ',
         'TH': 'ᆺ',
         'V': 'ᆸ',
         'W': 'ᆼ',
         'Y': 'ᆼ',
         'Z': 'ᆽ',
         'ZH': 'ᆽ'}

    return d.get(arpabet, arpabet)


def reconstruct(string):
    '''Some postprocessing rules'''
    pairs = [("그W", "ᄀW"),
             ("흐W", "ᄒW"),
             ("크W", "ᄏW"),
             ("ᄂYᅥ", "니어"),
             ("ᄃYᅥ", "디어"),
             ("ᄅYᅥ", "리어"),
             ("Yᅵ", "ᅵ"),
             ("Yᅡ", "ᅣ"),
             ("Yᅢ", "ᅤ"),
             ("Yᅥ", "ᅧ"),
             ("Yᅦ", "ᅨ"),
             ("Yᅩ", "ᅭ"),
             ("Yᅮ", "ᅲ"),
             ("Wᅡ", "ᅪ"),
             ("Wᅢ", "ᅫ"),
             ("Wᅥ", "ᅯ"),
             ("Wᅩ", "ᅯ"),
             ("Wᅮ", "ᅮ"),
             ("Wᅦ", "ᅰ"),
             ("Wᅵ", "ᅱ"),
             ("ᅳᅵ", "ᅴ"),
             ("Y", "ᅵ"),
             ("W", "ᅮ")
             ]
    for str1, str2 in pairs:
        string = string.replace(str1, str2)
    return string


def compose(letters):
    # insert placeholder
    letters = re.sub("(^|[^\u1100-\u1112])([\u1161-\u1175])", r"\1ᄋ\2", letters)

    string = letters # assembled characters
    # c+v+c
    syls = set(re.findall("[\u1100-\u1112][\u1161-\u1175][\u11A8-\u11C2]", string))
    for syl in syls:
        string = string.replace(syl, j2h(*syl))

    # c+v
    syls = set(re.findall("[\u1100-\u1112][\u1161-\u1175]", string))
    for syl in syls:
        string = string.replace(syl, j2h(*syl))

    return string


def group(inp):
    '''For group_vowels=True
    Contemporarily, Korean speakers don't distinguish some vowels.
    '''
    inp = inp.replace("ᅢ", "ᅦ")
    inp = inp.replace("ᅤ", "ᅨ")
    inp = inp.replace("ᅫ", "ᅬ")
    inp = inp.replace("ᅰ", "ᅬ")

    return inp


def _get_examples():
    '''For internal use'''
    text = open('rules.txt', 'r', encoding='utf8').read().splitlines()
    examples = []
    for line in text:
        if line.startswith("->"):
            examples.extend(re.findall("([ㄱ-힣][ ㄱ-힣]*)\[([ㄱ-힣][ ㄱ-힣]*)]", line))
    _examples = []
    for inp, gt in examples:
        for each in gt.split("/"):
            _examples.append((inp, each))

    return _examples


def get_rule_id2text():
    '''for verbose=True'''
    rules = open(os.path.dirname(os.path.abspath(__file__)) + '/rules.txt', 'r', encoding='utf8').read().strip().split("\n\n")
    rule_id2text = dict()
    for rule in rules:
        rule_id, texts = rule.splitlines()[0], rule.splitlines()[1:]
        rule_id2text[rule_id.strip()] = "\n".join(texts)
    return rule_id2text


def gloss(verbose, out, inp, rule):
    '''displays the process and relevant information'''
    if verbose and out != inp and out != re.sub("/[EJPB]", "", inp):
        print(compose(inp), "->", compose(out))
        print("\033[1;31m", rule, "\033[0m")


def to_jamo(string):
    return h2j(string)


def replace_idioms(string, before, verbose=False):
    file = "kr_idioms_before.txt" if before else "kr_idioms_after.txt"
    idioms_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
    rule = f"from {file}"
    out = string

    for line in open(idioms_path, 'r', encoding="utf8"):
        line = line.split("#")[0].strip()
        if "===" in line:
            str1, str2 = line.split("===")
            out = re.sub(str1, str2, out)
    gloss(verbose, out, string, rule)

    return out


def convert_eng(string, cmu=cmudict.dict()):
    '''Convert a string such that English words inside are turned into Hangul.
    string: input string.
    cmu: cmu dict object.

    >>> convert_eng("그 사람 좀 old school이야", cmu)
    그 사람 좀 올드 스쿨이야
    '''
    eng_words = set(re.findall("[A-Za-z']+", string))
    for eng_word in eng_words:
        word = eng_word.lower()
        if word not in cmu:
            continue

        arpabets = cmu[word][0] # https://en.wikipedia.org/wiki/ARPABET
        phonemes = adjust(arpabets)
        ret = ""
        for i in range(len(phonemes)):
            p = phonemes[i] # phoneme
            p_prev = phonemes[i - 1] if i > 0 else "^"
            p_next = phonemes[i + 1] if i < len(phonemes) - 1 else "$"
            p_next2 = phonemes[i + 1] if i < len(phonemes) - 2 else "$"

            # desginated sets
            short_vowels = ("AE", "AH", "AX", "EH", "IH", "IX", "UH")
            vowels = "AEIOUY"
            consonants = "BCDFGHJKLMNPQRSTVWXZ"
            syllable_final_or_consonants = "$BCDFGHJKLMNPQRSTVWXZ"

            # 외래어 표기법 https://ko.dict.naver.com/help.nhn?page=4-1-3-1#dtl_cts
            #  1항. 무성 파열음 ([p], [t], [k])
            # 1. 짧은 모음 다음의 어말 무성 파열음([p], [t], [k])은 받침으로 적는다.
            # 2. 짧은 모음과 유음·비음([l], [r], [m], [n]) 이외의 자음 사이에 오는 무성 파열음([p], [t], [k])은 받침으로 적는다.
            # 3. 위 경우 이외의 어말과 자음 앞의 [p], [t], [k]는 '으'를 붙여 적는다.

            if p in "PTK":
                if p_prev[:2] in short_vowels and p_next == "$":  # 1
                    ret += to_jongseong(p)
                elif p_prev[:2] in short_vowels and p_next[0] not in "AEIOULRMN":  # 2
                    ret += to_jongseong(p)
                elif p_next[0] in "$BCDFGHJKLMNPQRSTVWXYZ":  # 3
                    ret += to_choseong(p)
                    ret += "ᅳ"
                else:
                    ret += to_choseong(p)

            # 2항. 유성 파열음([b], [d], [g])
            # 어말과 모든 자음 앞에 오는 유성 파열음은 '으'를 붙여 적는다.
            elif p in "BDG":
                ret += to_choseong(p)
                if p_next[0] in syllable_final_or_consonants:
                    ret += "ᅳ"

            # 3항. 마찰음([s], [z], [f], [v], [θ], [ð], [ʃ], [ʒ])
            # 1. 어말 또는 자음 앞의 [s], [z], [f], [v], [θ], [ð]는 '으'를 붙여 적는다.
            # 2. 어말의 [ʃ]는 '시'로 적고, 자음 앞의 [ʃ]는 '슈'로, 모음 앞의 [ʃ]는 뒤따르는 모음에 따라 '샤', '섀', '셔', '셰', '쇼', '슈', '시'로 적는다.
            # 3. 어말 또는 자음 앞의 [ʒ]는 '지'로 적고, 모음 앞의 [ʒ]는 'ㅈ'으로 적는다.
            elif p in ("S", "Z", "F", "V", "TH", "DH", "SH", "ZH"):
                ret += to_choseong(p)

                if p in ("S", "Z", "F", "V", "TH", "DH"):  # 1
                    if p_next[0] in syllable_final_or_consonants:
                        ret += "ᅳ"
                elif p == "SH":  # 2
                    if p_next[0] in "$":
                        ret += "ᅵ"
                    elif p_next[0] in consonants:
                        ret += "ᅲ"
                    else:
                        ret += "Y"
                elif p == "ZH":  # 3
                    if p_next[0] in syllable_final_or_consonants:
                        ret += "ᅵ"

            # 4항. 파찰음([ʦ], [ʣ], [ʧ], [ʤ])
            # 1. 어말 또는 자음 앞의 [ʦ], [ʣ]는 '츠', '즈'로 적고, [ʧ], [ʤ]는 '치', '지'로 적는다.
            # 2. 모음 앞의 [ʧ], [ʤ]는 'ㅊ', 'ㅈ'으로 적는다.
            elif p in ("TS", "DZ", "CH", "JH",):
                ret += to_choseong(p)  # 2

                if p_next[0] in syllable_final_or_consonants:  # 1
                    if p in ("TS", "DZ"):
                        ret += "ᅳ"
                    else:
                        ret += "ᅵ"

            # 5항. 비음([m], [n], [ŋ])
            # 1. 어말 또는 자음 앞의 비음은 모두 받침으로 적는다.
            # 2. 모음과 모음 사이의 [ŋ]은 앞 음절의 받침 'ㆁ'으로 적는다.
            elif p in ("M", "N", "NG"):
                if p in "MN" and p_next[0] in vowels:
                    ret += to_choseong(p)
                else:
                    ret += to_jongseong(p)

            # 6항. 유음([l])
            # 1. 어말 또는 자음 앞의 [l]은 받침으로 적는다.
            # 2. 어중의 [l]이 모음 앞에 오거나, 모음이 따르지 않는 비음([m], [n]) 앞에 올 때에는 'ㄹㄹ'로 적는다.
            # 3. 다만, 비음([m], [n]) 뒤의 [l]은 모음 앞에 오더라도 'ㄹ'로 적는다.
            elif p == "L":
                if p_prev == "^":  # initial
                    ret += to_choseong(p)
                elif p_next[0] in "$BCDFGHJKLPQRSTVWXZ":  # 1
                    ret += to_jongseong(p)
                elif p_prev in "MN":  # 3
                    ret += to_choseong(p)
                elif p_next[0] in vowels:  # 2
                    ret += "ᆯᄅ"
                elif p_next in "MN" and p_next2[0] not in vowels:  # 2
                    ret += "ᆯ르"

            # custom
            elif p == "ER":
                if p_prev[0] in vowels:
                    ret += "ᄋ"
                ret += to_jungseong(p)
                if p_next[0] in vowels:
                    ret += "ᄅ"
            elif p == "R":
                if p_next[0] in vowels:
                    ret += to_choseong(p)

            # 8항. 중모음1) ([ai], [au], [ei], [ɔi], [ou], [auə])
            # 중모음은 각 단모음의 음가를 살려서 적되, [ou]는 '오'로, [auə]는 '아워'로 적는다.
            elif p[0] in "AEIOU":
                ret += to_jungseong(p)

            else:
                ret += to_choseong(p)

        ret = reconstruct(ret)
        ret = compose(ret)
        ret = re.sub("[\u1100-\u11FF]", "", ret) # remove hangul jamo
        string = string.replace(eng_word, ret)
    return string


def convert_num(string):
    '''Convert a annotated string such that arabic numerals inside are spelled out.
    >>> convert_num("우리 3시/B 10분/B에 만나자.")
    우리 세시/B 십분/B에 만나자.
    '''
    global BOUND_NOUNS

    # Bound Nouns
    tokens = set(re.findall("([\d][\d,]*)([ㄱ-힣]+)/B", string))
    for token in tokens:
        num, bn = token
        if bn in BOUND_NOUNS:
            spelledout = num_to_hangeul(num, sino=False)
        else:
            spelledout = num_to_hangeul(num, sino=True)
        string = string.replace(f"{num}{bn}/B", f"{spelledout}{bn}/B")

    # digit by digit for remaining digits
    digits = "0123456789"
    names = "영일이삼사오육칠팔구"
    for d, n in zip(digits, names):
        string = string.replace(d, n)

    return string


def hangeulize(string, verbose=False):
    # Replace letters before conversion to hangeul
    string = replace_idioms(string, True, verbose)
    # Lower case letters to Hangul
    string = convert_eng(string)
    # Replace letters after hengeul conversion
    string = replace_idioms(string, False, verbose)
    # Spell out arabic numbers
    string = convert_num(string)
    # Convert to Jamo characters
    inp = h2j(string)

    # TODO: It is unclear to me what purpose these played in inference. I doubt they help at all, but I need to verify.
    # # 6. special
    # for func in (jyeo, ye, consonant_ui, josa_ui, vowel_ui, \
    #                 jamo, rieulgiyeok, rieulbieub, verb_nieun, \
    #                 balb, palatalize, modifying_rieul):
    #     inp = func(inp, descriptive, verbose)
    # inp = re.sub("/[PJEB]", "", inp)

    return inp
