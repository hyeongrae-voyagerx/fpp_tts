""" from https://github.com/keithito/tacotron """
import re
from unicodedata import normalize

from .cleaners import collapse_whitespace
from .symbols import lang_to_dict, lang_to_dict_inverse
from .symbols import lang_to_symbols

from .external_kr_cleaner import normalize as normalize_kr
def korean_cleaners(text):
    '''Pipeline for Korean text, including collapses whitespace.'''
    text = normalize_kr(text)
    text = collapse_whitespace(text)
    text = normalize('NFKD', text)
    return text


def text_to_sequence(raw_text, lang="ko_KR_spaces", return_phones=False):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
        text: string to convert to a sequence
        lang: language of the input text
    Returns:
        List of integers corresponding to the symbols in the text
    '''
    _symbol_to_id = lang_to_dict(lang)
    raw_text = raw_text.replace("\u200b", " ")
    raw_text = raw_text.replace('_x000D_','')
    text = collapse_whitespace(raw_text)
    if lang == 'ko_KR_spaces':    
        text = normalize_kr(text)

        text = normalize('NFKD', text)
        text = " " + text + " "
        sequence = []
        for symbol in text:
            if symbol in _symbol_to_id:
                sequence.append(_symbol_to_id[symbol])
            else:
                continue            
        # sequence = [_symbol_to_id[symbol] for symbol in text]
        tone = [0 for i in sequence]
        
    elif lang == 'ko_KR':    
        text = normalize_kr(text)

        text = normalize('NFKD', text)
        sequence = []
        for symbol in text:
            if symbol in _symbol_to_id:
                sequence.append(_symbol_to_id[symbol])
            else:
                continue            
        # sequence = [_symbol_to_id[symbol] for symbol in text]
        tone = [0 for i in sequence]
    
    elif lang == 'ko_KR_sep_interword':    
        text = normalize_kr(text)
        space_sep = text.split(" ")
        c = []
        for x in space_sep:
            c.append(("").join([b + "|" for b in x]))
        text = (" ").join(c)

        text = normalize('NFKD', text)
        sequence = []
        for symbol in text:
            if symbol in _symbol_to_id:
                sequence.append(_symbol_to_id[symbol])
            else:
                continue            
        # sequence = [_symbol_to_id[symbol] for symbol in text]
        tone = [0 for i in sequence]
    
    elif lang == 'ko_KR_syllables':
        text = normalize_kr(text)
        text = collapse_whitespace(text)
        sequence = []
        for symbol in text:
            if symbol in _symbol_to_id:
                sequence.append(_symbol_to_id[symbol])
            else:
                continue
        tone = [0 for i in sequence]
        
    elif lang == "ko_pron":
        from ko_pron import romanise
        text = normalize_kr(text)
        text = collapse_whitespace(text)

        # if text starts with punctuation marks remove them all
        text = re.sub(r"^[.,!?]+", "", text)
        # remove any spaces between punctuation marks and words
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        # remove repeated punctuation marks
        text = re.sub(r"([.,!?])\1+", r"\1", text)
        # add space after punctuation marks if it is not at end of word
        text = re.sub(r"([.,!?])(?=[^\s])", r"\1 ", text)

        # # remove all punctuation marks
        # text = re.sub(r"[.,!?]", "", text)
        # save punctuation mark locations by word index
        new_text = ""
        for i, word in enumerate(text.split(" ")):
            punctuation = None
            if word != "":
                if re.match(r"[.,!?]", word[-1]):
                    if len(word) == 1:
                        pass
                    punctuation = word[-1]
                ph_word = romanise(word[:-1], 'rrr') if punctuation else romanise(word, 'rrr')        
                ph_word = ph_word + punctuation if punctuation else ph_word
                new_text = new_text + ph_word + " "
        text = new_text.strip()
        # add a full stop at the end if there is none
        text = text + "." if not re.match(r"[.,!?]", text[-1]) else text
        sequence = []
        for idx,symbol in enumerate(text):
            if re.match(r"[.,!?]", symbol):
                symbol = text[idx-1] + symbol
                if symbol in _symbol_to_id:
                    sequence.pop() if len(sequence) > 0 else None
                    sequence.append(_symbol_to_id[symbol])
            else:
                if symbol in _symbol_to_id:
                    sequence.append(_symbol_to_id[symbol])
                else:
                    continue
        tone = [0 for i in sequence]

    elif lang == "ko_pron_with_tone":
        
        from ko_pron import romanise

        puncs_pad = lang_to_symbols['common']
        puncs_pad_tone_dict = {value: index for index, value in enumerate(puncs_pad)}
        
        text = normalize_kr(text)
        text = collapse_whitespace(text)

        # if text starts with punctuation marks remove them all
        text = re.sub(r"^[.,!?]+", "", text)
        # remove any spaces between punctuation marks and words
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        # remove repeated punctuation marks
        text = re.sub(r"([.,!?])\1+", r"\1", text)
        # add space after punctuation marks if it is not at end of word
        text = re.sub(r"([.,!?])(?=[^\s])", r"\1 ", text)       
        
        new_text = ""
        for i, word in enumerate(text.split(" ")):
            punctuation = None
            if word != "":
                if re.match(r"[.,!?]", word[-1]):
                    if len(word) == 1:
                        pass
                    punctuation = word[-1]
                ph_word = romanise(word[:-1], 'rrr') if punctuation else romanise(word, 'rrr')        
                ph_word = ph_word + punctuation if punctuation else ph_word
                new_text = new_text + ph_word + " "
        text = new_text.strip()
        # add a full stop at the end if there is none
        text = text + "." if not re.match(r"[.,!?]", text[-1]) else text
        #replace full stop with space if it is at the end of the sentence
        # text = re.sub(r"\.$", " ", text)
        sequence = []
        tone = []
        for idx,symbol in enumerate(text[:-1]):
            #  if next symbol is alphabet then tone is 0
            # if symbol is alphabet or space or - or … then append to sequence
            if re.match(r"[a-zA-Z]", text[idx]) or re.match(r"\s", text[idx]) or re.match(r"-", text[idx]) or re.match(r"…", text[idx]):
                sequence.append(_symbol_to_id[symbol])
                if re.match(r"[a-zA-Z]", text[idx+1]):
                    # tone.append(len(puncs_pad_tone_dict))
                    tone.append(_symbol_to_id[text[idx+1]])
                # if next symbol is in puncs_pad list 
                elif text[idx+1] in puncs_pad:
                    # tone.append(puncs_pad_tone_dict[text[idx+1]])
                    tone.append(_symbol_to_id[text[idx+1]])
                else:
                    continue
            else:
                continue
        assert len(sequence) == len(tone), f"sequence and tone length mismatch: {len(sequence)} != {len(tone)}"

    elif lang == "ko_pronwiht_tone_interspersed":
        
        from ko_pron import romanise

        puncs_pad = lang_to_symbols['common']
        puncs_pad_tone_dict = {value: index for index, value in enumerate(puncs_pad)}
        
        text = normalize_kr(text)
        text = collapse_whitespace(text)

        # if text starts with punctuation marks remove them all
        text = re.sub(r"^[.,!?]+", "", text)
        # remove any spaces between punctuation marks and words
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        # remove repeated punctuation marks
        text = re.sub(r"([.,!?])\1+", r"\1", text)
        # add space after punctuation marks if it is not at end of word
        text = re.sub(r"([.,!?])(?=[^\s])", r"\1 ", text)       
        
        new_text = ""
        for i, word in enumerate(text.split(" ")):
            punctuation = None
            if word != "":
                if re.match(r"[.,!?]", word[-1]):
                    if len(word) == 1:
                        pass
                    punctuation = word[-1]
                ph_word = romanise(word[:-1], 'rrr') if punctuation else romanise(word, 'rrr')        
                ph_word = ph_word + punctuation if punctuation else ph_word
                new_text = new_text + ph_word + " "
        text = new_text.strip()
        # add a full stop at the end if there is none
        text = text + "." if not re.match(r"[.,!?]", text[-1]) else text
        #replace full stop with space if it is at the end of the sentence
        # text = re.sub(r"\.$", " ", text)
        sequence = []
        tone = []
        for idx,symbol in enumerate(text[:-1]):
            #  if next symbol is alphabet then tone is 0
            # if symbol is alphabet or space or - or … then append to sequence
            if re.match(r"[a-zA-Z]", text[idx]) or re.match(r"\s", text[idx]) or re.match(r"-", text[idx]) or re.match(r"…", text[idx]):
                sequence.append(_symbol_to_id[symbol])
                if re.match(r"[a-zA-Z]", text[idx]):
                    sequence.append(_symbol_to_id["|"])
                    # tone.append(len(puncs_pad_tone_dict))
                    tone.append(_symbol_to_id["|"])
                if re.match(r"[a-zA-Z]", text[idx+1]):
                    # tone.append(len(puncs_pad_tone_dict))
                    tone.append(_symbol_to_id[text[idx+1]])
                # if next symbol is in puncs_pad list 
                elif text[idx+1] in puncs_pad:
                    # tone.append(puncs_pad_tone_dict[text[idx+1]])
                    tone.append(_symbol_to_id[text[idx+1]])
                else:
                    continue
            else:
                continue
        assert len(sequence) == len(tone), f"sequence and tone length mismatch: {len(sequence)} != {len(tone)}"

    elif lang == "ko_pron_with_tone_and_pos":
        
        from ko_pron import romanise
        
        puncs_pad = lang_to_symbols['common']
        puncs_pad_tone_dict = {value: index for index, value in enumerate(puncs_pad)}
        
        text = normalize_kr(text)
        text = collapse_whitespace(text)

        # if text starts with punctuation marks remove them all
        text = re.sub(r"^[.,!?]+", "", text)
        # remove any spaces between punctuation marks and words
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        # remove repeated punctuation marks
        text = re.sub(r"([.,!?])\1+", r"\1", text)
        # add space after punctuation marks if it is not at end of word
        text = re.sub(r"([.,!?])(?=[^\s])", r"\1 ", text)       
        
        new_text = ""
        for i, word in enumerate(text.split(" ")):
            punctuation = None
            if word != "":
                if re.match(r"[.,!?]", word[-1]):
                    if len(word) == 1:
                        pass
                    punctuation = word[-1]
                ph_word = romanise(word[:-1], 'rr') if punctuation else romanise(word, 'rr')        
                ph_word = ph_word + punctuation if punctuation else ph_word
                new_text = new_text + ph_word + " "
        text = new_text.strip()
        # add a full stop at the end if there is none
        text = text + "." if not re.match(r"[.,!?]", text[-1]) else text
        #add a space before every punctuation mark
        text = re.sub(r"([.,!?])", r" \1", text)
        #remove space after punctuation mark
        text = re.sub(r"([.,!?])\s", r"\1", text)
        #replace full stop with space if it is at the end of the sentence
        # text = re.sub(r"\.$", " ", text)
        sequence = []
        tone = []
        pos = []
        current_pos = 0
        for idx,symbol in enumerate(text[:-1]):
            
            #  if next symbol is alphabet then tone is 0
            # if symbol is alphabet or - or … then append to sequence
            if re.match(r"[a-zA-Z]", text[idx]) or re.match(r"\s", text[idx]) or re.match(r"-", text[idx]) or re.match(r"…", text[idx]):
                sequence.append(_symbol_to_id[symbol])
                pos.append(current_pos)
                tone.append(_symbol_to_id[text[idx+1]])
                # if match space, then increment pos
                if re.match(r"\s", text[idx]) and text[idx+1] not in puncs_pad:
                    current_pos += 1
                # if re.match(r"[a-zA-Z]", text[idx+1]):
                #     tone.append(_symbol_to_id[text[idx+1]])
                # # if next symbol is in puncs_pad list 
                # elif text[idx+1] in puncs_pad:
                #     tone.append(_symbol_to_id[text[idx+1]])
                else:
                    continue
            else:
                continue
        assert len(sequence) == len(tone), f"sequence and tone length mismatch: {len(sequence)} != {len(tone)}"
        assert len(sequence) == len(pos), f"sequence and pos length mismatch: {len(sequence)} != {len(pos)}"

        if return_phones:
            return sequence, tone, pos, text
        else:
            return sequence, tone, pos
    
    elif lang == "ko_pron_with_tone_and_pos_word_size":
        
        from ko_pron import romanise

        puncs_pad = lang_to_symbols['common']
        puncs_pad_tone_dict = {value: index for index, value in enumerate(puncs_pad)}
        
        text = normalize_kr(text)
        text = collapse_whitespace(text)

        # if text starts with punctuation marks remove them all
        text = re.sub(r"^[.,!?]+", "", text)
        # remove any spaces between punctuation marks and words
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        # remove repeated punctuation marks
        text = re.sub(r"([.,!?])\1+", r"\1", text)
        # add space after punctuation marks if it is not at end of word
        text = re.sub(r"([.,!?])(?=[^\s])", r"\1 ", text)       
        
        new_text = ""
        for i, word in enumerate(text.split(" ")):
            punctuation = None
            if word != "":
                if re.match(r"[.,!?]", word[-1]):
                    if len(word) == 1:
                        pass
                    punctuation = word[-1]
                ph_word = romanise(word[:-1], 'rr') if punctuation else romanise(word, 'rr')        
                ph_word = ph_word + punctuation if punctuation else ph_word + " "
                new_text = new_text + ph_word
        text = new_text.strip()
        # add a full stop at the end if there is none
        text = text + "." if not re.match(r"[.,!?]", text[-1]) else text
        # remove space after punctuation mark and before punctuation mark
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        text = re.sub(r"([.,!?])\s", r"\1", text)
        sequence = []
        tone = []
        pos = []
        current_pos = 0

        prev_idx = 0
        for idx,symbol in enumerate(text[:-1]):
            tone.append(puncs_pad_tone_dict[symbol] if symbol in puncs_pad else 0)
            current_tone = puncs_pad_tone_dict[symbol] if symbol in puncs_pad else 0
            if current_tone != 0:
                # change all the previous tones to current tone
                tone[prev_idx:idx] = [current_tone] * idx
                prev_idx = idx
            if re.match(r"[a-zA-Z]", text[idx]):
                sequence.append(_symbol_to_id[symbol])
                pos.append(current_pos)
                # if match space or punctuation, then increment pos
                if re.match(r"\s", text[idx]) or text[idx] in puncs_pad:
                    current_pos += 1
                else:
                    continue
            else:
                continue

        assert len(sequence) == len(tone), f"sequence and tone length mismatch: {len(sequence)} != {len(tone)}"
        assert len(sequence) == len(pos), f"sequence and pos length mismatch: {len(sequence)} != {len(pos)}"

        if return_phones:
            return sequence, tone, pos, text
        else:
            return sequence, tone, pos
        
    elif lang == "ko_pron_with_tone_and_pos_word_size_eos_sos":
        
        from ko_pron import romanise

        puncs_pad = lang_to_symbols['common']
        puncs_pad_tone_dict = {value: index for index, value in enumerate(puncs_pad)}
        
        text = normalize_kr(text)
        text = collapse_whitespace(text)

        # if text starts with punctuation marks remove them all
        text = re.sub(r"^[.,!?]+", "", text)
        # remove any spaces between punctuation marks and words
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        # remove repeated punctuation marks
        text = re.sub(r"([.,!?])\1+", r"\1", text)
        # add space after punctuation marks if it is not at end of word
        text = re.sub(r"([.,!?])(?=[^\s])", r"\1 ", text)       
        
        new_text = ""
        for i, word in enumerate(text.split(" ")):
            punctuation = None
            if word != "":
                if re.match(r"[.,!?]", word[-1]):
                    if len(word) == 1:
                        pass
                    punctuation = word[-1]
                ph_word = romanise(word[:-1], 'rr') if punctuation else romanise(word, 'rr')        
                ph_word = ph_word + punctuation if punctuation else ph_word
                new_text = new_text + ph_word + " "
        text = new_text.strip()
        # add a full stop at the end if there is none
        text = text + "." if not re.match(r"[.,!?]", text[-1]) else text
        # sequence = []
        # tone = []
        # pos = []
        # current_pos = 1
        # word_lengths = [len(word) for word in text.split(" ")]
        # for idx,symbol in enumerate(text[:-1]):
        #     #  if next symbol is alphabet then tone is 0
        #     # if symbol is alphabet or - or … then append to sequence
        #     if re.match(r"[a-zA-Z]", text[idx]):
        #         sequence.append(_symbol_to_id[symbol])
        #         pos.append(current_pos)
        #         tone.append(_symbol_to_id[text[idx+1]])
        #         if text[idx+1] in puncs_pad:
        #             current_pos += 1

        #         else:
        #             continue
        #     else:
        #         continue

        sequence = []
        tone = []
        pos = []
        current_pos = 0

        prev_symbol_idx = -1
        count = 0
        punc_count = 0
        for idx,symbol in enumerate(text):
            if (symbol in lang_to_symbols[lang]) and (symbol not in puncs_pad):
                count = count + 1
                sequence.append(_symbol_to_id[symbol])
                pos.append(current_pos)
                # if match space or punctuation, then increment pos
            if text[idx] in puncs_pad:
                punc_count = punc_count + 1
                if text[idx] == ".":
                    current_pos = 0
                else:
                    current_pos += 1
                tone.append([puncs_pad_tone_dict[symbol]] * (idx - prev_symbol_idx - 1))
                prev_symbol_idx = idx
            else:
                continue
            
        tone = [item for sublist in tone for item in sublist]
        sequence = sequence
        pos = pos

        # sequence = [_symbol_to_id['<SOS>']] + sequence
        # tone = [0] + tone
        # pos = [0] + pos
        assert len(sequence) == len(tone), f"sequence and tone length mismatch: {len(sequence)} != {len(tone)}"
        assert len(sequence) == len(pos), f"sequence and pos length mismatch: {len(sequence)} != {len(pos)}"

        if return_phones:
            return sequence, tone, pos, text
        else:
            return sequence, tone, pos

    elif lang == "raw_korean_with_tone_and_pos_word_size": 
        
        # from .external_kr_cleaner import jamo_to_korean
        # text = jamo_to_korean(text)
        puncs_pad = lang_to_symbols['common']
        puncs_pad_tone_dict = {value: index for index, value in enumerate(puncs_pad)}
        text = normalize_kr(text)
        text = collapse_whitespace(text)
        text = normalize('NFKD', text)
        # if text starts with punctuation marks remove them all
        text = re.sub(r"^[.,!?]+", "", text)
        # remove any spaces between punctuation marks and words
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        # remove repeated punctuation marks
        text = re.sub(r"([.,!?])\1+", r"\1", text)
        # add space after punctuation marks if it is not at end of word
        # text = re.sub(r"([.,!?])(?=[^\s])", r"\1 ", text)       
        # remove space after punctuation mark
        text = re.sub(r"([.,!?])\s", r"\1", text)

        text = text.strip()
        # add a full stop at the end if there is none
        text = text + "." if not re.match(r"[.,!?]", text[-1]) else text

        sequence = []
        tone = []
        pos = []
        current_pos = 0

        prev_symbol_idx = -1
        count = 0
        punc_count = 0
        for idx,symbol in enumerate(text):
            if symbol in lang_to_symbols['raw_korean_with_tone_and_pos_word_size']:
                count = count + 1
                sequence.append(_symbol_to_id[symbol])
                pos.append(current_pos)
                # if match space or punctuation, then increment pos
            if text[idx] in puncs_pad:
                punc_count = punc_count + 1
                current_pos += 1
                tone.append([puncs_pad_tone_dict[symbol]] * (idx - prev_symbol_idx - 1))
                prev_symbol_idx = idx
            else:
                continue
            
        tone = [item for sublist in tone for item in sublist]
        sequence = sequence
        pos = pos

        assert len(sequence) == len(tone), f"sequence and tone length mismatch: {len(sequence)} != {len(tone)}"
        assert len(sequence) == len(pos), f"sequence and pos length mismatch: {len(sequence)} != {len(pos)}"

        if return_phones:
            return sequence, tone, pos, text
        else:
            return sequence, tone, pos
    
    elif lang == "raw_korean_with_tone_and_pos_word_size_with_sos": 
        # this sos is actually eos ; typo ; added to end of sentence!!
        # from .external_kr_cleaner import jamo_to_korean
        # text = jamo_to_korean(text)
        puncs_pad = lang_to_symbols['common']
        puncs_pad_tone_dict = {value: index for index, value in enumerate(puncs_pad)}
        text = normalize_kr(text)
        text = collapse_whitespace(text)
        text = normalize('NFKD', text)
        # if text starts with punctuation marks remove them all
        text = re.sub(r"^[.,!?]+", "", text)
        # remove any spaces between punctuation marks and words
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        # remove repeated punctuation marks
        text = re.sub(r"([.,!?])\1+", r"\1", text)
        # add space after punctuation marks if it is not at end of word
        # text = re.sub(r"([.,!?])(?=[^\s])", r"\1 ", text)       
        # remove space after punctuation mark
        text = re.sub(r"([.,!?])\s", r"\1", text)

        text = text.strip()
        # add a full stop at the end if there is none
        text = text + "." if not re.match(r"[.,!?]", text[-1]) else text

        sequence = []
        tone = []
        pos = []
        current_pos = 0

        prev_symbol_idx = -1
        count = 0
        punc_count = 0
        for idx,symbol in enumerate(text):
            if symbol in lang_to_symbols['raw_korean_with_tone_and_pos_word_size_with_sos']:
                count = count + 1
                sequence.append(_symbol_to_id[symbol])
                pos.append(current_pos)
                # if match space or punctuation, then increment pos
            if text[idx] in puncs_pad:
                punc_count = punc_count + 1
                current_pos += 1
                tone.append([puncs_pad_tone_dict[symbol]] * (idx - prev_symbol_idx - 1))
                prev_symbol_idx = idx
            else:
                continue
            
        tone = [item for sublist in tone for item in sublist]
        # max tone index is puncs_pad_tone_dict length
        tone = tone + [len(puncs_pad_tone_dict)]

        sequence = sequence + [_symbol_to_id['<SOS>']]
        # max pos embedding max leng is 200
        pos = pos + [199]

        assert len(sequence) == len(tone), f"sequence and tone length mismatch: {len(sequence)} != {len(tone)}"
        assert len(sequence) == len(pos), f"sequence and pos length mismatch: {len(sequence)} != {len(pos)}"

        if return_phones:
            return sequence, tone, pos, text
        else:
            return sequence, tone, pos
        
    elif lang == "raw_korean_with_tone_and_pos_word_size_with_eos_and_sos": 
        # this sos is actually eos ; typo ; added to end of sentence!!
        # from .external_kr_cleaner import jamo_to_korean
        # text = jamo_to_korean(text)
        puncs_pad = lang_to_symbols['common']
        puncs_pad_tone_dict = {value: index for index, value in enumerate(puncs_pad)}
        text = normalize_kr(text)
        text = collapse_whitespace(text)
        text = normalize('NFKD', text)
        # if text starts with punctuation marks remove them all
        text = re.sub(r"^[.,!?]+", "", text)
        # remove any spaces between punctuation marks and words
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        # remove repeated punctuation marks
        text = re.sub(r"([.,!?])\1+", r"\1", text)
        # add space after punctuation marks if it is not at end of word
        # text = re.sub(r"([.,!?])(?=[^\s])", r"\1 ", text)       
        # remove space after punctuation mark
        text = re.sub(r"([.,!?])\s", r"\1", text)

        text = text.strip()
        # add a full stop at the end if there is none
        text = text + "." if not re.match(r"[.,!?]", text[-1]) else text

        sequence = []
        tone = []
        pos = []

        current_pos = 0

        prev_symbol_idx = -1
        count = 0
        punc_count = 0
        for idx,symbol in enumerate(text):
            if symbol in lang_to_symbols[lang]:
                count = count + 1
                sequence.append(_symbol_to_id[symbol])
                pos.append(current_pos)
                # if match space or punctuation, then increment pos
            if text[idx] in puncs_pad:
                punc_count = punc_count + 1
                current_pos += 1
                tone.append([puncs_pad_tone_dict[symbol]] * (idx - prev_symbol_idx - 1))
                prev_symbol_idx = idx
            else:
                continue
            
        tone = [item for sublist in tone for item in sublist]

        # max tone index is puncs_pad_tone_dict length
        tone = [len(puncs_pad_tone_dict)] + tone + [len(puncs_pad_tone_dict)]
        sequence = [_symbol_to_id['<SOS>']] + sequence + [_symbol_to_id['<SOS>']]
        # max pos embedding max leng is 200
        pos = [199] + pos + [199]

        assert len(sequence) == len(tone), f"sequence and tone length mismatch: {len(sequence)} != {len(tone)}"
        assert len(sequence) == len(pos), f"sequence and pos length mismatch: {len(sequence)} != {len(pos)}"

        if return_phones:
            return sequence, tone, pos, text
        else:
            return sequence, tone, pos
        
    elif lang == 'en_US':
        _curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')
        sequence = []

        while len(text):
            m = _curly_re.match(text)

            if m is not None:
                ar = m.group(1)
                sequence += [_symbol_to_id[symbol] for symbol in ar]
                ar = m.group(2)
                sequence += [_symbol_to_id[symbol] for symbol in ar.split()]
                text = m.group(3)
            else:
                sequence += [_symbol_to_id[symbol] for symbol in text]
                break

        tone = [0 for i in sequence]

    else:
        raise RuntimeError('Wrong type of lang')

    assert len(sequence) == len(tone)
    if return_phones:
        return sequence, tone, text
    else:
        return sequence, tone


def sequence_to_text(sequence, lang):
    '''Converts a sequence of IDs back to a string'''
    _id_to_symbol = lang_to_dict_inverse(lang)
    result = ''
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text
