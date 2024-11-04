import os
import random
import string
from tqdm import tqdm
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import einx
from einops import rearrange, reduce


from jamo import h2j


# seed everything

def seed_everything(seed = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def lens_to_mask(
    t,
    length: int | None = None
):

    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device = t.device)
    return einx.less('n, b -> b n', seq, t)

def mask_from_start_end_indices(
    seq_len,
    start,
    end
):
    max_seq_len = seq_len.max().item()  
    seq = torch.arange(max_seq_len, device = start.device).long()
    return einx.greater_equal('n, b -> b n', seq, start) & einx.less('n, b -> b n', seq, end)

def mask_from_frac_lengths(
    seq_len,
    frac_lengths
):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min = 0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)

def maybe_masked_mean(
    t,
    mask = None
):

    if not exists(mask):
        return t.mean(dim = 1)

    t = einx.where('b n, b n d, -> b n d', mask, t, 0.)
    num = reduce(t, 'b n d -> b d', 'sum')
    den = reduce(mask.float(), 'b n -> b', 'sum')

    return einx.divide('b d, b -> b d', num, den.clamp(min = 1.))


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(
    text: list[str],
    padding_value = -1
):
    text = [h2j(t) for t in text]
    list_tensors = [torch.tensor([*bytes(t, 'UTF-8')]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value = padding_value, batch_first = True)
    return text

# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value = -1
):
    text = [h2j(t) for t in text]
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value = padding_value, batch_first = True)
    return text


# save spectrogram
def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin='lower', aspect='auto')
    plt.colorbar()
    plt.savefig(path)
    plt.close()


# seedtts testset metainfo: utt, prompt_text, prompt_wav, gt_text, gt_wav
def get_seedtts_testset_metainfo(metalst):
    f = open(metalst); lines = f.readlines(); f.close()
    metainfo = []
    for line in lines:
        if len(line.strip().split('|')) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split('|')
        elif len(line.strip().split('|')) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split('|')
            gt_wav = os.path.join(os.path.dirname(metalst), "wavs", utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)
        metainfo.append((utt, prompt_text, prompt_wav, gt_text, gt_wav))
    return metainfo


# librispeech test-clean metainfo: gen_utt, ref_txt, ref_wav, gen_txt, gen_wav
def get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path):
    f = open(metalst); lines = f.readlines(); f.close()
    metainfo = []
    for line in lines:
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split('\t')

        # ref_txt = ref_txt[0] + ref_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        ref_spk_id, ref_chaptr_id, _ =  ref_utt.split('-')
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + '.flac')

        # gen_txt = gen_txt[0] + gen_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        gen_spk_id, gen_chaptr_id, _ =  gen_utt.split('-')
        gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + '.flac')

        metainfo.append((gen_utt, ref_txt, ref_wav, " " + gen_txt, gen_wav))

    return metainfo


# padded to max length mel batch
def padded_mel_batch(ref_mels):
    max_mel_length = torch.LongTensor([mel.shape[-1] for mel in ref_mels]).amax()
    padded_ref_mels = []
    for mel in ref_mels:
        padded_ref_mel = F.pad(mel, (0, max_mel_length - mel.shape[-1]), value = 0)
        padded_ref_mels.append(padded_ref_mel)
    padded_ref_mels = torch.stack(padded_ref_mels)
    padded_ref_mels = rearrange(padded_ref_mels, 'b d n -> b n d')
    return padded_ref_mels


# get wav_res_ref_text of seed-tts test metalst
# https://github.com/BytedanceSpeech/seed-tts-eval

def get_seed_tts_test(metalst, gen_wav_dir, gpus):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        if len(line.strip().split('|')) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split('|')
        elif len(line.strip().split('|')) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split('|')

        if not os.path.exists(os.path.join(gen_wav_dir, utt + '.wav')):
            continue
        gen_wav = os.path.join(gen_wav_dir, utt + '.wav')
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)

        test_set_.append((gen_wav, prompt_wav, gt_text))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]
    
    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i*wav_per_job:(i+1)*wav_per_job]))

    return test_set


# get librispeech test-clean cross sentence test

def get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth = False):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split('\t')

        if eval_ground_truth:
            gen_spk_id, gen_chaptr_id, _ =  gen_utt.split('-')
            gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + '.flac')
        else:
            if not os.path.exists(os.path.join(gen_wav_dir, gen_utt + '.wav')):
                raise FileNotFoundError(f"Generated wav not found: {gen_utt}")
            gen_wav = os.path.join(gen_wav_dir, gen_utt + '.wav')

        ref_spk_id, ref_chaptr_id, _ =  ref_utt.split('-')
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + '.flac')

        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]
    
    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i*wav_per_job:(i+1)*wav_per_job]))

    return test_set


# load asr model

def load_asr_model(lang, ckpt_dir = ""):
    if lang == "zh":
        from funasr import AutoModel
        model = AutoModel(
            model = os.path.join(ckpt_dir, "paraformer-zh"), 
            # vad_model = os.path.join(ckpt_dir, "fsmn-vad"), 
            # punc_model = os.path.join(ckpt_dir, "ct-punc"),
            # spk_model = os.path.join(ckpt_dir, "cam++"), 
            disable_update=True,
            )  # following seed-tts setting
    elif lang == "en":
        from faster_whisper import WhisperModel
        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return model

# WER Evaluation, the way Seed-TTS does

def run_asr_wer(args):
    rank, lang, test_set, ckpt_dir = args

    if lang == "zh":
        import zhconv
        torch.cuda.set_device(rank)
    elif lang == "en":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    else:
        raise NotImplementedError("lang support only 'zh' (funasr paraformer-zh), 'en' (faster-whisper-large-v3), for now.")

    asr_model = load_asr_model(lang, ckpt_dir = ckpt_dir)
    
    from zhon.hanzi import punctuation
    punctuation_all = punctuation + string.punctuation
    wers = []

    from jiwer import compute_measures
    for gen_wav, prompt_wav, truth in tqdm(test_set):
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, 'zh-cn')
        elif lang == "en":
            segments, _ = asr_model.transcribe(gen_wav, beam_size=5, language="en")
            hypo = ''
            for segment in segments:
                hypo = hypo + ' ' + segment.text

        # raw_truth = truth
        # raw_hypo = hypo

        for x in punctuation_all:
            truth = truth.replace(x, '')
            hypo = hypo.replace(x, '')

        truth = truth.replace('  ', ' ')
        hypo = hypo.replace('  ', ' ')

        if lang == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()

        measures = compute_measures(truth, hypo)
        wer = measures["wer"]

        # ref_list = truth.split(" ")
        # subs = measures["substitutions"] / len(ref_list)
        # dele = measures["deletions"] / len(ref_list)
        # inse = measures["insertions"] / len(ref_list)

        wers.append(wer)

    return wers

# filter func for dirty data with many repetitions

def repetition_found(text, length = 2, tolerance = 10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i:i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


# load model checkpoint for inference

def load_checkpoint(model, ckpt_path, device, use_ema = True):
    from ema_pytorch import EMA
    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device)

    if use_ema == True:
        ema_model = EMA(model, include_online_model = False).to(device)
        if ckpt_type == "safetensors":
            ema_model.load_state_dict(checkpoint)
        else:
            ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        ema_model.copy_params_from_ema_to_model()
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    return model