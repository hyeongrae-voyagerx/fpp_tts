import torch
import torch.nn.functional as F
import webdataset as wds
import numpy as np
from scipy.stats import betabinom
import os.path as osp
from glob import glob
from utils.baseline_tp import symbol_to_id
from unicodedata import normalize

from .text_utils import text_to_sequence

_static_mel_dir = "/data/tts/korean_va/static_mels"

class KoreanVADataset:
    def __init__(self, data_dir):
        files = glob(osp.join(data_dir, "*.tar"))
        self.dataset = wds.WebDataset(files).decode()

    @classmethod
    def get_dataloader(cls, dataset_kwargs, dataloader_kwargs):
        return wds.WebLoader(cls(**dataset_kwargs).dataset, collate_fn=collate_fn, **dataloader_kwargs)
    

def repair_keys(source):
    for sample in source:
        sample = {key.split(".")[0]: val if val != "N" else None for key, val in sample.items()}
        sample["name"] = sample["__key__"]
        yield sample


def collate_fn(batch):
    text_data, audio_data, mel_data, pitch_data, energy_data, speaker_data = \
        fetch(batch, "raw_text.txt", "audio.pyd", "mel.pyd", "pitch.pyd", "energy.pyd", "speaker.pyd")
    
    text, text_lens = text_data
    mel, mel_lens = mel_data
    pitch, _ = pitch_data
    energy, _ = energy_data
    speaker, _ = speaker_data
    attn_prior = get_attn_priors(mel_lens, text_lens)
    audio, audio_lens = audio_data

    static_mels = []
    for item in batch:
        speaker_string = "".join([stringify(item, s) for s in ("speaker.pyd", "emotion.txt", "sensitivity.txt", "style.txt", "character.txt", "character_emotion.txt")])
        static_mel = load_static_mel(_static_mel_dir, speaker_string)
        static_mels.append(static_mel)
    static_mels, static_mels_len = merge_to_tensor(static_mels, 1)
    
    return [text, text_lens, mel, mel_lens, static_mels, static_mels_len, pitch, energy, speaker, attn_prior, audio]


def fetch(batch, *keys):
    fetch_result = []
    for key in keys:
        data_list = [item_dict[key] for item_dict in batch]

        if isinstance(data_list[0], str):
            if key == "raw_text.txt":
                text_data = [torch.tensor([symbol_to_id[c] for c in normalize("NFKD", text) if c in symbol_to_id]) for text in data_list]
                # text_data = [torch.tensor(text_to_sequence(text)[0]) for text in data_list]
                data, data_len = merge_to_tensor(text_data, 0)
        elif isinstance(data_list[0], torch.Tensor):
            if (variant_dim:=variant_len(data_list)) is not None:
                data, data_len = merge_to_tensor(data_list, variant_dim)
            else:
                data, data_len = torch.stack(data_list), None
        fetch_result.append([data, data_len])
    return fetch_result

def variant_len(data_list):
    assert len({item.ndim for item in data_list}) == 1
    shapes = [data.shape for data in data_list]

    variant_dim = []
    for dim, dim_shapes in enumerate(zip(*shapes)):
        if len(set(dim_shapes)) > 1:
            variant_dim.append(dim)

    assert len(variant_dim) < 2
    return variant_dim[0] if variant_dim else None


def merge_to_tensor(data_list, variant_dim):
    lengths = [data.shape[variant_dim] for data in data_list]
    max_len_data = max(lengths)
    padded_data = torch.stack([F.pad(data, [0, max_len_data-data.shape[variant_dim]]) for data in data_list])
    return padded_data, torch.tensor(lengths)

def get_attn_priors(mel_lens, text_lens):
    max_mel_lens = max(mel_lens)
    max_text_lens = max(text_lens)
    attn_priors = torch.zeros((len(mel_lens), max_mel_lens, max_text_lens))
    for i, (mlen, tlen) in enumerate(zip(mel_lens, text_lens)):
        attn_prior = beta_binomial_prior_distribution(tlen, mlen)
        attn_priors[i, :mlen, :tlen] = attn_prior

    return attn_priors

def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M + 1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))

def load_static_mel(directory, filename_string):
    path = osp.join(directory, f"{filename_string}.pt")
    mel_spec = torch.load(path, weights_only=False)
    return mel_spec

def stringify(item, key):
    match key:
        case "speaker.pyd":
            return str(item[key].item())
        case "emotion.txt":
            if item[key] == "N": return "None"
            else: return item[key]
        case "sensitivity.txt":
            if item[key] == "N": return "None"
            else: return item[key]
        case "style.txt":
            if item[key] == "N": return "None"
            else: return item[key]
        case "character.txt":
            if item[key] == "N": return "None"
            else: return item[key]
        case "character_emotion.txt":
            if item[key] == "N": return "None"
            else: return item[key]
        case _:
            raise KeyError(f"Unknown key: {key}")