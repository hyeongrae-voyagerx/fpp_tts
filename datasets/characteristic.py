import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from scipy.stats import betabinom
from models.modules.nv_blocks import TacotronSTFT
import os.path as osp
import crepe
import math
from utils.baseline_tp import symbol_to_id
from unicodedata import normalize

SR = 22050
static_mel = None

class CharacteristicDataset:
    def __init__(self, data_dir):
        self.metadata = osp.join(data_dir, "metadata.txt")
        self.dataset = []
        with open(self.metadata, "r") as fr:
            lines = fr.read().split("\n")
        if len(lines[-1]) == 0:
            lines = lines[:-1]
        for file, text in map(lambda x: x.split("|"), lines):
            wav, sr = torchaudio.load(osp.join(data_dir, file))
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)[0]
            else:
                wav = wav[0]
            pitch_path=osp.join(data_dir, file).replace(".wav", ".pitch")
            # self.save_pitch(wav, pitch_path)
            self.dataset.append({"wav": wav, "text": text, "pitch": normalize_pitch(torch.load(pitch_path, weights_only=False))})
        
        global static_mel
        static_mel = get_mel(self.dataset[0]["wav"][None])
    
    @classmethod
    def get_dataloader(cls, dataset_kwargs, dataloader_kwargs):
        return DataLoader(cls(**dataset_kwargs).dataset, collate_fn=collate_fn, **dataloader_kwargs)
    
    @staticmethod
    def save_pitch(wav, save_name):
        if not osp.exists(save_name):
            mel = get_mel(wav)
            pitch = estimate_pitch(wav.numpy(), SR, mel.shape[1])
            torch.save(pitch, save_name)

    

def collate_fn(batch):
    text = [torch.tensor([symbol_to_id[c] for c in normalize("NFKD", item["text"])]) for item in batch]
    text, text_lens = merge_to_tensor(text, 0)

    audio = [item["wav"] for item in batch]
    mel_list = [get_mel(a) for a in audio]
    mel, mel_lens = merge_to_tensor(mel_list, 1)

    pitch_list = [item["pitch"] for item in batch]
    pitch, pitch_lens = merge_to_tensor(pitch_list, 1)

    energy_list = [get_energy(m) for m in mel_list]
    energy, energy_lens = merge_to_tensor(energy_list, 0)

    attn_prior = get_attn_priors(mel_lens, text_lens)

    static_mels = [static_mel for _ in range(len(mel_list))]
    static_mels, static_mel_lens = merge_to_tensor(static_mels, 1)
    audio, audio_lens = merge_to_tensor(audio, 0)

    return [text, text_lens, mel, mel_lens, static_mels, static_mel_lens, pitch, energy, audio_lens, attn_prior, audio]



def merge_to_tensor(data_list, variant_dim):
    lengths = [data.shape[variant_dim] for data in data_list]
    max_len_data = max(lengths)
    padded_data = torch.stack([F.pad(data, [0, max_len_data-data.shape[variant_dim]]) for data in data_list])
    return padded_data, torch.tensor(lengths)


def get_mel(wav):
    if wav.ndim == 1:
        wav = wav[None]
    mel = stft.mel_spectrogram(wav)[0]
    return mel

def normalize_pitch(pitch, mean=218.14, std=67.24):
    zeros = pitch==0.0
    pitch_shift = pitch-mean
    pitch_norm = pitch_shift / std
    pitch_norm[zeros] == 0.0
    return pitch_norm

def estimate_pitch(snd, sr, mel_len, n_formants=1):
    _, freq, conf, _ = crepe.predict(snd, sr, viterbi=True, step_size=snd.shape[0] / mel_len / sr * 1000, verbose=0,
                                        model_capacity="full")
    freq[conf <= 0.45] = 0.
    len_diff = (mel_len - freq.shape[0]) / 2
    if len_diff < 0:
        freq = freq[-math.ceil(len_diff): math.floor(len_diff)]
    elif len_diff > 0:
        freq = np.pad(freq, ((math.floor(len_diff), math.ceil(len_diff))))
    pitch_mel = torch.from_numpy(freq).unsqueeze(0)

    return pitch_mel.float()

def get_energy(mel):
    energy = torch.norm(mel.float(), dim=0, p=2)
    return energy

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

stft = TacotronSTFT(
    filter_length=1024,
    hop_length=256,
    win_length=1024,
    n_mel_channels=80,
    sr=SR,
    mel_fmin=0.0,
    mel_fmax=8000.0,
)