from pathlib import Path, PureWindowsPath
import torch
import json
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import librosa
import copy


def obj_to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))


def get_free_gpu(gpus):
    device = None
    min_mem = float("inf")
    for gpu in gpus:
        gpu = torch.device(f"cuda:{gpu}")
        mem = torch.cuda.memory_reserved()
        if mem < min_mem:
            min_mem = mem
            device = gpu

    return device


def uniform(mean, std, size):
    floor, ceil = mean - std, mean + std
    return (ceil - floor) * torch.rand(size=size) + floor


def get_sampler(dist):
    if dist == "normal":
        return torch.normal
    elif dist == "uniform":
        return uniform


def right_pad_sequence(
    sequence, dim, max_val=None, get_lens=False, dtype=torch.float, fill_min=False
):
    if max_val is None:
        max_val = max([x.size(dim) for x in sequence])

    new_shape = [len(sequence)]
    for i, size in enumerate(sequence[0].shape):
        if i == dim:
            new_shape.append(max_val)
        else:
            new_shape.append(size)
    new_shape = tuple(new_shape)

    # There's got to be a better way, but this should be very fast
    if get_lens:
        lens = torch.zeros(new_shape[0], dtype=torch.long)

    if fill_min:
        fill_val = torch.min(sequence[0])
    else:
        fill_val = 0.0
    padded_batch = torch.full(size=new_shape, fill_value=fill_val, dtype=dtype)
    for i, sample in enumerate(sequence):
        temp_len = None

        if dim == 0:
            temp_len = sample.shape[0]
            padded_batch[i, 0:temp_len, ...] = sample
        elif dim == 1:
            temp_len = sample.shape[1]
            padded_batch[i, :, 0:temp_len, ...] = sample
        elif dim == 2:
            temp_len = sample.shape[2]
            padded_batch[i, :, :, 0:temp_len, ...] = sample
        elif dim == 3:
            temp_len = sample.shape[3]
            padded_batch[i, :, :, :, 0:temp_len, ...] = sample
        elif dim == 4:
            temp_len = sample.shape[4]
            padded_batch[i, :, :, :, :, 0:temp_len, ...] = sample

        if get_lens:
            lens[i] = temp_len

    if get_lens:
        return padded_batch, max_val, lens

    return padded_batch, max_val


def inner_sample(tensor, size):
    start = torch.randint(high=tensor.size(-1) - size, size=(1,))
    return tensor[..., start : start + size]


def copy_config_re(new_cfg, old_cfg, ignore=set()):
    """Recursively copy values from an older config file to a newer config file. This ensures that any new
    config values are filled in with their required values."""
    new_cfg = copy.deepcopy(new_cfg)
    for att, val in new_cfg.__dict__.items():
        if att in old_cfg.__dict__ and att not in ignore:
            child = getattr(val, "__dict__", None)
            if child is None:
                new_cfg.__dict__[att] = old_cfg.__dict__[att]
            else:
                new_cfg.__dict__[att] = copy_config_re(val, old_cfg.__dict__[att])
        # else:
        #     new_cfg.__dict__[att] = old_cfg.__dict__[att]
    return new_cfg


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
        fig.colorbar(im, ax=axs)
    return plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == "__main__":
    pass
