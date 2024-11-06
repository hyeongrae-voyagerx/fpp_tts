import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

import os
import os.path as osp

from .neptune import init as neptune_init

def save_audio(audio: torch.Tensor, path="piui.wav", sr=22050):
    audio = audio.cpu().detach()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(path, audio, sr)

class DummyLogger:
    def add_scalar(self, *args, **kwargs):
        pass
    
    def add_audio(self, *args, **kwargs):
        pass

class LossFormatter:
    def __init__(self):
        self.losses = {}
        self.loss_items = {}

    def add(self, loss, name):
        self.losses[name] = loss
        self.loss_items[name] = loss.item() if isinstance(loss, torch.Tensor) else loss

    def __str__(self):
        return f"{self.item():.3f} = " + \
            " + ".join([f"{key}: {item:5.3f}" for key, item in self.loss_items.items()])

    def item(self):
        return sum([loss for loss in self.loss_items.values()])
    
    def remove(self, name):
        self.losses.pop(name, None)
        self.loss_items.pop(name, None)
    
    def __truediv__(self, frac):
        self.loss /= frac
        return self
    
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone() for name, param in model.named_parameters()}
        self.original_params = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = self.decay * self.shadow[name].data + (1 - self.decay) * param.data

    def apply_shadow(self):
        self.original_params = {name: param.clone() for name, param in self.model.named_parameters() if param.requires_grad}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.original_params:
                param.data.copy_(self.original_params[name].data)

    @staticmethod
    def ema_mode(func):
        def wrapper(self, *args, **kwargs):
            if hasattr(self, "ema"):
                self.ema.apply_shadow()
            out = func(self, *args, **kwargs)
            if hasattr(self, "ema"):
                self.ema.restore()
            return out
        
        return wrapper
    
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

def draw_mel_pitch(mel_batch, pitch_batch, pitch_true_batch, path):
    mel_batch, pitch_batch, pitch_true_batch = to_numpy(mel_batch, pitch_batch, pitch_true_batch)
    directory = osp.dirname(path)
    os.makedirs(directory, exist_ok=True)

    for i in range(mel_batch.shape[0]):
        mel, pitch, pitch_true = mel_batch[i], pitch_batch[i][0], pitch_true_batch[i][0]

        plt.figure(figsize=(10, 4))
        plt.imshow(mel, aspect="auto", origin="lower")
        plt.colorbar(format="%+2.f dB")
        plt.tight_layout()
        mel_filename = path.replace(".png", f"_mel_{i+1}.png")
        plt.savefig(mel_filename)
        plt.cla()
        plt.close()
        plt.figure(figsize=(10, 2))
        plt.plot(np.arange(pitch.shape[0]), pitch, "r")
        plt.plot(np.arange(pitch_true.shape[0]), pitch_true, "g")
        plt.tight_layout()
        pitch_filename = path.replace(".png", f"_pitch_{i+1}.png")
        plt.savefig(pitch_filename)
        plt.cla()
        plt.close()


def to_numpy(*items):
    numpys = []
    for item in items:
        if isinstance(item, torch.Tensor): item = item.detach().cpu().numpy()
        numpys.append(item)
    return numpys