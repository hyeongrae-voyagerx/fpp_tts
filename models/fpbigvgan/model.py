import torch
import torch.nn as nn
import torch.optim as optim

from ..fastpitch import FastPitch
from ..nvBigVGAN import BigVGAN

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FPBigVGAN(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        fastpitch_cfg = model_cfg.fastpitch_config
        bigvgan_cfg = model_cfg.bigvgan_config
        self.fastpitch = FastPitch(fastpitch_cfg)
        self.bigvgan = BigVGAN(bigvgan_cfg)

        self.load_fastpitch(model_cfg.fp_load)
        self.load_bigvgan(model_cfg.bv_load)

        self.voc_sample_len = model_cfg.voc_sample_len
        self.hop_len = model_cfg.bigvgan_config.hop_length

        self.device = _device


    def training_step(self, batch, batch_idx, return_pred=False, do_train=True):
        with torch.no_grad():
            self.fastpitch.eval()
            _, y_pred = self.fastpitch.validation_step(batch, batch_idx, return_pred=True)
        mel_pred = y_pred[0].permute(0, 2, 1)

        audio, ids = self.rand_slice_segments(batch[-1].unsqueeze(1), batch[-3], self.voc_sample_len)
        mel = self.slice_segments(mel_pred, ids // self.hop_len, self.voc_sample_len // self.hop_len)
        # audio = batch[-1].unsqueeze(1)
        loss = self.bigvgan.shared_step({"mel": mel, "audio": audio})
        return loss

    @staticmethod
    def parse_batch(batch):
        x = list(batch) # text, text_lens, mel, mel_lens, style_mel, style_mel_lens, pitch, energy, speaker, prior, audio
        # y = [batch[key] for key in ["mel", "text_lens", "mel_lens"]]
        y = [batch[2], batch[1], batch[3]]

        return x, y

    def rand_slice_segments(self, x, x_lengths=None, segment_size=4):
        b, d, t = x.size()
        if x_lengths is None:
            x_lengths = t
        ids_str_max = x_lengths - segment_size + 1
        ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
        ret = self.slice_segments(x, ids_str, segment_size)
        return ret, ids_str
    
    @staticmethod
    def slice_segments(x, ids_str, segment_size=4):
        ret = torch.zeros_like(x[:, :, :segment_size])
        for i in range(x.size(0)):
            idx_str = max(0, ids_str[i])
            idx_end = idx_str + segment_size
            ret[i] = x[i, :, idx_str:idx_end]
        return ret
    
    def load_fastpitch(self, path):
        state_dict = torch.load(path, weights_only=False)
        self.fastpitch.load_state_dict(state_dict["model"])

    def load_bigvgan(self, path):
        state_dict = torch.load(path, weights_only=False)
        self.bigvgan.load_state_dict(state_dict)