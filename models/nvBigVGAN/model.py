import pytorch_lightning as pl
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.nn import functional as F
from auraloss.freq import MultiResolutionSTFTLoss
import itertools

from .model_config import register_configs
from .models import BigVGAN as Generator
from .models import MultiPeriodDiscriminator, MultiResolutionDiscriminator, feature_loss, generator_loss, discriminator_loss

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BigVGAN(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.automatic_optimization = False
        self.model_cfg = model_cfg
        self.c_mel = model_cfg.c_mel
        self.MRLoss = None
        self.sr = model_cfg.sr

        self.gen = Generator(model_cfg)
        self.mpd = MultiPeriodDiscriminator(model_cfg)
        self.mrd = MultiResolutionDiscriminator(model_cfg)

        if hasattr(model_cfg, "opts"):
            self.opts = (
                getattr(optim, model_cfg.opts[0])(self.gen.parameters(), model_cfg.lrs[0], **model_cfg.opt_args[0]),
                getattr(optim, model_cfg.opts[1])(itertools.chain(self.mpd.parameters(), self.mrd.parameters()), model_cfg.lrs[1], **model_cfg.opt_args[1])
            )
            self.schs = (
                getattr(optim.lr_scheduler, model_cfg.sch)(self.opts[0], **model_cfg.sch_args[0]),
                getattr(optim.lr_scheduler, model_cfg.sch)(self.opts[1], **model_cfg.sch_args[1]),
            )
        self.device = _device

    @staticmethod
    def parse_batch(batch):
        x = batch["mel"]
        y = batch["audio"]
        return x, y

    def encode_save(self, y, path):
        path = f"{path}.wav"
        if y.dim() > 2:
            y = y[0, ...]
        elif y.dim() < 2:
            y = y[None, ...]
        torchaudio.save(path, y, self.sr)
        return path

    def forward(self, data):
        if data.dim() < 3:
            data = data[None, ...]

        return self.gen(data)

    def shared_step(self, batch, train=False):
        mel, audio = self.parse_batch(batch)
        opt_g, opt_d = self.optimizers()
        schs = self.lr_schedulers()
        if self.MRLoss is None:
            self.MRLoss = MultiResolutionSTFTLoss(device=self.device)

        # Discriminator step
        # self.toggle_optimizer(opt_d)
        _audio = self.gen(mel)

        if _audio.size(-1) < audio.size(-1):
            padded = torch.zeros_like(audio)
            padded[..., :_audio.size(-1)] = _audio
            _audio = padded
        else:
            _audio = _audio[..., : audio.size(-1)]

        _audio_mpd_real, _audio_mpd_fake, _, _ = self.mpd(audio, _audio.detach())
        loss_mpd, losses_mpd_real, losses_mpd_fake = discriminator_loss(_audio_mpd_real, _audio_mpd_fake)

        _audio_mrd_real, _audio_mrd_fake, _, _ = self.mrd(audio, _audio.detach())
        loss_mrd, losses_mrd_real, losses_mrd_fake = discriminator_loss(_audio_mrd_real, _audio_mrd_fake)

        loss_d = loss_mrd + loss_mpd

        if train:
            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()
        # self.untoggle_optimizer(opt_d)

        # Generator step
        # self.toggle_optimizer(opt_g)
        _audio_mpd_real, _audio_mpd_fake, fmap_mpd_real, fmap_mpd_fake = self.mpd(audio, _audio)
        fm_mpd_loss = feature_loss(fmap_mpd_real, fmap_mpd_fake)
        loss_g_mpd, losses_g_mpd = generator_loss(_audio_mpd_fake)

        _audio_mrd_real, _audio_mrd_fake, fmap_mrd_real, fmap_mrd_fake = self.mrd(audio, _audio)
        fm_mrd_loss = feature_loss(fmap_mrd_real, fmap_mrd_fake)
        loss_g_mrd, losses_g_mrd = generator_loss(_audio_mrd_fake)

        mel_loss = self.MRLoss(audio, _audio) * self.c_mel

        loss_g_total = loss_g_mpd + loss_g_mrd + fm_mpd_loss + fm_mrd_loss + mel_loss

        if train:
            opt_g.zero_grad()
            self.manual_backward(loss_g_total)
            opt_g.step()
            if schs:
                for sch in schs:
                    sch.step()
        # self.untoggle_optimizer(opt_g)

        return {
            "loss_d_total": loss_d.detach(),
            "loss_mpd": loss_mpd,
            "loss_mrd": loss_mrd,
            "loss_g_total": loss_g_total.detach(),
            "mel_loss": mel_loss.detach(),
            "fm_loss": fm_mpd_loss.detach() + fm_mrd_loss.detach(),
            "loss_g": loss_g_mpd.detach() + loss_g_mrd.detach()
        }

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, train=True)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, train=False)

    def onnx_split(self, enc, window):
        enc = enc.permute(0, 2, 1)  # permute to (batch_size, sequence_length, hidden_size)
        window = window.to(enc.device)
        num_windows = enc.size(1) // window
        start_idx = torch.arange(0, num_windows * window, step=window, device=enc.device).unsqueeze(1).expand(
            num_windows, window).reshape(-1)
        index = start_idx + torch.arange(window, device=enc.device).unsqueeze(0).expand(num_windows, window).reshape(-1)
        windows_test = torch.index_select(enc.view(-1, enc.size(2)), 0, index)
        windows_test = windows_test.view(enc.size(0), num_windows, window, enc.size(2)).permute(1, 3, 2, 0)
        return windows_test.squeeze()

    def predict_step(self, mels, mel_lens=None, extras=None, batch_idx=None):
        with torch.no_grad():
            if mel_lens is not None:
                mel_lens = mel_lens.to(self.device)
            else:
                mel_lens = torch.tensor([mels.size(-1)], device=self.device,)
                mels = mels.to(self.device)

            if mels.dim() <= 2:
                mels = mels[None, ...]

            mels = mels.to(self.device)
            single_mel = torch.swapaxes(mels, 0, 1).reshape(1, 80, -1)
            single_mel = single_mel[..., :mel_lens.sum()]
            output = self.gen(single_mel)

            # window = 100
            # last_len = single_mel.size(-1) % window
            # enc = F.pad(single_mel, (0, window - last_len))
            # # start = time.time()
            # windows = self.onnx_split(enc, torch.tensor(window))
            #
            # outs = []
            # for slice in windows:
            #     outs.append(self.gen(slice[None, ...]).detach())
            #
            # output = torch.concat(outs, -1)

        return output[0, ...], extras

    def configure_optimizers(self):
        """
        You can impliment this function however you prefer, but this is an implimentation that allows the optimizer
        parameters to be defined in the config file.
        :return: list of optimizers
        """
        opts = []
        for opt_name, opt_args, lr, params in zip(
                self.model_cfg.opts,
                self.model_cfg.opt_args,
                self.model_cfg.lrs,
                self.model_cfg.params,
        ):
            opt_args = eval(opt_args)
            opt_args["lr"] = lr
            if isinstance(params, list):
                opt_args["params"] = itertools.chain(eval(f"self{params[0]}.parameters()"),
                                                     eval(f"self{params[1]}.parameters()"))
            else:
                opt_args["params"] = eval(f"self{params}.parameters()")
            opt = eval(f"torch.optim.{opt_name}")
            opt = opt(**opt_args)
            opts.append(opt)

        schs = []
        if self.model_cfg.sch is not None:
            for opt in opts:
                tr_sch_dict = eval(self.model_cfg.sch_args)
                tr_sch = eval(f"torch.optim.lr_scheduler.{self.model_cfg.sch}")
                tr_sch = tr_sch(opt, **tr_sch_dict)
                if self.cfg.trainer_cfg.sch == "CyclicLR":
                    tr_sch._scale_fn_custom = tr_sch._scale_fn_ref()
                    tr_sch._scale_fn_ref = None
                tr_sch_dict = {"scheduler": tr_sch, "interval": "step"}
                schs.append(tr_sch_dict)

        if len(schs) > 0:
            return opts, schs
        return opts

    def optimizers(self):
        return self.opts
    
    def lr_schedulers(self):
        return self.schs
    


def get_model():
    return BigVGAN


@hydra.main(config_path=None, config_name="model_config", version_base=None)
def get_model_details(model_cfg):
    model = get_model()(model_cfg)
    summary = pl.utilities.model_summary.ModelSummary(model, max_depth=1)
    print(summary)


if __name__ == "__main__":
    register_configs()
    get_model_details()
