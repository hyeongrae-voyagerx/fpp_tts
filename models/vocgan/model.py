import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import itertools

from .generator import ModifiedGenerator
from .multiscale import MultiScaleDiscriminator
from ..modules.stft_loss import MultiResolutionSTFTLoss

_device = torch.device("cuda" if torch.cuda.is_available else "cpu")

class VocGAN(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.gen = ModifiedGenerator(
            model_cfg.n_mel_channels,
            model_cfg.n_residual_layers,
            ratios=model_cfg.generator_ratio,
            mult=model_cfg.mult,
            out_band=model_cfg.out_channels,
        )
        self.dis = MultiScaleDiscriminator()
        self.stft_loss = MultiResolutionSTFTLoss()
        self.criterion = torch.nn.MSELoss()
        self.sr = model_cfg.sr
        self.hop_len = model_cfg.hop_len
        self.automatic_optimization = False
        self.opt_g = getattr(optim, model_cfg.opts[0], optim.Adam)(
            self.gen.parameters(),
            model_cfg.lrs[0],
            **model_cfg.opt_args[0]
        )
        self.opt_d = getattr(optim, model_cfg.opts[1], optim.Adam)(
            self.dis.parameters(),
            model_cfg.lrs[1],
            **model_cfg.opt_args[1]
        )
        self.schs = [
            optim.lr_scheduler.LinearLR(self.opt_g, **model_cfg.sch_args[0]),
            optim.lr_scheduler.LinearLR(self.opt_d, **model_cfg.sch_args[1]),
        ]
        self.register_buffer("global_step", getattr(model_cfg, "global_step", torch.tensor(0)))
        self.device = _device

    @staticmethod
    def parse_batch(batch):
        x = batch[2]
        # x = batch["mel"]
        y = batch[-1]
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

    def shared_step(self, batch, train=False, optimizers=None, schedulers=None, meta=None):
        mel, audio = self.parse_batch(batch)
        meta = {} if meta is None else meta

        fake_audio = self.gen(mel)
        if fake_audio.size(-1) < audio.size(-1):
            padded = torch.zeros_like(audio)
            padded[..., :fake_audio.size(-1)] = fake_audio
            fake_audio = padded
        else:
            fake_audio = fake_audio[..., : audio.size(-1)]

        sc_loss, mag_loss = self.stft_loss(fake_audio.squeeze(1), audio.squeeze(1))
        meta["sc_loss"] = sc_loss
        meta["mag_loss"] = mag_loss
        loss_g = sc_loss + mag_loss

        adv_loss = torch.tensor(0.0).type_as(loss_g)
        gstep = self.global_step.item() if isinstance(self.global_step, torch.Tensor) else self.global_step
        if (
            gstep // 2 > self.model_cfg.dis_start_step // (mel.size(0))
        ):
            real_d = self.dis(audio)
            fake_d = self.dis(fake_audio)

            for feats_fake, score_fake in fake_d:
                adv_loss += self.criterion(score_fake, torch.ones_like(score_fake))
            adv_loss = adv_loss / len(fake_d)

            if self.model_cfg.feat_loss:
                for (feats_fake, score_fake), (feats_real, _) in zip(
                    fake_d, real_d
                ):
                    for feat_f, feat_r in zip(feats_fake, feats_real):
                        adv_loss += self.model_cfg.feat_match * torch.mean(
                            torch.abs(feat_f - feat_r)
                        )

            weighted_adv_loss = self.model_cfg.adv_weight * adv_loss
            meta["weighted_adv_loss"] = weighted_adv_loss
            loss_g += weighted_adv_loss

        if "loss" in meta:
            meta["loss"] += loss_g
        else:
            meta["loss"] = loss_g

        if train:
            self.opt_g.zero_grad()
            meta["loss"].backward()
            self.opt_g.step()

        if (
            gstep // 2 > self.model_cfg.dis_start_step // mel.size(0)
        ):
            fake_audio = self.gen(mel)[..., : audio.size(-1)]
            fake_audio = fake_audio.detach()
            loss_d_sum = 0.0
            # TODO: Enable more discriminator training repetitions
            for _ in range(1):
                fake_d = self.dis(fake_audio)
                real_d = self.dis(audio)
                loss_d_real = 0.0
                loss_d_fake = 0.0
                for (_, score_fake), (_, score_real) in zip(fake_d, real_d):
                    loss_d_real += self.criterion(
                        score_real, torch.ones_like(score_real)
                    )
                    loss_d_fake += self.criterion(
                        score_fake, torch.zeros_like(score_fake)
                    )
                loss_d_real = loss_d_real / len(real_d)
                loss_d_fake = loss_d_fake / len(fake_d)
                meta["loss_d_real"] = loss_d_real
                meta["loss_d_fake"] = loss_d_fake
                meta["loss_d"] = loss_d_real + loss_d_fake
                if train:
                    self.opt_d.zero_grad()
                    meta["loss_d"].backward()
                    self.opt_d.step()

        if train:
            for sch in self.schs:
                sch.step()

        return meta

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, train=True)

    def validation_step(self, batch, batch_idx):
        meta = self.shared_step(batch)
        return meta

    def predict_step(self, mels, mel_lens=None, extras=None, batch_idx=None, **kwargs):
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


def get_model():
    return VocGAN