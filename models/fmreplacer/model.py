import datetime as dt
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .monotonic_align import maximum_path
from .modules.cfm import CFM
from .modules.text_encoder import TextEncoder
from .utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
    AttentionBinarizationLoss,
    AttentionCTCLoss
)
from .modules.alf import AlignmentLearningFramework



class MatchaTTS(nn.Module):
    def __init__(
        self,
        model_cfg
    ):
        super().__init__()

        self.n_vocab = model_cfg.n_vocab
        self.n_feats = model_cfg.encoder_params.n_feats
        self.prior_loss = model_cfg.use_prior_loss
        self.aligner = model_cfg.aligner
        if model_cfg.aligner == "alf":
            self.alf = AlignmentLearningFramework(
                n_tokens=model_cfg.n_vocab,
                feature_size=model_cfg.encoder_params.n_feats,
                encoding_size=model_cfg.encoder_params.n_channels
            )
            self.bin_loss = AttentionBinarizationLoss()
            self.ctc_loss = AttentionCTCLoss()

        self.encoder = TextEncoder(
            model_cfg.encoder_params,
            model_cfg.dur_pred_params,
            model_cfg.n_vocab,
            model_cfg.use_saln,
            spk_emb_dim=128,
        )

        self.decoder = CFM(
            in_channels=2*model_cfg.encoder_params.n_feats,
            out_channel=model_cfg.encoder_params.n_feats,
            cfm_params=model_cfg.cfm_params,
            decoder_params=model_cfg.decoder_params,
            use_saln=model_cfg.use_saln,
            spk_emb_dim=128,
            cfg_rate=model_cfg.cfg_rate
        )
        
        self.update_data_statistics(model_cfg.data_statistics)

        

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, style_mel, style_mel_length, n_timesteps=20, temperature=1.0, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            spks (bool, optional): speaker ids.
                shape: (batch_size,)
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.

        Returns:
            dict: {
                "encoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Average mel spectrogram generated by the encoder
                "decoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Refined mel spectrogram improved by the CFM
                "attn": torch.Tensor, shape: (batch_size, max_text_length, max_mel_length),
                # Alignment map between text and mel spectrogram
                "mel": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Denormalized mel spectrogram
                "mel_lengths": torch.Tensor, shape: (batch_size,),
                # Lengths of mel spectrograms
                "rtf": float,
                # Real-time factor
        """
        # For RTF computation
        t = dt.datetime.now()

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask, style_mel_mask = self.encoder(x, x_lengths, style_mel, style_mel_length)
        if self.aligner == "alf":
            duration = nn.functional.softplus(logw).squeeze(1)
            mu_y, y_lens, dur_pred = regulate_len(duration, mu_x.transpose(1, 2), pace=1.0, mel_max_len=None)
            mu_y = mu_y.transpose(1, 2)
            y_max_length = y_lens.max()
            y_max_length_ = fix_len_compatibility(y_max_length)
            y_mask = sequence_mask(y_lens, y_max_length).unsqueeze(1)
        elif self.aligner == "mas":
            w = torch.exp(logw) * x_mask
            w_ceil = torch.ceil(w) * length_scale
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = y_lengths.max()
            y_max_length_ = fix_len_compatibility(y_max_length)

            # Using obtained durations `w` construct alignment map `attn`
            y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

            # Align encoded text and get mu_y
            mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
            mu_y = mu_y.transpose(1, 2)
            encoder_outputs = mu_y[:, :, :y_max_length]

        # Generate sample tracing the probability flow
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature, style_mel)
        # decoder_outputs = decoder_outputs[:, :, :y_max_length]

        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)

        return {
            "encoder_outputs": encoder_outputs if self.aligner=="mas" else None,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length] if self.aligner=="mas" else None,
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths if self.aligner=="mas" else None,
            "rtf": rtf,
        }

    def forward(self, x, x_lengths, y, y_lengths, style_mel, style_mel_lengths, spks=None, out_size=None, cond=None, durations=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            spks (torch.Tensor, optional): speaker ids.
                shape: (batch_size,)
        """
        if style_mel is None:
            y, y_lengths, style_mel, style_mel_lengths = self.split_y(y, y_lengths)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask, style_mel_mask = self.encoder(x, x_lengths, style_mel, style_mel_lengths)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        style_mel_mask = (~style_mel_mask).float().unsqueeze(1)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        return_dict = {}

        if self.aligner == "alf":
            dur_attn, attn_soft, attn_hard, attn_logprobs = self.alf(x, x_lengths, y, y_lengths)
            dur_pred = nn.functional.softplus(logw)
            dur_loss = duration_loss(dur_pred, dur_attn, x_lengths, "l1")
            bin_loss = self.bin_loss(attn_hard, attn_soft)
            ctc_loss = self.ctc_loss(attn_logprobs, x_lengths, y_lengths)
            return_dict["dur_loss"] = dur_loss
            return_dict["bin_loss"] = bin_loss
            return_dict["ctc_loss"] = ctc_loss
            
            mu_y, y_lens, _ = regulate_len(dur_attn, mu_x.transpose(1, 2), pace=1.0, mel_max_len=None)
            mu_y = mu_y.transpose(1, 2)
        elif self.aligner == "mas":
            # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
            attn = self.get_mas(mu_x=mu_x, attn_mask=attn_mask, y=y)

            # Compute loss between predicted log-scaled durations and those obtained from MAS
            # refered to as prior loss in the paper
            logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
            dur_loss = duration_loss(logw, logw_, x_lengths)

            # Cut a small segment of mel-spectrogram in order to increase batch size
            #   - "Hack" taken from Grad-TTS, in case of Grad-TTS, we cannot train batch size 32 on a 24GB GPU without it
            #   - Do not need this hack for Matcha-TTS, but it works with it as well
            if out_size is not None:
                attn, y, y_mask = self.segment_y(y_lengths, out_size, attn, y)

            # Align encoded text with mel-spectrogram and get mu_y segment
            mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
            mu_y = mu_y.transpose(1, 2)
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
            return_dict["dur_loss"] = dur_loss
            return_dict["prior_loss"] = prior_loss
        else:
            raise ValueError(f"Unknown aligner: {self.aligner}")
        # Compute loss of the decoder
        # y = torch.cat((style_mel, y), dim=-1)
        # y_mask = torch.cat((style_mel_mask, y_mask), dim=-1)
        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, style_mel=style_mel, style_mel_mask=style_mel_mask, cond=cond)
        return_dict["diff_loss"] = diff_loss
        return_dict["loss"] = sum([v for k, v in return_dict.items() if "loss" in k])
        return return_dict
    
    def training_step(self, batch, idx):
        x, x_len, y, y_len, style_mel, style_mel_len = self.parse_batch(batch)

        return_dict = self(
            x=x,
            x_lengths=x_len,
            y=y,
            y_lengths=y_len,
            style_mel=style_mel,
            style_mel_lengths=style_mel_len,
        )

        return return_dict

    def predict_step(self, inputs, return_vars=("mel_out", )):
        x, style_mel = inputs
        x_length = torch.tensor(x.shape[-1:], dtype=torch.long, device=x.device)
        style_len = torch.tensor(style_mel.shape[-1:], dtype=torch.long, device=x.device)
        
        synthesis_result = self.synthesise(x, x_length, n_timesteps=32, style_mel=style_mel, style_mel_length=style_len)
        enc_out, dec_out = synthesis_result["encoder_outputs"], synthesis_result["decoder_outputs"]
        attn = synthesis_result["attn"]
        mel_out, mel_len = synthesis_result["mel"], synthesis_result["mel_lengths"]

        return get_return_tuple(locals(), return_vars)

    def segment_y(self, y_lengths, out_size, attn, y):
        max_offset = (y_lengths - out_size).clamp(0)
        offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
        out_offset = torch.LongTensor(
            [torch.tensor(random.choice(range(start, end)) if end > start else 0) for start, end in offset_ranges]
        ).to(y_lengths)
        attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
        y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

        y_cut_lengths = []
        for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
            y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
            y_cut_lengths.append(y_cut_length)
            cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
            y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
            attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

        y_cut_lengths = torch.LongTensor(y_cut_lengths)
        y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

        attn, y, y_mask = attn_cut, y_cut, y_mask
        return attn, y, y_mask

    @torch.no_grad()
    def get_mas(self, mu_x, attn_mask, y):
        const = -0.5 * math.log(2 * math.pi) * self.n_feats
        factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
        y_square = torch.matmul(factor.transpose(1, 2), y**2)
        y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
        mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
        log_prior = y_square - y_mu_double + mu_square + const

        attn = maximum_path(log_prior, attn_mask.squeeze(1))
        attn = attn.detach()  # b, t_text, T_mel

        return attn

    @staticmethod
    def parse_batch(batch):
        x, x_len = batch[0], batch[1]
        y, y_len = batch[2], batch[3]
        style_mel, style_mel_len = batch[4], batch[5]

        return x, x_len, y, y_len, style_mel, style_mel_len




    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    @staticmethod
    def get_mask_from_lens(lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths)

        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).type_as(lengths)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
        return mask
    
def get_return_tuple(locals_, keys):
    if isinstance(keys, str):
        return locals_[keys]
    if len(keys) == 0: return
    if len(keys) == 1: return locals_[keys[0]]
    return tuple(locals_[key] for key in keys)

def regulate_len(
    durations, enc_out, pace: float = 1.0, mel_max_len = None
):
    """If target=None, then predicted durations are applied"""
    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).long()
    dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype)

    range_ = torch.arange(max_len).to(enc_out.device)[None, :, None]
    mult = (reps_cumsum[:, :, :-1] <= range_) & (reps_cumsum[:, :, 1:] > range_)
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)

    if mel_max_len is not None:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)
    return enc_rep, dec_lens, reps.float()
