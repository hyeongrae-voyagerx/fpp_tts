# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from typing import Optional

import torch
import itertools
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time


from ..modules.nv_blocks import ConvReLUNorm
from ..modules.nv_utils import get_mask_from_lens, mask_from_lens
from ..modules.alignment import b_mas, mas_width1
from ..modules.attention import ConvAttention
from ..modules.transformer import FFTransformer, TransformerLayerWithSALN, PositionalEmbedding

from ..modules.loss_function import FPCFMLoss
from ..modules.attn_loss_function import AttentionBinarizationLoss
from ..modules.style_encoder import MelStyleEncoder
from .cfm import CFM
from .dit import DiT

_device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = F.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = F.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (
        torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)
    ).float()
    pitch_nelems = (
        torch.gather(pitch_nonzero_cums, 2, dce)
        - torch.gather(pitch_nonzero_cums, 2, dcs)
    ).float()

    pitch_avg = torch.where(
        pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems
    )
    return pitch_avg

class FPCFM(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.criterion = FPCFMLoss(
            dur_predictor_loss_scale=model_cfg.dur_loss_scale,
            pitch_predictor_loss_scale=model_cfg.pitch_loss_scale,
            attn_loss_scale=model_cfg.attn_loss_scale,
            use_mel_loss=model_cfg.use_mel_loss,
            loss_to_train=model_cfg.loss_to_train
        )
        self.attention_kl_loss = AttentionBinarizationLoss()

        self.encoder = FFTransformer(
            n_layer=self.model_cfg.in_fft_layers,
            n_head=self.model_cfg.in_fft_heads,
            d_model=self.model_cfg.symbols_embed_dims,
            d_head=self.model_cfg.in_fft_head_dims,
            d_inner=self.model_cfg.in_fft_filter_size,
            kernel_size=self.model_cfg.in_fft_kernel_size,
            dropout=self.model_cfg.in_fft_dropout,
            dropatt=self.model_cfg.in_fft_att_dropout,
            dropemb=self.model_cfg.in_fft_emb_dropout,
            embed_input=True,
            d_embed=self.model_cfg.symbols_embed_dims,
            n_embed=self.model_cfg.n_symbols,
            padding_idx=self.model_cfg.padding_idx,
            saln=model_cfg.saln,
        )

        if self.model_cfg.n_speakers > 1:
            self.speaker_emb = nn.Embedding(
                self.model_cfg.n_speakers, self.model_cfg.symbols_embed_dims
            )
        else:
            self.speaker_emb = None
        self.speaker_emb_weight = self.model_cfg.speaker_emb_weight

        self.duration_predictor = TemporalPredictor(
            self.model_cfg.in_fft_output_size,
            filter_size=self.model_cfg.dur_pred_filter_size,
            kernel_size=self.model_cfg.dur_pred_kernel_size,
            dropout=self.model_cfg.dur_pred_dropout,
            n_layers=self.model_cfg.dur_pred_n_layers,
        )

        # self.decoder = FFTransformer(
        #     n_layer=self.model_cfg.out_fft_layers,
        #     n_head=self.model_cfg.out_fft_heads,
        #     d_model=self.model_cfg.symbols_embed_dims,
        #     d_head=self.model_cfg.out_fft_head_dims,
        #     d_inner=self.model_cfg.out_fft_filter_size,
        #     kernel_size=self.model_cfg.out_fft_kernel_size,
        #     dropout=self.model_cfg.out_fft_dropout,
        #     dropatt=self.model_cfg.out_fft_att_dropout,
        #     dropemb=self.model_cfg.out_fft_emb_dropout,
        #     embed_input=False,
        #     d_embed=self.model_cfg.symbols_embed_dims,
        #     saln=model_cfg.saln,
        # )
        decoder_module = DiT(
            dim=self.model_cfg.symbols_embed_dims,
            depth=self.model_cfg.out_fft_layers,
            heads=self.model_cfg.out_fft_heads,
            dim_head=self.model_cfg.out_fft_head_dims,
            dropout=self.model_cfg.out_fft_dropout,
            ff_mult=self.model_cfg.out_fft_filter_size//self.model_cfg.symbols_embed_dims,
            mel_dim=self.model_cfg.n_mel_channels,
            latent_dim=self.model_cfg.symbols_embed_dims,
        )
        self.decoder = CFM(decoder_module)

        self.pitch_predictor = TemporalPredictor(
            self.model_cfg.in_fft_output_size,
            filter_size=self.model_cfg.pitch_pred_filter_size,
            kernel_size=self.model_cfg.pitch_pred_kernel_size,
            dropout=self.model_cfg.pitch_pred_dropout,
            n_layers=self.model_cfg.pitch_pred_n_layers,
            n_predictions=self.model_cfg.pitch_cond_formants,
            n_vocab=self.model_cfg.n_symbols,
        )

        self.pitch_emb = nn.Conv1d(
            self.model_cfg.pitch_cond_formants,
            self.model_cfg.symbols_embed_dims,
            kernel_size=self.model_cfg.pitch_embed_kernel_size,
            padding=int((self.model_cfg.pitch_embed_kernel_size - 1) / 2),
        )

        # Store values precomputed for training data within the model
        self.register_buffer("pitch_mean", torch.zeros(1))
        self.register_buffer("pitch_std", torch.zeros(1))

        self.energy_conditioning = self.model_cfg.energy_conditioning
        if self.model_cfg.energy_conditioning:
            self.energy_predictor = TemporalPredictor(
                self.model_cfg.in_fft_output_size,
                filter_size=self.model_cfg.energy_pred_filter_size,
                kernel_size=self.model_cfg.energy_pred_kernel_size,
                dropout=self.model_cfg.energy_pred_dropout,
                n_layers=self.model_cfg.energy_pred_n_layers,
                n_predictions=1,
                n_vocab=self.model_cfg.n_symbols,
            )

            self.energy_emb = nn.Conv1d(
                1,
                self.model_cfg.symbols_embed_dims,
                kernel_size=self.model_cfg.energy_embed_kernel_size,
                padding=int((self.model_cfg.energy_embed_kernel_size - 1) / 2),
            )

        self.proj = nn.Linear(
            self.model_cfg.out_fft_output_size, self.model_cfg.n_mel_channels, bias=True
        )

        self.attention = ConvAttention(
            self.model_cfg.n_mel_channels,
            0,
            self.model_cfg.symbols_embed_dims,
            use_query_proj=True,
            align_query_enc_type="3xconv",
        )

        self.saln = model_cfg.saln
        if self.saln:
            self.style_encoder = MelStyleEncoder()

        self.sr = model_cfg.sr
        self.mel_hop_size = model_cfg.mel_hop_size
        self.seconds_to_mel = model_cfg.sr / model_cfg.mel_hop_size

        self.punctuation_map = {
            # model_cfg.comma_encoding: torch.tensor((0.15, 0.4)),
            model_cfg.period_encoding: torch.tensor((0.4, 1.5)),
            model_cfg.question_mark_encoding: torch.tensor((0.4, 1.5)),
            model_cfg.exclamation_mark_encoding: torch.tensor((0.4, 1.5)),
            model_cfg.colon_encoding: torch.tensor((0.4, 1.5)),
            model_cfg.semicolon_encoding: torch.tensor((0.4, 1.5))
        }

        self.freeze_by_loss(model_cfg.loss_to_train)
        self.loss_to_train = model_cfg.loss_to_train

        self.opt = getattr(optim, model_cfg.optimizers[0], optim.Adam)(self.parameters(), model_cfg.lrs[0], **model_cfg.opt_args[0])
        self.sch = getattr(optim.lr_scheduler, model_cfg.sch[0], optim.lr_scheduler.LinearLR)(self.opt, **model_cfg.sch_args[0])
        self.register_buffer("global_step", getattr(model_cfg, "global_step", torch.tensor(0)))
        self.device = _device

    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = torch.zeros_like(attn)
            for ind in range(b_size):
                hard_attn = mas_width1(
                    attn_cpu[ind, 0, : out_lens[ind], : in_lens[ind]]
                )
                attn_out[ind, 0, : out_lens[ind], : in_lens[ind]] = torch.tensor(
                    hard_attn, device=attn.get_device()
                )
        return attn_out

    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(
                attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1
            )
        return torch.from_numpy(attn_out).type_as(attn)

    def forward(self, inputs, use_gt_pitch=True, pace=1.0, max_duration=75):
        (
            inputs,
            input_lens,
            mel_tgt,
            mel_lens,
            style_mel,
            style_mel_lens,
            pitch_dense,
            energy_dense,
            speaker,
            attn_prior,
            audiopaths,
        ) = inputs

        mel_max_len = mel_tgt.size(2)

        if self.saln:
            if style_mel is None:
                style_mel = mel_tgt
                style_mel_lens = mel_lens
            style_mel_mask = get_mask_from_lens(style_mel_lens) if style_mel_lens is not None else None
            style_vector = self.style_encoder(style_mel, style_mel_mask)

            # Input FFT
            enc_out, enc_mask = self.encoder(inputs, conditioning=style_vector)

        else:
            # Calculate speaker embedding
            if self.speaker_emb is None:
                spk_emb = 0
            else:
                spk_emb = self.speaker_emb(speaker).unsqueeze(1)
                spk_emb.mul_(self.speaker_emb_weight)

            # Input FFT
            enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        # Alignment
        text_emb = self.encoder.word_emb(inputs)

        # make sure to do the alignments before folding
        attn_mask = mask_from_lens(input_lens)[..., None] == 0
        # attn_mask should be 1 for unused timesteps in the text_enc_w_spkvec tensor

        attn_soft, attn_logprob = self.attention(
            mel_tgt,
            text_emb.permute(0, 2, 1),
            mel_lens,
            attn_mask,
            key_lens=input_lens,
            keys_encoded=enc_out,
            attn_prior=attn_prior,
        )

        attn_hard = self.binarize_attention_parallel(attn_soft, input_lens, mel_lens)

        # Viterbi --> durations
        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        dur_tgt = attn_hard_dur

        # assert torch.all(torch.eq(dur_tgt.sum(dim=1), mel_lens)), \
        #     f'{mel_lens}\n{dur_tgt.sum(dim=1)}\n{inputs}'

        # if not torch.all(torch.eq(dur_tgt.sum(dim=1), mel_lens)):
        #     print(dur_tgt.sum(dim=1) - mel_lens)

        # (torch.all(torch.eq(dur_tgt.sum(dim=1), mel_lens)) == False).nonzero(as_tuple=True)[0]

        # Predict durations
        log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        # Predict pitch
        pitch_pred = self.pitch_predictor(enc_out, enc_mask).permute(0, 2, 1)
        # Average pitch over characters
        pitch_tgt = average_pitch(pitch_dense, dur_tgt)

        if use_gt_pitch and pitch_tgt is not None:
            pitch_emb = self.pitch_emb(pitch_tgt)
        else:
            pitch_emb = self.pitch_emb(pitch_pred)
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        # Predict energy
        if self.energy_conditioning:
            energy_pred = self.energy_predictor(enc_out, enc_mask).squeeze(-1)

            # Average energy over characters
            energy_tgt = average_pitch(energy_dense.unsqueeze(1), dur_tgt)
            energy_tgt = torch.log(1.0 + energy_tgt)

            energy_emb = self.energy_emb(energy_tgt)
            energy_tgt = energy_tgt.squeeze(1)
            enc_out = enc_out + energy_emb.transpose(1, 2)
        else:
            energy_pred = None
            energy_tgt = None

        len_regulated, dec_lens, _ = regulate_len(dur_tgt, enc_out, pace, mel_max_len)

        # Output FFT
        if self.saln:
            dec_loss = self.decoder(
                mel_tgt.transpose(1, 2), len_regulated, style_vector=style_vector, lens=dec_lens
            )
        else:
            dec_loss = self.decoder(mel_tgt.transpose(1, 2), len_regulated, dec_lens)
        
        return (
            dec_loss,
            None, # before: dec_mask
            dur_pred,
            log_dur_pred,
            pitch_pred,
            pitch_tgt,
            energy_pred,
            energy_tgt,
            attn_soft,
            attn_hard,
            attn_hard_dur,
            attn_logprob,
        )


    def onnx_split(self, enc, window):
        window = window.to(self.device)
        num_windows = enc.size(1) // window
        start_idx = torch.arange(0, num_windows * window, step=window, device=self.device).unsqueeze(1).expand(num_windows, window).reshape(-1)
        index = start_idx + torch.arange(window, device=self.device).unsqueeze(0).expand(num_windows, window).reshape(-1)
        windows_test = torch.index_select(enc.view(-1, 384), 0, index)
        windows_test = windows_test.view(num_windows, window, 384)

        return windows_test

    def predict_complete(self, inputs, style_mel, pace, pitch_addition, energy_addition,  window=torch.tensor(700), min_pause=torch.tensor(0.4), **kwargs):
        enc, lens, style, durs = self.predict_encoder_step(inputs, style_mel=style_mel, pace=pace, pitch_addition=pitch_addition, energy_addition=energy_addition, min_pause=min_pause)
        durs = torch.round(durs * self.mel_hop_size) / self.sr

        last_len = enc.size(1) % window
        enc = F.pad(enc, (0, 0, 0, window - last_len))
        # windows_test = torch.split(enc, window, dim=1)
        # windows_test = torch.concat(windows_test, dim=0)

        start = time.time()
        windows = self.onnx_split(enc, window)
        print(time.time() - start)

        # assert((windows_test - windows).sum() == 0)

        style = style.repeat(windows.size(0), 1, 1)
        lens = window.repeat(windows.size(0))
        lens[-1] = last_len
        outputs = self.predict_decoder_step(windows, lens, style)

        return outputs[0], outputs[1], durs

    def predict_encoder_step(self, inputs, style_mel, pace, pitch_addition, energy_addition, min_pause):
        return self.predict_step(inputs.to(self.device), style_mel=style_mel.to(self.device), pace=pace.to(self.device), pitch_addition=pitch_addition.to(self.device), energy_addition=energy_addition.to(self.device), subset="encoder", min_pause=min_pause, improve=True)

    def predict_decoder_step(self, inputs, dec_lens, enc_style_vector):
        return self.predict_step(inputs.to(self.device), enc_style_vector=enc_style_vector.to(self.device), dec_lens=dec_lens.to(self.device), subset="decoder", improve=True)

    def clamp_punctuation(self, text_encoding, durs, min_pause):
        min_pause = min_pause.type_as(durs)
        for punc, minmax in self.punctuation_map.items():
            minmax = minmax.type_as(min_pause)
            punc_indices = (text_encoding == punc).nonzero(as_tuple=True)
            space_indices = (punc_indices[0], punc_indices[-1] + 1)
            durs[space_indices] = torch.clamp(durs[space_indices], min=self.seconds_to_mel * min_pause, max=self.seconds_to_mel * minmax[1])

        return durs

    def increase_question(self, text_encoding, pitch):
        pitch = pitch[:, 0, :]
        q_indices = (text_encoding == 10).nonzero(as_tuple=True)
        # b0_indices = (q_indices[0], q_indices[-1])
        # pitch[b0_indices] += 1.
        max = pitch.max()
        # b1_indices = (q_indices[0], q_indices[-1] - 1)
        # pitch[b1_indices] = 0.8*max
        b2_indices = (q_indices[0], q_indices[-1] -2)
        pitch[b2_indices] = 0.8*max

        # random_pitch = torch.rand(pitch.shape) * 4 - 2
        # pitch += random_pitch
        return pitch[:, None, :]

    def exclamation_pitch(self, text_encoding, pitch):
        pitch = pitch[:, 0, :]
        q_indices = (text_encoding == self.model_cfg.question_mark_encoding).nonzero(as_tuple=True)
        b3_indices = (q_indices[0], q_indices[-1] -2)
        b1_indices = (q_indices[0], q_indices[-1] - 1)
        pitch[b1_indices] = pitch[b3_indices]
        b2_indices = (q_indices[0], q_indices[-1] -2)
        pitch[b2_indices] = pitch[b3_indices]

        # random_pitch = torch.rand(pitch.shape) * 4 - 2
        # pitch += random_pitch
        return pitch[:, None, :]

    def increase_exclamation(self, text_encoding, energy):
        # energy = energy[:, 0, :]
        q_indices = (text_encoding == self.model_cfg.exclamation_mark_encoding).nonzero(as_tuple=True)
        for i in range(1, q_indices[-1][0]):
            b1_indices = (q_indices[0], q_indices[-1] - i)
            energy[b1_indices] *= 0.97
        # b2_indices = (q_indices[0], q_indices[-1] -2)
        # pitch[b2_indices] += 0.3

        # random_pitch = torch.rand(pitch.shape) * 4 - 2
        # pitch += random_pitch
        return energy

    def predict_step(
        self,
        inputs,
        style_mel=None,   # This is very messy, but it allows for simpler export to ONNX
        pace=1.0,
        pitch_addition=torch.tensor(0.0),
        energy_addition=torch.tensor(0.0),
        dur_tgt=None,
        pitch_tgt=None,
        energy_tgt=None,
        pitch_transform=None,
        max_duration=20,
        speaker=0,
        return_all=False,
        batch_idx=None,
        subset=None,
        dec_lens=None,
        enc_style_vector=None,
        min_pause=torch.tensor(0.4),
        improve=False,
        return_vars = ("mel_out", "dec_lens", "pitch_pred")
    ):
        if isinstance(inputs, list):
            if len(inputs) > 2:
                (inputs, _, _, _, mel_tgt, mel_lens, pitch_true, _, _, _, _,) = inputs

                if mel_tgt.dim() == 2:
                    mel_tgt = mel_tgt[None, ...]
                if mel_lens.dim() == 0:
                    mel_tgt = mel_tgt[..., :mel_lens]
                    mel_lens = None
                    style_mel = mel_tgt
                else:
                    mel_lens = mel_lens.to(self.device)
                mel_tgt = mel_tgt.to(self.device)
        elif isinstance(inputs, tuple):
            inputs, style_mel = inputs
        if inputs.dim() == 1:
            inputs = inputs[None, ...]

        if pitch_addition.dim() == 0 or pitch_addition.shape[-1] == 1:
            pitch_addition = torch.full_like(inputs, pitch_addition, dtype=torch.float32)
        else:
            assert(pitch_addition.shape[-1] == inputs.shape[-1])
        if energy_addition.dim() == 0 or energy_addition.shape[-1] == 1:
            energy_addition = torch.full_like(inputs, energy_addition, dtype=torch.float32)
        else:
            assert(energy_addition.shape[-1] == inputs.shape[-1])

        inputs = inputs.to(self.device)

        if subset is None or subset == "encoder":
            if self.saln:
                if style_mel is None:
                    mel_mask = (
                        get_mask_from_lens(mel_lens, max_len=mel_tgt.shape[-1]) if mel_lens is not None else None
                    )
                    enc_style_vector = self.style_encoder(mel_tgt, mel_mask)
                else:
                    enc_style_vector = self.style_encoder(style_mel)
                enc_out, enc_mask = self.encoder(inputs, conditioning=enc_style_vector)
            else:
                if self.speaker_emb is None:
                    spk_emb = 0
                else:
                    speaker = torch.ones(inputs.size(0)).long().to(inputs.device) * speaker
                    spk_emb = self.speaker_emb(speaker).unsqueeze(1)
                    spk_emb.mul_(self.speaker_emb_weight)
                enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

            # Predict durations
            log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
            dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 3, max_duration)
            if improve:
                dur_pred = self.clamp_punctuation(inputs, dur_pred, min_pause)

            # Pitch over chars
            pitch_pred = self.pitch_predictor(enc_out, enc_mask, pitch=True).permute(0, 2, 1)
            if improve:
                pitch_pred += pitch_addition
                old_pitch = pitch_pred.clone().detach()
                # pitch_pred = self.increase_question(inputs, pitch_pred)
            # pitch_pred = self.exclamation_pitch(inputs, pitch_pred)

            if pitch_transform is not None:
                if self.pitch_std[0] == 0.0:
                    # XXX LJSpeech-1.1 defaults
                    mean, std = 218.14, 67.24
                else:
                    mean, std = self.pitch_mean[0], self.pitch_std[0]
                pitch_pred = pitch_transform(
                    pitch_pred, enc_mask.sum(dim=(1, 2)), mean, std
                )

            if pitch_tgt is None:
                pitch_emb = self.pitch_emb(pitch_pred).transpose(1, 2)
            else:
                pitch_emb = self.pitch_emb(pitch_tgt).transpose(1, 2)

            enc_out = enc_out + pitch_emb

            # Predict energy
            if self.energy_conditioning:

                if energy_tgt is None:
                    energy_pred = self.energy_predictor(enc_out, enc_mask).squeeze(-1)
                    # energy_pred = self.increase_exclamation(inputs, energy_pred)
                    energy_addition = energy_addition.type_as(energy_pred)
                    energy_pred += energy_addition
                    energy_emb = self.energy_emb(energy_pred.unsqueeze(1)).transpose(1, 2)
                else:
                    energy_emb = self.energy_emb(energy_tgt).transpose(1, 2)

                enc_out = enc_out + energy_emb
            else:
                energy_pred = None

            len_regulated, dec_lens, dur_pred = regulate_len(
                dur_pred if dur_tgt is None else dur_tgt, enc_out, pace, mel_max_len=None
            )

            if subset == "encoder":
                return len_regulated, dec_lens, enc_style_vector, dur_pred
            else:
                inputs = len_regulated
        
        if subset is None or subset == "decoder":
            len_regulated = inputs
            if self.saln:
                mel_out, _ = self.decoder.sample(
                    len_regulated, style_vector=enc_style_vector, cfg_strength=0
                )
            else:
                mel_out, _ = self.decoder(len_regulated, dec_lens)

            # mel_out = self.proj(dec_out)
            # # mel_lens = dec_mask.squeeze(2).sum(axis=1).long()
            # mel_out = mel_out.permute(0, 2, 1)  # For inference.py
            return get_return_tuple(locals(), return_vars)

    @staticmethod
    def parse_batch(batch):
        x = list(batch) # text, text_lens, mel, mel_lens, style_mel, style_mel_lens, pitch, energy, speaker, prior, audio
        # y = [batch[key] for key in ["mel", "text_lens", "mel_lens"]]
        y = [batch[2], batch[1], batch[3]]

        return x, y

    def encode_save(self, y, path):
        """
        Takes a models output or ground-truth, encodes or decodes it to a valid format, and saves the artifact
        to the specified path. i.e. takes an audio timeseries, encodes it as an audio file, and saves it.
        :param y: raw model output
        :param path: string representing the directory and filename without the extension
        :return: the path of the saved file with the extension added
        """
        y = y.squeeze() + 100
        y = librosa.power_to_db(y, ref=np.max)
        librosa.display.specshow(y, y_axis="mel", fmax=8000, x_axis="time")
        plt.title("Mel Spectrogram")
        plt.colorbar(format="%+2.0f dB")
        final_path = f"{path}.png"
        plt.savefig(final_path)
        plt.close()
        return final_path

    def training_step(self, batch, batch_idx, return_pred=False, do_train=True):
        x, y = self.parse_batch(batch)

        y_pred = self(x)
        meta = self.criterion(y_pred, y)

        gstep = self.global_step.item() if isinstance(self.global_step, torch.Tensor) else self.global_step
        if (
            self.model_cfg.kl_loss_start_step is not None
            and gstep >= self.model_cfg.kl_loss_start_step
            and "attn" in self.loss_to_train
        ):
            _, _, _, _, _, _, _, _, attn_soft, attn_hard, _, _ = y_pred
            binarization_loss = self.attention_kl_loss(attn_hard, attn_soft)
            kl_weight = (
                min(
                    (gstep - self.model_cfg.kl_loss_start_step)
                    / self.model_cfg.kl_loss_warmup_steps,
                    1.0,
                )
                * self.model_cfg.kl_loss_weight
            )
            meta["kl_weight"] = kl_weight
            meta["binarization_loss"] = binarization_loss
            meta["kl_loss"] = binarization_loss * kl_weight
            meta["loss"] += kl_weight * binarization_loss
        else:
            if "attn" in self.loss_to_train:
                meta["binarization_loss"] = torch.zeros_like(meta["loss"])
                meta["kl_loss"] = torch.zeros_like(meta["loss"])
        if do_train:
            self.opt.zero_grad()
            meta["loss"].backward()
            self.opt.step()
            if hasattr(self, "sch"):
                self.sch.step()

        return meta if not return_pred else (meta, y_pred)

    def validation_step(self, batch, batch_idx, return_pred=False):
        x, y = self.parse_batch(batch)

        y_pred = self(x)
        meta = self.criterion(y_pred, y)
        _, _, _, _, _, _, _, _, attn_soft, attn_hard, _, _ = y_pred
        binarization_loss = self.attention_kl_loss(attn_hard, attn_soft)
        gstep = self.global_step.item() if isinstance(self.global_step, torch.Tensor) else self.global_step
        if gstep >= self.model_cfg.kl_loss_start_step:
            kl_weight = (
                min(
                    (gstep - self.model_cfg.kl_loss_start_step)
                    / self.model_cfg.kl_loss_warmup_steps,
                    1.0,
                )
                * self.model_cfg.kl_loss_weight
            )
        else:
            kl_weight = 0
        if "attn" in self.loss_to_train:
            meta["binarization_loss"] = binarization_loss
            meta["kl_loss"] = binarization_loss * kl_weight
            meta["loss"] += kl_weight * binarization_loss

        return meta if not return_pred else (meta, y_pred)

    def test_step(self, batch, batch_idx):
        x, y = self.parse_batch(batch)

        y_pred = self(x)
        meta = self.criterion(y_pred, y)

        _, _, _, _, _, _, _, _, attn_soft, attn_hard, _, _ = y_pred
        binarization_loss = self.attention_kl_loss(attn_hard, attn_soft)
        gstep = self.global_step.item() if isinstance(self.global_step, torch.Tensor) else self.global_step
        if gstep >= self.model_cfg.kl_loss_start_step:
            kl_weight = (
                min(
                    (gstep - self.model_cfg.kl_loss_start_step)
                    / self.model_cfg.kl_loss_warmup_steps,
                    1.0,
                )
                * self.model_cfg.kl_loss_weight
            )
        else:
            kl_weight = 0

        meta["kl_loss"] = binarization_loss * kl_weight
        meta["loss"] += kl_weight * binarization_loss

        return meta
    
    def freeze_by_loss(self, loss_to_train: tuple):
        if set(loss_to_train) == {"cfm", "attn", "dur", "pitch", "energy"}:
            return
        
        for p in self.parameters(): p.requires_grad=False
        if "pitch" in loss_to_train:
            for p in self.encoder.parameters(): p.requires_grad=True
            for p in self.pitch_predictor.parameters(): p.requires_grad=True
        if "dur" in loss_to_train:
            for p in self.duration_predictor.parameters(): p.requires_grad=True
        if "energy" in loss_to_train:
            for p in self.energy_predictor.parameters(): p.requires_grad=True
        if "attn" in loss_to_train:
            for p in self.attention.parameters(): p.requires_grad=True
        if "mel" in loss_to_train:
            if hasattr(self, "style_encoder"):
                for p in self.style_encoder.parameters(): p.requires_grad=True
            for p in self.encoder.parameters(): p.requires_grad=True
            for p in self.decoder.parameters(): p.requires_grad=True
            for p in self.proj.parameters(): p.requires_grad=True


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
    
    average_pitch = staticmethod(average_pitch)


def regulate_len(
    durations, enc_out, pace: float = 1.0, mel_max_len: Optional[int] = None
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


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(
        self, input_size, filter_size, kernel_size, dropout, n_layers=2, n_predictions=1, n_vocab=None
    ):
        super(TemporalPredictor, self).__init__()

        # if n_vocab is not None:
        #     self.emb = nn.Embedding(n_vocab, input_size)
        self.pe = PositionalEmbedding(input_size)

        self.layers = nn.Sequential(
            *[
                ConvReLUNorm(
                    input_size if i == 0 else filter_size,
                    filter_size,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for i in range(n_layers)
            ]
        )
        self.n_predictions = n_predictions
        self.fc = nn.Linear(filter_size, self.n_predictions, bias=True)


    def forward(self, enc_out, enc_out_mask, pitch=False):
        # if hasattr(self, "emb"):
        #     enc_out = self.emb(enc_out)
        if pitch:
            pos_seq = torch.arange(enc_out.shape[1], device=enc_out.device, dtype=enc_out.dtype)
            enc_out = enc_out + self.pe(pos_seq) * enc_out_mask
        out = enc_out * enc_out_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out) * enc_out_mask
        return out


def get_return_tuple(locals_, keys):
    if len(keys) == 0: return
    if len(keys) == 1: return locals_[keys[0]]
    return tuple(locals_[key] for key in keys)

