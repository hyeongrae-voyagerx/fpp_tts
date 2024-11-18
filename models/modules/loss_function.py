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

import torch
import torch.nn.functional as F
from torch import nn
from .nv_utils import mask_from_lens
from .attn_loss_function import AttentionCTCLoss


class FastPitchLoss(nn.Module):
    def __init__(
        self,
        dur_predictor_loss_scale=1.0,
        pitch_predictor_loss_scale=1.0,
        attn_loss_scale=1.0,
        energy_predictor_loss_scale=0.1,
        use_mel_loss=True,
        loss_to_train=("mel", "attn", "dur", "pitch", "energy")
    ):
        super(FastPitchLoss, self).__init__()
        self.dur_predictor_loss_scale = dur_predictor_loss_scale
        self.pitch_predictor_loss_scale = pitch_predictor_loss_scale
        self.energy_predictor_loss_scale = energy_predictor_loss_scale
        self.attn_loss_scale = attn_loss_scale
        self.attn_ctc_loss = AttentionCTCLoss()
        self.use_mel_loss = use_mel_loss
        self.loss_to_train = loss_to_train

    def forward(self, model_out, targets, is_training=True, meta_agg="mean"):
        (
            mel_out,
            dec_mask,
            dur_pred,
            log_dur_pred,
            pitch_pred,
            pitch_tgt,
            energy_pred,
            energy_tgt,
            attn_soft,
            attn_hard,
            attn_dur,
            attn_logprob,
        ) = model_out
        (mel_tgt, in_lens, out_lens) = targets
        mel_tgt.requires_grad = False
        # (B,H,T) => (B,T,H)
        mel_tgt = mel_tgt.transpose(1, 2)
        dur_lens = in_lens
        dur_tgt = attn_dur
        dur_mask = mask_from_lens(dur_lens, max_len=dur_tgt.size(1))

        if "dur" in self.loss_to_train:
            log_dur_tgt = torch.log(dur_tgt.float() + 1)
            loss_fn = F.mse_loss
            dur_pred_loss = loss_fn(log_dur_pred, log_dur_tgt, reduction="none")
            dur_pred_loss = (dur_pred_loss * dur_mask).sum() / dur_mask.sum()
        else: dur_pred_loss = 0
        
        if "mel" in self.loss_to_train:
            ldiff_mel = mel_tgt.size(1) - mel_out.size(1)
            mel_out = F.pad(mel_out, (0, 0, 0, ldiff_mel, 0, 0), value=0.0)
            mel_mask = mel_tgt.ne(0).float()
            loss_fn = F.l1_loss
            mel_loss = loss_fn(mel_out, mel_tgt, reduction="none")
            mel_loss = (mel_loss * mel_mask).sum() / mel_mask.sum()
        else: mel_loss = 0
        
        if "pitch" in self.loss_to_train:
            ldiff = pitch_tgt.size(2) - pitch_pred.size(2)
            pitch_pred = F.pad(pitch_pred, (0, ldiff, 0, 0, 0, 0), value=0.0)
            # pitch_loss = F.mse_loss(pitch_tgt, pitch_pred, reduction="none")
            pitch_loss = F.l1_loss(pitch_tgt, pitch_pred, reduction="none")
            pitch_loss = (pitch_loss * dur_mask.unsqueeze(1)).sum() / dur_mask.sum()
        else: pitch_loss = 0

        if "energy" in self.loss_to_train and energy_pred is not None:
            energy_pred = F.pad(energy_pred, (0, ldiff, 0, 0), value=0.0)
            # energy_loss = F.mse_loss(energy_tgt, energy_pred, reduction="none")
            energy_loss = F.l1_loss(energy_tgt, energy_pred, reduction="none")
            energy_loss = (energy_loss * dur_mask).sum() / dur_mask.sum()
        else: energy_loss = 0

        # Attention loss
        if "attn" in self.loss_to_train:
            attn_loss = self.attn_ctc_loss(attn_logprob, in_lens, out_lens)
        else: attn_loss = 0

        loss = (
            dur_pred_loss * self.dur_predictor_loss_scale
            + pitch_loss * self.pitch_predictor_loss_scale
            + energy_loss * self.energy_predictor_loss_scale
            + attn_loss * self.attn_loss_scale
        )

        if self.use_mel_loss:
            loss += mel_loss

        meta = {"loss": loss}
        if "dur" in self.loss_to_train:
            meta["duration_predictor_loss"] = dur_pred_loss.detach()
            meta["dur_error"] = (torch.abs(dur_pred - dur_tgt).sum() / dur_mask.sum()).detach()
        if "mel" in self.loss_to_train:
            meta["mel_loss"] = mel_loss.detach()
        if "pitch" in self.loss_to_train:
            meta["pitch_loss"] = pitch_loss.detach() * self.pitch_predictor_loss_scale
        if "energy" in self.loss_to_train and energy_pred is not None:
            meta["energy_loss"] = energy_loss * self.energy_predictor_loss_scale
        if "attn" in self.loss_to_train:
            meta["attn_loss"] = attn_loss.detach() * self.attn_loss_scale

        assert meta_agg in ("sum", "mean")
        if meta_agg == "sum":
            bsz = mel_out.size(0)
            meta = {k: v * bsz for k, v in meta.items()}
        return meta

class FPCFMLoss(FastPitchLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, model_out, targets, is_training=True, meta_agg="mean"):
        (
            dec_loss,
            dec_mask,
            dur_pred,
            log_dur_pred,
            pitch_pred,
            pitch_tgt,
            energy_pred,
            energy_tgt,
            attn_soft,
            attn_hard,
            attn_dur,
            attn_logprob,
        ) = model_out
        # TODO: implement loss_to_train
        (mel_tgt, in_lens, out_lens) = targets
        mel_tgt.requires_grad = False
        # (B,H,T) => (B,T,H)
        mel_tgt = mel_tgt.transpose(1, 2)
        dur_lens = in_lens
        dur_tgt = attn_dur
        dur_mask = mask_from_lens(dur_lens, max_len=dur_tgt.size(1))

        if "dur" in self.loss_to_train:
            log_dur_tgt = torch.log(dur_tgt.float() + 1)
            loss_fn = F.mse_loss
            dur_pred_loss = loss_fn(log_dur_pred, log_dur_tgt, reduction="none")
            dur_pred_loss = (dur_pred_loss * dur_mask).sum() / dur_mask.sum()
        else: dur_pred_loss = 0
        
        if "cfm" in self.loss_to_train:
            cfm_loss = dec_loss
        else: cfm_loss = 0
        
        if "pitch" in self.loss_to_train:
            ldiff = pitch_tgt.size(2) - pitch_pred.size(2)
            pitch_pred = F.pad(pitch_pred, (0, ldiff, 0, 0, 0, 0), value=0.0)
            # pitch_loss = F.mse_loss(pitch_tgt, pitch_pred, reduction="none")
            pitch_loss = F.l1_loss(pitch_tgt, pitch_pred, reduction="none")
            pitch_loss = (pitch_loss * dur_mask.unsqueeze(1)).sum() / dur_mask.sum()
        else: pitch_loss = 0

        if "energy" in self.loss_to_train and energy_pred is not None:
            energy_pred = F.pad(energy_pred, (0, ldiff, 0, 0), value=0.0)
            # energy_loss = F.mse_loss(energy_tgt, energy_pred, reduction="none")
            energy_loss = F.l1_loss(energy_tgt, energy_pred, reduction="none")
            energy_loss = (energy_loss * dur_mask).sum() / dur_mask.sum()
        else: energy_loss = 0

        # Attention loss
        if "attn" in self.loss_to_train:
            attn_loss = self.attn_ctc_loss(attn_logprob, in_lens, out_lens)
        else: attn_loss = 0

        loss = (
            dur_pred_loss * self.dur_predictor_loss_scale
            + pitch_loss * self.pitch_predictor_loss_scale
            + energy_loss * self.energy_predictor_loss_scale
            + attn_loss * self.attn_loss_scale
        )

        if self.use_mel_loss:
            loss += cfm_loss

        meta = {"loss": loss}
        if "dur" in self.loss_to_train:
            meta["duration_predictor_loss"] = dur_pred_loss.detach()
            meta["dur_error"] = (torch.abs(dur_pred - dur_tgt).sum() / dur_mask.sum()).detach()
        if "cfm" in self.loss_to_train:
            meta["cfm_loss"] = cfm_loss.detach()
        if "pitch" in self.loss_to_train:
            meta["pitch_loss"] = pitch_loss.detach() * self.pitch_predictor_loss_scale
        if "energy" in self.loss_to_train and energy_pred is not None:
            meta["energy_loss"] = energy_loss * self.energy_predictor_loss_scale
        if "attn" in self.loss_to_train:
            meta["attn_loss"] = attn_loss.detach() * self.attn_loss_scale

        assert meta_agg in ("sum", "mean")
        if meta_agg == "sum":
            bsz = pitch_tgt.size(0)
            meta = {k: v * bsz for k, v in meta.items()}
        return meta
