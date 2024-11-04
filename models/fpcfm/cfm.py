"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
from typing import Callable
from random import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torchdiffeq import odeint

from einops import rearrange

from .utils import exists
from ..modules.nv_utils import mask_from_lens


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma = 0.,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method = 'euler'  # 'midpoint'
        ),
        audio_drop_prob = 0.3,
        cond_drop_prob = 0.2,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.),
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs


    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        inp, # mel or raw wave
        latent,
        style_vector,
        *,
        lens = None,
    ):
        batch, seq_len, dtype, device, σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device = device)
        

        # mel is x1
        x1 = inp
        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype = dtype, device = self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = rearrange(time, 'b -> b 1 1')
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # transformer and cfg training with a drop rate
        drop_saln = random() < self.audio_drop_prob  # p_drop in voicebox paper
        # if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
        #     drop_saln = True
            
        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(x = φ, latent = latent, time = time, style_vector=style_vector, drop_saln = drop_saln)

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        # TODO check twice
        loss_mask = mask_from_lens(lens=lens)
        loss = (loss * loss_mask[..., None]).mean((1, 2)) * loss.shape[1] / lens

        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        latent,
        style_vector,
        *,
        steps = 32,
        cfg_strength = 1., 
        sway_sampling_coef = None,
    ):
        self.eval()

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(x = x, latent = latent, time = t, style_vector=style_vector, drop_saln = False)
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(x = x, latent = latent, time = t, style_vector=style_vector, drop_saln = True)
            return pred + (pred - null_pred) * cfg_strength
        # pred = self.transformer(x = φ, latent = latent, time = time, style_vector=style_vector, drop_saln = drop_saln)
        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = torch.randn((*latent.shape[:-1], 80), device=self.device)

        t_start = 0

        t = torch.linspace(t_start, 1, steps, device = self.device)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        
        sampled = trajectory[-1]
        out = sampled


        return out, trajectory
