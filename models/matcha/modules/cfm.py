from abc import ABC

import torch
import torch.nn.functional as F

from .decoder import Decoder
from .dit import DiT


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        use_saln=True,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.use_saln = use_saln
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, style=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, style=style, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, style, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        sb, sd, st = style.shape

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        mask = torch.cat((torch.ones_like(style)[:, :1], mask), dim=-1)
        for step in range(1, len(t_span)):
            if step == 1:
                x = torch.cat((style, x), dim=-1)
                mu = torch.cat((torch.zeros_like(style), mu), dim=-1)
            else:
                x[..., :st] = style
                mu[..., :st] = torch.zeros_like(style)
                
            dphi_dt = self.estimator(x, mask, mu, t)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        out = sol[-1]
        out = out[..., st:]
        return out

    def compute_loss(self, x1, mask, mu, style_mel=None, style_mel_mask=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, mu_t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        y = torch.cat((style_mel, y), dim=-1)
        mu = torch.cat((torch.zeros_like(style_mel), mu), dim=-1)
        y_mask = torch.cat((style_mel_mask, mask), dim=-1)
        u_hat = self.estimator(y, y_mask, mu, t.squeeze())
        u_hat = u_hat[..., -mu_t:]
        loss = F.mse_loss(u_hat, u, reduction="none") * mask
        loss = loss.mean((1, 2)) * mu_t / mask.sum((1, 2))
        loss = loss.mean()
        return loss, y


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, use_saln=True, spk_emb_dim=128):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            use_saln=use_saln,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels = in_channels
        # Just change the architecture of the estimator here
        if decoder_params.name == "unet":
            self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params.params._dict())
        elif decoder_params.name == "dit":
            self.estimator = DiT(**decoder_params.params._dict())