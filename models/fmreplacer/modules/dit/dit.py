import torch
import torch.nn as nn
from einops import repeat, pack
from x_transformers.x_transformers import RotaryEmbedding

from .ditblock import DiTBlock, TimestepEmbedding, AdaLayerNormZero_Final, InputEmbedding

class DiT(nn.Module):
    def __init__(self, *, 
                 dim, depth = 8, heads = 8, dim_head = 64, dropout = 0.1, ff_mult = 4,
                 mel_dim = 100, long_skip_connection = False,
    ):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        self.dim = dim
        self.depth = depth
        self.input_embed = InputEmbedding(mel_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)
        
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    ff_mult = ff_mult,
                    dropout = dropout
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias = False) if long_skip_connection else None
        
        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
        self,
        x, # nosied input audio : float['b n d']
        mask, # mask : bool['b n']
        mu,  # text expanded by duration : float['b n d']
        time,  # time step : float['b'] | float['']
    ):
        batch, seq_len = x.shape[0], x.shape[2]
        if time.ndim == 0:
            time = repeat(time, ' -> b', b = batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        x = pack([x, mu], "b * t")[0].transpose(1, 2)
        x = self.input_embed(x)
        mask = mask.squeeze(1).to(torch.bool)
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x
        
        for block in self.transformer_blocks:
            x = block(x, t, mask = mask, rope = rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim = -1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        output = output.transpose(1, 2)
        return output


