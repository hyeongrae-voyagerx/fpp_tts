from dataclasses import dataclass, field, asdict
from frozendict import frozendict

@dataclass
class EncoderParams:
    n_feats: int = 80
    n_channels: int = 192
    filter_channels: int = 768
    filter_channels_dp: int = 256
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: int = 0.1
    spk_emb_dim: int = 64
    n_spks: int = 1
    prenet: bool = True

@dataclass
class DurPredParams:
    filter_channels_dp: int = 256
    kernel_size: int = 1
    p_dropout: float = 0.1

@dataclass
class CFMParams:
    solver: str = "euler"
    sigma_min: float = 1e-4

@dataclass
class DecoderParams:
    channels: tuple[int] = (256, 256)
    dropout: float = 0.05
    attention_head_dim: int = 64
    n_blocks: int = 1
    num_mid_blocks: int = 2
    num_heads: int = 2
    act_fn: str = "snakebeta"

    def _dict(self):
        return asdict(self)

@dataclass
class ModelConfig:
    project_name: str = "v6x/fp-pitch"
    exp_name: None | str = None
    resume: None | str = None
    
    n_vocab: int = 200
    use_saln: bool = True
    use_prior_loss: bool = True
    encoder_params: EncoderParams = field(default_factory=EncoderParams)
    dur_pred_params: DurPredParams = field(default_factory=DurPredParams)
    cfm_params: CFMParams = field(default_factory=CFMParams)
    decoder_params: DecoderParams = field(default_factory=DecoderParams)
    data_statistics: None | tuple[float] = None # (mean, std)
    aligner: str = "alf"
    opt: str = "Adam"
    opt_args: frozendict = frozendict(lr=1e-4, betas=(0.9, 0.99))