from typing import Optional, List
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from frozendict import frozendict

@dataclass
class FastPitchConfig:
    n_mel_channels: int = 80
    max_seq_len: int = 2048

    n_symbols: int = 200
    padding_idx: int = 0
    symbols_embed_dims: int = 384

    in_fft_layers: int = 6
    in_fft_heads: int = 1
    in_fft_head_dims: int = 64
    in_fft_kernel_size: int = 3
    in_fft_filter_size: int = 1536
    in_fft_output_size: int = 384
    in_fft_dropout: float = 0.1
    in_fft_att_dropout: float = 0.1
    in_fft_emb_dropout: float = 0.0

    out_fft_layers: int = 6
    out_fft_heads: int = 1
    out_fft_head_dims: int = 64
    out_fft_kernel_size: int = 3
    out_fft_filter_size: int = 1536
    out_fft_output_size: int = 384
    out_fft_dropout: float = 0.1
    out_fft_att_dropout: float = 0.1
    out_fft_emb_dropout: float = 0.0

    dur_pred_kernel_size: int = 3
    dur_pred_filter_size: int = 256
    dur_pred_dropout: float = 0.1
    dur_pred_n_layers: int = 2
    dur_loss_scale: float = 0.1

    pitch_pred_kernel_size: int = 3
    pitch_pred_filter_size: int = 256
    pitch_pred_dropout: float = 0.1
    pitch_pred_n_layers: int = 2
    pitch_cond_formants: int = 1
    pitch_loss_scale: float = 0.1

    energy_conditioning: bool = True
    energy_pred_kernel_size: int = 3
    energy_pred_filter_size: int = 256
    energy_pred_dropout: float = 0.1
    energy_pred_n_layers: int = 2

    attn_loss_scale: float = 1.0

    pitch_embed_kernel_size: int = 3
    energy_embed_kernel_size: int = 3
    speaker_emb_weight: float = 1.0
    n_speakers: int = 1

    saln: bool = True

    jit: bool = False

    kl_loss_start_step: int = 0 
    kl_loss_warmup_steps: int = 1
    kl_loss_weight: float = 1.0

    sr: int = 22050
    mel_hop_size: int = 256

    period_encoding: int = 5
    comma_encoding: int = 4
    question_mark_encoding: int = 8
    exclamation_mark_encoding: int = 1
    space_encoding: int = 9
    colon_encoding: int = 6
    semicolon_encoding: int = 7

    use_mel_loss: bool = True
    loss_to_train: tuple = ("mel", "attn", "dur", "pitch", "energy") # ("pitch", ) # Choice(s) from ("mel", "attn", "dur", "pitch", "energy")

    optimizers: tuple[str] = ("Adam", )
    lrs: tuple[float] = (1e-5, )
    opt_args: tuple[dict] = ({"betas": (0.5, 0.9)}, )
    # Leave the params as an empy string to
    params: List = field(default_factory=lambda: [""])

    # TODO: Support multiple schedulers
    sch: Optional[str] = ("LinearLR", )
    sch_args: str = ({"start_factor": 0.0001, "total_iters": 1}, )

@dataclass
class BigVGANConfig:
    project_name: str = "v6x/fp-pitch"
    resume: Optional[str] = None # "VOC-492"
    best: bool = False

    exp_name: Optional[str] = None

    resblock: str = "1"
    p_dropout: float = 0.1,
    resblock_kernel_sizes: tuple[int] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_rates: tuple[int] = (8, 8, 2, 2)
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: tuple[int] = (16, 16, 4, 4)
    use_spectral_norm: bool = False
    filter_length: float = 1024
    hop_length: int = 256
    win_length: int = 1024
    segment_size: int = 8192
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = 8000
    num_mels: int = 80
    sr: int = 22050
    discriminator_channel_mult: int = 1

    activation: str = "snakebeta"
    snake_logscale: bool = True
    resolutions: tuple[tuple[int]] = ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))
    mpd_reshapes: tuple[int] = (2, 3, 5, 7, 11)

    c_mel: int = 45

    opts: tuple = ("AdamW", "AdamW") # field(default_factory=lambda: ["AdamW", "AdamW"])
    lrs: tuple = (2e-4, 2e-4) # field(default_factory=lambda: [2e-4, 2e-4])
    opt_args: tuple = ({"betas": (0.8, 0.99)}, {"betas": (0.8, 0.99)}) # field(
    #     default_factory=lambda: ['{"betas": (0.8, 0.99)}', '{"betas": (0.8, 0.99)}']
    # )
    # Leave the params as an empy string to
    params: List = field(default_factory=lambda: [".gen", [".mrd", ".mpd"]])

    # TODO: Support multiple schedulers
    sch: Optional[str] = "LinearLR"
    sch_args: tuple[dict] = ({"start_factor": 0.0001, "total_iters": 1000}, {"start_factor": 0.0001, "total_iters": 1000})


@dataclass
class ModelConfig:
    fastpitch_config = FastPitchConfig
    bigvgan_config = BigVGANConfig

    voc_sample_len: int = 16384
    fp_load: str = "results3/latest_mhome.pt"
    bv_load: str = "bv.ckpt"

def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="model_config", node=ModelConfig)
