from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    project_name: str = "v6x/fp-pitch"
    exp_name: None | str = None
    resume: None | str = None
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