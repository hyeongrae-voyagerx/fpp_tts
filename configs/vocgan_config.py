import os
from typing import Optional, List
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from pathlib import Path


@dataclass
class ModelConfig:
    project_name: str = "v6x/fp-pitch"
    resume: Optional[str] = None #"VOC-582"
    best: bool = True

    exp_name: Optional[str] = None

    feat_match: float = 10.0
    adv_weight: int = 1
    use_subband_stft_loss: bool = False
    feat_loss: bool = False
    out_channels: int = 1
    sr: int = 22050
    hop_len: int = 256
    generator_ratio: tuple = (4, 4, 2, 2, 2, 2) # for 256 hop size and 22050 sample rate
    mult: int = 256
    n_residual_layers: int = 10
    num_d: int = 3
    ndf: int = 16
    n_layers: int = 3
    downsampling_factor: int = 4
    disc_out: int = 512
    dis_start_step: int = 1 # 3200000 # This is the actual number of samples, so it is the same regardless of batch size
    rep_disc: int = 1

    loss_fft_sizes: List = field(
        default_factory=lambda: [1024, 2048, 512]
    )  # List of FFT size for STFT-based loss.
    loss_hop_sizes: List = field(
        default_factory=lambda: [120, 240, 50]
    )  # List of hop size for STFT-based loss
    loss_win_lengths: List = field(
        default_factory=lambda: [600, 1200, 240]
    )  # List of window length for STFT-based loss.
    loss_window: str = "hann_window"  # Window function for STFT-based loss

    sub_loss_fft_sizes: List = field(
        default_factory=lambda: [384, 683, 171]
    )  # List of FFT size for STFT-based loss.
    sub_loss_hop_sizes: List = field(
        default_factory=lambda: [30, 60, 10]
    )  # List of hop size for STFT-based loss
    sub_loss_win_lengths: List = field(
        default_factory=lambda: [150, 300, 60]
    )  # List of window length for STFT-based loss.
    sub_loss_window: str = "hann_window"  # Window function for STFT-based loss

    # These should match the dataset config options
    n_mel_channels: int = 80

    # opts: List = field(default_factory=lambda: ["Adam", "Adam"])
    opts: tuple = ("Adam", "Adam")
    # lrs: List = field(default_factory=lambda: [1e-4, 1e-4])
    lrs: tuple = (1e-4, 1e-4)
    # opt_args: List = field(
    #     default_factory=lambda: ['{"betas": (0.5, 0.9)}', '{"betas": (0.5, 0.9)}']
    # )
    opt_args: tuple = ({"betas": (0.5, 0.9)}, {"betas": (0.5, 0.9)})
    # Leave the params as an empy string to
    params: List = field(default_factory=lambda: [".gen", ".dis"])

    # TODO: Support multiple schedulers
    sch: Optional[str] = "LinearLR"
    # sch_args: str = '{"start_factor": 0.0004, "total_iters": 1000}'
    sch_args: tuple = (
        {"start_factor": 0.0004, "total_iters": 1000},
        {"start_factor": 0.0004, "total_iters": 1000}
    )


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="model_config", node=ModelConfig)
