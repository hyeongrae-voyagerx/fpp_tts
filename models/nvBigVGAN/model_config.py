import os
from typing import Optional, List
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from pathlib import Path


@dataclass
class ModelConfig:
    project_name: str = "v6x/fp-pitch"
    resume: Optional[str] = None # "VOC-492"
    best: bool = False

    exp_name: Optional[str] = None

    resblock: str = "1"
    p_dropout: float = 0.1,
    resblock_kernel_sizes: List = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates: List = field(default_factory=lambda:[8, 8, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: List = field(default_factory=lambda: [16, 16, 4, 4])
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
    resolutions: List = field(default_factory=lambda: [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]])
    mpd_reshapes: List = field(default_factory=lambda: [2, 3, 5, 7, 11])

    c_mel: int = 45

    opts: List = field(default_factory=lambda: ["AdamW", "AdamW"])
    lrs: List = field(default_factory=lambda: [2e-4, 2e-4])
    opt_args: List = field(
        default_factory=lambda: ['{"betas": (0.8, 0.99)}', '{"betas": (0.8, 0.99)}']
    )
    # Leave the params as an empy string to
    params: List = field(default_factory=lambda: [".gen", [".mrd", ".mpd"]])

    # TODO: Support multiple schedulers
    sch: Optional[str] = "LinearLR"
    sch_args: str = '{"start_factor": 0.0001, "total_iters": 1000}'


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="model_config", node=ModelConfig)
