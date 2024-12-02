from .fastpitch import FastPitch
from .vocgan import VocGAN
from .nvBigVGAN import BigVGAN
from .fpcfm import FPCFM
from .fpbigvgan import FPBigVGAN
from .matcha import MatchaTTS
from .fmreplacer import FMReplacer

from utils.neptune import load_weight
import torch

_model_dict = {
    "fastpitch": FastPitch,
    "vocgan": VocGAN,
    "bigvgan": BigVGAN,
    "fpcfm": FPCFM,
    "fpbigvgan": FPBigVGAN,
    "matcha": MatchaTTS,
    "fmreplacer": FMReplacer
}

def get_model(model_config):
    Model = _model_dict[model_config.name]
    model = Model(model_config)

    if getattr(model_config, "resume", False):
        weight_path = load_weight(model_config)
        if weight_path is not None:
            w = torch.load(weight_path, weights_only=False)
            model.load_state_dict(w["model"])
            print(f"Load Pretrained weight from: {model_config.resume}")
    return model