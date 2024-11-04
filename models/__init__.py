from .fastpitch import FastPitch
from .vocgan import VocGAN
from .nvBigVGAN import BigVGAN
from .fpcfm import FPCFM
from .fpbigvgan import FPBigVGAN

_model_dict = {
    "fastpitch": FastPitch,
    "vocgan": VocGAN,
    "bigvgan": BigVGAN,
    "fpcfm": FPCFM,
    "fpbigvgan": FPBigVGAN
}

def get_model(model_config, load=None):
    Model = _model_dict[model_config.name]
    return Model(model_config)