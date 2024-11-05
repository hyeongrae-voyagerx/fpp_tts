from dataclasses import dataclass
from importlib import import_module
from .train_config import TrainerConfig
from .data_config import DataConfig

@dataclass
class Config:
    model_name: str

    trainer_config: TrainerConfig
    data_config: DataConfig

    def __post_init__(self):
        module = import_module(f".{self.model_name}_config", "configs")
        self.model_config = getattr(module, "ModelConfig")
        self.model_config.name = self.model_name

    def __getitem__(self, key):
        match key:
            case "model":
                return self.model_config
            case "trainer":
                return self.trainer_config
            case "data":
                return self.data_config
            case _:
                raise KeyError(f"Unknown key: {key}")
    
    @classmethod
    def fastpitch_base(cls):
        return cls(
            model_name="fastpitch",
            trainer_config=TrainerConfig.fastpitch_base(),
            data_config=DataConfig.fastpitch_base()
        )
    
    @classmethod
    def fastpitch_char_base(cls):
        return cls(
            model_name="fastpitch",
            trainer_config=TrainerConfig.fastpitch_char_base(),
            data_config=DataConfig.fastpitch_char_base()
        )
    
    @classmethod
    def fpcfm_base(cls):
        return cls(
            model_name="fpcfm",
            trainer_config=TrainerConfig.fpcfm_base(),
            data_config=DataConfig.fpcfm_base()
        )
    
    @classmethod
    def vocgan_base(cls):
        return cls(
            model_name="vocgan",
            trainer_config=TrainerConfig.vocgan_base(),
            data_config=DataConfig.vocgan_base()
        )
    
    @classmethod
    def bigvgan_base(cls):
        return cls(
            model_name="bigvgan",
            trainer_config=TrainerConfig.vocgan_base(),
            data_config=DataConfig.vocgan_base()
        )
    
    @classmethod
    def fpbigvgan_base(cls):
        return cls(
            model_name="fpbigvgan",
            trainer_config=TrainerConfig.fpbigvgan_base(),
            data_config=DataConfig.fpbigvgan_base()
        )
    
    @classmethod
    def matcha_base(cls):
        return cls(
            model_name="matcha",
            trainer_config=TrainerConfig.matcha_base(),
            data_config=DataConfig.matcha_base()
        )