from .config import Config

def get_configs(model_name):
    config = getattr(Config, f"{model_name}_base")()
    model_config = config["model"]
    trainer_config = config["trainer"]
    data_config = config["data"]

    return model_config, trainer_config, data_config