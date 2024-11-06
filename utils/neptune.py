import neptune
from pathlib import Path
from contextlib import contextmanager
import pickle as pkl
import os

_cache_dir = "./cache"

def init(model_config):
    run = neptune.init_run(
        project=model_config.project_name,
        name=model_config.exp_name,
        with_id=model_config.resume,
        mode="async"
    )

def load_weight(model_config):
    run = neptune.init_run(
        project=model_config.project_name,
        name=model_config.exp_name,
        with_id=model_config.resume,
        mode="read-only"
    )

    weight_path = _get_weight_from_run(run)
    return weight_path

def _get_weight_from_run(run):
    data = run.get_structure()
    best_path_string = data["training"]["model"]["best_model_path"].fetch()
    best_path = Path(best_path_string)
    cache_dir = best_path.parent.resolve()
    if not cache_dir.exists():
        cache_dir = Path(_cache_dir, data["sys"]["id"].fetch())
        cache_dir.mkdir(exist_ok=True, parents=True)

    if not best_path.exists():
        data["training"]["model"]["checkpoints"][best_path.stem].download(str(best_path))
    return best_path

@contextmanager
def cache_write(key, item, delete=False):
    if "." not in key.split("/")[-1]:
        key = key + ".pkl"
    file_path = Path(_cache_dir, key)
    with open(str(file_path), "wb") as fw:
        pkl.dump(item)
    
    yield str(file_path)

    if delete and file_path.exists():
        os.remove(file_path)


class NeptuneLogger:
    def __init__(self, model_config):
        self.run = init(model_config)

    def log(self, key: str, item: str | float | int):
        self.run[key].log(item)
    
    def upload(self, key: str, item, empty_cache=False):
        with cache_write(key, item, empty_cache) as file:
            self.run[key].upload(file)