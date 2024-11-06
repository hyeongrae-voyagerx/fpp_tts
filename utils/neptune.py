import neptune
from pathlib import Path
from contextlib import contextmanager
import pickle as pkl
import os
import torchaudio
import torch

_cache_dir = "./cache"

def init(model_config):
    run = neptune.init_run(
        project=model_config.project_name,
        name=model_config.exp_name,
        with_id=model_config.resume,
        mode="async"
    )
    return run

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
    if _getitem_nest(data, "training", "model", "best_model_path") is not None:
        breakpoint()
        best_path_string = _fetch(data["training"]["model"]["best_model_path"])
        best_path = Path(best_path_string)
        cache_dir = best_path.parent.resolve()
        if not cache_dir.exists():
            cache_dir = Path(_cache_dir, _fetch(data["sys"]["id"]))
            cache_dir.mkdir(exist_ok=True, parents=True)

        if not best_path.exists():
            data["training"]["model"]["checkpoints"][best_path.stem].download(str(best_path))
        return best_path
    else:
        return None

@contextmanager
def cache_write(key, item, delete=False, **kwargs):
    if "." not in key.split("/")[-1]:
        key = key + ".pkl"
    file_path = Path(_cache_dir, key)
    file_path.parent.mkdir(exist_ok=True, parents=True)
    if key.endswith(".wav"):
        torchaudio.save(file_path, item, kwargs.get("sr", 22050))
    else:
        torch.save(item, str(file_path))
    
    yield str(file_path)

    if delete and file_path.exists():
        os.remove(file_path)

def _getitem_nest(dict_: dict, *args):
    if len(args) > 1:
        if args[0] not in dict_:
            return None
        return _getitem_nest(dict_[args[0]], *args[1:])
    elif len(args) == 1:
        return dict_.get(args[0], None)
    else:
        raise KeyError("| The number of args should be > 0")


class NeptuneLogger:
    def __init__(self, model_config):
        self._run = init(model_config)

    def log(self, key: str, item: str | float | int | dict, override=True):
        if not override and self.exists(key):
            return
        if isinstance(item, (str, float, int)):
            self._run[key].log(item)
        elif isinstance(item, dict):
            if len(item) > 50:
                print("| [WARNING] many items to save. isn't it state_dict?")
            for k, v in item.items():
                key_ = "/".join([key, k])
                if isinstance(v, (str, float, int, dict)):
                    self.log(key_, v)
                else:
                    self.upload(key_, v, empty_cache=True)
        
    
    def upload(self, key: str, item, empty_cache=False, **kwargs):
        with cache_write(key, item, empty_cache, **kwargs) as file:
            self._run[key].upload(file, wait=empty_cache) # wait if synchronous

    def save_audio(self, key, wav: torch.Tensor, sr=22050):
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if not key.endswith(".wav"):
            key += ".wav"
        self.upload(key, wav.detach().cpu(), empty_cache=True, sr=sr)

    def exists(self, key):
        data = self._run.get_structure()
        return _getitem_nest(data, *key.split("/")) is not None


def _fetch(item: neptune.attributes.series.series.Series | neptune.attributes.atoms.atom.Atom):
    if isinstance(item, neptune.attributes.atoms.atom.Atom):
        return item.fetch()
    elif isinstance(item, neptune.attributes.series.series.Series):
        return item.fetch_last()
    else:
        breakpoint()
        raise TypeError(f"| Unknown type to fetch in neptune: {type(item)}")

if __name__ == "__main__":
    model_config = type("Temp", (object, ), dict(project_name="v6x/fp-pitch", exp_name=None, resume="FPP-17"))
    logger = NeptuneLogger(model_config)
    wav, sr = torchaudio.load("piui.wav")
    logger.save_audio("wave.wav", wav, sr)
    breakpoint()