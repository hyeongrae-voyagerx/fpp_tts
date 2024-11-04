from .korean_va import KoreanVADataset
from .characteristic import CharacteristicDataset

_dataset_dict = {
    "korean_va": KoreanVADataset,
    "characteristic": CharacteristicDataset
}

def get_dataloaders(dataset_config):
    Dataset = _dataset_dict[dataset_config.name]
    test_kwargs = getattr(dataset_config, "test_kwargs", None)
    return {
        "train": Dataset.get_dataloader(**dataset_config.train_kwargs),
        "valid": Dataset.get_dataloader(**dataset_config.valid_kwargs),
        "test": Dataset.get_dataloader(**test_kwargs) if test_kwargs is not None else None
    }
