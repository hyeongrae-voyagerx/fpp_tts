from dataclasses import dataclass, field, asdict

@dataclass
class DataConfig:
    name: str = "korean_va"

    train_kwargs: dict = field(default_factory=lambda: {
        "dataset_kwargs": {
            "data_dir": "/data/tts/korean_va/train/"
        },
        "dataloader_kwargs": {
            "batch_size": 32,
            "num_workers": 32,
            "drop_last": True,
        }
    })
    valid_kwargs: dict = field(default_factory=lambda: {
        "dataset_kwargs": {
            "data_dir": "/data/tts/korean_va/val/"
        },
        "dataloader_kwargs": {
            "batch_size": 8,
            "num_workers": 8,
            "drop_last": True,
        }
    })
    
    @classmethod
    def fastpitch_base(cls):
        return cls(
            name = "korean_va",
            train_kwargs = {
                "dataset_kwargs": {
                    "data_dir": "/data/tts/korean_va/train/"
                },
                "dataloader_kwargs": {
                    "batch_size": 32,
                    "num_workers": 32,
                    "drop_last": True,
                }
            },
            valid_kwargs = {
                "dataset_kwargs": {
                    "data_dir": "/data/tts/korean_va/val/"
                },
                "dataloader_kwargs": {
                    "batch_size": 32,
                    "num_workers": 32,
                    "drop_last": True,
                }
            }
        )
    
    @classmethod
    def fastpitch_char_base(cls):
        return cls(
            name = "characteristic",
            train_kwargs = {
                "dataset_kwargs": {
                    "data_dir": "/home/hyeongrae/data/m_homeshopping/"
                },
                "dataloader_kwargs": {
                    "batch_size": 16,
                    "num_workers": 16,
                    "drop_last": True,
                }
            },
            valid_kwargs = {
                "dataset_kwargs": {
                    "data_dir": "/home/hyeongrae/data/m_homeshopping/"
                },
                "dataloader_kwargs": {
                    "batch_size": 32,
                    "num_workers": 16,
                    "drop_last": True,
                }
            }
        )
    
    @classmethod
    def fpcfm_base(cls):
        return cls(
            name = "korean_va",
            train_kwargs = {
                "dataset_kwargs": {
                    "data_dir": "/data/tts/korean_va/train/"
                },
                "dataloader_kwargs": {
                    "batch_size": 32,
                    "num_workers": 16,
                    "drop_last": True,
                }
            },
            valid_kwargs = {
                "dataset_kwargs": {
                    "data_dir": "/data/tts/korean_va/val/"
                },
                "dataloader_kwargs": {
                    "batch_size": 32,
                    "num_workers": 16,
                    "drop_last": True,
                }
            }
        )
    
    @classmethod
    def vocgan_base(cls):
        return cls(
            train_kwargs = {
                "dataset_kwargs": {
                    "data_dir": "/data/tts/korean_va/train/"
                },
                "dataloader_kwargs": {
                    "batch_size": 8,
                    "num_workers": 8,
                    "drop_last": True,
                }
            }
        )
    
    @classmethod
    def fpbigvgan_base(cls):
        return cls(
            name = "characteristic",
            train_kwargs = {
                "dataset_kwargs": {
                    "data_dir": "/home/hyeongrae/data/m_homeshopping/"
                },
                "dataloader_kwargs": {
                    "batch_size": 12,
                    "num_workers": 12,
                    "shuffle": True,
                    "drop_last": True,
                }
            },
            valid_kwargs = {
                "dataset_kwargs": {
                    "data_dir": "/home/hyeongrae/data/m_homeshopping/"
                },
                "dataloader_kwargs": {
                    "batch_size": 8,
                    "num_workers": 8,
                    "drop_last": True,
                }
            }
        )

    @classmethod
    def matcha_base(cls):
        return cls(
            name="korean_va",
            train_kwargs = {
                "dataset_kwargs": {
                    "data_dir": "/data/tts/korean_va/train/"
                },
                "dataloader_kwargs": {
                    "batch_size": 32,
                    "num_workers": 16,
                    "drop_last": True,
                }
            },
            valid_kwargs = {
                "dataset_kwargs": {
                    "data_dir": "/data/tts/korean_va/val/"
                },
                "dataloader_kwargs": {
                    "batch_size": 32,
                    "num_workers": 16,
                    "drop_last": True,
                }
            }
        )
