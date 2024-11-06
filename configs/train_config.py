from dataclasses import dataclass

@dataclass
class TrainerConfig:
    load: bool | str = False
    ckpt_every: int = 25000
    use_ema: bool = True
    num_train_step: int = 5
    best_loss_key: str = "loss"

    @classmethod
    def fastpitch_base(cls):
        return cls(
            use_ema = False,
            num_train_step = 2_000_000
        )

    @classmethod
    def fastpitch_char_base(cls):
        return cls(
            use_ema = False,
            ckpt_every = 5000,
            num_train_step = 2_000_000,
            best_loss_key = "mel_loss"
        )
    
    @classmethod
    def fpcfm_base(cls):
        return cls(
            use_ema = False,
            ckpt_every = 20000,
            num_train_step = 5_000_000
        )
    
    @classmethod
    def vocgan_base(cls):
        return cls(
            use_ema=False,
            num_train_step = 1_000_000
        )
    
    @classmethod
    def fpbigvgan_base(cls):
        return cls(
            use_ema=False,
            num_train_step = 1_000_000,
            best_loss_key = "loss_g_total"
        )
    
    @classmethod
    def matcha_base(cls):
        return cls(
            use_ema=False,
            num_train_step = 1_000_000,
        )