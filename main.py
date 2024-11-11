import torch

import os
import os.path as osp
import argparse

from configs import get_configs
from models import get_model
from datasets import get_dataloaders
from utils import DummyLogger, LossFormatter, EMA, draw_mel_pitch, save_audio, NeptuneLogger

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# _start_weight = "/home/hyeongrae/vrew_tts/training/fast_pitch_bigv_gan/cache/TTS-952/last.ckpt"
# _start_weight = "/home/hyeongrae/vrew_tts/training/fast_pitch_bigv_gan/cache/TTS-1935/last.ckpt"
_start_weight = "results1/latest_bi_pt.pt"

class Trainer:
    def __init__(self, args):
        model_config, trainer_config, data_config = get_configs(args.model)
        self.logger = NeptuneLogger(model_config) if not args.debug else DummyLogger()
        self.dataloaders = get_dataloaders(data_config)
        self.model = get_model(model_config=model_config).to(device=_device)
        if getattr(trainer_config, "ema", False):
            self.ema = EMA(self.model, decay=0.999)

        self.save_dir = f"results{args.gpu}"
        self.step = 1
        self.ckpt_every = trainer_config.ckpt_every
        self.best_loss_key = getattr(trainer_config, "best_loss_key", "loss")

        self.debug = args.debug
        self.num_train_step = trainer_config.num_train_step
        self.model_name = args.model
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train_step(self, data, step):
        self.model.train()
        loss = LossFormatter()
        for i in range(len(data)):
            if data[i] is None:
                continue
            data[i] = data[i].to(device=_device, non_blocking=True)
        try:
            model_out = self.model.training_step(data, step)
        except KeyboardInterrupt:
            breakpoint()
        except:
            print("piui")
            loss.add(0, "loss")
            return loss
        model_out.pop("loss")
        for item in filter(lambda x: "loss" in x, model_out):
            loss.add(model_out[item], item)

        if hasattr(self, "ema"):
            self.ema.update()
        
        return loss

    def train(self):
        self.loss_best = float("inf")
        while self.step < self.num_train_step:
            for step, data in enumerate(self.dataloaders["train"], self.step):
                loss = self.train_step(data, step)
                for k, v in loss.loss_items.items():
                    self.logger.log(f"training/{k}", v)
                print(f"\r | [{step:06d}] | loss = {loss}", end="")
                
                self.step = step
                if step % self.ckpt_every == 0 or self.debug:
                    avg_loss = self.validate()
                    print(f"| val_loss = {avg_loss:.3f} (best = {self.loss_best:.3f})")
                    if not self.debug and avg_loss < self.loss_best:
                        self.loss_best = avg_loss
                        self.save()
                    

    def save(self):
        state_dict = {
            'model': self.model.state_dict(),
            'ema': self.ema.shadow if hasattr(self, "ema") else None,
            'step': self.step
        }
        # ckpt_path = os.path.join(self.save_dir, f"ckpt_{self.step}.pt")
        ckpt_path = os.path.join(self.save_dir, f"latest.pt")
        self.logger.upload(f"training/model/checkpoints/best.pt", state_dict)
        self.logger.log("training/model/best_model_path", "/home/hyeongrae/fpp_tts/cache/training/model/checkpoints/best.pt", override=False)
        torch.save(state_dict, ckpt_path)

    def load(self, path=None, load_step=False):
        ckpt_path = osp.join(self.save_dir, f"latest.pt") if path is None else path
        state_dict = torch.load(ckpt_path, weights_only=False)
        self.model.load_state_dict(state_dict["model"])
        if hasattr(self, "ema"):
            self.ema.shadow = state_dict["ema"]
        self.step = state_dict["step"] if load_step else 1

    def load_baseline_fp(self):
        if osp.exists(_start_weight):
            state_dict = torch.load(_start_weight, weights_only=False)["model"]#["state_dict"]
        else:
            print(f"| Fail to load {_start_weight}, just use last.ckpt in the project directory")
            state_dict = torch.load("last.ckpt", weights_only=False)["state_dict"]
        fp_state_dict = {k.replace("fastpitch.", ""): v for k, v in state_dict.items() if k.startswith("fastpitch")}
        self.model.load_state_dict(fp_state_dict, strict=False)

    @EMA.ema_mode
    @torch.no_grad()
    def validate(self):
        print()
        self.model.eval()
        total_loss = 0
        for idx, data in enumerate(self.dataloaders["valid"], 1):
            for i in range(len(data)):
                if data[i] is None:
                    continue
                data[i] = data[i].to(device=_device, non_blocking=True)
            model_out = self.model.training_step(data, idx, do_train=False)
            total_loss += model_out[self.best_loss_key].item()
            avg_loss = total_loss / idx
            
            if idx > 100:
                break
        # if avg_loss < self.loss_best:
        #     self.save_sample(data)
        return avg_loss

    @torch.no_grad()
    def save_sample(self, data):
        for i in range(len(data)):
            if data[i] is None:
                continue
            data[i] = data[i].to(device=_device, non_blocking=True)
        if self.model_name == "fastpitch":
            mel_out, pitch_pred, pitch_true, dur_pred = self.model.predict_step(data, return_vars=("mel_out", "pitch_pred", "pitch_true", "dur_pred"))
            pitch_true = self.model.average_pitch(pitch_true.to(dur_pred.device), dur_pred)
            draw_mel_pitch(mel_out, pitch_pred, pitch_true, osp.join(self.save_dir, "result.png"))
        if self.model_name == "vocgan":
            audio_out, _ = self.model.predict_step(data[2])
            save_audio(audio_out, osp.join(self.save_dir, "result.wav"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", default=0, type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--debug", "-D", action="store_true")
    parser.add_argument('--no_ema', dest='use_ema', action='store_const', const=False, default=True)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    trainer = Trainer(args)
    trainer.train()