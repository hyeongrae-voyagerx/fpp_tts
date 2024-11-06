import torch
import argparse
import os.path as osp

from configs import get_configs
from models import get_model
from datasets.text_utils import text_to_sequence
from utils import save_audio
from utils.baseline_tp import symbol_to_id
from jamo import h2j

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Inferencer:
    def __init__(self, args):
        fp_config, _, _ = get_configs("fastpitch")
        # fp_config, _, _ = get_configs("fpcfm")
        voc_config, _, _ = get_configs("bigvgan")

        self.fp = get_model(model_config=fp_config).to(device=_device)
        self.voc = get_model(model_config=voc_config).to(device=_device)
        self.load(args.fp_load, args.voc_load)
        self.style = args.style
        self.fp_load = args.fp_load
        self.voc_load = args.voc_load

    def load(self, fp, voc):
        fp_state_dict = torch.load(fp, weights_only=False)
        self.fp.load_state_dict(fp_state_dict["model"], strict=False)

        voc_state_dict = torch.load(voc, weights_only=False)
        if "model" in voc_state_dict:
            voc_state_dict = voc_state_dict["model"]
        if any("bigvgan." in item for item in voc_state_dict):
            voc_state_dict = {k.replace("bigvgan.", ""): v for k, v in voc_state_dict.items() if k.startswith("bigvgan.")}
        self.voc.load_state_dict(voc_state_dict)

    def infer(self, text=None, baseline=False):
        if baseline:
            self.load_baseline_fp()
            inputs = self.get_inputs(text, baseline=True)
        else:
            self.load(self.fp_load, self.voc_load)
            inputs = self.get_inputs(text, baseline=False)
        mel_out = self.fp.predict_step(inputs, return_vars=("mel_out", ))
        if mel_out.shape[-1] == 80:
            mel_out = mel_out.transpose(-2, -1)
        gen_audio, _ = self.voc.predict_step(mel_out)
        fname = "m_" + (f"baseline_{text[:10].replace(' ', '_')}.wav" if baseline else f"gen_{text[:10].replace(' ', '_')}.wav")
        path = osp.join("test", fname)

        save_audio(gen_audio, path)

    def get_inputs(self, text=None, baseline=False):
        if text is None:
            text = "초등학교, 중학교, 고등학교를 거쳐 대학교 과정까지."
        text = " " + text
        if baseline or True:
            text = [symbol_to_id[c] for c in h2j(text)]
            t = torch.tensor(text)[None].to(device=_device)
        else:
            t = torch.tensor(text_to_sequence(text)[0])[None].to(device=_device)
        m = torch.load(self.style, weights_only=False)[None].to(device=_device)

        return t, m

    def load_baseline_fp(self):
        state_dict = torch.load("last.ckpt", weights_only=False)["state_dict"]
        fp_state_dict = {k.replace("fastpitch.", ""): v for k, v in state_dict.items() if k.startswith("fastpitch")}
        self.fp.load_state_dict(fp_state_dict, strict=False)



def main(args):
    inferencer = Inferencer(args)
    with open("test.txt", "r") as f:
        texts = f.read().split("\n")

    for t in texts:
        inferencer.infer(t, baseline=False)
        # inferencer.infer(t, baseline=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp_load", required=True, type=str)
    parser.add_argument("--voc_load", required=True, type=str)
    # parser.add_argument("--style", type=str, default="m_homeshopping_static.tensor")
    parser.add_argument("--style", type=str, default="/data/tts/korean_va/static_mels/8Happy반갑다NoneNoneNone.pt")
    # parser.add_argument("--style", type=str, default="/data/tts/korean_va/static_mels/13Happy즐겁다NoneNoneNone.pt")

    args = parser.parse_args()
    main(args)