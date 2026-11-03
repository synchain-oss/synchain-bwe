"""LavaSR's bandwidth-extension model, loaded and run on this machine.

The weights are the enhancer_v2 part of the Hugging Face repository YatharthS/LavaSR at a fixed revision, checked by
SHA-256 before anything is loaded. The network is Vocos (a mel front end, a ConvNeXt backbone and an iSTFT head); the
three parts are built here directly from the expected classes with the numbers in config.yaml, which must name exactly
those classes and only plain numbers and strings, and the weights are read with torch.load(weights_only=True). Nothing
else from the download is imported or run.

The model is fed what LavaSR's own pipeline feeds it: the input resampled to 16 kHz and back up to 48 kHz. Only its raw
guess is used; LavaSR's merge, which would keep the 16 kHz copy below 8 kHz and so throw away anything the input has
between 8 kHz and its own cutoff, is not.
"""
import hashlib
import io
import os
from pathlib import Path

import numpy as np

REPO = "YatharthS/LavaSR"
REVISION = "b98dc8be472da45ab7b6346ad7997e1dfeb5911d"
SUBDIR = "enhancer_v2"
FILES = {
    "config.yaml": "0d970f5cdc1913730c417b49e476bb09bb8b874583d113f71a7f10c8bb3a4b7d",
    "pytorch_model.bin": "d100db961b2c125d77a52a12215c689e44cd9926a72f117513395b7e25e6de12",
}
PARTS = {
    "feature_extractor": ("vocos.feature_extractors.MelSpectrogramFeatures",
                          {"sample_rate", "n_fft", "hop_length", "n_mels", "padding", "f_min", "f_max", "norm", "mel_scale"}),
    "backbone": ("vocos.models.VocosBackbone", {"input_channels", "dim", "intermediate_dim", "num_layers"}),
    "head": ("vocos.heads.ISTFTHead", {"dim", "n_fft", "hop_length", "padding"}),
}
HOP = 512


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _folder(folder):
    folder = Path(folder)
    if not (folder / "config.yaml").exists() and (folder / SUBDIR / "config.yaml").exists():
        folder = folder / SUBDIR
    return folder


def read_verified(folder):
    """{name: bytes} of config.yaml and pytorch_model.bin, each read once and checked by SHA-256, so what is loaded is
    exactly what was checked."""
    folder = _folder(folder)
    data = {}
    for name, digest in FILES.items():
        p = folder / name
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing")
        data[name] = p.read_bytes()
        if hashlib.sha256(data[name]).hexdigest() != digest:
            raise ValueError(f"{p} is not the file this version expects (SHA-256 differs); refusing to load it")
    return data


def verify(folder):
    """The folder holding config.yaml and pytorch_model.bin, after checking both."""
    folder = _folder(folder)
    for name, digest in FILES.items():
        p = folder / name
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing")
        if sha256(p) != digest:
            raise ValueError(f"{p} is not the file this version expects (SHA-256 differs); refusing to load it")
    return folder


def fetch(local_dir=None):
    """Downloads the two files at the fixed revision if they are not there yet, checks them and returns their folder."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import hf_hub_download

    paths = []
    for name in FILES:
        kw = {"repo_id": REPO, "filename": f"{SUBDIR}/{name}", "revision": REVISION}
        if local_dir:
            kw["local_dir"] = str(local_dir)
        try:
            paths.append(hf_hub_download(local_files_only=True, **kw))
        except Exception:
            paths.append(hf_hub_download(**kw))
    return verify(Path(paths[0]).parent)


def _check_config(cfg):
    if set(cfg) != set(PARTS):
        raise ValueError(f"unexpected model config sections: {sorted(cfg)}")
    for key, (cls, allowed) in PARTS.items():
        part = cfg[key]
        if set(part) != {"class_path", "init_args"} or part["class_path"] != cls:
            raise ValueError(f"unexpected model config for {key}: {part.get('class_path')}")
        args = part["init_args"]
        if not set(args) <= allowed or not all(isinstance(v, (int, float, str)) and not isinstance(v, bool) for v in args.values()):
            raise ValueError(f"unexpected model config arguments for {key}: {args}")


def _head_forward(head, x):
    """Vocos's iSTFT head as LavaSR runs it: the magnitude clipped at 1e3 instead of Vocos's 1e2."""
    import torch

    x = head.out(x).transpose(1, 2)
    mag, p = x.chunk(2, dim=1)
    mag = torch.clip(torch.exp(mag), max=1e3)
    return head.istft(mag * (torch.cos(p) + 1j * torch.sin(p)))


class Guesser:
    """Guesses the full band of a 48 kHz signal from what it has below 8 kHz."""

    def __init__(self, folder, threads=None):
        import torch
        import yaml
        from vocos.feature_extractors import MelSpectrogramFeatures
        from vocos.heads import ISTFTHead
        from vocos.models import VocosBackbone
        from vocos.pretrained import Vocos

        data = read_verified(folder)
        cfg = yaml.safe_load(data["config.yaml"].decode("utf-8"))
        _check_config(cfg)
        if threads:
            torch.set_num_threads(int(threads))
        model = Vocos(MelSpectrogramFeatures(**cfg["feature_extractor"]["init_args"]),
                      VocosBackbone(**cfg["backbone"]["init_args"]),
                      ISTFTHead(**cfg["head"]["init_args"]))
        state = torch.load(io.BytesIO(data["pytorch_model.bin"]), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model.eval()
        self.torch = torch

    def model_input(self, x, sr):
        """What LavaSR feeds the model: x (mono float32 at sr) to 16 kHz and back to 48 kHz."""
        import torchaudio

        t = self.torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        return torchaudio.functional.resample(torchaudio.functional.resample(t, sr, 16000), 16000, 48000)[0].numpy()

    def __call__(self, segment, length=None):
        """The guess for a stretch of model_input(), `length` samples long (default: the stretch's). The model gives whole
        512-sample frames only, so the stretch is padded with zeros to whole frames plus one; that also covers a length
        a few samples longer than the stretch."""
        torch = self.torch
        w = torch.from_numpy(np.ascontiguousarray(segment, dtype=np.float32))[None]
        w = torch.nn.functional.pad(w, (0, (-w.shape[-1]) % HOP + HOP))
        with torch.no_grad():
            out = _head_forward(self.model.head, self.model.backbone(self.model.feature_extractor(w)))
        out = out[0].numpy().astype(np.float64)
        length = len(segment) if length is None else length
        if len(out) < length:
            raise RuntimeError(f"the model returned {len(out)} samples where {length} were needed")
        return out[:length]
