import pytest
import yaml

from synchain_bwe import model

CONFIG = """
feature_extractor:
  class_path: vocos.feature_extractors.MelSpectrogramFeatures
  init_args: {sample_rate: 44100, n_fft: 2048, hop_length: 512, n_mels: 80, padding: same, f_min: 0, f_max: 8000, norm: slaney, mel_scale: slaney}
backbone:
  class_path: vocos.models.VocosBackbone
  init_args: {input_channels: 80, dim: 512, intermediate_dim: 1536, num_layers: 8}
head:
  class_path: vocos.heads.ISTFTHead
  init_args: {dim: 512, n_fft: 2048, hop_length: 512, padding: same}
"""


def cfg():
    return yaml.safe_load(CONFIG)


def test_expected_config_passes():
    model._check_config(cfg())


@pytest.mark.parametrize("change", [
    lambda c: c.update({"extra": {"class_path": "os.system"}}),
    lambda c: c["head"].update({"class_path": "builtins.eval"}),
    lambda c: c["backbone"]["init_args"].update({"layer_scale_init_value": [1, 2]}),
    lambda c: c["backbone"]["init_args"].update({"num_layers": True}),
    lambda c: c["feature_extractor"].update({"dict_kwargs": {}}),
])
def test_unexpected_config_is_refused(change):
    c = cfg()
    change(c)
    with pytest.raises(ValueError):
        model._check_config(c)


def test_wrong_files_are_refused(tmp_path):
    (tmp_path / "config.yaml").write_text(CONFIG)
    (tmp_path / "pytorch_model.bin").write_bytes(b"not the weights")
    with pytest.raises(ValueError, match="SHA-256"):
        model.verify(tmp_path)
    with pytest.raises(FileNotFoundError):
        model.verify(tmp_path / "nowhere")
    with pytest.raises(ValueError, match="SHA-256"):
        model.read_verified(tmp_path)
    with pytest.raises(FileNotFoundError):
        model.read_verified(tmp_path / "nowhere")
