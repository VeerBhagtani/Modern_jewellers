import pytest

from arouse.config import ConfigError
from arouse.model.config import PRESETS, ModelConfig, get_preset
from arouse.tokenizer import TokenizerTrainingConfig
from tests.conftest import ROOT


def test_default_is_v01_preset_about_100m():
    cfg = ModelConfig()
    assert cfg == get_preset("arouse-v0.1")
    assert (cfg.n_layers, cfg.d_model, cfg.head_dim, cfg.ffn_size) == (12, 768, 64, 2048)
    # 32768*768 embedding + 12 * (4*768^2 attn + 3*768*2048 mlp + 2*768 norms) + 768 final norm
    assert cfg.num_parameters() == 110_119_680
    assert 90e6 < cfg.num_parameters() < 130e6


@pytest.mark.parametrize("path,preset", [("configs/model.yaml", "arouse-v0.1"), ("configs/model_tiny.yaml", "arouse-tiny")])
def test_yaml_files_match_presets(path, preset):
    assert ModelConfig.from_yaml(ROOT / path) == PRESETS[preset]


def test_yaml_roundtrip(tmp_path):
    cfg = get_preset("arouse-tiny")
    cfg.to_yaml(tmp_path / "m.yaml")
    assert ModelConfig.from_yaml(tmp_path / "m.yaml") == cfg


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="unknown keys"):
        ModelConfig.from_dict({"n_layer": 12})


@pytest.mark.parametrize("key,value", [("n_layers", "12"), ("n_layers", True), ("d_model", 768.0), ("tie_embeddings", 1)])
def test_wrong_types_rejected(key, value):
    with pytest.raises(ConfigError, match=key):
        ModelConfig.from_dict({key: value})


def test_int_coerced_for_float_fields():
    cfg = ModelConfig.from_dict({"rope_theta": 500000})
    assert isinstance(cfg.rope_theta, float) and cfg.rope_theta == 500000.0


@pytest.mark.parametrize(
    "changes",
    [
        {"d_model": 770},  # not divisible by 12 heads
        {"d_model": 36, "n_heads": 12},  # head_dim 3: odd, RoPE needs even
        {"n_kv_heads": 5},  # 12 % 5 != 0
        {"n_kv_heads": 0},
        {"dropout": 1.0},
        {"vocab_size": 100},  # below 64 special + 256 bytes
        {"n_layers": 0},
        {"ffn_hidden_size": -1},
        {"norm_eps": 0.0},
    ],
)
def test_invalid_configs_rejected(changes):
    with pytest.raises(ConfigError):
        ModelConfig(**changes)


def test_replace_revalidates():
    with pytest.raises(ConfigError):
        ModelConfig().replace(n_heads=7)


def test_ffn_auto_and_override():
    assert ModelConfig(d_model=1024, n_heads=16).ffn_size == 2816  # 8/3*1024=2730 -> 2816
    assert ModelConfig(ffn_hidden_size=3000).ffn_size == 3000


def test_resolved_is_explicit_and_equivalent():
    cfg = ModelConfig()
    r = cfg.resolved()
    assert r.n_kv_heads == 12 and r.ffn_hidden_size == 2048
    assert r.num_parameters() == cfg.num_parameters()
    # resolved values survive a change to d_model (auto values would not)
    assert r.replace(d_model=1536).ffn_size == 2048


def test_gqa_and_untied_param_accounting():
    base = ModelConfig()
    gqa = ModelConfig(n_kv_heads=4)
    assert base.num_parameters() - gqa.num_parameters() == 12 * 2 * 768 * (768 - 256)
    untied = ModelConfig(tie_embeddings=False)
    assert untied.num_parameters() - base.num_parameters() == 32768 * 768


def test_bias_param_accounting():
    d, f = 768, 2048
    assert ModelConfig(bias=True).num_parameters() - ModelConfig().num_parameters() == 12 * (4 * d + 2 * f + d)


def test_scaling_is_config_only():
    big = ModelConfig(name="arouse-350m", d_model=1024, n_layers=24, n_heads=16, n_kv_heads=8)
    assert 300e6 < big.num_parameters() < 400e6


def test_unknown_preset():
    with pytest.raises(ConfigError):
        get_preset("arouse-500b")


def test_summary_mentions_size():
    assert "110.1M" in ModelConfig().summary()


@pytest.mark.parametrize("path", ["configs/tokenizer.yaml", "configs/tokenizer_tiny.yaml"])
def test_tokenizer_configs_load(path):
    cfg = TokenizerTrainingConfig.from_yaml(ROOT / path)
    assert cfg.vocab_size >= 320


def test_tokenizer_vocab_matches_model_vocab():
    assert TokenizerTrainingConfig.from_yaml(ROOT / "configs/tokenizer.yaml").vocab_size == ModelConfig().vocab_size
    assert (
        TokenizerTrainingConfig.from_yaml(ROOT / "configs/tokenizer_tiny.yaml").vocab_size
        == get_preset("arouse-tiny").vocab_size
    )


@pytest.mark.parametrize("changes", [{"vocab_size": 300}, {"min_frequency": 0}, {"max_bytes": 0}])
def test_invalid_tokenizer_training_config(changes):
    with pytest.raises(ConfigError):
        TokenizerTrainingConfig(**changes)
