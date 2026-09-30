import json

import pytest

torch = pytest.importorskip("torch")

from arouse.config import ConfigError  # noqa: E402
from arouse.model.config import get_preset  # noqa: E402
from arouse.model.transformer import ArouseTransformer  # noqa: E402
from arouse.training.checkpoint import LATEST, latest_checkpoint  # noqa: E402
from arouse.training.config import TrainingConfig  # noqa: E402
from arouse.training.optim import build_optimizer, lr_at  # noqa: E402
from arouse.training.trainer import Trainer  # noqa: E402
from tests.conftest import ROOT  # noqa: E402


class Crash(Exception):
    pass


@pytest.fixture
def run_cfg(tmp_path, tiny_data_dir):
    model_yaml = tmp_path / "model.yaml"
    get_preset("arouse-tiny").replace(context_length=64).to_yaml(model_yaml)

    def make(**over):
        base = dict(run_name="t", model_config=str(model_yaml), data_dir=str(tiny_data_dir),
                    out_dir=str(tmp_path / "run"), seed=7, batch_size=4, max_steps=12, lr=3e-3, min_lr=3e-4,
                    warmup_steps=2, log_interval=100, eval_interval=100, eval_batches=1,
                    checkpoint_interval=100, keep_checkpoints=2, threads=2)
        return TrainingConfig(**{**base, **over})

    return make


def quiet(_msg: str) -> None:
    pass


# --- schedule / optimizer ----------------------------------------------------------


def test_lr_schedule():
    cfg = TrainingConfig(run_name="x", model_config="m", data_dir="d", out_dir="o",
                         max_steps=100, warmup_steps=10, lr=1.0, min_lr=0.1)
    assert lr_at(0, cfg) == pytest.approx(0.1)
    assert lr_at(9, cfg) == pytest.approx(1.0)  # end of warmup = peak
    assert lr_at(10, cfg) == pytest.approx(1.0)
    assert lr_at(55, cfg) == pytest.approx(0.55)  # cosine midpoint
    assert lr_at(100, cfg) == pytest.approx(0.1) and lr_at(500, cfg) == pytest.approx(0.1)
    after = [lr_at(s, cfg) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(after, after[1:]))


def test_weight_decay_groups():
    m = ArouseTransformer(get_preset("arouse-tiny"))
    cfg = TrainingConfig(run_name="x", model_config="m", data_dir="d", out_dir="o")
    decay, no_decay = build_optimizer(m, cfg).param_groups
    assert all(p.dim() == 2 for p in decay["params"]) and all(p.dim() == 1 for p in no_decay["params"])
    assert sum(p.numel() for g in (decay, no_decay) for p in g["params"]) == m.num_parameters()  # tied once
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0


@pytest.mark.parametrize("bad", [{"batch_size": 0}, {"min_lr": 1.0, "lr": 0.5}, {"warmup_steps": 1000},
                                 {"precision": "int8"}, {"beta2": 1.0}, {"run_name": "a b"}, {"seq_len": 1}])
def test_training_config_validation(bad):
    with pytest.raises(ConfigError):
        TrainingConfig(**{"run_name": "x", "model_config": "m", "data_dir": "d", "out_dir": "o", **bad})


def test_train_tiny_yaml_loads():
    cfg = TrainingConfig.from_yaml(ROOT / "configs/train_tiny.yaml")
    assert cfg.model_config == "configs/model_tiny.yaml"


# --- training loop -------------------------------------------------------------------


def test_training_reduces_loss_and_writes_artifacts(run_cfg):
    cfg = run_cfg(max_steps=30, eval_interval=30, checkpoint_interval=10, keep_checkpoints=2)
    last = Trainer(cfg, log=quiet).train()
    out = cfg.out_dir
    rows = [json.loads(line) for line in open(f"{out}/metrics.jsonl")]
    assert [r["step"] for r in rows] == list(range(1, 31))
    assert rows[-1]["loss"] < rows[0]["loss"] - 1.0
    assert "val/mixture" in last and "val/agent" in last
    ckpts = sorted(p.name for p in __import__("pathlib").Path(out).glob("step_*"))
    assert ckpts == ["step_0000020", "step_0000030"]  # rotation kept the newest 2
    assert open(f"{out}/{LATEST}").read().strip() == "step_0000030"
    info = json.load(open(f"{out}/run_info.json"))
    assert info["num_parameters"] > 0 and info["data_fingerprint"]


def test_resume_after_crash_is_exact(run_cfg, tmp_path):
    """Crash at step 6, resume, finish at 12: identical to an uninterrupted run."""
    straight = run_cfg(out_dir=str(tmp_path / "a"), checkpoint_interval=6)
    Trainer(straight, log=quiet).train()

    crashy = run_cfg(out_dir=str(tmp_path / "b"), checkpoint_interval=6)

    def crash_after_first_save(msg: str) -> None:
        if msg.startswith("saved") and "step_0000006" in msg:
            raise Crash

    with pytest.raises(Crash):
        Trainer(crashy, log=crash_after_first_save).train()
    assert latest_checkpoint(tmp_path / "b").name == "step_0000006"
    Trainer(crashy, log=quiet).train()

    wa = torch.load(tmp_path / "a" / "step_0000012" / "model.pt", weights_only=True)
    wb = torch.load(tmp_path / "b" / "step_0000012" / "model.pt", weights_only=True)
    assert wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)
    la = [json.loads(x)["loss"] for x in open(tmp_path / "a" / "metrics.jsonl")]
    lb = [json.loads(x)["loss"] for x in open(tmp_path / "b" / "metrics.jsonl")]
    assert la == lb


def test_incomplete_checkpoint_is_ignored(run_cfg, tmp_path):
    cfg = run_cfg(max_steps=4, checkpoint_interval=2)
    Trainer(cfg, log=quiet).train()
    out = tmp_path / "run"
    (out / "step_0000099.tmp").mkdir()  # a save that crashed half-way
    assert latest_checkpoint(out).name == "step_0000004"


def test_resume_rejects_different_model(run_cfg, tmp_path):
    cfg = run_cfg(max_steps=3, checkpoint_interval=3)
    Trainer(cfg, log=quiet).train()
    other = tmp_path / "other.yaml"
    get_preset("arouse-tiny").replace(context_length=64, n_layers=2).to_yaml(other)
    with pytest.raises(ConfigError, match="model config"):
        Trainer(cfg.replace(model_config=str(other), max_steps=5), log=quiet).train()


def test_grad_accumulation_and_bf16(run_cfg):
    cfg = run_cfg(max_steps=3, grad_accum_steps=2, precision="bf16")
    last = Trainer(cfg, log=quiet).train()
    assert last["step"] == 3 and last["tokens"] > 4 * 63  # two micro-batches of target tokens
    assert torch.isfinite(torch.tensor(last["loss"]))


def test_seq_len_longer_than_context_rejected(run_cfg):
    with pytest.raises(ConfigError, match="seq_len"):
        Trainer(run_cfg(seq_len=65), log=quiet)


def test_checkpoint_serves_through_inference_engine(run_cfg, tmp_path):
    import arouse

    cfg = run_cfg(max_steps=3, checkpoint_interval=3)
    Trainer(cfg, log=quiet).train()
    engine = arouse.load(str(tmp_path / "run"))  # run dir -> LATEST checkpoint
    info = engine.info()
    assert info["trained"] is True and info["train_steps"] == 3
    assert isinstance(engine.generate("hi", arouse.inference.SamplingParams(max_new_tokens=3)).text, str)
