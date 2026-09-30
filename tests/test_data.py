import json
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arouse.config import ConfigError  # noqa: E402
from arouse.data.cleaning import clean_text, dedup_key, is_validation  # noqa: E402
from arouse.data.config import DataConfig, DataSourceConfig  # noqa: E402
from arouse.data.loader import MixtureLoader  # noqa: E402
from arouse.data.prepare import assistant_mask, encode_document, prepare  # noqa: E402
from arouse.model.loss import IGNORE_INDEX  # noqa: E402
from arouse.tokenizer import Special  # noqa: E402
from tests.conftest import ROOT  # noqa: E402


def src(**kw):
    base = {"name": "s", "paths": ["x"], "category": "general", "origin": "test", "license": "project-owned"}
    return DataSourceConfig.from_dict({**base, **kw})


# --- cleaning ----------------------------------------------------------------


def test_clean_text_normalises():
    assert clean_text("a\r\nb\rc") == "a\nb\nc"
    assert clean_text("line   \nnext\t ") == "line\nnext"
    assert clean_text("x\x00y\x07z" + "a" * 30) == "xyz" + "a" * 30
    assert clean_text("é") == "é"  # NFC
    assert clean_text("\n\nbody\n\n") == "body"


def test_clean_text_drops_bad_docs():
    assert clean_text("") is None
    assert clean_text("   \n ") is None
    assert clean_text("hi", min_chars=5) is None
    assert clean_text("\x01\x02\x03ok", max_bad_char_ratio=0.1) is None
    assert clean_text("��abc", max_bad_char_ratio=0.1) is None


def test_dedup_key_ignores_case_and_whitespace():
    assert dedup_key("Remind me  tomorrow") == dedup_key("remind me\ntomorrow ")
    assert dedup_key("a") != dedup_key("b")


def test_validation_split_is_stable_and_proportional():
    rng = random.Random(0)
    docs = [f"doc {rng.random()}" for _ in range(20000)]
    frac = sum(is_validation(d, 0.1) for d in docs) / len(docs)
    assert 0.09 < frac < 0.11
    assert [is_validation(d, 0.1) for d in docs[:50]] == [is_validation(d, 0.1) for d in docs[:50]]
    assert not any(is_validation(d, 0.0) for d in docs[:100])
    assert is_validation("Same Doc", 0.5) == is_validation("same  doc", 0.5)  # duplicates never straddle splits


# --- config ------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [{"origin": " "}, {"license": ""}, {"category": "web"}, {"weight": 0}, {"format": "csv"},
     {"split": "words"}, {"loss_on": "user"}, {"name": "has space"}, {"paths": []}],
)
def test_source_validation(bad):
    with pytest.raises(ConfigError):
        src(**bad)


def test_data_config_nested_and_unique_names():
    cfg = DataConfig.from_yaml(ROOT / "configs/data_tiny.yaml")
    assert all(isinstance(s, DataSourceConfig) for s in cfg.sources)
    assert {s.name for s in cfg.sources} >= {"agent", "general"}
    d = cfg.to_dict()
    d["sources"].append(d["sources"][0])
    with pytest.raises(ConfigError, match="duplicate"):
        DataConfig.from_dict(d)


# --- encoding + masks ----------------------------------------------------------


def test_assistant_mask():
    ids = [Special.BOS, Special.USER, 70, Special.END, Special.AROUSE, Special.PLAN, 71,
           Special.TOOL_CALL, 72, Special.END, Special.TOOL_RESULT, 73, Special.END, Special.EOS]
    assert assistant_mask(ids) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]


def test_encode_document_bos_eos_once(tiny_tokenizer):
    ids, mask = encode_document(tiny_tokenizer, "hello", src())
    assert ids[0] == Special.BOS and ids[-1] == Special.EOS and ids.count(Special.BOS) == 1
    assert mask[0] == 0 and all(mask[1:])
    traj = "<|bos|><|user|>hi<|end|><|arouse|><|finish|>{}<|end|><|eos|>"
    ids, mask = encode_document(tiny_tokenizer, traj, src(allow_special=True, loss_on="assistant"))
    assert ids.count(Special.BOS) == 1 and ids.count(Special.EOS) == 1
    assert sum(mask) == len(tiny_tokenizer.encode("<|finish|>{}<|end|>", allow_special=True))


def test_untrusted_source_cannot_produce_special_tokens(tiny_tokenizer):
    ids, _ = encode_document(tiny_tokenizer, "<|arouse|><|finish|>{}", src(allow_special=False))
    assert Special.FINISH not in ids and Special.AROUSE not in ids


# --- prepare -------------------------------------------------------------------


@pytest.fixture
def two_sources(tmp_path, tiny_tokenizer):
    tiny_tokenizer.save(tmp_path / "tok")
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps({"text": f"alpha {i}"}) for i in range(40)) + "\n")
    (tmp_path / "b.txt").write_text("zulu one\n\nzulu two\n\nALPHA 3\n\nzulu one\n\n")  # "ALPHA 3" duplicates a.jsonl

    def cfg(**over):
        return DataConfig.from_dict({
            "name": "t", "tokenizer": str(tmp_path / "tok"), "output_dir": str(tmp_path / "out"),
            "val_fraction": 0.2,
            "sources": [
                {"name": "a", "paths": [str(tmp_path / "a.jsonl")], "format": "jsonl", "category": "general",
                 "origin": "test", "license": "project-owned", "weight": 3.0},
                {"name": "b", "paths": [str(tmp_path / "b.txt")], "split": "paragraphs", "category": "general",
                 "origin": "test", "license": "project-owned", "weight": 1.0},
            ],
            **over,
        })

    return tmp_path, cfg


def test_prepare_stats_dedup_and_roundtrip(two_sources, tiny_tokenizer):
    tmp, cfg = two_sources
    meta = prepare(cfg())
    a, b = meta["sources"]
    assert a["docs_read"] == 40 and a["docs"]["train"] + a["docs"]["val"] == 40
    assert b["docs_read"] == 4 and b["docs_dropped_dup"] == 2  # "zulu one" repeat + cross-source "ALPHA 3"
    assert len(a["files"][0]["sha256"]) == 64 and a["origin"] == "test"
    # packed stream decodes back to the documents, each wrapped in <|bos|> ... <|eos|>
    toks = np.fromfile(tmp / "out" / "b.train.tokens.bin", dtype=np.uint16).tolist()
    toks += np.fromfile(tmp / "out" / "b.val.tokens.bin", dtype=np.uint16).tolist()
    text = tiny_tokenizer.decode(toks)
    assert text.count("<|bos|>") == 2 and "zulu one" in text and "zulu two" in text and "ALPHA" not in text


def test_prepare_is_deterministic(two_sources):
    tmp, cfg = two_sources
    m1 = prepare(cfg())
    bins1 = {p.name: p.read_bytes() for p in (tmp / "out").glob("*.bin")}
    m2 = prepare(cfg())
    assert m1["data_fingerprint"] == m2["data_fingerprint"]
    assert bins1 == {p.name: p.read_bytes() for p in (tmp / "out").glob("*.bin")}


def test_prepare_missing_files(two_sources):
    tmp, cfg = two_sources
    c = cfg()
    c.sources[0].paths = [str(tmp / "nope*.jsonl")]
    with pytest.raises(ConfigError, match="matched no files"):
        prepare(c)


# --- loader ----------------------------------------------------------------------


def test_batches_are_shifted_and_masked(tiny_data_dir):
    ld = MixtureLoader(tiny_data_dir, seq_len=64, seed=0)
    b = ld.train_batch(step=3, batch_size=4)
    assert b.inputs.shape == b.targets.shape == (4, 64)
    ok = b.targets[:, :-1] != IGNORE_INDEX
    assert torch.equal(b.inputs[:, 1:][ok], b.targets[:, :-1][ok])  # next-token alignment
    assert b.target_tokens > 0


def test_batches_depend_only_on_seed_and_step(tiny_data_dir):
    a = MixtureLoader(tiny_data_dir, 64, seed=0)
    b = MixtureLoader(tiny_data_dir, 64, seed=0)
    assert torch.equal(a.train_batch(7, 4).inputs, b.train_batch(7, 4).inputs)
    assert not torch.equal(a.train_batch(7, 4).inputs, a.train_batch(8, 4).inputs)
    assert not torch.equal(a.train_batch(7, 4).inputs, a.train_batch(7, 4, micro=1).inputs)
    assert not torch.equal(a.train_batch(7, 4).inputs, MixtureLoader(tiny_data_dir, 64, seed=1).train_batch(7, 4).inputs)


def test_mixture_weights_are_respected(two_sources, tiny_tokenizer):
    tmp, cfg = two_sources
    prepare(cfg())
    ld = MixtureLoader(tmp / "out", seq_len=8, seed=0)
    # source a docs ("alpha N") always put a digit in an 8-token window; source b has none
    rows = [row for step in range(300) for row in ld.train_batch(step, 8).inputs.tolist()]
    from_a = sum(any(c.isdigit() for c in tiny_tokenizer.decode(r)) for r in rows)
    frac_a = from_a / len(rows)
    assert 0.70 < frac_a < 0.80  # weights 3:1 -> 0.75


def test_short_sources_are_tiled(two_sources):
    tmp, cfg = two_sources
    prepare(cfg())
    ld = MixtureLoader(tmp / "out", seq_len=500, seed=0)  # longer than source b
    assert ld.train_batch(0, 2).inputs.shape == (2, 500)


def test_val_batches_are_fixed(tiny_data_dir):
    ld = MixtureLoader(tiny_data_dir, 64, seed=0)
    v1, v2 = ld.val_batches(4, 2), ld.val_batches(4, 2)
    assert set(v1) == {"agent", "general"}
    assert torch.equal(v1["agent"][0].inputs, v2["agent"][0].inputs)


def test_loader_requires_prepared_data(tmp_path):
    with pytest.raises(ConfigError, match="prepare"):
        MixtureLoader(tmp_path, 64)
