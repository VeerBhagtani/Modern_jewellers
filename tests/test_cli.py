import json

from arouse.cli import main
from tests.conftest import ROOT, TINY_DATA


def test_model_info(capsys):
    assert main(["model", "info", "--config", str(ROOT / "configs/model.yaml")]) == 0
    assert "110.1M" in capsys.readouterr().out


def test_tokenizer_train_encode_decode(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)  # config corpus paths are relative to the repo root
    out = tmp_path / "tok"
    assert main(["tokenizer", "train", "--config", "configs/tokenizer_tiny.yaml", "--output-dir", str(out)]) == 0
    assert (out / "tokenizer.json").exists()
    capsys.readouterr()

    text = '<|user|>Remind me at 08:00<|end|>'
    assert main(["tokenizer", "encode", "--tokenizer", str(out), "--allow-special", text]) == 0
    ids = json.loads(capsys.readouterr().out)
    assert ids[0] == 9 and ids[-1] == 3

    assert main(["tokenizer", "decode", "--tokenizer", str(out), *map(str, ids)]) == 0
    assert capsys.readouterr().out.rstrip("\n") == text

    assert main(["tokenizer", "stats", "--tokenizer", str(out), str(TINY_DATA / "general.txt")]) == 0
    assert "bytes/token=" in capsys.readouterr().out


def test_model_init_and_export(tmp_path, tiny_tokenizer, capsys):
    import json as _json

    tok_dir = tmp_path / "tok"
    tiny_tokenizer.save(tok_dir)
    src, out = tmp_path / "m", tmp_path / "exported"
    assert main(["model", "init", "--config", str(ROOT / "configs/model_tiny.yaml"), "--tokenizer", str(tok_dir),
                 "--out", str(src)]) == 0
    assert main(["model", "export", "--model", str(src), "--out", str(out), "--notes", "release"]) == 0
    meta = _json.loads((out / "meta.json").read_text())
    assert meta["notes"] == "release" and meta["trained"] is False and (out / "model.pt").exists()
