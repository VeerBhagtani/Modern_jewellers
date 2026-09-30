"""Tiny tokenizer corpus: provenance integrity, reproducibility, grammar + safety invariants."""

import hashlib
import importlib.util
import json

from arouse.tokenizer import Special
from arouse.tokenizer.special_tokens import split_on_special
from arouse.tokenizer.trainer import iter_documents
from tests.conftest import ROOT, TINY_DATA

JSON_BODY_AFTER = {Special.CONTEXT, Special.TOOLS, Special.TOOL_RESULT, Special.TOOL_ERROR,
                   Special.TOOL_CALL, Special.ASK_USER, Special.FINISH, Special.FAIL}


def _manifest():
    return json.loads((TINY_DATA / "MANIFEST.json").read_text())


def test_manifest_covers_every_file_with_correct_hash():
    m = _manifest()
    listed = {f["file"] for f in m["files"]}
    on_disk = {p.name for p in TINY_DATA.iterdir() if p.name != "MANIFEST.json"}
    assert listed == on_disk
    for f in m["files"]:
        assert hashlib.sha256((TINY_DATA / f["file"]).read_bytes()).hexdigest() == f["sha256"], f["file"]
        assert f["origin"] and f["license"]
    assert m["external_sources"] == []


def test_generator_is_reproducible(tmp_path):
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "build_tiny_tokenizer_corpus.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    for name in gen.HAND_WRITTEN:  # generator hashes hand-written files in place
        (tmp_path / name).write_bytes((TINY_DATA / name).read_bytes())
    gen.build(tmp_path, seed=1234)
    for name in ["scheduling.jsonl", "agent_trajectories.jsonl", "MANIFEST.json"]:
        assert (tmp_path / name).read_bytes() == (TINY_DATA / name).read_bytes(), name


def _trajectories():
    return list(iter_documents(TINY_DATA / "agent_trajectories.jsonl"))


def _segments(text):
    """[(special_token, body_text)] for a rendered trajectory."""
    ids = {t.text: t for t in Special}
    out, cur = [], None
    for piece, is_special in split_on_special(text):
        if is_special:
            cur = ids[piece]
            out.append([cur, ""])
        else:
            assert out, "text before first special token"
            out[-1][1] += piece
    return [tuple(x) for x in out]


def test_trajectory_grammar():
    trajs = _trajectories()
    assert len(trajs) == 300
    for t in trajs:
        segs = _segments(t)
        assert segs[0][0] == Special.BOS and segs[-1][0] == Special.EOS
        for tok, body in segs:
            if tok in JSON_BODY_AFTER:
                json.loads(body)
        # every arouse turn ends with exactly one action
        for i, (tok, _) in enumerate(segs):
            if tok == Special.AROUSE:
                j = i + 1
                while segs[j][0] in (Special.PLAN, Special.VERIFY):
                    j += 1
                assert segs[j][0] in (Special.TOOL_CALL, Special.ASK_USER, Special.FINISH, Special.FAIL)
                assert segs[j + 1][0] == Special.END


def test_no_false_completion_in_data():
    """<|finish|> may only follow a successful tool result, never an error."""
    finishes = 0
    for t in _trajectories():
        last_obs = None
        for tok, _ in _segments(t):
            if tok in (Special.TOOL_RESULT, Special.TOOL_ERROR):
                last_obs = tok
            if tok == Special.FINISH:
                finishes += 1
                assert last_obs == Special.TOOL_RESULT
    assert finishes > 0


def test_ambiguous_requests_ask_instead_of_guessing():
    asked = 0
    for t in _trajectories():
        segs = _segments(t)
        user = next(body for tok, body in segs if tok == Special.USER)
        if user.startswith("Remind me") and " at " not in user and "hour" not in user:
            asked += 1
            assert Special.ASK_USER in [tok for tok, _ in segs]
            assert Special.TOOL_CALL not in [tok for tok, _ in segs]
    assert asked > 0


def test_scheduled_dates_are_correct():
    """Every scheduler.create call is in the future and matches its repeat rule / request."""
    from datetime import datetime, timedelta

    checked = 0
    for t in _trajectories():
        segs = _segments(t)
        now = datetime.fromisoformat(json.loads(next(b for k, b in segs if k == Special.CONTEXT))["now"])
        user = next(b for k, b in segs if k == Special.USER)
        for tok, body in segs:
            call = json.loads(body) if tok == Special.TOOL_CALL else None
            if not call or call["tool"] != "scheduler.create":
                continue
            a = call["arguments"]
            when = datetime.fromisoformat(f"{a['date']}T{a['time']}")
            assert when > now, user
            rep = a.get("repeat")
            if rep and rep["freq"] == "weekly":
                assert ["MO", "TU", "WE", "TH", "FR", "SA", "SU"][when.weekday()] == rep["by_day"][0]
                assert when - now <= timedelta(days=7)
            if rep and rep["freq"] == "monthly":
                assert when.day == 1
            if "tomorrow" in user:
                assert when.date() == (now + timedelta(days=1)).date()
            if user.startswith("In "):
                assert when - now == timedelta(hours=int(user.split()[1]))
            checked += 1
    assert checked > 100
