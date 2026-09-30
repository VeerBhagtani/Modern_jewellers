"""Browser test of the chat UI. Skipped unless Playwright + a Chromium binary are available."""

import os
import shutil
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sync_api = pytest.importorskip("playwright.sync_api")

from arouse.api.server import make_server  # noqa: E402
from arouse.inference import InferenceEngine  # noqa: E402
from arouse.model.config import get_preset  # noqa: E402
from arouse.model.transformer import ArouseTransformer  # noqa: E402


def _chromium() -> str | None:
    for c in (os.environ.get("AROUSE_CHROMIUM"), "/opt/pw-browsers/chromium", shutil.which("chromium")):
        if c and Path(c).exists():
            return c
    return None


@pytest.fixture(scope="module")
def page(tiny_tokenizer):
    exe = _chromium()
    if exe is None:
        pytest.skip("no Chromium binary")
    torch.manual_seed(0)
    srv = make_server(InferenceEngine(ArouseTransformer(get_preset("arouse-tiny")), tiny_tokenizer), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe)
        pg = browser.new_page(viewport={"width": 390, "height": 740})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.goto(f"http://127.0.0.1:{srv.server_port}/")
        yield pg, errors
        browser.close()
    srv.shutdown()
    srv.server_close()


def test_status_shows_untrained(page):
    pg, _ = page
    pg.wait_for_selector("#status.warn")
    assert "UNTRAINED" in pg.inner_text("#status")
    assert "random weights" in pg.inner_text("#note")


def test_send_and_stream_reply(page):
    pg, errors = page
    pg.click("#settingsBtn")
    pg.fill("#maxTokens", "12")
    pg.click("#settingsBtn")
    pg.fill("#input", "Remind me <|finish|> tomorrow")
    pg.keyboard.press("Enter")
    pg.wait_for_selector(".msg.assistant .meta", timeout=30000)
    user = pg.locator(".msg.user").last
    assert user.inner_text() == "Remind me <|finish|> tomorrow"  # literal, not a token chip
    assert user.locator(".tok").count() == 0
    assert "tokens" in pg.locator(".msg.assistant .meta").last.inner_text()
    assert not pg.is_disabled("#input")
    assert errors == []


def test_no_horizontal_scroll_on_phone(page):
    pg, _ = page
    assert pg.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")


def test_new_chat_clears(page):
    pg, _ = page
    pg.click("#newBtn")
    assert pg.locator(".msg").count() == 0
