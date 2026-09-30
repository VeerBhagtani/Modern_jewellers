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
    model = ArouseTransformer(get_preset("arouse-tiny").replace(context_length=1024))
    srv = make_server(InferenceEngine(model, tiny_tokenizer), port=0)
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


def test_send_shows_final_answer_and_steps(page):
    pg, errors = page
    pg.fill("#input", "Remind me <|finish|> tomorrow")
    pg.keyboard.press("Enter")
    pg.wait_for_selector(".msg.assistant details.steps", timeout=60000)
    user = pg.locator(".msg.user").last
    assert user.inner_text() == "Remind me <|finish|> tomorrow"  # literal text, never a control token
    assert "step" in pg.locator(".msg.assistant details.steps summary").last.inner_text()
    assert not pg.is_disabled("#sendBtn")
    assert errors == []


def test_suggestion_chip_sends(page):
    pg, _ = page
    pg.click("#newBtn")
    chips = pg.locator(".sg")
    assert chips.count() >= 4
    n_before = pg.locator(".msg.user").count()
    chips.first.click()
    pg.wait_for_function("document.querySelectorAll('.msg.assistant details').length >= 1", timeout=60000)
    assert pg.locator(".msg.user").count() == n_before + 1


def test_no_horizontal_scroll_on_phone(page):
    pg, _ = page
    assert pg.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")


def test_new_chat_clears(page):
    pg, _ = page
    pg.click("#newBtn")
    assert pg.locator(".msg").count() == 0 and pg.locator(".sg").count() >= 4
