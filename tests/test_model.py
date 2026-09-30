import math

import pytest

torch = pytest.importorskip("torch")

from arouse.model.config import ModelConfig, get_preset  # noqa: E402
from arouse.model.embeddings import RotaryEmbedding  # noqa: E402
from arouse.model.io import ModelLoadError, load_pretrained, save_pretrained  # noqa: E402
from arouse.model.loss import IGNORE_INDEX, lm_loss  # noqa: E402
from arouse.model.norm import RMSNorm  # noqa: E402
from arouse.model.transformer import ArouseTransformer  # noqa: E402

TINY = get_preset("arouse-tiny")


def make(cfg=TINY, seed=0, impl="sdpa"):
    torch.manual_seed(seed)
    return ArouseTransformer(cfg, attn_impl=impl)


def rand_ids(cfg=TINY, b=2, t=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (b, t), generator=g)


@pytest.mark.parametrize(
    "cfg",
    [
        TINY,
        TINY.replace(tie_embeddings=False),
        TINY.replace(bias=True),
        TINY.replace(n_kv_heads=None),
        TINY.replace(n_kv_heads=1, ffn_hidden_size=200),
        ModelConfig(),  # the real v0.1: 110,119,680
    ],
)
def test_param_count_matches_config(cfg):
    assert ArouseTransformer(cfg).num_parameters() == cfg.num_parameters()


def test_forward_shapes_and_initial_loss():
    m = make()
    ids = rand_ids()
    out = m(ids, targets=rand_ids(seed=2))
    assert out.logits.shape == (2, 32, TINY.vocab_size)
    # Random init ~ uniform prediction: loss close to ln(vocab)
    assert abs(out.loss.item() - math.log(TINY.vocab_size)) < 0.3
    assert m(ids).loss is None


@pytest.mark.parametrize("impl", ["sdpa", "manual"])
def test_causality(impl):
    m = make(impl=impl).eval()
    ids = rand_ids()
    changed = ids.clone()
    changed[:, 20:] = (changed[:, 20:] + 7) % TINY.vocab_size
    a, b = m(ids).logits, m(changed).logits
    torch.testing.assert_close(a[:, :20], b[:, :20])
    assert not torch.allclose(a[:, 20:], b[:, 20:])


def test_manual_and_sdpa_attention_agree():
    ids = rand_ids()
    sd = make(impl="sdpa").eval()
    man = make(impl="manual").eval()
    man.load_state_dict(sd.state_dict())
    torch.testing.assert_close(sd(ids).logits, man(ids).logits, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("impl", ["sdpa", "manual"])
def test_kv_cache_matches_full_forward(impl):
    m = make(impl=impl).eval()
    ids = rand_ids(b=1, t=24)
    full = m(ids).logits
    cache = m.new_kv_cache()
    parts = [m(ids[:, :10], start_pos=0, kv_cache=cache).logits]  # prefill
    for t in range(10, 24):  # then one token at a time
        parts.append(m(ids[:, t : t + 1], start_pos=t, kv_cache=cache).logits)
    torch.testing.assert_close(torch.cat(parts, dim=1), full, atol=1e-5, rtol=1e-4)


def test_kv_cache_position_mismatch_rejected():
    m = make().eval()
    cache = m.new_kv_cache()
    m(rand_ids(b=1, t=5), kv_cache=cache)
    with pytest.raises(ValueError, match="kv_cache"):
        m(rand_ids(b=1, t=1), start_pos=3, kv_cache=cache)


def test_context_overflow_rejected():
    with pytest.raises(ValueError, match="context_length"):
        make()(rand_ids(t=TINY.context_length + 1))


def test_rope_is_relative_and_norm_preserving():
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64, theta=10000.0)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def rot(x, pos):
        return rope(x, start_pos=pos)

    s1 = (rot(q, 10) * rot(k, 7)).sum()
    s2 = (rot(q, 33) * rot(k, 30)).sum()  # same distance 3
    torch.testing.assert_close(s1, s2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rot(q, 17).norm(), q.norm())
    torch.testing.assert_close(rot(q, 0), q)  # position 0 = identity


def test_rmsnorm_unit_rms():
    x = torch.randn(4, 8, 32) * 5 + 3
    y = RMSNorm(32, 1e-6)(x)
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(4, 8), atol=1e-4, rtol=1e-4)


def test_tied_and_untied_heads():
    tied = make()
    assert tied.lm_head.weight is tied.embed.weight
    untied = make(TINY.replace(tie_embeddings=False))
    assert untied.lm_head.weight is not untied.embed.weight


def test_residual_projection_init_is_scaled():
    m = make(ModelConfig(d_model=256, n_heads=4, n_layers=8, vocab_size=1024))
    std_q = m.layers[0].attn.wq.weight.std().item()
    std_o = m.layers[0].attn.wo.weight.std().item()
    assert std_o == pytest.approx(std_q / math.sqrt(16), rel=0.1)


def test_loss_masking():
    logits = torch.randn(2, 6, 50)
    targets = torch.randint(0, 50, (2, 6))
    masked = targets.clone()
    masked[:, :3] = IGNORE_INDEX
    expected = torch.nn.functional.cross_entropy(logits[:, 3:].reshape(-1, 50), targets[:, 3:].reshape(-1))
    torch.testing.assert_close(lm_loss(logits, masked), expected)
    assert lm_loss(logits, torch.full_like(targets, IGNORE_INDEX)).item() == 0.0


def test_dropout_only_in_training():
    m = make(TINY.replace(dropout=0.3))
    ids = rand_ids()
    m.train()
    assert not torch.allclose(m(ids).logits, m(ids).logits)
    m.eval()
    torch.testing.assert_close(m(ids).logits, m(ids).logits)


def test_bf16_autocast_forward():
    m = make()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(rand_ids(), targets=rand_ids(seed=3))
    assert torch.isfinite(out.loss)


def test_gradients_reach_every_parameter():
    m = make(TINY.replace(bias=True, tie_embeddings=False))
    m(rand_ids(), targets=rand_ids(seed=4)).loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_can_overfit_one_batch():
    """End-to-end learning signal: tiny model memorises one batch."""
    m = make()
    ids = rand_ids(b=2, t=33, seed=5)
    x, y = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(80):
        loss = m(x, targets=y).loss
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert first > 7.0 and loss.item() < 0.5


def test_save_load_pretrained(tmp_path, tiny_tokenizer):
    m = make().eval()
    save_pretrained(tmp_path, m, tiny_tokenizer, trained=False, notes="test")
    loaded, tok, meta = load_pretrained(tmp_path)
    ids = rand_ids()
    torch.testing.assert_close(loaded(ids).logits, m(ids).logits)
    assert tok.fingerprint() == tiny_tokenizer.fingerprint()
    assert meta["trained"] is False and meta["num_parameters"] == TINY.num_parameters()


def test_load_rejects_wrong_tokenizer(tmp_path, tiny_tokenizer):
    import json

    save_pretrained(tmp_path, make(), tiny_tokenizer, trained=False)
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["tokenizer_fingerprint"] = "0" * 16
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ModelLoadError, match="fingerprint"):
        load_pretrained(tmp_path)
    with pytest.raises(ModelLoadError, match="meta.json"):
        load_pretrained(tmp_path / "nope")


def test_vocab_smaller_than_tokenizer_rejected(tmp_path, tiny_tokenizer):
    small = make(TINY.replace(vocab_size=400))
    with pytest.raises(ModelLoadError, match="vocab_size"):
        save_pretrained(tmp_path, small, tiny_tokenizer, trained=False)
