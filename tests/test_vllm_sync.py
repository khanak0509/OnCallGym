import contextlib

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
vg_mod = pytest.importorskip("trl.generation.vllm_generation")

from training.vllm_sync import make_safe_peft_iter  # noqa: E402


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 8)
        self.q_proj = torch.nn.Linear(8, 8)
        self.lm_head = torch.nn.Linear(8, 10, bias=False)

    def forward(self, x):
        return self.lm_head(self.q_proj(self.embed(x)))


class FakeDist:
    is_fsdp = False

    def gather_params(self, params):
        return contextlib.nullcontext()


def make_vg():
    torch.manual_seed(0)
    cfg = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], init_lora_weights=False)
    model = peft.get_peft_model(Tiny(), cfg)
    vg = object.__new__(vg_mod.VLLMGeneration)
    vg.model, vg._dist = model, FakeDist()
    return vg, model


def test_matches_trl_names_and_merged_values():
    vg, model = make_vg()
    base = model.base_model.model.q_proj.base_layer.weight.detach().clone()
    ref = [(n, p.clone()) for n, p in vg_mod.VLLMGeneration._iter_named_params(vg)]
    got = list(make_safe_peft_iter(vg)())
    assert [n for n, _ in got] == [n for n, _ in ref]
    for (_, a), (_, b) in zip(got, ref, strict=True):
        assert torch.equal(a, b)
    merged = dict(got)["q_proj.weight"]
    assert not torch.allclose(merged, base)
    assert not model.base_model.model.q_proj.merged
    assert torch.allclose(model.base_model.model.q_proj.base_layer.weight, base, atol=1e-6)
    assert not torch.allclose(merged, base)


def test_unmerges_when_consumer_stops_early():
    vg, model = make_vg()
    g = make_safe_peft_iter(vg)()
    next(g)
    assert model.base_model.model.q_proj.merged
    g.close()
    assert not model.base_model.model.q_proj.merged


def test_trl_sync_weights_uses_the_patched_iterator():
    vg, model = make_vg()
    base = model.base_model.model.q_proj.base_layer.weight.detach().clone()
    sent = {}

    class Client:
        def update_named_params(self, metadata, params):
            sent.update({n: t.clone() for n, t in params})

        def reset_prefix_cache(self):
            sent["_reset"] = True

    class Acc:
        is_main_process = True

    vg.mode, vg.enable_sleep_mode, vg.accelerator = "server", False, Acc()
    vg._weight_metadata, vg.vllm_client = None, Client()
    calls = []
    safe = make_safe_peft_iter(vg)

    def spy():
        calls.append(1)
        yield from safe()

    vg._iter_named_params = spy
    vg.sync_weights()
    assert calls and sent.get("_reset")
    assert not torch.allclose(sent["q_proj.weight"], base)
    assert not model.base_model.model.q_proj.merged
