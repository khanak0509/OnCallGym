"""The safe LoRA sync yields the same names as TRL's iterator, merged values that the unmerge
can't touch, and leaves the model unmerged. Needs torch, peft and trl (the training venv)."""

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
    assert not torch.allclose(merged, base)  # the LoRA delta is in what vLLM gets
    # model is back to unmerged, and the yielded tensor was not changed by the unmerge
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
