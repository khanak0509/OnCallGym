"""Race-free LoRA weight sync to a vLLM server for TRL 1.18 + vLLM >= 0.28.

TRL's PEFT path is a generator that calls merge_adapter(), yields views of the live parameters
and calls unmerge_adapter() after the last one. vLLM's packed_nccl_broadcast_producer pulls that
generator inside `torch.cuda.stream(side_stream)` blocks, alternating between two side streams,
and torch.cat's the views there. Nothing orders the merge (queued on one stream) or the unmerge
(queued on another) against the copies, so on a T4 some layers reach vLLM as base weights. We
measured vLLM sampling from roughly "LoRA at half strength" after the second sync, with a mean
trainer-vs-vLLM logprob gap of 0.8-1.1 nats per token instead of ~0.15.

This replacement synchronizes the device after merging, yields clones (so the unmerge can't
change what is still being copied) and synchronizes again before unmerging.
"""

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_safe_peft_iter(vg):
    """vg: trainer.vllm_generation (trl.generation.vllm_generation.VLLMGeneration)."""
    model = vg.model
    if getattr(vg._dist, "is_fsdp", False):
        raise NotImplementedError("safe sync only covers the single-process PEFT path")

    def it():
        model.merge_adapter()
        _sync()
        try:
            for name, param in model.named_parameters():
                name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                if model.prefix in name:
                    continue
                if "original_module" in name or (
                    "modules_to_save." in name and "modules_to_save.default." not in name
                ):
                    continue
                name = vg._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])
                yield name, param.data.clone()
            _sync()
        finally:
            _sync()
            model.unmerge_adapter()
            _sync()

    return it
