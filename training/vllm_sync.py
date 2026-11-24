import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_safe_peft_iter(vg):
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
