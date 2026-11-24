import concurrent.futures as cf

from .episode import Episode, tool_schemas
from .loop import AgentLoop

TEMPLATE_KWARGS = {"enable_thinking": False}


class TokenRenderer:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.tools = tool_schemas()

    def prompt_ids(self, messages):
        return self.tok.apply_chat_template(
            messages,
            tools=self.tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            **TEMPLATE_KWARGS,
        )

    def suffix_ids(self, env_messages):
        dummy = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        text = self.tok.apply_chat_template(
            dummy + env_messages,
            tools=self.tools,
            add_generation_prompt=True,
            tokenize=False,
            **TEMPLATE_KWARGS,
        )
        start = text.rfind("<|im_start|>assistant\ny")
        if start < 0:
            start = text.rfind("<|im_start|>assistant")
        cut = text.index("<|im_end|>", start) + len("<|im_end|>")
        return self.tok.encode(text[cut:], add_special_tokens=False)

    def text(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)


def run_episodes(
    specs,
    generate,
    tokenizer,
    max_turn_tokens=768,
    max_model_len=8192,
    start_workers=8,
    verify_workers=8,
):
    r = TokenRenderer(tokenizer)
    eps = [Episode(s["family"], int(s["variant"]), int(s.get("seed", s["variant"]))) for s in specs]
    loops = [AgentLoop(e) for e in eps]
    with cf.ThreadPoolExecutor(start_workers) as ex:
        list(ex.map(lambda lp: lp.start(), loops))
    state = []
    for lp in loops:
        p = r.prompt_ids(lp.messages)
        state.append(
            {"prompt_ids": p, "seq": list(p), "completion_ids": [], "logprobs": [], "env_mask": []}
        )
    try:
        while True:
            active = [i for i, lp in enumerate(loops) if not lp.done]
            if not active:
                break
            for i in list(active):
                if len(state[i]["seq"]) + max_turn_tokens > max_model_len:
                    loops[i].ep._end("context_exhausted")
                    active.remove(i)
            if not active:
                break
            outs = generate([state[i]["seq"] for i in active], max_turn_tokens)
            for i, (ids, lps) in zip(active, outs, strict=True):
                st = state[i]
                st["seq"] += ids
                st["completion_ids"] += ids
                st["logprobs"] += list(lps)
                st["env_mask"] += [1] * len(ids)
                env_msgs = loops[i].step(r.text(ids))
                if env_msgs and not loops[i].done:
                    suffix = r.suffix_ids(env_msgs)
                    st["seq"] += suffix
                    st["completion_ids"] += suffix
                    st["logprobs"] += [0.0] * len(suffix)
                    st["env_mask"] += [0] * len(suffix)
        with cf.ThreadPoolExecutor(verify_workers) as ex:
            breakdowns = list(ex.map(lambda e: e.finish(), eps))
    finally:
        with cf.ThreadPoolExecutor(start_workers) as ex:
            list(ex.map(lambda e: e.close(), eps))
    results = []
    for e, st, b in zip(eps, state, breakdowns, strict=True):
        results.append(
            {
                **{k: st[k] for k in ("prompt_ids", "completion_ids", "logprobs", "env_mask")},
                "breakdown": b.to_dict(),
                "trajectory": e.trajectory,
                "variant": e.fault.variant_id,
            }
        )
    return results
