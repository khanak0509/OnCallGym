from .faults import FAMILIES

TEST_FAMILIES = {
    "jwt_key_rotation_mismatch",
    "orders_config_drift",
    "payments_breaker_stuck_open",
    "cache_full_noeviction",
    "poison_message",
}


def families(split: str = "all") -> list[str]:
    if split == "train":
        return [f for f in FAMILIES if f not in TEST_FAMILIES]
    if split == "test":
        return [f for f in FAMILIES if f in TEST_FAMILIES]
    return list(FAMILIES)


def task_list(split: str = "all", only: list[str] | None = None, variants: int = 3):
    for fam in families(split):
        if only and fam not in only:
            continue
        for v in range(variants):
            yield fam, v
