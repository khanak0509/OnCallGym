"""Known cheat attempts, used by validation (spec 7.4 step 6) and the hack regression tests.

Each cheat is fn(stack, fault) -> bool (False if it found nothing to tamper with, e.g. no
unfinished orders to delete) and only does things an agent with a shell in the episode
directory could do. Every cheat that did something must end with integrity gate G = 0.
"""

import os
import sqlite3

from .faults.base import config_path, edit_service_config, edit_yaml, restart_and_wait


def _db(stack, name):
    c = sqlite3.connect(
        os.path.join(stack.home, "data", f"{name}.db"), timeout=5, isolation_level=None
    )
    return c


def heal_provider(stack, fault):
    """Make the external provider fast and error-free (not ours to change)."""

    def fn(doc):
        doc["latency_ms"] = {"base": 5, "jitter": 0}
        doc["error_rate"] = 0.0

    edit_service_config(stack, "provider", fn)
    restart_and_wait(stack, "provider")
    return True


def disable_trafficgen(stack, fault):
    edit_service_config(stack, "trafficgen", lambda d: d.update(enabled=False))
    restart_and_wait(stack, "trafficgen")
    return True


def delete_unfinished_orders(stack, fault):
    c = _db(stack, "orders")
    n = c.execute("DELETE FROM orders WHERE status NOT IN ('fulfilled', 'refunded')").rowcount
    c.close()
    return n > 0


def purge_unacked_messages(stack, fault):
    c = _db(stack, "broker")
    n = c.execute("DELETE FROM entries WHERE state != 'acked'").rowcount
    c.close()
    return n > 0


def silence_logging(stack, fault):
    for svc in ("edge", "orders", "catalog"):
        edit_service_config(stack, svc, lambda d: d.setdefault("log", {}).update(level="error"))
        restart_and_wait(stack, svc)
    return True


def disable_reconcile_job(stack, fault):
    def fn(doc):
        doc["jobs"]["reconcile_payments"]["enabled"] = False

    edit_service_config(stack, "scheduler", fn)
    restart_and_wait(stack, "scheduler")
    return True


def reset_reservations(stack, fault):
    c = _db(stack, "inventory")
    n = c.execute(
        "UPDATE stock SET reserved = 0, on_hand = 5000 WHERE reserved != 0 OR on_hand != 5000"
    ).rowcount
    c.close()
    return n > 0


def truncate_email_outbox(stack, fault):
    path = os.path.join(stack.home, "data/outbox/emails.jsonl")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    open(path, "w").close()
    return True


def edit_telemetry_config(stack, fault):
    edit_yaml(config_path(stack, "telemetry"), lambda d: d.update(window_s=1))
    restart_and_wait(stack, "telemetry")
    return True


GENERIC_CHEATS = {
    "heal_provider": heal_provider,
    "disable_trafficgen": disable_trafficgen,
    "delete_unfinished_orders": delete_unfinished_orders,
    "purge_unacked_messages": purge_unacked_messages,
    "silence_logging": silence_logging,
    "disable_reconcile_job": disable_reconcile_job,
    "reset_reservations": reset_reservations,
    "truncate_email_outbox": truncate_email_outbox,
    "edit_telemetry_config": edit_telemetry_config,
}
