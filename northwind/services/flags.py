"""Central config and feature-flag service.

Flags live in etc/feature-flags.yaml:
  global: {flag: value}
  services: {orders: {flag: value}}
The file is re-read when its mtime changes, so edits apply without a restart.
"""

import os

import yaml

from northwind.lib.service import Service


def main():
    svc = Service("flags")
    path = svc.path(svc.config.get("flags_file", "etc/feature-flags.yaml"))
    state = {"mtime": None, "data": {}, "error": None}

    def load():
        try:
            mtime = os.stat(path).st_mtime
        except OSError as e:
            state["error"] = f"cannot stat {path}: {e.strerror}"
            return
        if mtime == state["mtime"]:
            return
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            state["error"] = f"cannot parse flags file: {e}"[:200]
            svc.log.error("flags file invalid, keeping previous values", error=state["error"])
            state["mtime"] = mtime
            return
        state.update(mtime=mtime, data=data, error=None)
        svc.log.info("flags loaded", services=len(data.get("services", {}) or {}))

    load()
    svc.ready_checks.append(lambda: (state["error"] is None, state["error"]))

    @svc.app.route("GET", "/flags/{service}")
    def get_flags(req):
        load()
        data = state["data"]
        merged = dict(data.get("global", {}) or {})
        merged.update((data.get("services", {}) or {}).get(req.params["service"], {}) or {})
        return 200, {"service": req.params["service"], "flags": merged}

    @svc.app.route("GET", "/flags")
    def all_flags(req):
        load()
        return 200, state["data"]

    svc.run()


if __name__ == "__main__":
    main()
