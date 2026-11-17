import json
import os

import yaml

from northwind import schema
from northwind.lib.db import connect
from northwind.provision import provision
from northwind.topology import BLOCK_SIZE, SERVICES, ports


def test_ports_unique_within_block():
    p = ports(30000)
    assert len(set(p.values())) == len(p)
    assert all(30000 <= v < 30000 + BLOCK_SIZE for v in p.values())


def test_provision_is_deterministic(tmp_path):
    a = provision(str(tmp_path / "a"), str(tmp_path / "pa"), 30000, seed=7)
    b = provision(str(tmp_path / "b"), str(tmp_path / "pb"), 30000, seed=7)
    assert a["skus"] == b["skus"]
    va = json.load(open(tmp_path / "pa" / "verifier.json"))
    vb = json.load(open(tmp_path / "pb" / "verifier.json"))
    assert va["seed_hashes"] == vb["seed_hashes"]
    assert va["probe_users"] == vb["probe_users"]


def test_provision_layout(tmp_path):
    home, private = tmp_path / "h", tmp_path / "p"
    provision(str(home), str(private), 30000, seed=1)
    for name in SERVICES:
        cfg = yaml.safe_load(open(home / "etc" / f"{name}.yaml"))
        assert cfg["listen"]["port"] == 30000 + SERVICES[name][0]
        assert (home / "etc" / "env" / f"{name}.env").exists()
    # verifier secrets are not inside the deployment directory
    assert not any("verifier" in f for _, _, fs in os.walk(home) for f in fs)
    manifest = json.load(open(home / "data/seed/manifest.json"))
    probe = json.load(open(private / "verifier.json"))["probe_users"]
    assert not {u["email"] for u in probe} & {u["email"] for u in manifest["traffic_users"]}
    for db, required in schema.REQUIRED.items():
        c = connect(str(home / "data" / f"{db}.db"))
        assert schema.schema_version(c) >= required
        c.close()
