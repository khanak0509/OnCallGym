import importlib


def test_verifiers_v1_imports():
    vf = importlib.import_module("verifiers.v1")
    assert vf is not None
