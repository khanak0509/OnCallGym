from northwind.services.telemetry import p_quantile


def test_p_quantile():
    buckets = [(10.0, 50), (100.0, 100), (float("inf"), 100)]
    assert p_quantile(buckets, 0.5) == 10.0
    assert 10.0 < p_quantile(buckets, 0.95) <= 100.0
    assert p_quantile([(10.0, 0), (float("inf"), 0)], 0.95) is None
