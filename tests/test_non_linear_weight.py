from signal_engine.regime import non_linear_weight


def test_non_linear_weight_increase_on_high_metric():
    base = 0.1
    low = non_linear_weight(base, metric=0.01, thr=0.03)
    high = non_linear_weight(base, metric=0.10, thr=0.03)
    assert high > low

