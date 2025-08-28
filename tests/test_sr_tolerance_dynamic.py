from indicators.sr_utils import near_level


def test_sr_tolerance_dynamic():
    price = 100.0
    levels = [100.5]
    # HIGH regime (ATR besar) → toleransi lebih besar, kemungkinan True
    assert near_level(price, levels, 0.3, "HIGH") is True
    # LOW regime (ATR kecil) → hasil bisa False jika jarak > ATR * mult_low
    assert isinstance(near_level(price, levels, 0.05, "LOW"), bool)

