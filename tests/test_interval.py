from fractions import Fraction

import pytest

from llmcodec.interval import Interval


def test_unit_interval():
    u = Interval.unit()
    assert u.lower == 0
    assert u.upper == 1
    assert u.width == 1
    assert u.midpoint() == Fraction(1, 2)


def test_subdivide_equal_parts():
    u = Interval.unit()
    parts = [u.subdivide([1, 1, 1, 1], i) for i in range(4)]
    assert parts[0].lower == 0 and parts[0].upper == Fraction(1, 4)
    assert parts[1].lower == Fraction(1, 4) and parts[1].upper == Fraction(1, 2)
    assert parts[3].lower == Fraction(3, 4) and parts[3].upper == 1


def test_subdivide_unequal_parts():
    u = Interval.unit()
    # weights 1,3,6 -> boundaries 0.1, 0.4, 1.0
    assert u.subdivide([1, 3, 6], 0).upper == Fraction(1, 10)
    assert u.subdivide([1, 3, 6], 1).lower == Fraction(1, 10)
    assert u.subdivide([1, 3, 6], 1).upper == Fraction(4, 10)
    assert u.subdivide([1, 3, 6], 2).upper == 1


def test_find_slice_agrees_with_subdivide():
    u = Interval.unit()
    counts = [2, 5, 3, 7, 1]
    for i in range(len(counts)):
        sub = u.subdivide(counts, i)
        mid = sub.midpoint()
        assert u.find_slice(counts, mid) == i


def test_find_slice_handles_one():
    u = Interval.unit()
    counts = [1, 1, 1]
    assert u.find_slice(counts, Fraction(1)) == 2


def test_find_slice_rejects_outside():
    u = Interval.unit()
    with pytest.raises(ValueError):
        u.find_slice([1, 1], Fraction(2))


def test_nested_subdivision_is_exact():
    u = Interval.unit()
    iv = u.subdivide([1, 2, 3], 2)
    iv2 = iv.subdivide([1, 1], 0)
    assert iv2.lower == iv.lower
    assert iv2.upper == (iv.lower + iv.upper) / 2


def test_from_fraction_contains_value():
    for den in (10, 100, 1 << 16):
        for k in range(0, 11):
            value = Fraction(k, 10)
            iv = Interval.from_fraction(value, den)
            assert iv.contains(value)


def test_from_fraction_one_maps_to_last_unit():
    iv = Interval.from_fraction(Fraction(1), 16)
    assert iv.upper == 1
    assert iv.width == Fraction(1, 16)


def test_invalid_interval():
    with pytest.raises(ValueError):
        Interval(2, 1, 1)  # lo > hi
    with pytest.raises(ValueError):
        Interval(0, 1, 0)  # non-positive denominator
    with pytest.raises(ValueError):
        Interval(-1, 1, 2)  # negative lower
