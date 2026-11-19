"""Unit tests for :class:`~llmcodec.interval.BitCode` and ``Interval.dyadic``.

These exercise the dyadic (binary) encoding in isolation -- no model, no
vocabulary -- so they are fast and deterministic.  The load-bearing property is
*containment*: the returned cell must lie entirely inside the source interval,
which is what makes the code safe to decode.
"""

import random
from fractions import Fraction
from math import ceil, log2

import pytest

from llmcodec.interval import BitCode, Interval


def test_bitcode_bits_value_width_and_cell():
  """The four accessors are consistent with the ``(n, k)`` fields."""
  code = BitCode(3, 3)
  assert code.bits == "011"
  assert code.value == Fraction(3, 8)
  assert code.width == Fraction(1, 8)
  assert code.cell.lower == Fraction(3, 8)
  assert code.cell.upper == Fraction(1, 2)


def test_bitcode_bits_are_zero_padded():
  """Leading zeros are significant: the length encodes the grid depth."""
  assert BitCode(5, 1).bits == "00001"
  assert BitCode(5, 1).value == Fraction(1, 32)


def test_bitcode_unit_interval_has_empty_bits():
  """``n == 0`` is the unit interval; it has no bits at all."""
  code = BitCode(0, 0)
  assert code.bits == ""
  assert code.value == Fraction(0, 1)
  assert code.width == Fraction(1, 1)
  assert code.cell == Interval(0, 1, 1)


def test_dyadic_worked_example():
  """The two-symbol ``6:4`` example: ``A`` then ``B`` gives ``[9/25, 3/5]``."""
  code = Interval(9, 15, 25).dyadic()
  assert code == BitCode(3, 3)
  assert code.bits == "011"


def test_dyadic_unit_interval():
  """The whole unit interval is contained in the single ``n = 0`` cell."""
  assert Interval(0, 1, 1).dyadic() == BitCode(0, 0)


def test_dyadic_rejects_zero_width():
  """A zero-width interval contains no cell, so the walk cannot terminate."""
  with pytest.raises(ValueError):
    Interval(1, 1, 2).dyadic()


def test_dyadic_cell_is_contained():
  """The returned cell lies inside the source interval -- the core guarantee."""
  for lo_num, hi_num, den in [
      (9, 15, 25),
      (3, 7, 10),
      (36, 60, 100),
      (1, 2, 3),
      (0, 1, 3),
      (0, 1, 1),
      (7, 8, 9),
  ]:
    interval = Interval(lo_num, hi_num, den)
    code = interval.dyadic()
    cell = code.cell
    assert interval.lower <= cell.lower
    assert cell.upper <= interval.upper


def test_dyadic_value_is_inside_interval():
  """The code's own value ``k / 2**n`` is a valid point of the interval."""
  interval = Interval(9, 15, 25)
  assert interval.contains(interval.dyadic().value)


def test_dyadic_cell_width_matches_two_power():
  """The cell width is exactly ``2**-n``."""
  code = Interval(9, 15, 25).dyadic()
  assert code.cell.width == Fraction(1, 2**code.n)


def test_dyadic_length_bound():
  """Containment forces ``n >= ceil(-log2(width))`` (the information bound)."""
  rng = random.Random(0)
  for _ in range(500):
    den = rng.randint(2, 5000)
    lo_num = rng.randint(0, den - 1)
    hi_num = rng.randint(lo_num + 1, den)
    interval = Interval(lo_num, hi_num, den)
    code = interval.dyadic()
    bound = ceil(-log2(float(interval.width)))
    assert code.n >= bound
    # The overshoot is tiny: the cell is at most a couple of bits too long.
    assert code.n - bound <= 2


def test_dyadic_is_minimal_grid_depth():
  """No coarser dyadic grid contains a cell inside the interval."""
  rng = random.Random(1)
  for _ in range(500):
    den = rng.randint(2, 5000)
    lo_num = rng.randint(0, den - 1)
    hi_num = rng.randint(lo_num + 1, den)
    interval = Interval(lo_num, hi_num, den)
    n = interval.dyadic().n
    if n == 0:
      continue
    # On the next-coarser grid no *proper* cell fits inside the interval.
    step = 1 << (n - 1)
    for k in range(0, step):
      cell = Interval(k, k + 1, step)
      assert not (interval.lower <= cell.lower and cell.upper <= interval.upper)


def test_dyadic_leftmost_cell_is_chosen():
  """Among contained cells of depth ``n`` the leftmost (smallest ``k``) wins."""
  interval = Interval(9, 15, 25)
  code = interval.dyadic()
  step = 1 << code.n
  # The cell to the left, ``[(k-1)/2^n, k/2^n]``, starts below the interval, so
  # it is not contained; ``k`` is therefore the leftmost contained index.
  assert Fraction(code.k - 1, step) < interval.lower


def test_interval_dyadic_roundtrip_through_find_slice():
  """The code's value decodes back to the symbol sequence via ``find_slice``.

  This is the interval-level analogue of the codec round-trip: encode two
  symbols with ``subdivide``, take the dyadic code of the final interval, and
  decode by walking ``find_slice`` from the code's value.
  """
  weights = (6, 4)
  interval = Interval.unit().subdivide(weights, 0).subdivide(weights, 1)
  start = Interval.unit()
  first = start.find_slice(weights, interval.dyadic().value)
  second = start.subdivide(weights, first).find_slice(weights, interval.dyadic().value)
  assert (first, second) == (0, 1)
